import json 
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas  as pd
import uproot
import numpy as np
import awkward as ak
import scipy.constants as const
#auxiliary functions
#---------------------------------------------------------------
# Retrieving PMT QE
def _init_quantum_efficiency():
    """Loads and returns channel specific quantum efficiency values for
    neutron-veto PMTs. Returns a dictionary containing interpolated 
    QE for each channel
    """
    with open('nveto_pmt_qe.json', 'r') as f:
    # Load the contents of the file as a Python object
        nv_pmt_qe = json.load(f)
    res = {}

    wavelength = nv_pmt_qe['nv_pmt_qe_wavelength']
 
    for channel, qe in nv_pmt_qe['nv_pmt_qe'].items():
        res[channel] = interp1d(wavelength,
                                np.array(qe)/100, # QE-values in file are in percent...
                                bounds_error=False,
                                fill_value=0
                               )
    return res

def energytowavelenght(E):
    Joules_to_eV=1.602*1e-19
    return 1e9*const.h*const.c/(E*Joules_to_eV)
#---------------------------------------------------------------

def reading_aux_files(SPE_file,acc_file):
    with open(SPE_file,'r') as f:
        SPE = json.loads(f.read())
    x,y = SPE['charge'],SPE['SPE_values']
    with open(acc_file,'r') as f:
        acceptance = json.loads(f.read())
    acc= acceptance['acceptance']
    return x, y,acc



#---------------------------------------------------------------
# Sampling charge from SPE spectra

def _get_charge(pmthits, spe_pdfs,x_data):
    """Generates charge for each PMT hit. 
    :param pmthits: akward array containing the field "pmthitID"
    :param spe_pdfs: SPE pdfs
    """
    offsets = ak.num(pmthits['pmthitTime'])

    pmthitids = ak.ravel(pmthits['pmthitID'])
    pmts, n_hits = np.unique(pmthitids, return_counts=True)
    charge = _draw_charge(pmts, n_hits, spe_pdfs,x_data)#cdfs)
    pmtcharge = _map_charge(pmthitids, charge)

    pmthits['pmthitCharge'] = ak.unflatten(pmtcharge, offsets)
    return pmthits



#Nuovo: con pdf invece di cdf
def _draw_charge(pmts, n_draws, spe_pdfs,x_data):
    n_pmts = 120
    res = np.zeros((n_pmts,  
                    np.max(n_draws)), np.float32)

    for ch, n_hits in zip(pmts, n_draws):
        
        spe_pdf_channel = spe_pdfs[ch-2000]
        x_data_channel = x_data[ch-2000]
        res[ch-2000][:n_hits] = np.random.choice(x_data_channel, 
                                           n_hits,
                                           p=spe_pdf_channel
         )
    return res


def _map_charge(pmthitids, charge):
    """Function which maps drawn charge for each PMT into the correct 
    order.
    """
    indicies = np.zeros(120, np.int64)
    res = np.zeros(len(pmthitids), np.float32)
    for ind, ch in enumerate(pmthitids):
        ch -= 2000
        _charge_in_hit = charge[ch][indicies[ch]] 
        if _charge_in_hit >=  0:
            res[ind] = _charge_in_hit
        indicies[ch] += 1
    return res
# Creating straxen hitlets and events

import strax
def pseudo_hitlet_dtype():
    dtype = []
    dtype += strax.time_dt_fields
    dtype += [(('Psuedo hitlet area', 'area'), np.float32),
              (('Psuedo hitlet channel', 'channel'), np.int16)
             ]
    return dtype

def stack_hitlets(pseudo_hitlets):
    """Funciton which stacks delta pulse hitlets if they occure in the 
    same channel at the same time.
    """
    res = np.zeros(len(pseudo_hitlets), dtype=pseudo_hitlet_dtype())
    res['length'] = 1
    res['dt'] = 1
    return _stack_hitlets(pseudo_hitlets, res)

def _stack_hitlets(pseudo_hitlets, res):
    offset = -1
    current_time = -1
    current_channel = -1
    for hit in pseudo_hitlets:
        _is_same_delta_hit = hit['time'] == current_time and hit['channel'] == current_channel
        if _is_same_delta_hit:
            res[offset]['area'] += hit['area']
        else:
            offset += 1
            res[offset]['time'] = hit['time']
            res[offset]['channel'] = hit['channel']
            res[offset]['area'] = hit['area']
            current_time = hit['time']
            current_channel = hit['channel']
    return res[:(offset+1)]   

def convert_to_hitlets(
                       pmthits, 
                       source_rate=156,
                       max_time=int(3600*24*7),
                      ):
    """Function which converts GEANT4 output into pseudo hitlets. To 
    allow for pile-up draw event time stamps from a uniform distribution 
    mimicing the source rate.
    :param pmthits: awkward array storing information about the PMT hits.
    :param source_rate: Rate of the AmBe source in n/s.
    :param max_time: Maximum time allowed for events. Currently it is 
        set to one week to account for longer lived isotopes during 
        calibration.
    :returns: pseudo hitlets, mc_truth_extra_information
    """
    n_events = len(pmthits['pmthitTime'])
    offsets = ak.num(pmthits['pmthitTime'])
    event_times = np.random.uniform(0, n_events/source_rate, n_events)
    event_times *=10**9
    event_times = event_times.astype(np.int64) 
    # Add some unix time (20220526 19:44= otherwise sorting does not 
    # work
    event_times += np.int64(1653587233*10**9 )
    pmthits['event_start_times'] = event_times
    hit_times = pmthits['pmthitTime']*10**9 


    hit_times = ak.to_numpy(ak.ravel(hit_times)).astype(np.int64)
    event_times = np.repeat(event_times, offsets)
    hit_times += event_times

    res = np.zeros(len(hit_times), dtype=pseudo_hitlet_dtype())
    res['time'] = hit_times
    res['length'] = 1
    res['dt'] = 1
    res['area'] = ak.to_numpy(ak.ravel(pmthits['pmthitCharge']))
    res['channel'] = ak.to_numpy(ak.ravel(pmthits['pmthitID']))

    #mc_truth = get_mc_truth(pmthits, res)


    #Cut all hits which are more delayed than max time after the 
    # last event:
    # TODO: This is not 100 % correct as we need to do this actually
    # for ecah individual event....
    mask = (res['time'] - event_times.max())/10**9 < max_time
    res = np.sort(res[mask], order=('time', 'channel'))
    print(len(res))
    res = stack_hitlets(res)
    print(len(res))
    """
    if len(mc_truth):
        mc_truth = np.sort(mc_truth[mask], order=('time', 'channel'))
        mc_truth = stack_truth(mc_truth) """
    return res


import cutax
def convert_to_events(pseudo_hitlets):#WHY DON'T USE JUST CUTAX PLUGIN?????

    st_pseudo = cutax.contexts.xenonnt_online()
    p_events = st_pseudo.get_single_plugin('0', 'events_nv')
    # Events are only sorted by time within each file! 
    # Multiple files overlap in time!
    pseudo_events = []
    pseudo_event_positions = []
    pseudo_center_time_cut = []
    last_time_seen = None
    _pseudo_hitlets=pseudo_hitlets
    if last_time_seen:
        time_offset = last_time_seen - _pseudo_hitlets['time'].min()
        _pseudo_hitlets['time'] += time_offset + 2000
    
    last_time_seen = _pseudo_hitlets['time'].max()
    _pseudo_events = p_events.compute(_pseudo_hitlets, 0, _pseudo_hitlets['time'].max()+10)
    pseudo_events.append(_pseudo_events)

    pseudo_events = np.concatenate(pseudo_events)
    return pseudo_events
class Hitlet_nv(object):
    def __init__(self, path, g4_file):
        self.x_interp, self.pdf_per_channel, self.acceptance = reading_aux_files(path+'SPE_SR1_test_fuse.json',path+'acceptance_SR0_test_fuse.json')
        self.g4_file = g4_file
        self.quantum_efficiency = _init_quantum_efficiency()
    def nv_hitlets(self, number):
        data=[]
        for i in range(0,number):
            try:
                file_root = g4_file +str(i)+'.root'
                #print("Opening File :",file_root)
                root_file = uproot.open(file_root)
                root_data = root_file['events'].arrays(['eventid', 'pmthitTime', 'pmthitEnergy', 'pmthitID'])
                data.append(root_data)
            except:
                continue
    
        # trasformo data in awkward 
        pmthits = ak.concatenate(data)
        pmthits['pmthitWavelength']=energytowavelenght(pmthits['pmthitEnergy'])
        n_photons_wo_qe = ak.sum(ak.num(pmthits['pmthitTime'], axis=1))
        n_events_wo_qe = len(pmthits)
        #Prima di applicare la QE ho :
        print(f'Number of events w/o QE: {n_events_wo_qe}')
        print(f'Number of photons w/o QE: {n_photons_wo_qe}')

        #QE and CE Manipulation (ma anche Acceptance) 
        # Appiattisco l'array e applico i tagli (per velocizzare). Dopo uso offset per ricostruire la struttura originaria.
        offsets = ak.num(pmthits['pmthitWavelength'])
        pmthitchannel = ak.ravel(pmthits['pmthitID'])
        pmthitwavelengths = ak.ravel(pmthits['pmthitWavelength'])
        buffer = np.zeros(len(pmthitwavelengths), np.int16)

        for ch in range(2000, 2120):
            mask = pmthitchannel == ch
            qe = quantum_efficiency[str(ch)](pmthitwavelengths[mask])
            p_detected = np.random.binomial(1, qe*ce*spe_acc[ch-2000])# Acceptance is applied here, this mean cutting a trivial percentage of the pes, not be more correct to cut only (1-acc)% of lowest charges 
            buffer[mask] = p_detected
    
        #Buffer -> detection probability 0-1
        pmthits['pmthitCharge'] = ak.unflatten(buffer, offsets)
        _photons_detected = pmthits['pmthitCharge'] == 1
        #Tolgo gli hit che non sono stati rivelati
        pmthits["pmthitCharge"]=pmthits["pmthitCharge"][_photons_detected]
        pmthits["pmthitWavelength"]=pmthits["pmthitWavelength"][_photons_detected]
        pmthits["pmthitID"]=pmthits["pmthitID"][_photons_detected]
        pmthits["pmthitTime"]=pmthits["pmthitTime"][_photons_detected]

        n_photons_w_qe = ak.sum(ak.num(pmthits['pmthitTime'], axis=1))  #Calcolo il numero di hits sopravvissuti (numero di entries per ogni riga, sommati)
        n_events_w_qe = len(pmthits) #Calcolo il numero di eventi (numero di righe)
        print(f'Number of events w/ QE: {n_events_w_qe}')
        print(f'Number of photons w/ QE: {n_photons_w_qe}')

        #Campionamento della carica
        #Charge sampling.
        #pmthits = _get_charge(pmthits,y_cdfs) 
        pmthits = _get_charge(pmthits,pdf_per_channel,x_interp)
        n_photons_recorded = ak.sum(ak.num(pmthits['pmthitTime'][pmthits['pmthitCharge'] > 0], axis=1))
        n_events_recorded = len(pmthits)
        print(f'Number of events recorded: {n_events_recorded}')
        print(f'Number of photons recorded: {n_photons_recorded}') 
        # Carica a 0 registrata in alcuni hits.

        #*----------------Creo Hitlets e Events --------------*
        hitlets = convert_to_hitlets(pmthits)
        return hitlets
