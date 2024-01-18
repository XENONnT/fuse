import uproot
import os
import awkward as ak
from tqdm.notebook import tqdm
import numba
import numpy as np
import matplotlib.pyplot as plt

def convert_energy_to_wavelength(root_data):
    """Converts pmthitEnergy into a wavelength. Modifies awkward array 
    in place. Adds a new column 'pmthitWavelength'.
    
    :param root_data: Awkward array containing the 'pmthitEnergy' field.
    """
    root_data
    h = 4.135667696 * 10**-15 # eV*s
    c = 2.998*10**8
    root_data['pmthitWavelength'] = (h*c / root_data['pmthitEnergy'])*10**9

def load_qe():
    """Function which oads QE json file.
    """
    from wfsim import load_config
    resources = load_config(dict(detector='XENONnT_neutron_veto'))
    
    n_channels_in_file = len(resources.nv_pmt_qe['nv_pmt_qe'])
    average_qunatum_efficiency = np.zeros(len(resources.nv_pmt_qe['nv_pmt_qe']['2000']), 
                                          dtype=[(('qe', 'PMT quantum efficiency.'), np.float64), 
                                                 (('wavelength', 'Wavelength at which QE was measured.'), np.float64)])
    for quantum_efficiencies_per_channel in resources.nv_pmt_qe['nv_pmt_qe'].values():
        average_qunatum_efficiency['qe'] += quantum_efficiencies_per_channel
    average_qunatum_efficiency['qe'] /= n_channels_in_file*100
    average_qunatum_efficiency['wavelength'] = resources.nv_pmt_qe['nv_pmt_qe_wavelength']
    return average_qunatum_efficiency
    
    
def apply_qe_and_ce(pmthits, 
                    average_quantum_efficiency, 
                    ce_efficiency):
    """Function which applies quantum efficiency and collection 
    efficiency to data. Retruns a new awkward array where all pmthits
    not surviving have been cut.
    """
    offsets = ak.num(pmthits['pmthitWavelength'])
    qe = np.interp(ak.ravel(pmthits['pmthitWavelength']), 
                       average_quantum_efficiency['wavelength'], 
                       average_quantum_efficiency['qe'],
                      )
    detection_efficiency = qe * ce_efficiency
    detection_efficiency = np.random.binomial(1, detection_efficiency)
    detection_efficiency = ak.unflatten(detection_efficiency, offsets)

    # Make a new awkward array
    #TODO: unify array reduction, see get_charge
    branches = ['pmthitEnergy', 'pmthitTime', 'pmthitID']
    _is_detected_in_nveto = (detection_efficiency == 1) 
    _is_detected_in_nveto = _is_detected_in_nveto & (pmthits['pmthitID'] >= 2000)
    _is_detected_in_nveto = _is_detected_in_nveto & (pmthits['pmthitID'] < 2200)
    result = pmthits[branches][_is_detected_in_nveto]
    _has_one_phd = ak.sum(_is_detected_in_nveto, axis=1) >= 1
    result = result[_has_one_phd]  # Only keep events with at least 1 phd
    result['eventid'] = pmthits[_has_one_phd]['eventid']
    return result

def get_charge(pmthits, sigma=0.3, threshold=0.6):
    """Function which draws some charge for every collected hit and 
    applies threshold. The charge is drawn from a simple normal 
    distribution. Adds a new column "pmthitCharge" to the awkward
    array. 
    
    :param pmthits: awkward.array containing pmthit information.
    :param sigma: Width of the charge distribution.
    :param threshold: DAQ threshold in units of PE.
    :returns: new awkward array for which DAQ threshold has been 
        applied.
    """
    offsets = ak.num(pmthits['pmthitTime'])
    n_hits = ak.sum(offsets)
    charge = np.random.normal(1, 0.3, n_hits)
    pmthits['pmthitCharge'] = ak.unflatten(charge, offsets)
    
    branches = ['pmthitEnergy', 'pmthitTime', 'pmthitID', 'pmthitCharge']
    result = pmthits[branches][pmthits['pmthitCharge'] >= threshold]
    _has_one_phd = ak.sum((pmthits['pmthitCharge'] >= threshold), axis=1) > 0
    result = result[_has_one_phd]
    result['eventid'] = pmthits[_has_one_phd]['eventid']
    return result


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

@numba.njit()
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

def convert_to_hitlets(pmthits, source_rate=160, max_time=int(3600*24*7)):
    """Function which converts GEANT4 output into pseudo hitlets. To 
    allow for pile-up draw event time stamps from a uniform distribution 
    mimicing the source rate.
    
    :param pmthits: awkward array storing information about the PMT hits.
    :param source_rate: Rate of the AmBe source in n/s.
    :param max_time: Maximum time allowed for events. Currently it is 
        set to one week to account for longer lived isotopes during 
        calibration.
    :returns: pseudo hitlets
    """
    n_event = len(pmthits['pmthitTime'])
    offsets = ak.num(pmthits['pmthitTime'])
    event_times = np.random.uniform(0, max_time, n_events)
    event_times *=10**9
    event_times = event_times.astype(np.int64) 
    # Add some unix time (20220526 19:44= otherwise sorting does not 
    # work
    event_times += np.int64(1653587233*10**9 )
    hit_times = pmthits['pmthitTime']*10**9 
    hit_times = ak.to_numpy(ak.ravel(hit_times)).astype(np.int64)
    hit_times += np.repeat(event_times, offsets)
    
    res = np.zeros(len(hit_times), dtype=pseudo_hitlet_dtype())
    res['time'] = hit_times
    res['length'] = 1
    res['dt'] = 1
    res['area'] = ak.to_numpy(ak.ravel(pmthits['pmthitCharge']))
    res['channel'] = ak.to_numpy(ak.ravel(pmthits['pmthitID']))
    
    
    #Cut all events which are more delayed than 2 max times:
    mask = (res['time'] - res['time'].min())/10**9 < max_time
    res = np.sort(res[mask], order=('time', 'channel'))
    res = stack_hitlets(res)
    return res

path_root = '/dali/lgrandi/pkavrigin/2022-08-01_TopUTGamma/'
root_files = np.sort(os.listdir(path_root))[1:8]
file_suffix = [f'{i+1}MeV' for i in range(len(root_files))]

for s,f in zip(file_suffix, root_files):
    print(s, f)
    root_file = uproot.open(os.path.join(path_root, f))
    root_data = root_file['events'].arrays(['eventid', 'pmthitTime', 'pmthitEnergy', 'pmthitID'])


    average_qe = load_qe()
    ce_efficiency=0.75
    convert_energy_to_wavelength(root_data)
    n_photons_wo_qe = ak.sum(ak.num(root_data['pmthitTime'], axis=1))
    n_events_wo_qe = len(root_data)
    root_data = apply_qe_and_ce(root_data, 
                               average_qe, 
                               ce_efficiency)
    n_photons_wo_charge = ak.sum(ak.num(root_data['pmthitTime'], axis=1))
    n_events_wo_charge = len(root_data)
    root_data = get_charge(root_data)
    n_photons = ak.sum(ak.num(root_data['pmthitTime'], axis=1))
    n_events = len(root_data)

    pseudo_hitlets = convert_to_hitlets(root_data)
    np.save(f'/dali/lgrandi/wenz/MC/nveto/energy_spectra/energy_calibration_sim/pseudo_hitlets_nT_mc_tutcw5p9m_{s}.npy', 
        pseudo_hitlets)
    
from cutax.cuts.nveto.nveto_nr_calibration import CutNVCenterTimeNR

st_pseudo = cutax.contexts.xenonnt_v8()
p_events = st_pseudo.get_single_plugin('040000', 'events_nv')
st_pseudo.register(CutNVCenterTimeNR)
p_cut_center_time = st_pseudo.get_single_plugin('040000', 'cut_nv_center_time_ambe')
p_event_positions = st_pseudo.get_single_plugin('040000', 'event_positions_nv')

pseudo_events_per_energy = dict()
pseudo_center_time_cut_per_energy = dict()
pseudo_event_positions_per_energy = dict()

for i in range(len(file_suffix)):
    file = f'pseudo_hitlets_nT_mc_tutcw5p9m_{file_suffix[i]}.npy'
    path = '/dali/lgrandi/wenz/MC/nveto/energy_spectra/energy_calibration_sim/'
    pseudo_hitlets = np.load(os.path.join(path, file))

    # Events are only sorted by time within each file! 
    # Multiple files overlap in time!
    pseudo_events = p_events.compute(pseudo_hitlets, 0, pseudo_hitlets['time'].max()+10)
    pseudo_center_time_cut = p_cut_center_time.cut_by(pseudo_events)
    pseudo_event_positions = p_event_positions.compute(pseudo_events, pseudo_hitlets)
    pseudo_events_per_energy[file_suffix[i]] = pseudo_events
    pseudo_center_time_cut_per_energy[file_suffix[i]] = pseudo_center_time_cut
    pseudo_event_positions_per_energy[file_suffix[i]] = pseudo_event_positions
