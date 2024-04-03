import strax
import straxen
import numpy as np
import logging

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.neutron_veto.nvhitlets')

from ...common import FUSE_PLUGIN_TIMEOUT

import uproot as rt
import json
import scipy as scp
from scipy import interpolate
import scipy.constants as const
import awkward as ak
import pandas as pd
from sklearn.cluster import DBSCAN
import random as rd
from tqdm import tqdm
import tqdm.notebook as tq
import time
import cutax

##--------------------------------------------COMMENTS------------------------------------------------------------##

#This hitlet simulator is an extension of the work of Diego Ramirez, Daniel Wenz, Andrea Mancuso and Pavel Kavrigin.
#Functions of SPE charge sampling are in the PDF function based in the calibrations of nVeto done by Andrea Mancuso. Daniel Wenz provides a code that takes into account this functions to sample the charge, where this code is based. This leads to SPE a PDF function to sample charge in the hitlet simulator.
#We use the Quantum efficiencies QE for each PMT as a wavelength function for nVeto provided by Andrea Mancuso.

#WIKI NOTES:
#1.The details of this hitlet simulation leading to the main functions 'G4_nveto_hitlets' and 'G4_mveto_hitlets' for muon and neutron Vetos:
#https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:layos:snhitlet_from_geant4_output
#In the note mentioned above we test several approaches of charge sampling, for that we can use the function 'G4_to_Veto_hitlets_comparison'.

#2.THE LAST UPDATE, leading to the actual structure of the code is here :https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:layos:vetohitlets_v2 , this one is ongoing and we expect to test with SR1 AmBe runs.
#In the note mentioned above we test several approaches of charge sampling, for that we can use the function 'G4_to_Veto_hitlets_comparison'.

    
#--------------------------------------------------------HITLETS AUXILIAR FUNCTIONS--------------------------------------------------------------------#
def flat_list(l):
    return np.array([item for sublist in l for item in sublist ])

def energytowavelenght(E):
    Joules_to_eV=1.602*1e-19
    return 1e9*const.h*const.c/(E*Joules_to_eV)

#SPE parameters: ID, pe, SPE, acceptance, threshold_pe(based on acceptance)
def SPE_parameters(json_file_spe_model,file_spe_acceptance):
    with open(json_file_spe_model,'r') as f:
        data_spe = json.loads(f.read())
    with open(file_spe_acceptance) as f:
        acceptance = json.load(f)
    SPE_ch= pd.DataFrame(columns=['pmtID','pe','SPE','acceptance','threshold_pe'])
    SPE_ch['pmtID'],SPE_ch['pe'], SPE_ch['SPE'],SPE_ch['acceptance']=data_spe['pmtID'],data_spe['charge'],data_spe['SPE_values'],acceptance['acceptance']
    acceptance_ch= [threshold_acc(SPE_ch,i) for i in np.arange(2000,2120)]
    SPE_ch['threshold_pe']=acceptance_ch
    return SPE_ch
    
def threshold_acc(SPE_df, ID):
    SPE_ID=pd.DataFrame()
    SPE_ID['cumulative']=np.cumsum(SPE_df[SPE_df.pmtID==ID].SPE.values[0])
    SPE_ID['charges']= SPE_df[SPE_df.pmtID==ID].pe.values[0]
    accep= SPE_df[SPE_df.pmtID==ID].acceptance.values[0]
    threshold= min(SPE_ID[SPE_ID.cumulative>=(1-accep)].charges.values)
    return threshold
    
#To get nVeto plugin, it should be a best way to do that...
st = cutax.contexts.xenonnt_online()
strax_nv = st.get_single_plugin('0', 'events_nv')



#Quantum efficiency
def QE_nVeto(Q_E_nveto_file):
    with open(Q_E_nveto_file,'r') as f:
        data = json.loads(f.read())
    QE_array_n=[]
    #nVeto
    for i in np.arange(2000,2120):
        QE_array_n.append(interpolate.interp1d(data['nv_pmt_qe_wavelength'],data['nv_pmt_qe'][str(i)], bounds_error=False,fill_value=0))
    #Watertank_QE
    pmt_id= list(np.arange(2000,2120))
    QE_array=QE_array_n
    pd_dict= {"pmt_id":pmt_id,"QE":QE_array}
    return pd_dict

#Cluster for stacket hitlets
def channel_cluster_nv(t):
    db_cluster = DBSCAN(eps=8, min_samples=1)#As a preliminar value we fix distance between two photons arriving in the same pmt 8ns
    t_val=np.array(t)
    clusters=np.array(db_cluster.fit_predict(t_val.reshape(-1, 1)))
    return clusters
    
#Function to use in pivot_table module (clearly something we can optimize)
def recover_value(x):
    m_size= np.array(x).size
    if m_size==1:
        ret=x
    elif  m_size>1:
        ret=list(x)
    return ret
    
    
def type_pri_evt(x):
    if (type(x)==np.float32) or (type(x)==np.float64) or (type(x)==float):
        ret=x
    elif type(x)==list:
        ret=x[0]              
    return ret
#For building independent time hitlets to compute into events or hitlets time from a source (ex: AmBe)
def time_hitlets_nv(time,ids,freq):
    if type(time)==np.float64:
        ret=time
    elif type(time)==list:
        ret=min(time)               
    return ret + freq*ids
    
#Fonction to transform a hitlet dataframe output into 'hitlets_nv' ndarray
#Fonction to transform a hitlet dataframe output into 'hitlets_nv' ndarray
dtype = [('area', np.float64),
             ('amplitude', np.float64),
             ('time_amplitude', np.int16),
             ('entropy', np.float64),
             ('range_50p_area', np.float64),
             ('range_80p_area', np.float64),
             ('left_area', np.float64),
             ('low_left_area', np.float64),
             ('range_hdr_50p_area', np.float64),
             ('range_hdr_80p_area', np.float64),
             ('left_hdr', np.float64),
             ('low_left_hdr', np.float64),
             ('fwhm', np.float64),
             ('left', np.float64),
             ('fwtm', np.float64),
             ('low_left', np.float64),
            ]
dtype = dtype + strax.interval_dtype #-> Time, length, dt, channel
#A fuse plugin is a python class that inherits from strax.Plugin
#As naming convention we use CamelCase for the class name
def df_to_hit_array(data):
    result = np.zeros(len(data), dtype = dtype)
    result['time'] = data.time.values
    result['length']= np.array([1.]*len(data))
    result['dt'] =np.array([10.]*len(data))
    result['channel']=data.pmthitID.values
    result['area']= data.pe_area.values
    result=strax.sort_by_time(result)
    return result
    
#A fuse plugin is a python class that inherits from strax.Plugin
#As naming convention we use CamelCase for the class name
class NeutronVetoHitlets(strax.Plugin):
    #Each plugin has a version number
    #If the version number changes, fuse will know that it need to re-simulate the data
    __version__ = "0.0.1"
    
    #You need to tell fuse and strax what the plugin needs as input
    #In this case we need nv_pmthits
    depends_on = ("nv_pmthits")
    
    #You need to tell fuse and strax what the plugin provides as output
    #In this case we provide nv_hitlets
    #You can later use st.make(run_number, "nv_hitlets") to run the simulation
    provides = "nv_hitlets"
    
    #You need to tell fuse and strax what the data looks like
    #Data of the same data_kind can be combined via "horizontal" concatenation and need 
    #to have the same output length. 
    data_kind = 'nv_hitlets'
    
    #You also need to tell strax what columns the data has
    #A column needs a name and a numpy data type. I set everything to float64 here, we can reduce it later
    #I used the columns described here:
    # https://github.com/XENONnT/fuse/blob/f777d9c281d9e046c2f322a30b462a9b7dd8ee00/test_nv/Hitlet_nv_fuse.py#L114
    
    #We need to disable automatic rechunking for fuse plugins
    #As fuse is going from "leightweigt" data to "heavy" data,
    #automatic rechunking can lead to problems in later plugins
    rechunk_on_save = False

    #We need to specify when we want to save the data
    save_when = strax.SaveWhen.TARGET

    #strax uses a rather short timeout, lets increase it as 
    #some of the fuse simulation steps can take a while
    input_timeout = FUSE_PLUGIN_TIMEOUT
    
    #We need to tell strax what config options the plugin needs
    #We will use the great URLConfigs that are a part of straxen
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    deterministic_seed = straxen.URLConfig(
        default=True, type=bool,
        help='Set the random seed from lineage and run_id, or pull the seed from the OS.',
    )
    def __init__(self):
        self.path = '/home/layos/env_starter/fuse/private_nt_aux_files/sim_files/'#Have to put here the correct paths....
        self.QE_value = QE_nVeto(self.path+'nveto_pmt_qe.json')
        self.SPE_nVeto = SPE_parameters(self.path+'SPE_'+'SR1'+'_test_fuse.json',self.path+'acceptance_SR0_test_fuse.json')
        self.dtype=dtype
    #Get Quantum efficiency
    def QE_E(self,E,ID):
        WL= energytowavelenght(E)
        ind=ID-2000
        qe=self.QE_value['QE'][ind](WL)
        return qe
        
    def get_acceptance(self,ID):
        ind=ID-2000
        acc= self.SPE_nVeto.acceptance.values[ind]
        return acc
    #Get acceptance threshold
    def get_threshold_acc(self,ID):
        ind=ID-2000
        threshold= self.SPE_nVeto.threshold_pe.values[ind]
        return threshold
    
    #Sampling charge from SPE  
    def pe_charge_N(self,pmt_id):
        SPE_channel= self.SPE_nVeto[self.SPE_nVeto.pmtID==pmt_id]
        charge=rd.choices(SPE_channel.pe.values[0],SPE_channel.SPE.values[0],k=1)[0]    
        return charge

    #--------------------------- Hitlet function ------------------------------------------------------------#

    def _nv_hitlets(self,pmthits, CE_Scaling=0.75):
        
    #-------------------------------------------------Arguments---------------------------------------------------#
    
    #e_1,e_2= range of entries from root file
    #root_keys= for primaries or flags that you to keep if you don't wont any just root_keys=[]
    #QE_Scaling corrrespond to collection efficiency, no study has been done on the CE of muon Veto we use a default value close to the nVeto see https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:mancuso:hitletsimulator:collection_efficiency
    #period : this could be related to the rate of a source, or the rate for a time bin if we reconstruct an spectrum. If no source 1e9 is the default value (see Comments)
    #csv : if you want to save the hitlet when you run several chuncks
    #Name : to name this csv file
    #Isotopes : if we want to keep information about some activated isotopes (True) if False it cut these events.(See comments) 
        
    #----------------------------------------------Commments-----------------------------------------------------------------#:
    #1.There is no application of a threshold per channel based on the acceptation by default, but we keep the value in the data frame for each pmt, and one can do manually. This is in order to not condition the sampling, and compare it with the data with different cuts.
    #.2. The period is set by default at 1s to care about no pyle up or merge of hitlets if one want to do an analysis for rare events (independent non sourced ones). If we simulate a calibration or a constant flux this value has to be changed to real rate one.  
    #.3. The way the code works, if information of Isotopes is keeped we cannot recover G4 primary parameters after building 'event_nv'. We have to think in this case a different way to do it.
    #4.Stacked hitlets: using DBSCAN is maybe not required (as it makes the hitlet slower) and an easier approach can be used (some ideas???)
    
    
        #0.---------------Load GEANT output-------------------#
    


        #1. First step PHOTON to first dinode
        pmthits=ak.Array(pmthits)
        mask=pmthits['pmthitID']>=2000
        pmthits=pmthits[mask]
        print("Applying QE and CE")
        qe = 1e-2*np.vectorize(self.QE_E)(pmthits['pmthitEnergy'],pmthits['pmthitID'])#Applying Quantum efficiency for each pmt
        qe *= CE_Scaling #Applying collection efficiency
        qe = qe*np.vectorize(self.get_acceptance)(pmthits['pmthitID'])#Applying acceptance: for the approach in which SPE PDF has already applied a threshold for low charges
        pe = np.array([np.random.binomial(1, j, 1)[0] for j in qe])
        print("Loading hit survive")
        maks_qe = pe>0
        pmthits=pmthits[maks_qe]
        
        #2. Stacked hitlets: this correspond to hitlets in the same pmt with a time difference below some estimated time response of the Channel (8 ns, i.e. 4 samples).        
        print('Looking for stacket hitlets')
        #Here we set times related to the first hit, we only use that for stacket hitlets
        times=[]
        for i in (np.unique(pmthits.evtid)):
            mask= pmthits.evtid==i
            pmthits_evt=pmthits[mask]
            cluster_times_ns = pmthits_evt.pmthitTime - min(pmthits_evt.pmthitTime)
            times.append(cluster_times_ns)
        pmthits['cluster_times_ns'] = flat_list(times)
        pmthits['pe_area'] = np.vectorize(self.pe_charge_N)(pmthits['pmthitID'])
        column = ['evtid','pmthitID']
        df=ak.to_dataframe(pmthits)
        #return pmthits
        df_c = pd.pivot_table(df, index = column, aggfunc={'cluster_times_ns': lambda x : list(x)})
        clusters = [channel_cluster_nv(i) for i in df_c.cluster_times_ns.values]
        df_sorted = df.sort_values(['evtid','pmthitID'], ascending = [True,True])
        df_sorted.insert(len(df.columns), 'labels',flat_list(clusters))
        #We track the stacket hitlets using the labels and save the min time and max endtime

        #3.Creating hitlet dataframe
        #We track the stacket hitlets using the labels and save the min time and max endtime
        #***(Dataframe format is because group by is not already implemented into a ankward array)
        col_index = ['evtid','pmthitID','labels']
        mask = np.isin(df_sorted.columns.values, col_index, invert=True)
        col_hitlet=df_sorted.columns.values[mask]
        arg_dicio = dict.fromkeys(col_hitlet)
        for i in col_hitlet:
            if i == 'pe_area':
                arg_dicio[i]= np.sum
            elif i == 'time':
                arg_dicio[i]= np.min
            elif i == 'endtime':
                arg_dicio[i]= np.max
            else:
                arg_dicio[i]= lambda x: recover_value(x)  
        table_stack = pd.pivot_table(df_sorted, index = col_index, aggfunc=arg_dicio)
        print('Creating hitlet dataframe')
        hitlet_df = pd.DataFrame(columns=col_index )
        for j in tq.tqdm(range(len(col_index))): 
            hitlet_df[col_index[j]] = [table_stack.index[i][j] for i in range(len(table_stack))]
        for i in tq.tqdm(col_hitlet):
            hitlet_df.insert(len(hitlet_df.columns), i , table_stack[i].values)
        print('getting time of hitlets_nv')  
        
        #4.Transforming to hitlets_nv format
        print("Creating hitlets_nv")
        hitlets_nv = df_to_hit_array(hitlet_df)
        return hitlet_df
        
    def setup(self):

        #All plugins can report problmes or debug information via the logging feature
        #You can set the log level via the debug config option. 
        #WARNING messages are always shown whild DEBUG messages are only shown if debug is True
        if self.debug:
            log.setLevel('DEBUG')
            log.debug(f"Running NeutronVetoHitlets version {self.__version__} in debug mode")
        else: 
            log.setLevel('WARNING')

        #Many plugins need to generate random numbers for simulation the corresponding physics process
        #In fuse we want to make sure that the simulation is reproducible.
        #Therefore we have the default setting of deterministic_seed = True
        #In this case the random seed is generated from the run_id and the lineage
        #The lineage includes all plugins and their verions that are connected to the input of the 
        #current plugin as well as all tracked config options and the strax version. 
        #The run_id is a user input. More on the deterministic seed can be found in 
        #a dedicated notebook.
        #Please make sure that you use the random number generator self.rng when you need random numbers
        #later in the plugin. 
        if self.deterministic_seed:
            hash_string = strax.deterministic_hash((self.run_id, self.lineage))
            seed = int(hash_string.encode().hex(), 16)
            self.rng = np.random.default_rng(seed = seed)
            log.debug(f"Generating random numbers from seed {seed}")
        else: 
            self.rng = np.random.default_rng()
            log.debug(f"Generating random numbers with seed pulled from OS")


    #The compute method is the heart of the plugin. It is executed for each chunk of input data and 
    #must produce data in the format specified in the self.dtype variable.
    def compute(self, nv_pmthits):
        #Make sure your plugin can handle empty inputs 
        if len(nv_pmthits) == 0:
            return np.zeros(0, self.dtype)
        
        #All your NV goes here
        result = df_to_hit_array(self._nv_hitlets(nv_pmthits))
        return result





