import uproot as rt
import numpy as np
import json
import scipy as scp
from scipy import interpolate
import scipy.constants as const
import awkward as ak
import pandas as pd
import straxen
import strax
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

#2.THE LAST UPDATE, leading to the actual structure of the code is here : , this one is ongoing and we expect to test with SR1 AmBe runs.
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
def df_to_hit_array(data):
    hitlet_list=['time','length','dt','channel','area','amplitude','time_amplitude','entropy','range_50p_area','range_80p_area','left_area','low_left_area','range_hdr_50p_area','range_hdr_80p_area','left_hdr','low_left_hdr','fwhm','left','fwtm','low_left']
    df_hit = pd.DataFrame(columns=hitlet_list,index=None)
    for i in hitlet_list:
        df_hit[i] = np.array([0]*len(data)).astype('float32')
    df_hit=df_hit.drop(columns=['time','length','dt','channel','area'])
    df_hit.insert(0,'time',data.time.values.astype('int64'))
    df_hit.insert(1,'length', np.array([1.]*len(data)).astype('int32'))
    df_hit.insert(2,'dt', np.array([10.]*len(data)).astype('int16'))
    df_hit.insert(3,'channel',data.pmthitID.values.astype('int16'))
    df_hit.insert(4,'area',data.pe_area.values.astype('float32'))
    df_hit=df_hit.astype({'time_amplitude': 'int16'})    
    hitlets= strax.sort_by_time(df_hit.to_records())    
    return hitlets



#--------------------------------------------------------Water tank HITLET CLASS--------------------------------------------------------------------#


class Hitlet_nv(object):
    def __init__(self, path, g4_file):
        self.path = path
        self.QE_value = QE_nVeto(path+'nveto_pmt_qe.json')
        self.SPE_nVeto = SPE_parameters(path+'SPE_'+'SR1'+'_test_fuse.json',path+'acceptance_SR0_test_fuse.json')
        self.g4_file = g4_file
        
    #Get Quantum efficiency
    def QE_E(self,E,ID):
        WL= energytowavelenght(E)
        ind=ID-2000
        qe=self.QE_value['QE'][ind](WL)
        return qe
    
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

    def nv_hitlets(self, e_1, e_2, root_keys=['e_pri','xp_pri','yp_pri','zp_pri'], CE_Scaling=0.75, period=1e9,csv=False, Name=None, Isotopes=False):
        
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
    
        print("Opening File :", self.g4_file)
        root_file = rt.open(self.g4_file)
        #We chose the pmt data frame that has all recorded values for Vetos PMTs from GEANT4 simulation
        keys=['eventid','pmthitEnergy','pmthitTime','pmthitID'] + root_keys #Warning: if you select 'type_pri' pmt information will be lost and only you '_pri' keys corresponding to primaries
        df=ak.to_pandas(root_file['events'].arrays(keys, library='ak',entry_start=e_1, entry_stop=e_2))
        #We choose only events produced in nVeto and recorded by nVeto pmts (this is useful if a G4 sim is confined in all Water volume)
        event_list=np.unique(df.eventid.values)
        n_Veto_id= np.unique(df[(abs(df.xp_pri)<2000) & (abs(df.yp_pri)<2000) & (df.zp_pri<600) & (df.zp_pri>-2000)].eventid.values)
        df=df[(df.pmthitID>=2000)]
        df=df[df.eventid.isin(n_Veto_id)]

        #1. First step PHOTON to first dinode
               
        print("Applying QE and CE")
        qe = 1e-2*np.vectorize(self.QE_E)(df.pmthitEnergy.values,df.pmthitID.values)#Applying Quantum efficiency for each pmt
        qe *= CE_Scaling #Applying collection efficiency
        pe = np.array([np.random.binomial(1, j, 1)[0] for j in qe])
        print("Loading hit survive")
        df.insert(len(df.columns),'pe',pe)
        df=df[df.pe>0]
        df=df.drop(columns=['pe'])
        #Getting the acceptance threshold:
        df['threshold_acc']= np.vectorize(self.get_threshold_acc)(df.pmthitID)
        #Cutting long events
        if Isotopes==False:
            times=[]
            print("Cutting Long events or Isotopes")
            for i in (np.unique(df.eventid)):
                df_time=df[df.eventid==i]
                time_ns= df_time.pmthitTime.values*1e9
                cluster_times_ns = time_ns - min(time_ns)
                times.append(cluster_times_ns)
            df.insert(len(df.columns),'cluster_times_ns',flat_list(times))
            df=df[(df.cluster_times_ns<1e9) & (df.pmthitTime<1e10)]
        elif Isotopes==True:
            print('Recording Isotopes')
            
        #2. Stacked hitlets: this correspond to hitlets in the same pmt with a time difference below some estimated time response of the Channel (8 ns, i.e. 4 samples).
        print('Looking for stacket hitlets')
        charges = np.vectorize(self.pe_charge_N)(df.pmthitID.values)
        df.insert(len(df.columns),'pe_area', charges)
        column = ['eventid','pmthitID']
        df_c = pd.pivot_table(df,index = column,aggfunc={'cluster_times_ns': lambda x: list(x)})
        cc = [channel_cluster_nv(i) for i in (df_c.cluster_times_ns.values)]
        df = df.sort_values(['eventid','pmthitID'], ascending = [True,True])
        df.insert(len(df.columns), 'clusters_c', flat_list(cc))
        
        #3.Creating hitlet dataframe
        
        col_index = ['eventid','pmthitID','clusters_c']
        mask = np.isin(df.columns.values, col_index, invert=True)
        col_hitlet=df.columns.values[mask]
        arg_dicio = dict.fromkeys(col_hitlet)
        for i in col_hitlet:
            if i == 'pe_area':
                arg_dicio[i]= np.sum
            elif i in(root_keys):#Attention if you use flags cause you recover the first value....
                arg_dicio[i]= lambda x: type_pri_evt(x)
            arg_dicio[i]= lambda x: recover_value(x)  
        df_ch = pd.pivot_table(df, index = col_index, aggfunc=arg_dicio)
        print('Creating hitlet dataframe')
        hitlet_df = pd.DataFrame(columns=col_index )
        for j in tq.tqdm(range(len(col_index))): 
            hitlet_df[col_index[j]] = [df_ch.index[i][j] for i in range(len(df_ch))]
        for i in tq.tqdm(col_hitlet):
            if i == 'pmthitTime':
                hitlet_df.insert(len(hitlet_df.columns),'pmthitTime',np.vectorize(time_hitlets_nv)(df_ch.pmthitTime.values,hitlet_df.eventid.values,0))#To keep GEANT4 time but only the first one of the eventual stacket hitlet
            else:
                hitlet_df.insert(len(hitlet_df.columns),i,np.vectorize(type_pri_evt)(df_ch[i].values))
        print('getting time of hitlets_nv')
        times_spe_ns = np.vectorize(time_hitlets_nv)(df_ch.cluster_times_ns.values,hitlet_df.eventid.values,period)#This will be hitlet_nv time related to first photon
        hitlet_df.insert(len(hitlet_df.columns), 'time', times_spe_ns)
        
        #4.Transforming to hitlets_nv format
        print("Creating hitlets_nv")
        hitlets_nv = df_to_hit_array(hitlet_df)
        return hitlet_df, hitlets_nv, root_keys

    
#This function takes the necessary to run the SN_events_E, but is not need to run it more than once and save output in .csv file.   
def hitlet_nv_chunks(path_root, Number, e_min, e_max, path_aux_nv, periods=1e9, CE=0.75, hitlets_keys=['e_pri','xp_pri','yp_pri','zp_pri'], Name='/home/layos/'):
    #Same arguments of  nv_hitlets 
    #Number = number of chuncks
    data_nv=[]
    data_nv_df=[]
    e_pri_df=[]
    hitlets=[]
    count_ids=0
    for i in tq.tqdm(range(0,Number)):
        try:
            g4file=path_root+str(i)+'.root'
            root_file = rt.open(g4file)
            keys=['eventid','pmthitEnergy','pmthitTime','pmthitID'] + hitlets_keys
            #We chose the pmt data frame that has all recorded values for Vetos PMTs from GEANT4 simulation
            df=ak.to_pandas(root_file['events'].arrays(keys, library='ak', entry_start=e_min, entry_stop=e_max))
            event_list=np.unique(df.eventid.values)
            n_Veto_id= np.unique(df[(abs(df.xp_pri)<2000) & (abs(df.yp_pri)<2000) & (df.zp_pri<600) & (df.zp_pri>-2000)].eventid.values)
            nVeto_hitlets= Hitlet_nv(path_aux_nv,g4file)
            hitlets_nv= nVeto_hitlets.nv_hitlets(e_min, e_max, hitlets_keys, CE_Scaling = CE, period=periods)
            hitlets_df=hitlets_nv[0].assign(eventid = hitlets_nv[0].eventid.values + count_ids)
            hitlets_df = hitlets_df.assign(time=np.array([time_hitlets_nv(i,j,periods) for i,j in zip(hitlets_df.cluster_times_ns.values,hitlets_df.eventid.values)]))
            hitlets_df.insert(len(hitlets_df.columns),'n_chunk',np.array([i+1]*len(hitlets_df)))
            hitlets.append(hitlets_df)
            count_ids+=len(n_Veto_id)
        except:
            continue
    hitlets_df_nv = pd.concat(hitlets)
    hitlets_nv = df_to_hit_array(hitlets_df_nv)
    hitlets_df_nv.to_csv(Name+'_hitlet_nv.csv',encoding='utf-8',index=False)
    np.save(Name+'_hitlets_nv.npy', hitlets_nv, allow_pickle=True, fix_imports=True)
    return hitlets_df_nv, hitlets_nv, hitlets_keys
    


#-----------------------------------------FUNCTIONS TO RECOVER G4 primaries------------------------------------------------------------#

#To recover just simulated G4 event parameters
def sim_pri(path_root, Number, e_min, e_max, path_aux_nv, hitlets_keys=['e_pri','xp_pri','yp_pri','zp_pri','Save_flag','Save_type','Save_x','Save_y','Save_z','Save_e','Save_t'], Name_csv='/home/ldaniel/csv_files/'):
    e_pri_df=[]
    count_ids=0
    for i in tq.tqdm(range(0,Number)):
        try:
            root_file = rt.open(path_root+str(i+1)+'.root')
            keys=['eventid','pmthitEnergy','pmthitTime','pmthitID'] + hitlets_keys
            #We chose the pmt data frame that has all recorded values for Vetos PMTs from GEANT4 simulation
            df=ak.to_dataframe(root_file['events'].arrays(keys, library='ak', entry_start=e_min, entry_stop=e_max))
            event_list=np.unique(df.eventid.values)
            n_Veto_id= np.unique(df[(abs(df.xp_pri)<2000) & (abs(df.yp_pri)<2000) & (df.zp_pri<600) & (df.zp_pri>-2000)].eventid.values)
            events_sim_nv=df[df.eventid.isin(m_Veto_id)]
            events_sim_nv.insert(len(events_sim_nv.columns),'eventid_g4',events_sim_nv.eventid.values)#to keep original g4 eventid and chunck, for checking tasks
            events_sim_nv.insert(len(events_sim_nv.columns),'n_chunk',np.array([i+1]*len(df)))
            events_sim_nv=events_sim_nv.assign(eventid = events_sim_nv.eventid.values+count_ids)
            e_pri_df.append(events_sim_nv)
            count_ids+=len(n_Veto_id)
        except:
            continue
    df_sim = pd.concat(e_pri_df)
    df_sim.to_csv(Name_csv+'sim_pri.csv',encoding='utf-8',index=False)
    return df_sim
    
#To include G4 primaries into processed events_nv from hitlets   
def event_info_nv_from_hitlet(df_hitlet,pri_keys=['e_pri','xp_pri','yp_pri','zp_pri'] ,threshold=True):
    #from hitlet dataframe
    if 'n_chunk' not in df_hitlet.columns.values:
        df_hitlet['n_chunk']= [0]*len(df_hitlet)#this implies hitlets are from just one root file, n_chunk is set to 0
    if threshold==True:
        df_hitlet= df_hitlet[df_hitlet.pe_area>=df_hitlet.threshold_acc]
    elif threshold==False:
        print('No threshold per channel is applied')
    hitlet_nv = df_to_hit_array(df_hitlet)
    events_nv = strax_nv.compute(hitlet_nv,min(hitlet_nv['time']),max(hitlet_nv['time']))
    evt_cols=list(events_nv.dtype.names)
    evt_cols.remove('area_per_channel')
    events_nv_df= pd.DataFrame(events_nv, columns= evt_cols)
    events_nv_df.insert(len(events_nv_df.columns),'area_per_channel',list(events_nv['area_per_channel']))
    evt_pri= pd.pivot_table(df_hitlet, index= ['n_chunk','eventid'], aggfunc={'time':[np.min,np.max]})
    n_chunk= flat_list([[df_hitlet[df_hitlet.n_chunk==evt_pri.index[i][0]]['n_chunk'].values[0]]*len(events_nv_df[(events_nv_df.time<=int(evt_pri.values[i][0]))&(events_nv_df.time>=int(evt_pri.values[i][1]))].time.values) for i in range(len(evt_pri))])
    events_nv_df.insert(len(events_nv_df.columns), 'n_chunk', n_chunk )
    evt_id =flat_list([[df_hitlet[df_hitlet.eventid==evt_pri.index[i][1]]['eventid'].values[0]]*len(events_nv_df[(events_nv_df.time<=int(evt_pri.values[i][0]))&(events_nv_df.time>=int(evt_pri.values[i][1]))].time.values) for i in range(len(evt_pri))])
    events_nv_df.insert(len(events_nv_df.columns), 'eventid', evt_id )
    print('getting pri parameters')
    for j in tq.tqdm(pri_keys):
        pri_array=flat_list([[df_hitlet[(df_hitlet.eventid==evt_pri.index[i][1])&(df_hitlet.n_chunk==evt_pri.index[i][0])][j].values[0]]*len(events_nv_df[(events_nv_df.time<=int(evt_pri.values[i][0]))&(events_nv_df.time>=int(evt_pri.values[i][1]))].time.values) for i in range(len(evt_pri))])
        events_nv_df.insert(len(events_nv_df.columns), j, pri_array )
    return events_nv_df, events_nv
   
