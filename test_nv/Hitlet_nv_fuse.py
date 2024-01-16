import uproot as rt
import numpy as np
import concurrent.futures
import re
import collections
import glob
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy as scp
import scipy.stats as stats
from scipy import interpolate
import scipy.constants as const
import awkward as ak
import pandas as pd
import time
import inspect
import sys
import multiprocessing as mp
import straxen
import strax
from sklearn.cluster import DBSCAN
import random as rd
from tqdm import tqdm
import tqdm.notebook as tq
import time


##--------------------------------------------COMMENTS------------------------------------------------------------##

#This hitlet simulator is an extension of the work of Diego Ramirez, Daniel Wenz, Andrea Mancuso and Pavel Kavrigin.
#Functions of SPE charge sampling are in the PDF function based in the calibrations of nVeto done by Andrea Mancuso. Daniel Wenz provides a code that takes into account this functions to sample the charge. This leas to SPE_PDF function in the SN_hitlet.
#We use the QE efficiencies for each PMT as a wavelength function for nVeto provided by Andrea Mancuso.


#The details of this hitlet simulation leading to the main functions 'G4_nveto_hitlets' and 'G4_mveto_hitlets' for muon and neutron Vetos:
#https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:layos:snhitlet_from_geant4_output

#In the note mentioned above we test several approaches of charge sampling, for that we can use the function 'G4_to_Veto_hitlets_comparison'.
#However, for these first note an update is ongoing here: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:layos:vetohitlets_v2


#--------------------------------------------------------HITLETS AUXILIAR FUNCTIONS--------------------------------------------------------------------#
def flat_list(l):
    return np.array([item for sublist in l for item in sublist ])

def energytowavelenght(E):
    Joules_to_eV=1.602*1e-19 
    return 1e9*const.h*const.c/(E*Joules_to_eV)

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

@np.vectorize
def get_threshold_acc(ID):
    ind=ID-2000
    threshold= SPE_param.threshold_pe.values[ind]
    return threshold


#Quantum efficiency
def QE_f(ID,Q_E_file):
    with open(Q_E_file,'r') as f:
        data = json.loads(f.read())
    #We define some bounds because some photons can be out of the interval of QE wavelenght .json file
    bounds=min(data['nv_pmt_qe_wavelength']),max(data['nv_pmt_qe_wavelength'])
    f_channel= lambda x: interpolate.interp1d(data['nv_pmt_qe_wavelength'],data['nv_pmt_qe'][str(ID)])(x) if ((x>=bounds[0]) and (x<=bounds[1])) else 0.0
    return f_channel
def QE_function(Q_E_nveto_file):
    #We define some bounds because some photons can be out of the interval of QE wavelenght .json file
    QE_array_n=[]
    for i in np.arange(2000,2120):
        QE_array_n.append(np.vectorize(QE_f(i,Q_E_nveto_file)))
    #Watertank_QE
    pmt_id= np.arange(2000,2120)
    QE_ch= pd.DataFrame(columns=['pmtID','QE'])
    QE_ch['pmtID'],QE_ch['QE']=pmt_id,QE_array_n
    return QE_ch.to_records()

#Cluster for stacket hitlets
def channel_cluster_nv(t):
    db_cluster = DBSCAN(eps=8, min_samples=1)#As a preliminar value we fix distance between two photons arriving in the same pmt 8ns
    t_val=np.array(t)
    clusters=np.array(db_cluster.fit_predict(t_val.reshape(-1, 1)))
    return clusters
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
    df_hit.insert(4,'area',data.pe_charge.values.astype('float32'))
    df_hit=df_hit.astype({'time_amplitude': 'int16'})    
    hitlets= strax.sort_by_time(df_hit.to_records())    
    return hitlets

def event_info_nv_from_hitlet(df_hitlet,th_per_channel,pri_keys=['e_pri','xp_pri','yp_pri','zp_pri'], Separate_hits =False, cut_hits=[0,1e9],local=True,get_pri=False):
    #from hitlet dataframe
    strax_nv= straxen.straxen.plugins.nVETOEvents()
    hitlet_nv_df= df_hitlet[df_hitlet.pe_charge>th_per_channel]
    if Separate_hits==False:
        hitlet_nv_df= hitlet_nv_df[hitlet_nv_df.pe_charge>th_per_channel]
    elif Separate_hits==True:
        hitlet_nv_df= hitlet_nv_df[(hitlet_nv_df.pe_charge>th_per_channel)&(hitlet_nv_df.pmthitTime>=cut_hits[0])&(hitlet_nv_df.pmthitTime<=cut_hits[1])]
    hitlet_nv= df_to_hit_array(hitlet_nv_df)
    if local==True:#Plugin modified to work on local machine
        events_nv = strax_nv.compute_local(hitlet_nv,min(hitlet_nv['time']),max(hitlet_nv['time']))
    elif local==False:
        events_nv = strax_nv.compute(hitlet_nv,min(hitlet_nv['time']),max(hitlet_nv['time']))
    evt_cols=list(events_nv.dtype.names)
    evt_cols.remove('area_per_channel')
    events_nv_df= pd.DataFrame(events_nv, columns= evt_cols)
    events_nv_df.insert(len(events_nv_df.columns),'area_per_channel',list(events_nv['area_per_channel']))
    evt_pri= pd.pivot_table(hitlet_nv_df, index= ['n_chunk','eventid'], aggfunc={'time':[np.min,np.max]})
    n_chunk= flat_list([[hitlet_nv_df[hitlet_nv_df.n_chunk==evt_pri.index[i][0]]['n_chunk'].values[0]]*len(events_nv_df[(events_nv_df.time<=int(evt_pri.values[i][0]))&(events_nv_df.time>=int(evt_pri.values[i][1]))].time.values) for i in range(len(evt_pri))])
    events_nv_df.insert(len(events_nv_df.columns), 'n_chunk', n_chunk )
    evt_id =flat_list([[hitlet_nv_df[hitlet_nv_df.eventid==evt_pri.index[i][1]]['eventid'].values[0]]*len(events_nv_df[(events_nv_df.time<=int(evt_pri.values[i][0]))&(events_nv_df.time>=int(evt_pri.values[i][1]))].time.values) for i in range(len(evt_pri))])
    events_nv_df.insert(len(events_nv_df.columns), 'eventid', evt_id )
    if get_pri==True:
        print('getting pri parameters')
        for j in tq.tqdm(pri_keys):
            pri_array=flat_list([[hitlet_nv_df[(hitlet_nv_df.eventid==evt_pri.index[i][1])&(hitlet_nv_df.n_chunk==evt_pri.index[i][0])][j].values[0]]*len(events_nv_df[(events_nv_df.time<=int(evt_pri.values[i][0])) (events_nv_df.time>=int(evt_pri.values[i][1]))].time.values) for i in range(len(evt_pri))])
            events_nv_df.insert(len(events_nv_df.columns), j, pri_array )
    elif get_pri==False:
        print('not getting pri parameters')
    return events_nv_df, events_nv


def npe_th(path,pmt_id):
    spe=SPE_PDF(path+'parameters.json',path+'gain_version2.json',path+'efficiency_version2.json')
    index= spe['pmt_id'].index(pmt_id) 
    thr= spe['threshold'][index] 
    return thr

def time_hitlets_nv(time,ids,freq):
    if type(time)==np.float64:
        ret=time
    elif type(time)==list:
        ret=min(time)               
    return ret + freq*ids

#--------------------------------------------------------Water tank HITLET CLASS--------------------------------------------------------------------#

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

@np.vectorize
def get_threshold_acc(ID):
    ind=ID-2000
    threshold= SPE_param.threshold_pe.values[ind]
    return threshold

class Hitlet_nv(object):
    def __init__(self, path, g4_file):
        self.path = path
        self.QE_value = QE_function(path+'nveto_pmt_qe.json')
        self.SPE_nVeto = SPE_parameters(path+'SPE_'+'SR1'+'_test_fuse.json',path+'acceptance_SR0_test_fuse.json')
        self.g4_file = g4_file
    #Get Quantum efficiency
    #@np.vectorize
    def QE_E(self,E,ID):
        WL= energytowavelenght(E)
        QE_arr=self.QE_value[self.QE_value.pmtID==ID]
        qe=QE_arr.QE[0](WL)
        return qe
        
    def pe_charge_N(self,pmt_id):
        index= self.SPE_nVeto['pmt_id'].index(pmt_id)
        params= self.SPE_nVeto['g_param'][index]
        charge=rd.choices(self.SPE_nVeto['pe'][index],self.SPE_nVeto['pdf'][index],k=1)[0]    
        return charge
        
    def npe_threshold(self,pmt_id):
        index= self.SPE_nVeto['pmt_id'].index(pmt_id) 
        thr= self.SPE_nVeto['threshold'][index] 
        return thr

    #--------------------------- Hitlet function ------------------------------------------------------------#

    def nv_hitlets(self, e_1, e_2, root_keys=['e_pri','xp_pri','yp_pri','zp_pri'], QE_Scaling=0.75, period=1e9,csv=False, Name=None, Isotopes=False):
        
    #-------------------------------------------------Arguments---------------------------------------------------#
    
    #e_1,e_2= range of entries from root file
    #root_keys= for primaries or flags that you to keep if you don't wont any just root_keys=[]
    #QE_Scaling corrrespond to collection efficiency, no study has been done on the CE of muon Veto we use a default value close to the nVeto see https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:mancuso:hitletsimulator:collection_efficiency
    #period : this could be related to the rate of a source, or the rate for a time been if we reconstruct an spectrum. Pile_up_cut is related to this value
    #csv : if you want to save the hitlet
    #Name : to name this csv file
    #Isotopes : if we want to keep information about some activated isotopes (True) if False it cut these events.(See comments) 
    #Comments:
    #1.There is no application of a threshold per channel based on the acceptation by default, but we keep the value in the data frame for each pmt, and one can do manually. This is to no condition the sampling and compare it with the data real one.
    #.2. The period is set by default at 1s to care about no pyle up or merge of hitlets if one want to do an analysis for rare events. If we simulate a calibration or a constant flux this value has to be changed to real rate one.  
    #.3. The way the code works, if information of Isotopes is keeped we cannot recover G4 primary parameters after building 'event_nv'. We have to think in this case a different way to do it.
    #4.Stacked hitlets: using DBSCAN is maybe not required (as it makes the hitlet heavier and )
    
    
        #0.---------------Load GEANT output-------------------#
    
        print("Opening File :", self.g4_file)
        root_file = rt.open(self.g4_file)
        #We chose the pmt data frame that has all recorded values for Vetos PMTs from GEANT4 simulation
        keys=['eventid','pmthitEnergy','pmthitTime','pmthitID'] + root_keys #Pay attention if you select 'type_pri' pmt information will be lost and only you '_pri' keys corresponding to primaries
        df=ak.to_pandas(root_file['events'].arrays(keys, library='ak',entry_start=e_1, entry_stop=e_2))
        #We choose only events produced in nVeto and recorded by nVeto pmts (this is useful if a G4 sim is confined in all Water volume)
        event_list=np.unique(df.eventid.values)
        n_Veto_id= np.unique(df[(abs(df.xp_pri)<2000) & (abs(df.yp_pri)<2000) & (df.zp_pri<600) & (df.zp_pri>-2000)].eventid.values)
        df=df[(df.pmthitID>=2000)]
        df=df[df.eventid.isin(n_Veto_id)]

        #1. First step PHOTON to first dinode
               
        print("Applying QE and CE")
        qe = 1e-2*np.vectorize(self.QE_E)(df.pmthitEnergy.values,df.pmthitID.values)#Applying Quantum efficiency for each pmt
        qe *= QE_Scaling #Applying collection efficiency
        pe = np.array([np.random.binomial(1, j, 1)[0] for j in qe])
        print("Loading hit survive")
        df.insert(len(df.columns),'pe',pe)
        df=df[df.pe>0]
        df=df.drop(columns=['pe'])
        #Getting the acceptance threshold:
        df['threshold_acc']= get_threshold_acc(df.pmthitID)
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
        df.insert(len(df.columns),'pe_charge', charges)
        column = ['eventid','pmthitID']
        df_c = pd.pivot_table(df,index = column,aggfunc={'cluster_times_ns': lambda x: list(x)})
        cc = [channel_cluster_nv(i) for i in (df_c.cluster_times_ns.values)]
        df = df.sort_values(['eventid','pmthitID'], ascending = [True,True])
        df.insert(len(df.columns), 'clusters_c', flat_list(cc))
        
        #3.Creating hitlet dataframe
        
        col_index = ['eventid','pmthitID','clusters_c']
        col_hitlet = ['pmthitEnergy', 'pmthitTime' ] + root_keys + ['pe_charge','cluster_times_ns'] 
        arg_dicio = dict.fromkeys(col_hitlet)
        for i in col_hitlet:
            if i == 'pe_charge':
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
def event_nv_chunks(path_root, Number, e_min, e_max, path_aux_nv, periods, CE, th, Separate_hits =False, cut_hits=[0,1e9], local=True, hitlets_keys=['e_pri','xp_pri','yp_pri','zp_pri'], Name_csv='/home/ldaniel/csv_files/'):
    #Same arguments of evenyt_nv_pri and mv_hitlets +
    #Number = number of chuncks
    data_nv=[]
    data_nv_df=[]
    e_pri_df=[]
    hitlets=[]
    count_ids=0
    for i in tq.tqdm(range(0,Number)):
        try:
            root_file = rt.open(path_root+str(i+1)+'.root')
            keys=['eventid','pmthitEnergy','pmthitTime','pmthitID'] + hitlets_keys
            #We chose the pmt data frame that has all recorded values for Vetos PMTs from GEANT4 simulation
            df=ak.to_dataframe(root_file['events'].arrays(keys, library='ak', entry_start=e_min, entry_stop=e_max))
            event_list=np.unique(df.eventid.values)
            n_Veto_id= np.unique(df[(abs(df.xp_pri)<2000) & (abs(df.yp_pri)<2000) & (df.zp_pri<600) & (df.zp_pri>-2000)].eventid.values)
            events_sim_nv=df[df.eventid.isin(n_Veto_id)]
            #events_sim_nv.insert(len(events_sim_nv.columns),'eventid_g4',events_sim_nv.eventid.values)#to keep original g4 eventid and chunck, for checking tasks
            events_sim_nv.insert(len(events_sim_nv.columns),'n_chunk',np.array([i+1]*len(events_sim_nv)))
            #events_sim_nv=events_sim_nv.assign(eventid = events_sim_nv.eventid.values+count_ids)
            e_pri_df.append(events_sim_nv)
            nVeto_hitlets= Hitlet_nv(path_root+str(i+1)+'.root',path_aux_nv)
            hitlets_nv= nVeto_hitlets.nv_hitlets(e_min, e_max, hitlets_keys, QE_Scaling = CE, period=periods, csv=False, Name=None)
            hitlets_df=hitlets_nv[0].assign(eventid = hitlets_nv[0].eventid.values + count_ids)
            hitlets_df = hitlets_df.assign(time=np.array([time_hitlets_nv(i,j,periods) for i,j in zip(hitlets_df.cluster_times_ns.values,hitlets_df.eventid.values)]))
            hitlets_df.insert(len(hitlets_df.columns),'n_chunk',np.array([i+1]*len(hitlets_df)))
            hitlets.append(hitlets_df)
            event_nv_df,event_nv = event_info_nv_from_hitlet(hitlets_df,th,hitlets_keys)
            data_nv_df.append(event_nv_df)
            data_nv.append(event_nv)
            count_ids+=len(n_Veto_id)
        except:
            continue
    events_df_nv = pd.concat(data_nv_df)
    events_nv = np.concatenate(data_nv)
    df_sim = pd.concat(e_pri_df)
    hitlets_df_nv = pd.concat(hitlets)
    events_df_nv.to_csv(Name_csv+'event_nv.csv',encoding='utf-8',index=False)
    hitlets_df_nv.to_csv(Name_csv+'hitlet_nv.csv',encoding='utf-8',index=False)
    np.save(Name_csv+'event_nv.npy', events_nv, allow_pickle=True, fix_imports=True)
    df_sim.to_csv(Name_csv+'event_sim.csv',encoding='utf-8',index=False)
    return events_df_nv, df_sim, events_nv


def event_nv_df(hitlets, th_per_channel, Separate_hits =False, cut_hits=[0,1e9], local=True, get_pri=False):
    #Same arguments for hitlets adding possibility to save them into csv
    #hitlets= output from mv_hitlets
    #separate_hits= If there are more than one particle an one want to select by pmthitTime
        #cut_hits = In case separate hits is True values of the range in pmthitTime in seconds
    #local = if you modify straxen plugin to work on local machine...
    #get_pri = to recover parameters of generated primaries
    strax_nv= straxen.straxen.plugins.muVETOEvents()
    hitlet_nv_df,hitlet_nv,pri_keys=hitlets
    hitlet_nv_df,hitlet_nv= hitlet_nv_df[hitlet_nv_df.pe_charge>th_per_channel],hitlet_nv[hitlet_nv['area']>th_per_channel]
    if Separate_hits==False:
        hitlet_nv_df,hitlet_nv= hitlet_nv_df[hitlet_nv_df.pe_charge>th_per_channel],hitlet_nv[hitlet_nv['area']>th_per_channel]
    elif Separate_hits==True:
        hitlet_nv_df= hitlet_nv_df[(hitlet_nv_df.pe_charge>th_per_channel)&(hitlet_nv_df.pmthitTime>=cut_hits[0])&(hitlet_nv_df.pmthitTime<=cut_hits[1])]
        hitlet_nv= df_to_hit_array(hitlet_nv_df)
    if local==True:#Plugin modified to work on local machine
        events_nv = strax_nv.compute_local(hitlet_nv,min(hitlet_nv['time']),max(hitlet_nv['time']))
    elif local==False:
        events_nv = strax_nv.compute(hitlet_nv,min(hitlet_nv['time']),max(hitlet_nv['time']))
    evt_cols=list(events_nv.dtype.names)
    evt_cols.remove('area_per_channel')
    events_nv_df= pd.DataFrame(events_nv, columns= evt_cols)
    events_nv_df.insert(len(events_nv_df.columns),'area_per_channel',list(events_nv['area_per_channel']))
    evt_id =flat_list([[hitlet_nv_df[hitlet_nv_df.eventid==evt_pri.index[i]]['eventid'].values[0]]*len(events_nv_df[(events_nv_df.time<=int(evt_pri.values[i][0]))&(events_nv_df.time>=int(evt_pri.values[i][1]))].time.values) for i in range(len(evt_pri))])
    events_nv_df.insert(len(events_nv_df.columns), 'eventid', evt_id )
    if get_pri==True:
        print('getting pri parameters')
        for j in tq.tqdm(pri_keys):
            pri_array=flat_list([[hitlet_nv_df[hitlet_nv_df.eventid==evt_pri.index[i]][j].values[0]]*len(events_nv_df[(events_nv_df.time<=int(evt_pri.values[i][0]))&(events_nv_df.time>=int(evt_pri.values[i][1]))].time.values) for i in range(len(evt_pri))])
            events_nv_df.insert(len(events_nv_df.columns), j, pri_array ) 
    elif get_pri==False:
        print('not getting pri parameters')
    return events_nv_df, events_nv
        
    
    
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
    
    
   
