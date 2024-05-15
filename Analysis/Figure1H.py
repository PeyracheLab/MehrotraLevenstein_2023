#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:45:10 2023

@author: dhruv
"""

#import dependencies
import numpy as np 
import pandas as pd
import scipy.io
from Dependencies.functions import *
from Dependencies.wrappers import *
import os
from Dependencies import neuroseries as nts 
import matplotlib.pyplot as plt 
from scipy.stats import wilcoxon

#%% Data organization

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

readpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'
writepath = '/home/dhruv/Code/MehrotraLevenstein_2023/Analysis/OutputFiles'

dur_D = []
dur_V = []

for s in datasets:
          
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(readpath,s)

###Loading the data
    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(rawpath)
    
###Load cell depths
    filepath = os.path.join(path, 'Analysis')
    listdir = os.listdir(filepath)
    file = [f for f in listdir if 'CellDepth' in f]
    celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
    depth = celldepth['cellDep']
    
###Load cell types 
    file = [f for f in listdir if 'CellTypes' in f]
    celltype = scipy.io.loadmat(os.path.join(filepath,file[0]))
    
###Load global UP and DOWN states
    file = os.path.join(writepath, name +'.evt.py.dow')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        down_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
        
    file = os.path.join(writepath, name +'.evt.py.upp')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        up_ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

#%% Determine which units are in dorsal or ventral portion 

    data = pd.DataFrame()   
    data['depth'] = np.reshape(depth,(len(spikes.keys())),)
    data['level'] = pd.cut(data['depth'],2, precision=0, labels=[1,0]) #0 is dorsal, 1 is ventral
    data['celltype'] = np.nan
    data['gd'] = 0
    
    for i in range(len(spikes)):
        if celltype['gd'][i] == 1:
            data.loc[i,'gd'] = 1
            
    data = data[data['gd'] == 1]

#%% Split population activity into dorsal and ventral 

    mua = {}
    
    latency_dorsal = []
    latency_ventral = []
    
    # define mua for dorsal and ventral
    for i in range(2):
        mua[i] = []        
        for n in data[data['level'] == i].index:            
            mua[i].append(spikes[n].index.values)
        mua[i] = nts.Ts(t = np.sort(np.hstack(mua[i])))
        
#%% Compute PETH of dorsal and ventral MUA around global DOWN state
### And determine the threshold crossing

    binsize = 5
    nbins = 1000        
    neurons = list(mua.keys())
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2        
    cc = pd.DataFrame(index = times, columns = neurons)
    tsd_dn = down_ep.as_units('ms').start.values

    rates = []
    
    ddur = []
    vdur = []

    for i in neurons:
        spk2 = mua[i].restrict(up_ep).as_units('ms').index.values
        tmp = crossCorr(tsd_dn, spk2, binsize, nbins)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
        
        #Normalize PETH by mean rate
        fr = len(spk2)/up_ep.tot_length('s')
        rates.append(fr)
        cc[i] = tmp.values
        cc[i] = tmp.values/fr
       
        dd = cc[-250:250]
        
        #Threshold crossing
        tmp = dd[i].loc[5:] > 0.2 
        ends = tmp.where(tmp == True).first_valid_index()
                
        tmp2 = dd[i].loc[-150:-5] > 0.2 
        start = tmp2.where(tmp2 == True).last_valid_index()
        
        #Categorize the durations by anatomical position         
        if i == 0: 
            ddur.append(ends - start)
        else: 
            vdur.append(ends - start)
    
    #Compute dorsal and ventral DOWN state durations for all datasets         
    dur_D.append(ddur[0])
    dur_V.append(vdur[0])

#%% Plotting 

plt.scatter(dur_D, dur_V, color = 'k', zorder = 3) 
plt.gca().axline((min(min(dur_D),min(dur_V)),min(min(dur_D),min(dur_V)) ), slope=1, color = 'silver', linestyle = '--')
plt.xlabel('Dorsal DOWN duration (ms)')
plt.ylabel('Ventral DOWN duration (ms)')
plt.axis('square')

#%% Stats 

W,p = wilcoxon(dur_D,dur_V)




