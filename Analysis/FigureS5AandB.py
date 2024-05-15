#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:16:30 2024

@author: dhruv
"""
#import dependencies
import numpy as np 
import pandas as pd 
import scipy.io
import nwbmatic as ntm
import pynapple as nap 
import os
import matplotlib.pyplot as plt 
from itertools import combinations
from scipy.stats import pearsonr
from sklearn.manifold import Isomap

#%% Data organization

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

readpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'
writepath = '/home/dhruv/Code/MehrotraLevenstein_2023/Analysis/OutputFiles'

allPETH = pd.DataFrame()
uponset = []
peaktiming = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(readpath,s)

###Loading the data
    data = ntm.load_session(rawpath, 'neurosuite')
    data.load_neurosuite_xml(rawpath)
    spikes = data.spikes  
        
###Load cell types
    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)
    
    file = [f for f in listdir if 'CellDepth' in f]
    celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
    depth = celldepth['cellDep']
    
    
    file = [f for f in listdir if 'CellTypes' in f]
    celltype = scipy.io.loadmat(os.path.join(filepath,file[0]))
    
    pyr = []
    interneuron = []
    hd = []
        
    for i in range(len(spikes)):
        if celltype['ex'][i] == 1 and celltype['gd'][i] == 1:
            pyr.append(i)
            
    for i in range(len(spikes)):
        if celltype['fs'][i] == 1 and celltype['gd'][i] == 1:
            interneuron.append(i)
            
    for i in range(len(spikes)):
        if celltype['hd'][i] == 1 and celltype['gd'][i] == 1:
            hd.append(i)
            
###Load UP and DOWN states
    file = os.path.join(writepath, name +'.evt.py.dow')
    down_ep = data.read_neuroscope_intervals(name = 'DOWN', path2file = file)
    
    file = os.path.join(writepath, name +'.evt.py.upp')
    up_ep = data.read_neuroscope_intervals(name = 'UP', path2file = file)
    
#%% Compute PETH 

    cc2 = nap.compute_eventcorrelogram(spikes, nap.Ts(up_ep['start']), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
    tmp = pd.DataFrame(cc2)
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd2 = tmp[0:0.155]  
    
    #Only EX cells
    ee = dd2[pyr] 
    
#%% Concatenate all PETHs for Isomap
 
    if len(ee.columns) > 0:
                     
         tokeep = []
         peaks_keeping_ex = []
         peaktiming_ex = []    
    
         sess_uponset = []
                     
         for i in range(len(ee.columns)):
             a = np.where(ee.iloc[:,i] > 0.5)
             
             if len(a[0]) > 0:
                              
               tokeep.append(ee.columns[i])  
               peaks_keeping_ex.append(ee.iloc[:,i].max())
               peaktiming_ex.append(ee.iloc[:,i].idxmax())
               peaktiming.append(ee.iloc[:,i].idxmax())
               res = ee.iloc[:,i].index[a]
               sess_uponset.append(res[0])
               uponset.append(res[0])
    
                           
    allPETH = pd.concat([allPETH, ee[tokeep]], axis = 1)
     

#%% Isomap 

projection = Isomap(n_components = 2, n_neighbors = 50).fit_transform(allPETH.T.values)   

#%% Statistics and plotting  

r, p = pearsonr(projection[:,1], uponset)
r2, p2 = pearsonr(projection[:,1], peaktiming)

plt.figure()
plt.scatter(projection[:,1], uponset, label = 'r = ' + str(round(r,2)))
plt.gca().set_box_aspect(1)
plt.xlabel('dim 2')
plt.xticks([])
plt.ylabel('UP onset delay (s)')
plt.yticks([0, 0.14])
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)

plt.figure()
plt.scatter(projection[:,1],peaktiming, label = 'r = ' + str(round(r2,2)))
plt.gca().set_box_aspect(1)
plt.xlabel('dim 2')
plt.xticks([])
plt.ylabel('Timing of peak rate (s)')
plt.yticks([0.02, 0.14])
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)



