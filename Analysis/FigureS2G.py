#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:26:25 2024

@author: dhruv
"""

#import dependencies
import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import nwbmatic as ntm
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import mannwhitneyu

#%% Compute UP onset delay of entire PoSub dataset 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

readpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'
writepath = '/home/dhruv/Code/MehrotraLevenstein_2023/Analysis/OutputFiles'

range_uponset_posub = []
range_uponset_adn = []

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
    
###Compute PETH 
    cc2 = nap.compute_eventcorrelogram(spikes, nap.Ts(up_ep['start']), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
    tmp = pd.DataFrame(cc2)
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd2 = tmp[0:0.155]  
    
    #Only EX cells
    ee = dd2[pyr] 
    
###Compute UP onset delay
    if len(ee.columns) > 0:
                    
        tokeep = []
        sess_uponset = []
                            
        for i in range(len(ee.columns)):
            a = np.where(ee.iloc[:,i] > 0.5)
            if len(a[0]) > 0:
              tokeep.append(ee.columns[i])  
              res = ee.iloc[:,i].index[a]
              sess_uponset.append(res[0])
              
    range_uponset_posub.append(np.std(sess_uponset)) 

#%% Compute UP onset delay of entire ADn dataset 

data_directory = '/media/dhruv/LaCie1/dataAdrien'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    
###Loading the data
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    
###Get ADn neurons only
    spikes = data.spikes  
    spikes_by_location = spikes.getby_category('location')
    spikes_adn = spikes_by_location['adn']
    spikes = spikes_adn
    
###Load epochs
    epochs = data.epochs

###Load UP and DOWN states
    file = os.path.join(writepath, name +'.DM.evt.py.dow')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        down_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
    file = os.path.join(writepath, name +'.DM.evt.py.upp')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        up_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
        
###Compute PETH
    cc2 = nap.compute_eventcorrelogram(spikes, nap.Ts(up_ep['start']), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
    tmp = pd.DataFrame(cc2)
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd2 = tmp[-0.05:0.155]  
    
###Compute UP onset delay
    if len(dd2.columns) > 0:
                   
        tokeep = []
        sess_uponset = []
            
        for i in range(len(dd2.columns)):
            a = np.where(dd2[0:0.155].iloc[:,i] > 0.5)
            if len(a[0]) > 0:
              tokeep.append(dd2[0:0.155].columns[i])  
              res = dd2[0:0.155].iloc[:,i].index[a]
              sess_uponset.append(res[0])
              
    range_uponset_adn.append(np.std(sess_uponset)) 
             
#%% ADn v/s PoSub range of UP onset duration distribution

t, p = mannwhitneyu(range_uponset_adn, range_uponset_posub)

ADtype = pd.DataFrame(['ADn' for x in range(len(range_uponset_adn))])
PoSubtype = pd.DataFrame(['PoSub' for x in range(len(range_uponset_posub))])

range_onsets = pd.DataFrame()
range_onsets['SD'] = pd.concat([pd.Series(range_uponset_posub), pd.Series(range_uponset_adn)])
range_onsets['type'] = pd.concat([PoSubtype, ADtype])

sns.set_style('white')
palette = ['cornflowerblue', 'indianred']
ax = sns.violinplot( x = range_onsets['type'], y = range_onsets['SD'] , data = range_onsets, dodge = False,
                    palette = palette,cut = 2,
                    scale = "width", inner = None)
ax.tick_params(bottom = True, left = True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform = ax.transData))
sns.boxplot(x = range_onsets['type'], y = range_onsets['SD'] , data = range_onsets, saturation = 1, showfliers = False,
            width = 0.3, boxprops = {'zorder': 3, 'facecolor': 'none'}, ax = ax)
old_len_collections = len(ax.collections)
sns.swarmplot(x = range_onsets['type'], y = range_onsets['SD'], data = range_onsets, color = 'k', dodge = False, ax = ax)

for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('SD of UP onset delays (s)')
ax.set_box_aspect(1) 

