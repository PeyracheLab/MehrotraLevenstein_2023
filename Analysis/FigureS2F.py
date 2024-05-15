#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:58:44 2024

@author: dhruv
"""

#import dependencies
import numpy as np 
import pandas as pd 
import scipy.io
import os
import nwbmatic as ntm
import pynapple as nap 
import matplotlib.pyplot as plt 

#%% Data organization

data_directory = '/media/dhruv/LaCie1/dataAdrien'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
writepath = '/home/dhruv/Code/MehrotraLevenstein_2023/Analysis/OutputFiles'
s = 'Mouse32-140822'

name = s.split('/')[-1]
path = os.path.join(data_directory, s)

###Load the session
data = ntm.load_session(path, 'neurosuite')
data.load_neurosuite_xml(path)
channelorder = data.group_to_channel[0]

###Get ADn neurons
spikes = data.spikes
spikes_by_location = spikes.getby_category('location')
spikes_adn = spikes_by_location['adn']
    
###Get epochs
epochs = data.epochs

#%%Get cell depths

depth = np.arange(0, -160, -20)

filepath = os.path.join(path, 'Analysis')
listdir    = os.listdir(filepath)

file = [f for f in listdir if 'SpikeWaveF' in f]
mch = scipy.io.loadmat(os.path.join(filepath,file[0]))
maxch = mch['maxIx'][0]

#%% Load UP and DOWN states

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

#%% Compute Peri-event time Histogram (PETH)

depth_final = []
  
for i in spikes_adn:
    tmp = depth[maxch[i]-1] 
    depth_final.append(tmp)
    
FR = spikes_adn.restrict(up_ep)._metadata['rate']
ratesort = np.argsort(FR.values)

cc = nap.compute_eventcorrelogram(spikes_adn, nap.Ts(up_ep['start']), binsize = 0.005, windowsize = 0.255, ep = up_ep)    
dd = cc[-0.05:0.15]
tmp = pd.DataFrame(dd)
tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)

#%% Compute onset

if len(dd.columns) > 0:
    indexplot = []
            
for i in range(len(tmp.columns)):
    a = np.where(tmp.iloc[:,i] > 0.5)
        
    if len(a[0]) > 0:
        res = tmp.iloc[:,i].index[a]
        indexplot.append(res[0])

#%% Sort data by depth 

n = len(depth_final)
t2 = np.argsort(depth_final)
desc = t2[::-1][:n] 
finalRates = tmp[cc.columns[desc]]

#%% Plotting 

fig, ax = plt.subplots()
cax = ax.imshow(finalRates.T,extent=[-50 , 150, len(spikes_adn)+1 , 1], vmin = 0, vmax = 2, aspect = 'auto', cmap = 'inferno')
cbar = fig.colorbar(cax, ticks=[0, 2], label = 'Norm. rate')
cbar.ax.set_yticklabels(['0', '>=2'])
plt.scatter(np.array(indexplot)[desc]*1e3, np.arange(1.5,len(spikes_adn)+1,1), c = 'w', s = 4)
ax.set_ylabel('Cells sorted by D-V position (31)')
ax.set_yticks([])
ax.set_xlabel('Time from UP onset (ms)')
ax.set_xticks([-50, 0, 150])
plt.axvline(0, color = 'w', linestyle = '--')
ax.set_box_aspect(1)