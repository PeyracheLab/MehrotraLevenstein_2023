#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:45:53 2023

@author: dhruv
"""

#import dependencies
import numpy as np 
import pandas as pd 
import nwbmatic as ntm
import pynapple as nap 
import os
import matplotlib.pyplot as plt 
from scipy.stats import kendalltau

#%% Data organization

data_directory = '/media/dhruv/LaCie1/A7800'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

PMR = []
uponset = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    
###Load the data
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    spikes = data.spikes
    
###Load UP and DOWN states 
    file = os.path.join(path, name +'.evt.py.dow')
    down_ep = data.read_neuroscope_intervals(name = 'DOWN', path2file = file)
    
    file = os.path.join(path, name +'.evt.py.upp')
    up_ep = data.read_neuroscope_intervals(name = 'UP', path2file = file)
    
#%% Compute PETH 

    cc2 = nap.compute_eventcorrelogram(spikes, nap.Ts(up_ep['start']), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
    tmp = pd.DataFrame(cc2)
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd2 = tmp[0:0.155]
    
#%% Compute peak-to-mean ratio (PMR) and UP onset delay

    for i in range(len(dd2.columns)):
        a = np.where(dd2.iloc[:,i] > 0.5)
        if len(a[0]) > 0:
            PMR.append(dd2.iloc[:,i].max())
            res = dd2.iloc[:,i].index[a]
            uponset.append(res[0])
            
#%% Plotting 

binsize = 0.005 #In seconds
(counts,onsetbins,peakbins) = np.histogram2d(uponset, PMR,
                                             bins = [len(np.arange(0, 0.155, binsize)) + 1, len(np.arange(0, 0.155, binsize)) + 1],
                                                 range = [[-0.0025, 0.1575],[0.5, 3.6]])

masked_array = np.ma.masked_where(counts == 0, counts)
cmap = plt.cm.viridis 
cmap.set_bad(color='white')

plt.figure()
plt.imshow(masked_array.T, origin='lower', extent = [onsetbins[0],onsetbins[-1],peakbins[0],peakbins[-1]],
                                               aspect='auto', cmap = cmap)
plt.colorbar(ticks = [min(counts.flatten()) + 1, max(counts.flatten())], label = '# cells')
plt.xlabel('UP onset delay (s)')
plt.xticks([0, 0.15])
plt.ylabel('PMR')
plt.yticks([0.5, 3.5])
plt.gca().set_box_aspect(1)

y_est = np.zeros(len(uponset))
m, b = np.polyfit(uponset, PMR, 1)
for i in range(len(uponset)):
     y_est[i] = m*uponset[i]

plt.plot(uponset, y_est + b, color = 'r', zorder = 3)
plt.axhline(1, color ='k', linestyle = '--')

#%% Stats 

corr, p = kendalltau(uponset, PMR)
                