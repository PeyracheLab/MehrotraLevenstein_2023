#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:21:30 2023

@author: dhruv
"""
#import dependencies
import numpy as np 
import pandas as pd 
import scipy.io
import pynapple as nap 
import os
import matplotlib.pyplot as plt 
from scipy.stats import kendalltau

#%% Data organization

data_directory = '/media/DataDhruv/Recordings/WatsonBO'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

uponset = []
PMR = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)

#%% Load the relevant mat files 

    listdir = os.listdir(path)
    file = [f for f in listdir if 'spikes' in f]
    spikedata = scipy.io.loadmat(os.path.join(path,file[0]))  

    file = [f for f in listdir if 'events' in f]
    events = scipy.io.loadmat(os.path.join(path,file[0]))  
    
    file = [f for f in listdir if 'states' in f]
    states = scipy.io.loadmat(os.path.join(path,file[0])) 
    
    file = [f for f in listdir if 'CellClass' in f]
    cellinfo = scipy.io.loadmat(os.path.join(path,file[0])) 
    
#%% Parsing mat files for variables of interest 

###Load EX cells 
    ex = cellinfo['CellClass'][0][0][1][0]
    pyr = []
    
    for i in range(len(ex)):
        if ex[i] == 1:
            pyr.append(i)
            
###Load spikes (only for EX cells)
    spks = spikedata['spikes']
    spk = {}
    for i in range(len(spks[0][0][1][0])):
        spk[i] = nap.Ts(spks[0][0][1][0][i].flatten())
    spikes = nap.TsGroup(spk)
    spikes = spikes[pyr]
    
###Load sleep states
    sleepstate = states['SleepState']
    wake_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][0][:,0], end = sleepstate[0][0][0][0][0][0][:,1])
    nrem_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][1][:,0], end = sleepstate[0][0][0][0][0][1][:,1])
    rem_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][2][:,0], end = sleepstate[0][0][0][0][0][2][:,1])
    
###Load UP and DOWN states 
    slowwaves = events['SlowWaves']
    up_ep = nap.IntervalSet( start = slowwaves[0][0][2][0][0][0][:,0], end = slowwaves[0][0][2][0][0][0][:,1])
    down_ep = nap.IntervalSet( start = slowwaves[0][0][2][0][0][1][:,0], end = slowwaves[0][0][2][0][0][1][:,1])
    
#%% Compute PETH 

    cc = nap.compute_eventcorrelogram(spikes, nap.Ts(up_ep['start']), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
    tmp = pd.DataFrame(cc)
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd = tmp[0:0.155]
    
#%% Compute peak-to-mean ratio (PMR) and UP onset delay
    
    for i in range(len(dd.columns)):
        a = np.where(dd.iloc[:,i] > 0.5)
        if len(a[0]) > 0:
            PMR.append(dd.iloc[:,i].max())
            res = dd.iloc[:,i].index[a]
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
            
    