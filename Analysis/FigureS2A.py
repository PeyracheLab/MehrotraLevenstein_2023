#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:26:17 2024

@author: dhruv
"""

#import dependencies
import numpy as np 
import os
import nwbmatic as ntm
import pynapple as nap
import matplotlib.pyplot as plt 
from Dependencies.functions import *

#%% Load the session

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
readpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'
s = 'A3707-200317'

name = s.split('/')[-1]
path = os.path.join(data_directory, s)
rawpath = os.path.join(readpath,s)

data = ntm.load_session(rawpath, 'neurosuite')
data.load_neurosuite_xml(rawpath)
spikes = data.spikes  
epochs = data.epochs

#%% Example Epoch 

per = nap.IntervalSet(start = 1247.288, end = 1255.288)

#%% Load UP and DOWN states, as well as NREM epoch
     
file = os.path.join(rawpath, name +'.evt.py.dow')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    down_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

file = os.path.join(rawpath, name +'.evt.py.upp')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    up_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)    

#%% Load the LFP

channelorder = data.group_to_channel[0]
seq = channelorder[::8].tolist()

filepath = os.path.join(readpath, name)
listdir = os.listdir(filepath)
    
file = [f for f in listdir if  name + '.eeg' in f]
filename = [name + '.eeg']
matches = set(filename).intersection(set(file))

if (matches == set()) is False:
    lfpsig = nap.load_eeg(filepath + '/' + name + '.eeg' , channel = seq[int(len(seq)/2)], n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 
else: 
    lfpsig = nap.load_eeg(filepath + '/' + name + '.lfp' , channel = seq[int(len(seq)/2)], n_channels = data.nChannels, frequency = 1250, precision ='int16', bytes_size = 2) 

#%% Downsample and filter the LFP in the delta band

downsample =  5
lfpsig = lfpsig[::downsample]
lfp_filt_delta = nap.Tsd(lfpsig.index.values, butter_bandpass_filter(lfpsig, 0.5, 4, 1250/5, 2))

#%% Compute population rate 

bin_size = 0.01 #In seconds
smoothing_window = 0.02 #In seconds

### Bin spike trains
rates = spikes.count(bin_size, sws_ep)

### Smoothen spike trains   
total2 = rates.as_dataframe().rolling(window = 100 ,win_type='gaussian',
                                      center=True,min_periods=1, 
                                      axis = 0).mean(std= int(smoothing_window/bin_size))

total2 = total2.sum(axis =1)
total2 = nap.Tsd(total2)

### Perform thresholding for DOWN state
idx = total2.threshold(np.percentile(total2.values,20),'below')

#%% Plotting

plt.figure()
plt.plot((lfp_filt_delta.restrict(per)*0.01) + 55, zorder = 5, linewidth = 2)
plt.plot(total2.restrict(per), color = 'dimgray', alpha = 0.5)
plt.fill_between(total2.restrict(per).index.values, total2.restrict(per), color = 'dimgray', alpha = 0.5)
plt.plot((lfpsig.restrict(per)*0.01) + 55, color = 'k')
[plt.axvspan(x,y, facecolor = 'g', alpha = 0.5) for x,y in zip(down_ep['start'][691:696], down_ep['end'][691:696])]
