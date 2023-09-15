#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:05:14 2023

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

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

readpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'
writepath = '/home/dhruv/Code/MehrotraLevenstein_2023/Analysis/OutputFiles'

s = 'A3707-200317'
name = s.split('/')[-1]
path = os.path.join(data_directory, s)
rawpath = os.path.join(readpath,s)

#%% Loading the data

data = nap.load_session(rawpath, 'neurosuite')
data.load_neurosuite_xml(rawpath)
spikes = data.spikes  

#%% Load cell types
    
filepath = os.path.join(path, 'Analysis')
listdir    = os.listdir(filepath)
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
            
#%% Load UP and DOWN states

file = os.path.join(writepath, name +'.evt.py.dow')
down_ep = data.read_neuroscope_intervals(name = 'DOWN', path2file = file)

file = os.path.join(writepath, name +'.evt.py.upp')
up_ep = data.read_neuroscope_intervals(name = 'UP', path2file = file)

#%% Compute PETH 

cc2 = nap.compute_eventcorrelogram(spikes, nap.Tsd(up_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
tmp = pd.DataFrame(cc2)
tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)

#%% Example units

plt.figure()
plt.plot(tmp[pyr][23][-0.05:0.2])
plt.plot(tmp[pyr][50][-0.05:0.2])
plt.axhline(1, linestyle = '--', color = 'silver')
plt.axvline(0, color = 'k')
plt.yticks([1], ['mean rate'])
plt.xticks([0], ['DU'])
plt.gca().set_box_aspect(1)
