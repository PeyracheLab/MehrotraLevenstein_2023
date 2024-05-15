#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:44:38 2023

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
from scipy.stats import kendalltau

#%% Load the session 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
readpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'
writepath = '/home/dhruv/Code/MehrotraLevenstein_2023/Analysis/OutputFiles'
s = 'A3707-200317'

name = s.split('/')[-1]
path = os.path.join(data_directory, s)
rawpath = os.path.join(readpath,s)
filepath = os.path.join(path, 'Analysis')
listdir = os.listdir(filepath)

spikes, shank = loadSpikeData(path)
n_channels, fs, shank_to_channel = loadXML(rawpath)

#Load cell depths
file = [f for f in listdir if 'CellDepth' in f]
celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
depth = celldepth['cellDep']

#Load cell types 
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


#Load UP and DOWN states
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
    
#%% Compute Peri-event time Histogram (PETH)

binsize = 5
nbins = 1000        
neurons = list(spikes.keys())
times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2        
cc = pd.DataFrame(index = times, columns = neurons)
tsd_up = up_ep.as_units('ms').start.values

rates = []

for i in neurons:
    spk2 = spikes[i].restrict(up_ep).as_units('ms').index.values
    tmp = crossCorr(tsd_up, spk2, binsize, nbins)
   
    tmp = pd.DataFrame(tmp)
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
    #Normalize PETH by mean rate
    fr = len(spk2)/up_ep.tot_length('s')
    rates.append(fr)
    cc[i] = tmp.values
    cc[i] = tmp.values/fr
    dd = cc[0:150]
    
#Only EX cells
ee = dd[pyr]

#%% Compute Onset 

indexplot_ex = []
depths_keeping_ex = []

for i in range(len(ee.columns)):
    a = np.where(ee.iloc[:,i] > 0.5)
    if len(a[0]) > 0:
        depths_keeping_ex.append(depth.flatten()[ee.columns[i]])
        res = ee.iloc[:,i].index[a]
        indexplot_ex.append(res[0])

#Onset v/s depth correlation
coef_ex, p_ex = kendalltau(indexplot_ex,depths_keeping_ex)

#%% Compute line of best fit 

y_est_ex = np.zeros(len(depths_keeping_ex))
m_ex, b_ex = np.polyfit(indexplot_ex, depths_keeping_ex, 1)

for i in range(len(indexplot_ex)):
    y_est_ex[i] = m_ex*indexplot_ex[i]
    
#%%  Plotting

plt.figure()
plt.plot(ee[1][0:100], color = 'k', linewidth = 2)
plt.axhline(0.5, color = 'silver', linestyle = '--')
plt.xlabel('UP onset delay (ms)')
plt.ylabel('Rate')
plt.yticks([0, 0.5, 1.5])

plt.figure()
plt.scatter(indexplot_ex, depths_keeping_ex, color = 'cornflowerblue')
plt.plot(indexplot_ex, y_est_ex + b_ex, color = 'cornflowerblue')
plt.ylabel('Depth from top of probe (um)')
plt.yticks([0, -400, -800])
plt.xlabel('UP onset delay (ms)')
