#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:25:49 2023

@author: dhruv
"""

#import dependencies
import numpy as np 
import scipy.io
from Dependencies.functions import *
from Dependencies.wrappers import *
import os
from Dependencies import neuroseries as nts 
import matplotlib.pyplot as plt 

#%% Load the session 

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
s = 'A3707-200317'

name = s.split('/')[-1]
path = os.path.join(data_directory, s)
filepath = os.path.join(path, 'Analysis')
listdir = os.listdir(filepath)

spikes, shank = loadSpikeData(path)

#Load cell depths
file = [f for f in listdir if 'CellDepth' in f]
celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
depth = celldepth['cellDep']

#%% Example Epoch 

per = nts.IntervalSet(start = 1251288000, end = 1251688000) #20min 51s 288ms

## Sort neurons by depth
n = len(depth)
tmp = np.argsort(depth.flatten())
desc = tmp[::-1][:n]

#Plotting 
fig, ax = plt.subplots()
[plt.plot(spikes[i].restrict(per).as_units('ms').fillna(i), '|', color = 'k') for i in desc]


