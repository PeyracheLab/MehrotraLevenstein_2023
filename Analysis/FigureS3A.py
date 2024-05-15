#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:06:23 2024

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

#%% Data organization

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

readpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'
writepath = '/home/dhruv/Code/MehrotraLevenstein_2023/Analysis/OutputFiles'

dur_D = []
dur_V = []
pmeans = []
diff = []
sess = []

allends = []
allstarts = []
dends = []
dstart = []
vends = []
vstart = []

meanupdur = [] 
meandowndur = []
CVup = []
CVdown = []

allupdur = [] 
alldowndur = []

updist = pd.DataFrame()
downdist = pd.DataFrame()

uplogdist = pd.DataFrame()
downlogdist = pd.DataFrame()

#%% Load the data 

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(readpath,s)

###Load UP and DOWN states
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
    
#%% Compute durations of UP and DOWN states 

    dep = down_ep/1e6
    uep = up_ep/1e6
    
    updur = (uep['end'] - uep['start']) 
    meanupdur.append(np.mean(updur))
    CVup.append(np.std(updur)/np.mean(updur))
    allupdur.append([i for i in updur.values])
    
    downdur = (dep['end'] - dep['start'])
    alldowndur.append([i for i in downdur.values])
    meandowndur.append(np.mean(downdur))
    CVdown.append(np.std(downdur)/np.mean(downdur))
       
    
    upbins = np.linspace(0,8,60)
    downbins = np.linspace(0,2,60)
    logbins = np.linspace(np.log10(0.02), np.log10(50), 30)
    
    upd, _ = np.histogram(updur, upbins)
    upd = upd/sum(upd)
    
    downd, _  = np.histogram(downdur, downbins)
    downd = downd/sum(downd)
    
    uplogd,_ = np.histogram(np.log10(updur), logbins)
    uplogd = uplogd/sum(uplogd)
    
    downlogd,_ = np.histogram(np.log10(downdur), logbins)
    downlogd = downlogd/sum(downlogd)
    
    
    updist = pd.concat([updist, pd.Series(upd)], axis = 1)
    downdist = pd.concat([downdist, pd.Series(downd)], axis = 1)
            
    uplogdist = pd.concat([uplogdist, pd.Series(uplogd)], axis = 1)
    downlogdist = pd.concat([downlogdist, pd.Series(downlogd)], axis = 1)

#%% Plotting 

upbincenter = 0.5 * (upbins[1:] + upbins[:-1])
downbincenter = 0.5 * (downbins[1:] + downbins[:-1])
logbincenter = 0.5 * (logbins[1:] + logbins[:-1])

uperr = updist.std(axis=1)
downerr = downdist.std(axis=1)
uplogerr = uplogdist.std(axis=1)
downlogerr = downlogdist.std(axis=1)

plt.figure()
plt.xlabel('Duration (s)')
plt.xticks([0, 8])
plt.ylabel('P (duration)')
plt.yticks([])
plt.plot(upbincenter, updist.mean(axis = 1), color = 'r', label = 'UP')
plt.fill_between(upbincenter, updist.mean(axis = 1) - uperr, updist.mean(axis = 1) + uperr, color = 'r', alpha = 0.2)
plt.plot(downbincenter, downdist.mean(axis = 1), color = 'b', label = 'DOWN')
plt.fill_between(downbincenter, downdist.mean(axis = 1) - downerr, downdist.mean(axis = 1) + downerr, color = 'b', alpha = 0.2)
plt.legend(loc = 'upper right')

plt.figure()
plt.xlabel('Duration (s)')
plt.xticks([-1, 0, 1],[0.1, 1, 10])
plt.ylabel('P (duration)')
plt.yticks([])
plt.plot(logbincenter, uplogdist.mean(axis = 1), color = 'r', label = 'UP')
plt.fill_between(logbincenter, uplogdist.mean(axis = 1) - uplogerr, uplogdist.mean(axis = 1) + uplogerr, color = 'r', alpha = 0.2)
plt.plot(logbincenter, downlogdist.mean(axis = 1), color = 'b', label = 'DOWN')
plt.fill_between(logbincenter, downlogdist.mean(axis = 1) - downlogerr, downlogdist.mean(axis = 1) + downlogerr, color = 'b', alpha = 0.2)
plt.legend(loc = 'upper right')



