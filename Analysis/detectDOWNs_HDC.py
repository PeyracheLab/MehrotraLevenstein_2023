#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:21:06 2023

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

#%% Data organization

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

readpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'
writepath = '/home/dhruv/Code/MehrotraLevenstein_2023/Analysis/OutputFiles'

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(readpath,s)

###Loading the data
    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(rawpath)

#%% Fishing out wake and sleep epochs 

    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)
    file = [f for f in listdir if 'BehavEpochs' in f]
    behepochs = scipy.io.loadmat(os.path.join(filepath,file[0]))

    if s == 'A3701-191119':
        sleep1_ep = np.hstack([behepochs['sleepPreEp'][0][0][1],behepochs['sleepPreEp'][0][0][2]])
        sleep1_ep = nts.IntervalSet(sleep1_ep[:,0], sleep1_ep[:,1], time_units = 's').drop_short_intervals(0.0)

        sleep2_ep = np.hstack([behepochs['sleepPostEp'][0][0][1],behepochs['sleepPostEp'][0][0][2]])
        sleep2_ep = nts.IntervalSet(sleep2_ep[:,0], sleep2_ep[:,1], time_units = 's').drop_short_intervals(0.0)
        
        wake_ep = np.hstack([behepochs['wakeEp'][0][0][1],behepochs['wakeEp'][0][0][2]])
        wake_ep = nts.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)
        
            
    else: 
        wake_ep = np.hstack([behepochs['wake1Ep'][0][0][1],behepochs['wake1Ep'][0][0][2]])
        wake_ep = nts.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)
        
        sleep1_ep = np.hstack([behepochs['sleep1Ep'][0][0][1],behepochs['sleep1Ep'][0][0][2]])
        
        #check if it is not empty, then go to next step
        if sleep1_ep.size != 0:
            print('sleep1 exists')
            sleep1_ep = nts.IntervalSet(sleep1_ep[:,0], sleep1_ep[:,1], time_units = 's').drop_short_intervals(0.0)
                

        sleep2_ep = np.hstack([behepochs['sleep2Ep'][0][0][1],behepochs['sleep2Ep'][0][0][2]])
        
        #check if it is not empty, then go to next step
        if sleep2_ep.size != 0: 
            print('sleep2 exists')
            sleep2_ep = nts.IntervalSet(sleep2_ep[:,0], sleep2_ep[:,1], time_units = 's').drop_short_intervals(0.0)
        
        
        
        #if both sleep1 and sleep2 are not empty, merge them. Else make the non-empty epoch sleep_ep
    if (sleep1_ep.size !=0 and sleep2_ep.size !=0): 
        
        if (sleep1_ep.start.values[0] > sleep2_ep.start.values[0]):
            sleep1_ep, sleep2_ep = sleep2_ep, sleep1_ep
    
        sleep_ep = pd.concat((sleep1_ep, sleep2_ep)).reset_index(drop=True)
            
          
    elif sleep1_ep.size != 0:
        sleep_ep = sleep1_ep
            
    else: 
        sleep_ep = sleep2_ep       

#%% Load LFP

    file = os.path.join(rawpath, name + '.lfp')
    if os.path.exists(file):    
        lfp = loadLFP(os.path.join(rawpath, name + '.lfp'), n_channels, 1, 1250, 'int16')
    else: 
        lfp = loadLFP(os.path.join(rawpath, name + '.eeg'), n_channels, 1, 1250, 'int16')
    
    downsample = 5 
    lfp = lfp[::downsample]
    
    #Load accelerometer data and use it to refine sleep
    acceleration = loadAuxiliary(rawpath, 1, fs = 20000) 
    newsleep_ep = refineSleepFromAccel(acceleration, sleep_ep)
        
    file = os.path.join(rawpath, name +'.sws.evt')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        sws_ep1 = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
        
    new_sws_ep = newsleep_ep.intersect(sws_ep1)

#%% Detection of UP and DOWN states

    bin_size = 10000 #microseconds
    rates = []

    for e in new_sws_ep.index:
        ep = new_sws_ep.loc[[e]]
        bins = np.arange(ep.iloc[0,0], ep.iloc[0,1], bin_size)       
        r = np.zeros((len(bins)-1))
   
    #Create population rate/multi-unit activity (MUA)
        for n in spikes.keys(): 
            tmp = np.histogram(spikes[n].restrict(ep).index.values, bins)[0]
            r = r + tmp
        rates.append(pd.Series(index = bins[0:-1] + np.diff(bins)/2, data = r))
    rates = pd.concat(rates)
    
    #Smooth the MUA
    total2 = rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)
       
    #Apply the threshold
    idx = total2[total2<np.percentile(total2,20)].index.values   
    
    tmp2 = [[idx[0]]]
    
    for i in range(1,len(idx)):
        if (idx[i] - idx[i-1]) > bin_size:
            tmp2.append([idx[i]])
        elif (idx[i] - idx[i-1]) == bin_size:
            tmp2[-1].append(idx[i])
            
   #Exclusion criteria
    down_ep = np.array([[e[0],e[-1]] for e in tmp2 if len(e) > 1])
    down_ep = nts.IntervalSet(start = down_ep[:,0], end = down_ep[:,1])
    down_ep = down_ep.drop_short_intervals(bin_size)
    down_ep = down_ep.reset_index(drop=True)
    down_ep = down_ep.merge_close_intervals(bin_size*2)
    down_ep = down_ep.drop_short_intervals(bin_size*3)
    down_ep = down_ep.drop_long_intervals(bin_size*50)
   
    #Create the UP state   
    up_ep = nts.IntervalSet(down_ep['end'][0:-1], down_ep['start'][1:])
    down_ep = new_sws_ep.intersect(down_ep)
    up_ep = new_sws_ep.intersect(up_ep)
    
#%% Write the data as an event file

    start = down_ep.as_units('ms')['start'].values
    ends = down_ep.as_units('ms')['end'].values

    datatowrite = np.vstack((start,ends)).T.flatten()

    n = len(down_ep)

    texttowrite = np.vstack(((np.repeat(np.array(['PyDown start 1']), n)), 
                              (np.repeat(np.array(['PyDown stop 1']), n))
                              )).T.flatten()

    evt_file = writepath + '/' + name + '.evt.py.dow'
       
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()        

    start = up_ep.as_units('ms')['start'].values
    ends = up_ep.as_units('ms')['end'].values

    datatowrite = np.vstack((start,ends)).T.flatten()

    n = len(up_ep)

    texttowrite = np.vstack(((np.repeat(np.array(['PyUp start 1']), n)), 
                              (np.repeat(np.array(['PyUp stop 1']), n))
                              )).T.flatten()

    evt_file = writepath + '/' + name + '.evt.py.upp'
         
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()

    
            
    