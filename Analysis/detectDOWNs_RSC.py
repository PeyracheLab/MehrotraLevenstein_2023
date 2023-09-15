#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:16:31 2023

@author: dhruv
"""

#import dependencies
import numpy as np 
import os
import pynapple as nap
import matplotlib.pyplot as plt 

#%% Data organization

data_directory = '/media/dhruv/LaCie1/A7800'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    
###Loading the data
    data = nap.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    spikes = data.spikes
    
#%% Fishing out wake and sleep epochs
    
    epochs = data.epochs    
    sleep_ep = epochs['sleep']
    wake_ep = epochs['wake']

    file = os.path.join(path, name +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)

#%% Detection of UP and DOWN states
    
    bin_size = 0.01 #In seconds
    smoothing_window = 0.02 #In seconds
    
    #Create binned spike times
    rates = spikes.count(bin_size, sws_ep)
    
    #Smooth binned spike times
    total2 = rates.as_dataframe().rolling(window = 100, win_type = 'gaussian',
                                          center = True, min_periods = 1, 
                                          axis = 0).mean(std = int(smoothing_window/bin_size))
    
    #Create population rate/multi-unit activity (MUA)
    total2 = total2.sum(axis = 1)
    total2 = nap.Tsd(total2)
    
    #Apply the threshold
    idx = total2.threshold(np.percentile(total2.values,20),'below')
    
    #Exclusion criteria     
    down_ep = idx.time_support
    down_ep = nap.IntervalSet(start = down_ep['start'], end = down_ep['end'])
    down_ep = down_ep.drop_short_intervals(bin_size)
    down_ep = down_ep.merge_close_intervals(bin_size*2)
    down_ep = down_ep.drop_short_intervals(bin_size*3) 
    down_ep = down_ep.drop_long_intervals(bin_size*50)
    
    #Create the UP state   
    up_ep = nap.IntervalSet(down_ep['end'][0:-1], down_ep['start'][1:])
    down_ep = sws_ep.intersect(down_ep)
    
#%% Write the data as an event file

    data.write_neuroscope_intervals(extension = '.evt.py.dow', isets = down_ep, name = 'DOWN') 
    data.write_neuroscope_intervals(extension = '.evt.py.upp', isets = up_ep, name = 'UP') 