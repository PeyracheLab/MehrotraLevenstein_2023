#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:44:45 2024

@author: dhruv
"""

#import dependencies
import numpy as np 
import scipy.io
import os
import nwbmatic as ntm
import pynapple as nap

#%% Data organization

data_directory = '/media/dhruv/LaCie1/dataAdrien'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)

###Loading the data
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    channelorder = data.group_to_channel[0]
    spikes = data.spikes
    
#%% Fishing out wake and sleep epochs

    epochs = data.epochs    
    sleep_ep = epochs['sleep']
    wake_ep = epochs['wake']
    
#%% Get NREM epoch after opto correction
      
    file = os.path.join(path, name +'.sts.SWS')
    if os.path.exists(file):
        tmp = np.genfromtxt(file) / 1250
        sws_ep_withOpto = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
        
    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)
    file = [f for f in listdir if 'BehavEpochs' in f]
    behepochs = scipy.io.loadmat(os.path.join(filepath,file[0]))
    
    sleep1_ep = np.hstack([behepochs['sleepPreEp'][0][0][1],behepochs['sleepPreEp'][0][0][2]])
    sleep1_ep = nap.IntervalSet(sleep1_ep[:,0], sleep1_ep[:,1], time_units = 's').drop_short_intervals(0.0)
    
    sleep2_ep = np.hstack([behepochs['sleepPostEp'][0][0][1],behepochs['sleepPostEp'][0][0][2]])
    sleep2_ep = nap.IntervalSet(sleep2_ep[:,0], sleep2_ep[:,1], time_units = 's').drop_short_intervals(0.0)
    
    sws_ep = sws_ep_withOpto.intersect(sleep1_ep.union(sleep2_ep))
    
#%% Detection of UP and DOWN states

    spikes_by_location = spikes.getby_category('location')
    spikes_adn = spikes_by_location['adn']
    
    bin_size = 0.01 #In seconds
    smoothing_window = 0.02 #In seconds 
    
    #Create binned spike times
    rates = spikes_adn.count(bin_size, sws_ep)
    
    #Smooth binned spike times
    total2 = rates.as_dataframe().rolling(window = 100 ,win_type='gaussian',
                                          center=True,min_periods=1, 
                                          axis = 0).mean(std= int(smoothing_window/bin_size))
    
    #Create population rate/multi-unit activity (MUA)
    total2 = total2.sum(axis = 1)
    total2 = nap.Tsd(total2)
    
    #Apply the threshold
    idx = total2.threshold(np.percentile(total2.values,20),'below')
    
    #Exclusion criteria         
    down_ep_DM = idx.time_support
    down_ep_DM = nap.IntervalSet(start = down_ep_DM['start'], end = down_ep_DM['end'])
    down_ep_DM = down_ep_DM.drop_short_intervals(bin_size)
    down_ep_DM = down_ep_DM.merge_close_intervals(bin_size*2)
    down_ep_DM = down_ep_DM.drop_short_intervals(bin_size*5) #50 ms for ADn
    down_ep_DM = down_ep_DM.drop_long_intervals(bin_size*50)
    
    #Create the UP state      
    up_ep = nap.IntervalSet(down_ep_DM['end'][0:-1], down_ep_DM['start'][1:])
    down_ep = sws_ep.intersect(down_ep_DM)
    
#%% Write the data as an event file

    data.write_neuroscope_intervals(extension = '.evt.py.dow', isets = down_ep, name = 'DOWN') 
    data.write_neuroscope_intervals(extension = '.evt.py.upp', isets = up_ep, name = 'UP') 