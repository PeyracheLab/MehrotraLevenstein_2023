#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:29:12 2023

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
from scipy.stats import kendalltau, wilcoxon, mannwhitneyu
import seaborn as sns

#%% Data organization

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

readpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'
writepath = '/home/dhruv/Code/MehrotraLevenstein_2023/Analysis/OutputFiles'

allcoefs_up_ex = []
allcoefs_dn_ex = []

for s in datasets:
          
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(readpath,s)

###Loading the data
    spikes, shank = loadSpikeData(path)
    n_channels, fs, shank_to_channel = loadXML(rawpath)
    
###Load cell depths
    filepath = os.path.join(path, 'Analysis')
    listdir = os.listdir(filepath)
    file = [f for f in listdir if 'CellDepth' in f]
    celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
    depth = celldepth['cellDep']

###Load cell types 
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
        
#%% Compute Peri-event time Histogram (PETH) for UP onset

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

#%% Compute UP state Onset 

    indexplot_ex = []
    depths_keeping_ex = []
    
    for i in range(len(ee.columns)):
        a = np.where(ee.iloc[:,i] > 0.5)
        if len(a[0]) > 0:
            depths_keeping_ex.append(depth.flatten()[ee.columns[i]])
            res = ee.iloc[:,i].index[a]
            indexplot_ex.append(res[0])
    
    #Onset v/s depth correlation
    coef_ex, p_ex = kendalltau(indexplot_ex, depths_keeping_ex)
    allcoefs_up_ex.append(coef_ex)

#%% Compute PETH for DOWN onset

    binsize = 5
    nbins = 1000        
    neurons = list(spikes.keys())
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2        
    cc = pd.DataFrame(index = times, columns = neurons)
    tsd_dn = down_ep.as_units('ms').start.values
    
    rates = []
               
    for i in neurons:
        spk2 = spikes[i].restrict(up_ep).as_units('ms').index.values
        tmp = crossCorr(tsd_dn, spk2, binsize, nbins)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
        
        #Normalize PETH by mean rate
        fr = len(spk2)/up_ep.tot_length('s')
        rates.append(fr)
        cc[i] = tmp.values
        cc[i] = tmp.values/fr
       
        dd = cc[-250:250]
        
    #Only EX cells 
    ee = dd[pyr]

#%% Compute DOWN state onset 

    tmp_ex = ee.loc[5:] > 0.5
    
    tokeep_ex = tmp_ex.columns[tmp_ex.sum(0) > 0]
    ends_ex = np.array([tmp_ex.index[np.where(tmp_ex[i])[0][0]] for i in tokeep_ex])
    es_ex = pd.Series(index = tokeep_ex, data = ends_ex)
        
    tmp2_ex = ee.loc[-150:-5] > 0.5
    
    tokeep2_ex = tmp2_ex.columns[tmp2_ex.sum(0) > 0]
    start_ex = np.array([tmp2_ex.index[np.where(tmp2_ex[i])[0][-1]] for i in tokeep2_ex])
    st_ex = pd.Series(index = tokeep2_ex, data = start_ex)
        
    ix_ex = np.intersect1d(tokeep_ex,tokeep2_ex)
    ix_ex = [int(i) for i in ix_ex]
   
    depths_keeping_ex = depth[ix_ex]
    
    #Onset v/s depth correlation
    coef_ex, p_ex = kendalltau(st_ex[ix_ex], depths_keeping_ex)
    allcoefs_dn_ex.append(coef_ex)

#%% Summary data 

DUcorr = pd.DataFrame(allcoefs_up_ex)
UDcorr = pd.DataFrame(allcoefs_dn_ex)
DUtype = pd.DataFrame(['DU' for x in range(len(allcoefs_up_ex))])
UDtype = pd.DataFrame(['UD' for x in range(len(allcoefs_dn_ex))])

summary = pd.DataFrame()
summary['corr'] = pd.concat([DUcorr,UDcorr])
summary['Transition'] = pd.concat([DUtype,UDtype])

#%% Plotting 

sns.set_style('white')
palette = ['royalblue', 'lightsteelblue']
ax = sns.violinplot( x = summary['Transition'], y= summary['corr'] , data = summary, dodge = False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = summary['Transition'], y = summary['corr'] , data = summary, saturation = 1, showfliers = False,
            width = 0.3, boxprops = {'zorder': 3, 'facecolor': 'none'}, ax = ax)
old_len_collections = len(ax.collections)
sns.swarmplot(x = summary['Transition'], y = summary['corr'], data = summary, color = 'k', dodge = False, ax = ax)

for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.axhline(0, color = 'silver')
plt.ylabel('Delay v/s depth (R)')
ax.set_box_aspect(1)

#%% Stats 

w_up, p_up = wilcoxon(np.array(allcoefs_up_ex)-0)
w_dn, p_dn = wilcoxon(np.array(allcoefs_dn_ex)-0)
t, p = mannwhitneyu(allcoefs_up_ex, allcoefs_dn_ex)
