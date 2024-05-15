#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:44:08 2023

@author: dhruv
"""

#import dependencies
import numpy as np 
import pandas as pd 
import scipy.io
import nwbmatic as ntm
import pynapple as nap 
import os
import matplotlib.pyplot as plt 
from itertools import combinations
from scipy.stats import pearsonr
from sklearn.manifold import Isomap

#%% Data organization

data_directory = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Data/AdrianPoSub/###AllPoSub'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')

readpath = '/media/DataDhruv/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/PoSub-UPstate/Data'
writepath = '/home/dhruv/Code/MehrotraLevenstein_2023/Analysis/OutputFiles'

allPETH = pd.DataFrame()
uponset = []
PMR = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(readpath,s)

###Loading the data
    data = ntm.load_session(rawpath, 'neurosuite')
    data.load_neurosuite_xml(rawpath)
    spikes = data.spikes  
        
###Load cell types
    filepath = os.path.join(path, 'Analysis')
    listdir    = os.listdir(filepath)
    
    file = [f for f in listdir if 'CellDepth' in f]
    celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
    depth = celldepth['cellDep']
    
    
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
    down_ep = data.read_neuroscope_intervals(name = 'DOWN', path2file = file)
    
    file = os.path.join(writepath, name +'.evt.py.upp')
    up_ep = data.read_neuroscope_intervals(name = 'UP', path2file = file)
    
#%% Compute PETH 

    cc2 = nap.compute_eventcorrelogram(spikes, nap.Ts(up_ep['start']), binsize = 0.005, windowsize = 0.255, ep = up_ep, norm = True)
    tmp = pd.DataFrame(cc2)
    tmp = tmp.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    dd2 = tmp[0:0.155]  
    
    #Only EX cells
    ee = dd2[pyr] 
    
#%% Concatenate all PETHs for Isomap

    if len(ee.columns) > 0:
                    
        tokeep = []
                            
        for i in range(len(ee.columns)):
            a = np.where(ee.iloc[:,i] > 0.5)
            if len(a[0]) > 0:
              tokeep.append(ee.columns[i])  
              PMR.append(ee.iloc[:,i].max())
              res = ee.iloc[:,i].index[a]
              uponset.append(res[0])
              
    allPETH = pd.concat([allPETH, ee[tokeep]], axis = 1)

#%% Isomap 

projection = Isomap(n_components = 2, n_neighbors = 50).fit_transform(allPETH.T.values)   
H = PMR/(max(PMR))

norm_onset = uponset/max(uponset)
cmap = plt.cm.OrRd

plt.figure(figsize = (8,8))
plt.scatter(projection[:,0], projection[:,1], c = cmap(H))
plt.gca().set_box_aspect(1)
plt.xlabel('dim 1')
plt.xticks([])
plt.ylabel('dim 2')
plt.yticks([])

#%% Specific examples from Isomap
    
allPETH.columns = range(allPETH.columns.size)

summ = {}
summ['peth'] = allPETH
summ['p1'] = projection[:,0]
summ['p2'] = projection[:,1]

examples = [2, 208, 226, 378, 528, 828, 1018, 1061, 1081]  

for i in examples:

    plt.figure()
    plt.title('Example ' + str(i))
    plt.plot(summ['peth'][i])
    plt.axhline(y = 1, linestyle = '--', color = 'k')
    plt.xlabel('Time from DU (s)')
    plt.ylabel('Norm. rate')        
    plt.gca().set_box_aspect(1)

#%% Plot of PMR as a function of Isomap dimension 1 

r, p = pearsonr(projection[:,0], PMR)

plt.figure()
plt.scatter(projection[:,0], PMR, label = 'r = ' + str(round(r,2)))
plt.gca().set_box_aspect(1)
plt.xlabel('dim 1')
plt.xticks([])
plt.ylabel('PMR')
plt.yticks([0.5, 3])
plt.legend(loc = 'upper right')
    
#%% Gradient vector 

pairs = list(combinations(summ['peth'].columns, 2))

F_pmrr = []

for i,p in enumerate(pairs):
    diff_pmrr = PMR[p[0]] - PMR[p[1]]
           
    dx = summ['p1'][p[0]] - summ['p1'][p[1]]
    dy = summ['p2'][p[0]] - summ['p2'][p[1]]
       
    F_pmrr.append([diff_pmrr/dx, diff_pmrr/dy])
    
mean_pmrr = np.mean(F_pmrr, axis = 0)

origin = np.array([[0, 0], [0, 0]])
plt.figure()
plt.title('Gradient vector')
plt.xlim(0, 0.4)
plt.ylim(-0.4, 0)
plt.xlabel('dim 1')
plt.xticks([])
plt.ylabel('dim 2')
plt.yticks([])
plt.quiver(origin[0], origin[1],  mean_pmrr[0] ,  mean_pmrr[1], angles = 'xy', scale_units = 'xy', scale = 1)