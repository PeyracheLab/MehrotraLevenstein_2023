import numpy as np
import sys,os
import scipy.io
from Dependencies import neuroseries as nts 
import pandas as pd
import scipy.signal
from numba import jit
'''
Wrappers should be able to distinguish between raw data or matlab processed data
'''

def loadSpikeData(path, index=None, fs = 20000):
    """
    if the path contains a folder named /Analysis, 
    the script will look into it to load either
        - SpikeData.mat saved from matlab
        - SpikeData.h5 saved from this same script
    if not, the res and clu file will be loaded 
    and an /Analysis folder will be created to save the data
    Thus, the next loading of spike times will be faster
    Notes :
        If the frequency is not given, it's assumed 20kH
    Args:
        path : string
    Returns:
        dict, array    
    """    
    if not os.path.exists(path):
        print("The path "+path+" doesn't exist; Exiting ...")
        sys.exit()    
    new_path = os.path.join(path, 'Analysis/')
    if os.path.exists(new_path):
        new_path    = os.path.join(path, 'Analysis/')
        files        = os.listdir(new_path)
        if 'SpikeData.mat' in files:
            spikedata     = scipy.io.loadmat(new_path+'SpikeData.mat')
            shank         = spikedata['shank'] - 1
            if index is None:
                shankIndex     = np.arange(len(shank))
            else:
                shankIndex     = np.where(shank == index)[0]
            spikes         = {}    
            for i in shankIndex:    
                spikes[i]     = nts.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2], time_units = 's')
            a             = spikes[0].as_units('s').index.values    
            if ((a[-1]-a[0])/60.)/60. > 20. : # VERY BAD        
                spikes         = {}    
                for i in shankIndex:
                    spikes[i]     = nts.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2]*0.0001, time_units = 's')
            return spikes, shank
        elif 'SpikeData.h5' in files:            
            final_path = os.path.join(new_path, 'SpikeData.h5')            
            try:
                spikes = pd.read_hdf(final_path, mode='r')
                # Returning a dictionary | can be changed to return a dataframe
                toreturn = {}
                for i,j in spikes:
                    toreturn[j] = nts.Ts(t=spikes[(i,j)].replace(0,np.nan).dropna().index.values, time_units = 's')
                shank = spikes.columns.get_level_values(0).values[:,np.newaxis]
                return toreturn, shank
            except:
                spikes = pd.HDFStore(final_path, 'r')
                shanks = spikes['/shanks']
                toreturn = {}
                for j in shanks.index:
                    toreturn[j] = nts.Ts(spikes['/spikes/s'+str(j)])
                shank = shanks.values
                spikes.close()
                del spikes
                return toreturn, shank
            
        else:            
            print("Couldn't find any SpikeData file in "+new_path)
            print("If clu and res files are present in "+path+", a SpikeData.h5 is going to be created")

    # Creating /Analysis/ Folder here if not already present
    if not os.path.exists(new_path): os.makedirs(new_path)
    files = os.listdir(path)
    clu_files     = np.sort([f for f in files if 'clu' in f and f[0] != '.'])
    res_files     = np.sort([f for f in files if 'res' in f and f[0] != '.'])
    clu1         = np.sort([int(f.split(".")[-1]) for f in clu_files])
    clu2         = np.sort([int(f.split(".")[-1]) for f in res_files])
    if len(clu_files) != len(res_files) or not (clu1 == clu2).any():
        print("Not the same number of clu and res files in "+path+"; Exiting ...")
        sys.exit()
    count = 0
    spikes = []
    basename = clu_files[0].split(".")[0]
    for i, s in zip(range(len(clu_files)),clu1):
        clu = np.genfromtxt(os.path.join(path,basename+'.clu.'+str(s)),dtype=np.int32)[1:]
        if np.max(clu)>1:
            # print(i,s)
            res = np.genfromtxt(os.path.join(path,basename+'.res.'+str(s)))
            tmp = np.unique(clu).astype(int)
            idx_clu = tmp[tmp>1]
            idx_col = np.arange(count, count+len(idx_clu))            
            tmp = pd.DataFrame(index = np.unique(res)/fs,
                                columns = pd.MultiIndex.from_product([[s],idx_col]),
                                data = 0, 
                                dtype = np.uint16)
            for j, k in zip(idx_clu, idx_col):
                tmp.loc[res[clu==j]/fs,(s,k)] = np.uint16(k+1)
            spikes.append(tmp)
            count+=len(idx_clu)

            # tmp2 = pd.DataFrame(index=res[clu==j]/fs, data = k+1, ))
            # spikes = pd.concat([spikes, tmp2], axis = 1)


    # Returning a dictionnary
    toreturn =  {}
    shank = []
    for s in spikes:
        shank.append(s.columns.get_level_values(0).values)
        sh = np.unique(shank[-1])[0]
        for i,j in s:
            toreturn[j] = nts.Ts(t=s[(i,j)].replace(0,np.nan).dropna().index.values, time_units = 's')

    del spikes
    shank = np.hstack(shank)

    final_path = os.path.join(new_path, 'SpikeData.h5')
    store = pd.HDFStore(final_path)
    for s in toreturn.keys():
        store.put('spikes/s'+str(s), toreturn[s].as_series())
    store.put('shanks', pd.Series(index = list(toreturn.keys()), data = shank))
    store.close()

    return toreturn, shank


def loadXML(path):
    """
    path should be the folder session containing the XML file
    Function returns :
        1. the number of channels
        2. the sampling frequency of the dat file or the eeg file depending of what is present in the folder
            eeg file first if both are present or both are absent
        3. the mappings shanks to channels as a dict
    Args:
        path : string
    Returns:
        int, int, dict
    """
    if not os.path.exists(path):
        print("The path "+path+" doesn't exist; Exiting ...")
        sys.exit()
    listdir = os.listdir(path)
    xmlfiles = [f for f in listdir if f.endswith('.xml')]
    if not len(xmlfiles):
        print("Folder contains no xml files; Exiting ...")
        sys.exit()
    new_path = os.path.join(path, xmlfiles[0])
    
    from xml.dom import minidom    
    xmldoc         = minidom.parse(new_path)
    nChannels     = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
    fs_dat         = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data
    fs_eeg         = xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[0].firstChild.data    
    if os.path.splitext(xmlfiles[0])[0] +'.dat' in listdir:
        fs = fs_dat
    elif os.path.splitext(xmlfiles[0])[0] +'.eeg' in listdir:
        fs = fs_eeg
    else:
        fs = fs_eeg
    shank_to_channel = {}
    groups         = xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[0].getElementsByTagName('group')
    for i in range(len(groups)):
        shank_to_channel[i] = np.sort([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
    return int(nChannels), int(fs), shank_to_channel


def loadLFP(path, n_channels=90, channel=64, frequency=1250.0, precision='int16'):
    from Dependencies import neuroseries as nts 
    if type(channel) is not list:
        f = open(path, 'rb')
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2        
        n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
        duration = n_samples/frequency
        interval = 1/frequency
        f.close()
        with open(path, 'rb') as f:
            data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:,channel]
        timestep = np.arange(0, len(data))/frequency
        return nts.Tsd(timestep, data, time_units = 's')
    elif type(channel) is list:
        f = open(path, 'rb')
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2
        
        n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
        duration = n_samples/frequency
        f.close()
        with open(path, 'rb') as f:
            data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:,channel]
        timestep = np.arange(0, len(data))/frequency
        return nts.TsdFrame(timestep, data, time_units = 's')

    
def loadAuxiliary(path, n_probe = 1, fs = 20000):
    """
    Extract the acceleration from the auxiliary.dat for each epochs
    Downsampled at 100 Hz
    Args:
        path: string
        epochs_ids: list        
    Return: 
        TsdArray
    """     
    if not os.path.exists(path):
        print("The path "+path+" doesn't exist; Exiting ...")
        sys.exit()
    if 'Acceleration.h5' in os.listdir(os.path.join(path, 'Analysis')):
        accel_file = os.path.join(path, 'Analysis', 'Acceleration.h5')
        store = pd.HDFStore(accel_file, 'r')
        accel = store['acceleration'] 
        store.close()
        accel = nts.TsdFrame(t = accel.index.values*1e6, d = accel.values) 
        return accel
    else:
        aux_files = np.sort([f for f in os.listdir(path) if 'auxiliary' in f])
        if len(aux_files)==0:
            print("Could not find "+f+'_auxiliary.dat; Exiting ...')
            sys.exit()

        accel = []
        sample_size = []
        for i, f in enumerate(aux_files):
            new_path     = os.path.join(path, f)
            f             = open(new_path, 'rb')
            startoffile = f.seek(0, 0)
            endoffile     = f.seek(0, 2)
            bytes_size     = 2
            n_samples     = int((endoffile-startoffile)/(3*n_probe)/bytes_size)
            duration     = n_samples/fs        
            f.close()
            tmp         = np.fromfile(open(new_path, 'rb'), np.uint16).reshape(n_samples,3*n_probe)
            accel.append(tmp)
            sample_size.append(n_samples)
            del tmp

        accel = np.concatenate(accel)    
        factor = 37.4e-6
        # timestep = np.arange(0, len(accel))/fs
        # accel = pd.DataFrame(index = timestep, data= accel*37.4e-6)
        tmp  = []
        for i in range(accel.shape[1]):
            tmp.append(scipy.signal.resample_poly(accel[:,i]*factor, 1, 100))
        tmp = np.vstack(tmp).T
        timestep = np.arange(0, len(tmp))/(fs/100)
        tmp = pd.DataFrame(index = timestep, data = tmp)
#         accel_file = os.path.join(path, 'Analysis', 'Acceleration.h5')
#         store = pd.HDFStore(accel_file, 'w')
#         store['acceleration'] = tmp
#         store.close()
        accel = nts.TsdFrame(t = tmp.index.values*1e6, d = tmp.values) 
        return accel    
    
