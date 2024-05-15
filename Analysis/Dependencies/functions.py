from numba import jit
import numpy as np
import pandas as pd
from Dependencies import neuroseries as nts 
import sys, os
import scipy
from scipy import signal
from itertools import combinations

'''
Utilities functions
Feel free to add your own
'''

#########################################################
# CORRELATION
#########################################################
@jit(nopython=True)
def crossCorr(t1, t2, binsize, nbins):
    ''' 
        Fast crossCorr 
    '''
    nt1 = len(t1)
    nt2 = len(t2)
    if np.floor(nbins/2)*2 == nbins:
        nbins = nbins+1

    m = -binsize*((nbins+1)/2)
    B = np.zeros(nbins)
    for j in range(nbins):
        B[j] = m+j*binsize

    w = ((nbins/2) * binsize)
    C = np.zeros(nbins)
    i2 = 1

    for i1 in range(nt1):
        lbound = t1[i1] - w
        while i2 < nt2 and t2[i2] < lbound:
            i2 = i2+1
        while i2 > 1 and t2[i2-1] > lbound:
            i2 = i2-1

        rbound = lbound
        l = i2
        for j in range(nbins):
            k = 0
            rbound = rbound+binsize
            while l < nt2 and t2[l] < rbound:
                l = l+1
                k = k+1

            C[j] += k

    # for j in range(nbins):
    # C[j] = C[j] / (nt1 * binsize)
    C = C/(nt1 * binsize/1000)

    return C

def compute_EventCrossCorr(spks, evt, ep, binsize = 5, nbins = 1000, norm=False):
    """
    """
    neurons = list(spks.keys())
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
    cc = pd.DataFrame(index = times, columns = neurons)
    tsd1 = evt.restrict(ep).as_units('ms').index.values
    for i in neurons:
        spk2 = spks[i].restrict(ep).as_units('ms').index.values
        tmp = crossCorr(tsd1, spk2, binsize, nbins)
        fr = len(spk2)/ep.tot_length('s')
        if norm:
            cc[i] = tmp/fr
        else:
            cc[i] = tmp
    return cc
        

def refineSleepFromAccel(acceleration, sleep_ep):
    vl = acceleration[0].restrict(sleep_ep)
    vl = vl.as_series().diff().abs().dropna()    
    a, _ = scipy.signal.find_peaks(vl, 0.025)
    peaks = nts.Tsd(vl.iloc[a])
    duration = np.diff(peaks.as_units('s').index.values)
    interval = nts.IntervalSet(start = peaks.index.values[0:-1], end = peaks.index.values[1:])

    newsleep_ep = interval.iloc[duration>15.0]
    newsleep_ep = newsleep_ep.reset_index(drop=True)
    newsleep_ep = newsleep_ep.merge_close_intervals(100000, time_units ='us')

    newsleep_ep    = sleep_ep.intersect(newsleep_ep)

    return newsleep_ep

#########################################################
# LFP FUNCTIONS
#########################################################
def butter_bandpass(lowcut, highcut, fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import filtfilt
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y



