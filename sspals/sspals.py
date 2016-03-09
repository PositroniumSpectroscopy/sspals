#! python
from __future__ import print_function, division
"""
Copyright (c) 2015-2016, UNIVERSITY COLLEGE LONDON

@author: Adam Deller
"""
from scipy import integrate
from math import ceil
import numpy as np
import pandas as pd

''' 
    ------------
    process data 
    ------------
'''

def sub_offset(arr, n_bsub=100):
    ''' row-wise subtract the mean of the first 'n_bsub' number of points of arr (2D)
        
        return
            arr, offset    
    '''
    offset = np.mean(arr[:,:n_bsub], axis=1)
    arr = np.subtract(arr.T, offset).T
    return arr, offset

def saturated(arr):
    ''' find where arr (1D) is equal to its own max and min value'''
    s = np.logical_or(arr==arr.max(), arr==arr.min())
    return s
   
def splice(hi, low):
    '''splice together the hi and low gain values of a 2D dataset (assume hi saturated)'''
    mask = np.apply_along_axis(saturated, 1, hi)
    flask = mask.flatten() 
    vals = low.flatten()[np.where(flask)]          # replacement values
    tmp = hi.flatten()
    tmp[flask] = vals
    arr = np.reshape(tmp, np.shape(hi))
    return arr

''' 
    -------------
    validate data
    -------------
'''

def val_test(arr, min_range):
    ''' where arr (1D) exceeds min_range '''
    rng = abs(arr.max() - arr.min())
    return rng > min_range

def validate(arr, **kwargs):
    ''' filter out rows with range below min_range from 2D data set '''
    # options  
    min_range = kwargs.get('min_range', 0.2)
    mask = np.apply_along_axis(val_test, 1, arr, min_range)
    return arr[mask]

''' 
    ----------------------------
    combine hi and low gain data
    ----------------------------
'''

def chmx(hi, low, **kwargs):
    ''' Remove zero offset from hi and low gain data, invert and splice 
        together by swapping saturated values from the hi-gain channel 
        for those from the low-gain channel.  Apply along rows of 2D arrays.
        
        defaults:
            n_bsub = 100      # number of points to use to find offset
            invert = True     # assume a negative (PMT) signal
            validate = False  # only return rows above min_range
            min_range = 0.2   # for use with validate
    '''
    # options  
    invert = kwargs.get('invert', True)
    vtest = kwargs.get('validate', False)
    n_bsub = kwargs.get('n_bsub', 100)
    # remove offsets    
    hi = sub_offset(hi, n_bsub)[0]
    low  = sub_offset(low, n_bsub)[0]
    # combine hi/low data    
    arr = splice(hi, low)                  
    if invert:
        arr = np.negative(arr)
    if vtest:
        # validate data
        arr = validate(arr, **kwargs)
    return arr

''' 
    --------
    triggers
    --------
'''

def cfd(arr, dt, **kwargs):
    ''' Apply cfd algorithm to arr (1D). Return trigger time (t0).
    
        Defaults:
            scale = 0.8
            offset = 1.4E-8
            threshold = 0.04
    '''
    # options  
    scale = kwargs.get('scale', 0.8)
    offset = kwargs.get('offset', 1.4E-8)
    threshold = kwargs.get('threshold', 0.04)
    # offset number of points
    sub = int(offset /dt) 
    x = np.arange(len(arr)) * dt
    # add orig to inverted, rescaled and offset
    z = arr[:-sub]-arr[sub:]*scale
    # find where greater than threshold and passes through zero
    test = np.where(np.logical_and(arr[:-sub-1] > threshold, 
                                 np.bool_(np.diff(np.sign(z)))))[0]
    if len(test) > 0:
        ix = test[0]
        # interpolate to find t0
        t0 = z[ix]*(x[ix]-x[ix+1])/(z[ix+1]-z[ix])+x[ix] 
    else:
        # no triggers found
        t0 = np.nan
    return t0

def triggers(arr, dt, **kwargs):
    ''' apply cfd to each row of arr (2D) '''
    # apply cfd
    trigs = np.apply_along_axis(cfd, 1, arr, dt, **kwargs)
    return trigs

''' 
    ----------------
    delayed fraction
    ----------------
'''

def integral(arr, dt, t0, lims, corr=True):
    ''' integrate arr (1D) between bounds A and B '''
    a, b = lims
    ix_a = (a + t0)/dt
    ix_b = (b + t0)/dt
    if ix_b <= ix_a:
        raise ValueError("upper integration limit should be higher than lower limit.")
    ab = integrate.simps(arr[ceil(ix_a):ceil(ix_b)+1], None, dt)
    if corr:
        # boundary corrections
        corr1 = (arr[ceil(ix_a)]+arr[ceil(ix_a)-1])*(ceil(ix_a)-ix_a)*dt/2
        corr2 = (arr[ceil(ix_b)+1]+arr[ceil(ix_b)])*(ceil(ix_b)-ix_b)*dt/2 
        ab = ab + corr1 - corr2
    return ab

def dfrac(arr, dt, t0, **kwargs):
    ''' calculate the delayed fraction (BC/AC) for arr (1D) '''
    lims = kwargs.get('lims', [-1E-8, 3.5E-8, 6.0E-7])
    corr = kwargs.get('corr', True)
    AC = integral(arr, dt, t0, [lims[0], lims[2]], corr)
    BC = integral(arr, dt, t0, [lims[1], lims[2]], corr)
    DF = BC/AC
    return AC, BC, DF

def sspals_1D(arr, dt, **kwargs):
    ''' Calculate the trigger time (cfd) and delayed fraction (BC/AC) for
        arr (1D).  Return np.array([(t0, AC, BC, DF)]).
           
        Defaults:
            # cfd
            scale = 0.8
            offset = 1.4E-8
            threshold = 0.04
            
            # delayed fraction ABC
            lims=[-1.0E-8, 3.5E-8, 6.0E-7]                
    '''
    dtype=[('t0','float64'),('AC','float64'),('BC','float64'),('DF','float64')]
    t0 = cfd(arr, dt, **kwargs)
    if not np.isnan(t0):
        AC, BC, DF = dfrac(arr, dt, t0, **kwargs)
        output = np.array([(t0, AC, BC, DF)], dtype=dtype)
    else:
        output = np.array([(np.nan, np.nan, np.nan, np.nan)], dtype=dtype)
    return output 

def sspals(arr, dt, **kwargs):
    ''' Apply sspals_1D to each row or arr (2D).
    
        Defaults:
            drop_na = True       # remove empty rows
            
            # cfd
            scale = 0.8
            offset = 1.4E-8
            threshold = 0.04
            
            # delayed fraction ABC
            lims=[-1.0E-8, 3.5E-8, 6.0E-7]  
    '''
    dropna = kwargs.get('dropna', False)
    dfracs = pd.DataFrame(np.apply_along_axis(sspals_1D, 1, arr, dt, **kwargs)[:,0])
    if dropna:
        dfracs.dropna(axis=0, how='any')
    return dfracs

''' 
    -------
    S_gamma
    -------
'''

def signal(A, Aerr, B, Berr, rescale=100.0):
    ''' Calculate S = (B-A)/ B and uncertainty. Return S, S_err.
    '''
    S = rescale * (B - A) / B
    Serr = rescale * np.sqrt((Aerr / B)**2.0+(A*Berr/(B**2.0))**2.0)
    return S, Serr