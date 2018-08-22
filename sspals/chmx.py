#! python
""" sspals: python tools for analysing single-shot positron annihilation lifetime spectra

    Copyright (c) 2015-2018, UNIVERSITY COLLEGE LONDON
    @author: Adam Deller
"""
from __future__ import division
import numpy as np

#    ------------
#    process data
#    ------------

def sub_offset(arr, n_bsub=100, axis=1):
    ''' Subtract the mean of the first 'n_bsub' number of points for each row in arr.

        args:
            arr                   # numpy.array()
            n_bsub=100
            axis=1

        return:
            (arr - offset) :: numpy.array(dims=2), offset :: float64
    '''
    offset = np.array([np.mean(arr[:, :n_bsub], axis=axis)])
    arr = np.subtract(arr, offset.T)
    return arr, offset

def saturated(arr):
    ''' Find where arr (1D) is equal to its own max and min value.

        args:
            arr                   # numpy.array()

        return:
            numpy.array(dims=1, dtype=bool)
    '''
    sat = np.logical_or(arr == arr.max(), arr == arr.min())
    return sat

def splice(high, low, axis=1):
    ''' Splice together the high and low gain values of a 2D dataset (swap saturated sections
        in the high-gain channel for the corresponding values in the low-gain channel).

        args:
            high                   # numpy.array(dims=2)
            low                    # numpy.array(dims=2)
            axis=1                 # int

        return:
            numpy.array(dims=2)
    '''
    mask = np.apply_along_axis(saturated, axis, high)
    flask = mask.flatten()
    vals = low.flatten()[np.where(flask)]          # replacement values
    tmp = high.flatten()
    tmp[flask] = vals
    arr = np.reshape(tmp, np.shape(high))
    return arr

#    -------------
#    validate data
#    -------------

def val_test(arr, min_range):
    ''' Does the vertical range of arr (1D) exceed min_range?

        args:
            arr                   # numpy.array()
            min_range             # float64

        return:
            bool
    '''
    rng = abs(arr.max() - arr.min())
    return rng > min_range

def validate(arr, min_range, axis=1):
    ''' Remove rows from arr (2D) that have a vertical range < min_range.

        args:
            arr                   # numpy.array()
            min_range             # float64
            axis=1                # int

        return:
            numpy.array(dims=2)
    '''
    mask = np.apply_along_axis(val_test, axis, arr, min_range)
    return arr[mask]

#    ------------------------------
#    combine high and low gain data
#    ------------------------------

def chmx(high, low, axis=1, **kwargs):
    ''' Remove zero offset from high and low gain data, invert and splice
        together by swapping saturated values from the hi-gain channel
        for those from the low-gain channel.  Apply along rows of 2D arrays.

        args:
            high                   # numpy.array(dims=2)
            low                    # numpy.array(dims=2)
            axis=1                 # int

        kwargs:
            n_bsub=100             # number of points to use to find offset
            invert=True            # invert signal (e.g., PMT)
            min_range=None         # remove rows where vertical range < min_range
            axis=1                 # int

        return:
            numpy.array(dims=2)
    '''
    # options
    invert = kwargs.get('invert', True)
    n_bsub = kwargs.get('n_bsub', 100)
    min_range = kwargs.get('min_range', None)
    # remove offsets
    if n_bsub is not None and n_bsub > 0:
        high, _ = sub_offset(high, n_bsub, axis=axis)
        low, _ = sub_offset(low, n_bsub, axis=axis)
    # combine hi/low data
    arr = splice(high, low, axis=axis)
    if invert:
        arr = np.negative(arr)
    if min_range is not None:
        # validate data
        arr = validate(arr, min_range, axis=axis)
    return arr
