#! python
""" sspals: python tools for analysing single-shot positron annihilation lifetime spectra

    Copyright (c) 2015-2018, UNIVERSITY COLLEGE LONDON
    @author: Adam Deller
"""
from __future__ import division
import numpy as np

def cfd_1d(arr, dt, **kwargs):
    ''' Apply cfd algorithm to arr (1D) to find trigger time (t0).

        args:
            arr                   # numpy.array(dims=1)
            dt                    # float64

        kwargs:
            cfd_scale=0.8
            cfd_offset=1.4e-8
            cfd_threshold=0.04
            debug=False

        return:
            float64
    '''
    # options
    scale = kwargs.get('cfd_scale', 0.8)
    offset = kwargs.get('cfd_offset', 1.4E-8)
    threshold = kwargs.get('cfd_threshold', 0.04)
    debug = kwargs.get('debug', False)
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
        t0 = z[ix] * (x[ix] - x[ix + 1]) / (z[ix + 1] - z[ix]) + x[ix]
    else:
        # no triggers found
        if not debug:
            # fail quietly
            t0 = np.nan
        else:
            raise Warning("cfd failed to find a trigger.")
    return t0

def cfd(arr, dt, axis=1, **kwargs):
    ''' Apply cfd to each row of arr (2D) to find trigger times.

        args:
            arr                   # numpy.array(dims=2)
            dt                    # float64
            axis=1                # int

        kwargs:
            invert=False
            cfd_scale=0.8
            cfd_offset=1.4e-8
            cfd_threshold=0.04
            debug=False

        return:
            numpy.array(dims=1)
    '''
    invert = kwargs.get('invert', False)
    if invert:
        arr = np.negative(arr)
    # apply cfd
    return np.apply_along_axis(cfd, axis, arr, dt, **kwargs)
