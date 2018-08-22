#! python
""" sspals: python tools for analysing single-shot positron annihilation lifetime spectra

    Copyright (c) 2015-2018, UNIVERSITY COLLEGE LONDON
    @author: Adam Deller
"""
from __future__ import division
from math import floor, ceil
from scipy import integrate
import numpy as np
import pandas as pd
from .chmx import chmx
from .cfd import cfd_1d

#    ----------------
#    delayed fraction
#    ----------------

def integral(arr, dt, t0, lim_a, lim_b, **kwargs):
    ''' Simpsons integration of arr (1D) between t=lim_a and t=lim_b.

        args:
            arr                   # numpy.array(dims=1)
            dt                    # float64
            t0                    # float64
            lim_a                 # float64
            lim_b                 # float64

        kwargs:
            corr = True         # apply boundary corrections
            debug = False       # fail quietly, or not if True

        return:
            float64
    '''
    corr = kwargs.get('corr', True)
    debug = kwargs.get('debug', False)
    assert lim_b > lim_a, "lim_b must be greater than lim_a"
    # fractional index
    frac_a = (lim_a + t0) / dt
    frac_b = (lim_b + t0) / dt
    # nearest index
    ix_a = int(round(frac_a))
    ix_b = int(round(frac_b))
    try:
        int_ab = integrate.simps(arr[ix_a:ix_b], None, dt)
        if corr:
            # boundary corrections (trap rule)
            corr_a = dt * (ix_a - frac_a) * (arr[int(floor(frac_a))] + arr[int(ceil(frac_a))]) / 2.0
            corr_b = dt * (ix_b - frac_b) * (arr[int(floor(frac_b))] + arr[int(ceil(frac_b))]) / 2.0
            int_ab = int_ab + corr_a - corr_b
    except IndexError:
        if not debug:
            # fail quietly
            int_ab = np.nan
        else:
            raise
    except:
        raise
    return int_ab

def dfrac(arr, dt, t0, limits, **kwargs):
    ''' Calculate the delayed fraction (DF) (int B->C/ int A->C) for arr (1D).

        args:
            arr                   # numpy.array(dims=1)
            dt                    # float64
            t0                    # float64
            limits                # (A, B, C)

        kwargs:
            corr = True         # apply boundary corrections
            debug = False       # fail quietly, or not if True

        return:
            AC :: float64, BC :: float64, DF :: float64
    '''
    int_ac = integral(arr, dt, t0, limits[0], limits[2], **kwargs)
    int_bc = integral(arr, dt, t0, limits[1], limits[2], **kwargs)
    df = int_bc / int_ac
    return int_ac, int_bc, df

def sspals_1d(arr, dt, limits, **kwargs):
    ''' Calculate the trigger time (cfd) and delayed fraction (BC / AC) for
        arr (1D).

        args:
            arr                   # numpy.array(dims=1)
            dt                    # float64
            limits                # (A, B, C)

        kwargs:
            cfd_scale=0.8
            cfd_offset=1.4e-8
            cfd_threshold=0.04
            corr=True
            debug=False

        return:
            (t0, AC, BC, DF)
    '''
    t0 = cfd_1d(arr, dt, **kwargs)
    if not np.isnan(t0):
        int_ac, int_bc, df = dfrac(arr, dt, t0, limits, **kwargs)
        return (t0, int_ac, int_bc, df)
    else:
        return (np.nan, np.nan, np.nan, np.nan)

def sspals(arr, dt, limits, axis=1, **kwargs):
    ''' Apply sspals_1D to each row of arr (2D).

        args:
            arr                   # numpy.array(dims=1)
            dt                    # float64
            limits                # (A, B, C)
            axis=1                # int

        kwargs:
            cfd_scale=0.8
            cfd_offset=1.4e-8
            cfd_threshold=0.04
            corr=True
            debug=False
            dropna = False

        return:
            pandas.DataFrame(columns=[t0, AC, BC, DF])
    '''
    dropna = kwargs.get('dropna', False)
    data = np.apply_along_axis(sspals_1d, axis, arr, dt, limits, **kwargs)
    if axis == 0:
        data = data.T
    dfracs = pd.DataFrame(data, columns=['t0', 'AC', 'BC', 'DF'])
    if dropna:
        dfracs = dfracs.dropna(axis=0, how='any')
    return dfracs

def chmx_sspals(high, low, dt, axis=1, **kwargs):
    """ Combine high and low gain data (chmx).  Re-analyse each to find t0 (cfd
        trigger) and the delayed fraction (fd = BC/ AC) for limits=[A, B, C].

        args:
            high                               # np.array(dims=2)
            low                                # np.array(dims=2)
            dt                                 # float64

        kwargs:
            n_bsub=100                         # number of points to use to find offset
            invert=True                        # invert signal (e.g., PMT)
            min_range=None                     # remove rows where vertical range < min_range
            cfd_scale=0.8                      # cfd
            cfd_offset=1.4e-8
            cfd_threshold=0.04
            limits=[-1.0e-8, 3.5e-8, 6.0e-7]   # delayed fraction ABC
            corr=True                          # apply boundary corrections
            dropna=False                       # remove empty rows
            debug=False                        # nans in output? try debug=True.
            trad=False                         # return (t0, AC, DC, DF)
            index_name=measurement             # pd.DataFrame.index.name

        return:
            pd.DataFrame(['t0', 'fd', 'total']))
    """
    trad = kwargs.get('trad', False)
    index_name = kwargs.get('index_name', 'measurement')
    df = sspals(chmx(high, low, axis, **kwargs), dt, axis, **kwargs)
    if not trad:
        df = df[['t0', 'DF', 'AC']]
        df.rename(index=str, columns={"DF": "fd", "AC": "total"}, inplace=True)
    if index_name is not None:
        df.index.rename(index_name, inplace=True)
    return df

#    -------
#    S_gamma
#    -------

def signal(a_val, a_err, b_val, b_err, rescale=100.0):
    ''' Calculate S = (b - a)/ b and the uncertainty.

        args:
            a_val
            a_err
            b_val
            b_err

        kwargs:
            rescale = 100.0    # e.g., for percentage units.

        return:
            rescale * (S, S_err)
    '''
    sig = rescale * (b_val - a_val) / b_val
    sig_err = rescale * np.sqrt((a_err / b_val)**2.0 + (a_val*b_err/(b_val**2.0))**2.0)
    return sig, sig_err
