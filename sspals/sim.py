#! python
""" sspals: python tools for analysing single-shot positron annihilation lifetime spectra

    Copyright (c) 2015-2018, UNIVERSITY COLLEGE LONDON
    @author: Adam Deller
"""
from __future__ import division
from scipy.special import erf                               # the error function
import numpy as np

def sim(t, amp=1.0, sigma=2.0E-9, eff=0.3, tau=1.420461E-7, kappa=1.0E-8, **kwargs):
    ''' Approximate a realistic SSPALS spectra, f(t), where t is an array of time values
        (in seconds).

        Gaussian(V_0, sigma) implantation time distribution and formation of o-Ps,
        convolved with detector function -- see below.

        args:
            t                       # numpy.array()
            amp=1.0                 # scaling factor
            sigma=2 ns              # Gaussian width
            eff=0.3                 # o-Ps re-emmission efficiency
            tau=142.0461 ns         # o-Ps lifetime
            kappa=10 ns             # detector decay time

        kwargs:
            norm=True               # normalise to max value

        return:
            numpy.array()

    '''
    norm = kwargs.get('norm', True)
    # sim.
    yvals = np.exp(-t *(1.0 / tau + 1.0 / kappa)) * ( \
            eff * \
            np.exp((sigma**2.0/(2.0 * tau**2.0)) + t/ kappa) * \
            (1.0 + erf((t * tau - sigma**2.0)/(np.sqrt(2.0) * sigma * tau))) - \
            (1 + tau * (eff - 1) / kappa) * \
            np.exp((sigma**2.0/(2.0 * kappa**2.0)) + t/ tau) * \
            (1.0 + erf((t * kappa - sigma**2.0)/(np.sqrt(2.0) * sigma * kappa))))
    if norm:
        # normalise to peak value
        yvals = yvals / max(yvals)
    return amp * yvals
