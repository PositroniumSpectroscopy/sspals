#! python
'''
pytest
'''
import numpy as np
from sspals.sim import sim
from sspals.chmx import sub_offset, validate, chmx
from sspals.cfd import cfd_1D
from sspals.sspals import sspals_1D

def test_sub():
    """ background subtraction of the mean value of first 3 elements of 1-10.
    """
    assert (sub_offset(np.array([np.arange(1.0, 10.0)]), 3)[0][0] == \
            [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).all()

def test_chmx():
    """ combine arrays high/ low into chmx, ignoring saturated values in high.
    """
    high = np.array([[0.1, 0.6, 0.5, 3, 3, 3]])
    low = np.array([[0.0, 1.1, 2.1, 3, 4, 5]])
    assert (chmx(high, low, n_bsub=0, invert=False) == [[0.0, 0.6, 0.5, 3.0, 4.0, 5.0]]).all()

def test_cfd():
    """ constant fraction discriminator test on linear ramp.
    """
    x_vals = np.arange(1, 10)
    assert cfd_1D(x_vals, 1, cfd_offset=2, cfd_threshold=1, cfd_scale=0.6) == 2.0
def test_validate():
    """ check validation -> ignore empty rows.
    """
    arr = np.array([[0.0, 2.0], [-2.0, -1.5], [0.0, 10.0]])
    assert (validate(arr, min_range=0.5) == np.array([[0.0, 2.0], [0.0, 10.0]])).all()

def test_sspals():
    """ simulate sspals and measure the delayed fraction.
    """
    limits = [-1.0E-8, 3.5E-8, 6.0E-7]
    x_vals = np.arange(-100, 600, 1) * 1e-9
    y_vals = sim(x_vals, amp=1.0, sigma=2e-09, eff=0.4, tau_Ps=1.420461e-07, tau_d=1e-08)
    assert round(sspals_1D(y_vals, 1e-9, limits)[3], 6) == 0.354113
