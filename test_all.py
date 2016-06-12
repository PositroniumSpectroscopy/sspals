#! python
'''
pytest
'''
import numpy as np
import sspals

def test_chmx():
    low = np.array([[0, 1.1, 2.1, 3, 4, 5]])
    high = np.array([[0.1, 0.6, 0.5, 3, 3, 3]])
    assert (sspals.chmx(high, low, n_bsub=0, invert=False) == \
            [[ 0. ,  0.6,  0.5,  3. ,  4. ,  5. ]]).all()
