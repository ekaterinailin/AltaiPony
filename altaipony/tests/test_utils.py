import numpy as np
from ..flarelc import FlareLightCurve
from ..utils import k2sc_quality_cuts

def test_k2sc_quality_cuts():
    a = np.array([1, 3., 5, -np.inf])
    b = np.array([2, 4., np.nan, 20])
    c = np.array([3, np.inf , 20.1, 5])
    d = np.array([3, 20. , 20.1, 5])
    data = FlareLightCurve(time=a, pos_corr1=b, pos_corr2=c, flux=d)
    data2 = k2sc_quality_cuts(data)
    assert data2.time[0] == 1.
    assert data2.time.shape[0] == 1
