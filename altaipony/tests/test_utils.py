import numpy as np
from ..flarelc import FlareLightCurve
from ..utils import k2sc_quality_cuts, split_gaps

def test_k2sc_quality_cuts():
    a = np.array([1, 3., 5, -np.inf])
    b = np.array([2, 4., np.nan, 20])
    c = np.array([3, np.inf , 20.1, 5])
    d = np.array([3, 20. , 20.1, 5])
    data = FlareLightCurve(time=a, pos_corr1=b, pos_corr2=c, flux=d)
    data2 = k2sc_quality_cuts(data)
    assert data2.time[0] == 1.
    assert data2.time.shape[0] == 1

def test_split_gaps():
    
    # Run integration test that succeeds
    gaps = [(0, 20.), (21., 34), (37, 41)]
    splits = [1.5, 14., 39.]
    result = [(0, 1.5), (1.5, 14.0), (14.0, 20.0),
              (21.0, 34), (37, 39.0), (39.0, 41)]
    
    assert split_gaps(gaps, splits) == result
    
    # Run integration test that succeeds with the default input
    # should return the input 
    gaps = [(0, 20.), (21., 34), (37, 41)]
    splits = []
    
    assert split_gaps(gaps, splits) == gaps
    

    # Get instruction on how to pick good split values
    # when you mess it up:

    gaps = [(0, 20.), (21., 34), (37, 41)]


    # by passing NaN
    with pytest.raises(IndexError):
        splits = [1.5, np.nan, 39.]
        split_gaps(gaps, splits)

    # by passing inf
    with pytest.raises(IndexError):
        splits = [1.5, np.inf, 39.]
        split_gaps(gaps, splits)

    # by passing values outside of range
    with pytest.raises(IndexError):
        splits = [1.5, 99., 39.]
        split_gaps(gaps, splits)
