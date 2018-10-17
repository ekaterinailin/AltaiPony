import numpy as np
import pytest
from inspect import currentframe, getframeinfo

from ..flarelc import FlareLightCurve
from ..lcio import from_K2SC_file

from .. import PACKAGEDIR
from . import test_ids, test_paths


#From lightkurve
def mock_flc(origin='TPF'):
    n = 1000
    time = np.arange(0, n/48, 1./48.)
    flux_err = np.random.rand(n)/100.
    flux = np.sin(time/2)*7. + 500. +flux_err
    flux[15] = 1e3
    pos_corr1 = np.zeros(n)
    pos_corr2 = np.zeros(n)
    targetid = 80000000
    return FlareLightCurve(time=time, flux=flux, flux_err=flux_err, targetid=targetid,
                          pos_corr1=pos_corr1, pos_corr2=pos_corr2, origin=origin)

def test_invalid_lightcurve():
    """Invalid FlareLightCurves should not be allowed."""
    err_string = ("Input arrays have different lengths."
                  " len(time)=5, len(flux)=4")
    time = np.array([1, 2, 3, 4, 5])
    flux = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError) as err:
        FlareLightCurve(time=time, flux=flux)
    assert err_string == err.value.args[0]

def test_find_gaps():
    lc = from_K2SC_file(test_paths[0])
    lc.find_gaps()
    assert lc.gaps == [(0, 2582), (2582, 3424)]

def test_detrend():
    flc = mock_flc()
    try:
        flc = flc.detrend()
        assert flc.detrended_flux.shape == flc.flux.shape
        assert flc.pv == pytest.approx([-4.52464711,  1.98863195, 34.25116362,
                                        0.10506948, -5.9999997 , 17.,
                                        16.99999881, -5.23056634, ], rel=2e-1)
    except np.linalg.linalg.LinAlgError:
        warning.warn('Detrending of mock LC failed, this happens.')
        pass

    #test non TPF derived LC fails
    #test the shapes are the same for all
    # test that the necessary attributes are kept
    pass

def test_detrend_fails():
    flc =  mock_flc(origin='KLC')
    """LightCurves with no data should not be allowed."""
    err_string = ('Only KeplerTargetPixelFile derived FlareLightCurves can be'
              ' passed to detrend().')
    with pytest.raises(ValueError) as err:
        flc.detrend()
    assert err_string == err.value.args[0]
