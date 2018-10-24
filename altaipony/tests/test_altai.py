import numpy as np
import pytest

from ..altai import (find_flares,
                     find_flares_in_cont_obs_period,
                     chi_square,
                     equivalent_duration,)
from ..flarelc import FlareLightCurve

def test_find_flares():
     """
     Integration test of a mock example light curve is given in test_flarelc.
     Add unit tests!
     """
     pass

def test_find_flares_in_cont_obs_period():
     """
     Integration test of a mock example light curve is given in test_flarelc.
     Add unit tests!
     """
     pass

def test_chi_square():
    """Test an abvious example"""
    residual = np.full(5,1.)
    error = np.full(5,.1)
    assert chi_square(residual, error) == 100.

def test_equivalent_duration():
    """Test a triangle shaped flare in a toy light curve."""
    detrended_flux = np.full(1000,0.)
    detrended_flux[60:70] = np.array([9,8,7,6,5,4,3,2,1,0])
    detrended_flux = detrended_flux + 1.
    lc = FlareLightCurve(time=np.arange(1000)/86400.,
                         detrended_flux=detrended_flux,
                         detrended_flux_err=np.full(1000,1e-8))
    ed, ed_err = equivalent_duration(lc, 60, 70, err=True)
    assert ed == pytest.approx(45,rel=1e-8)
    assert ed_err == pytest.approx(2.665569e-08, rel=1e-4)
