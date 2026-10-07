"""Tests of ``estimate_detrended_noise``."""

import numpy as np
import pytest
import astropy.units as u
from astropy.time import Time

from altaipony.detrending import estimate_detrended_noise
from altaipony.flarelc import FlareLightCurve


class TestEstimateDetrendedNoise:

    def test_estimate_detrended_noise_simple(self):
        """Test noise estimation on simple Gaussian noise"""
        time = Time(np.linspace(10, 30, 2000), format='jd')
        
        np.random.seed(30)
        flux = np.random.normal(0, 40, 2000) + 200.
        
        flc = FlareLightCurve(time=time, flux=flux * (u.electron / u.s),
                             flux_err=np.ones(2000) * (u.electron / u.s))
        flc['detrended_flux'] = flux
        
        flc_est = estimate_detrended_noise(flc, mask_pos_outliers_sigma=2.5, 
                                          std_window=100)
        
        # Error should be close to input error of 40
        estimated_error = np.nanmedian(flc_est.detrended_flux_err)
        assert estimated_error == pytest.approx(40, abs=5)
    
    def test_estimate_detrended_noise_with_flare(self):
        """Test that noise estimation masks flares"""
        time = Time(np.linspace(10, 30, 2000), format='jd')
        
        np.random.seed(30)
        flux = np.random.normal(0, 40, 2000) + 200.
        flux[120:124] = [500, 380, 300, 270]  # Add flare
        
        flc = FlareLightCurve(time=time, flux=flux * (u.electron / u.s),
                             flux_err=np.ones(2000) * (u.electron / u.s))
        flc['detrended_flux'] = flux
        
        flc_est = estimate_detrended_noise(flc, mask_pos_outliers_sigma=2.5, 
                                          std_window=100)
        
        # Error should not be inflated by flare
        estimated_error = np.nanmedian(flc_est.detrended_flux_err)
        assert estimated_error == pytest.approx(40, abs=5)
    
    def test_estimate_detrended_noise_with_nans(self):
        """Test that noise estimation handles NaNs"""
        time = Time(np.linspace(10, 30, 2000), format='jd')
        
        np.random.seed(30)
        flux = np.random.normal(0, 40, 2000) + 200.
        flux[30:40] = np.nan  # Add NaNs
        
        flc = FlareLightCurve(time=time, flux=flux * (u.electron / u.s),
                             flux_err=np.ones(2000) * (u.electron / u.s))
        flc['detrended_flux'] = flux
        
        # Should not raise error
        flc_est = estimate_detrended_noise(flc, mask_pos_outliers_sigma=2.5, 
                                          std_window=100)
        
        assert flc_est is not None
        assert len(flc_est.detrended_flux_err) == 2000
