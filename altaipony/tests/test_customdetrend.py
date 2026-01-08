import pytest
import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u

from ..flarelc import FlareLightCurve
from ..customdetrend import (
    custom_detrending,
    estimate_detrended_noise,
    measure_flare,
    fit_spline,
)


class TestCustomDetrending:
    """Test suite for custom detrending functions"""
    
    @pytest.fixture
    def simple_flc(self):
        """Create a simple FlareLightCurve for testing"""
        time = Time(np.linspace(2450000, 2450010, 1000), format='jd')
        flux = np.ones(1000) * 3400 + np.random.normal(0, 15, 1000)
        flux_err = np.full(1000, 10.0)
        
        meta = {
            'TARGETID': 123456789,
            'MISSION': 'Kepler',
            'QUARTER': 5,
            'TIMEDEL': 0.0206833,
        }
        
        flc = FlareLightCurve(time=time, flux=flux * (u.electron / u.s), 
                             flux_err=flux_err * (u.electron / u.s), meta=meta)
        return flc
    
    @pytest.fixture
    def flc_with_variability(self):
        """
        Generate light curve with variability on several timescales.
        Mimics the old generate_lightcurve function but with proper metadata.
        """
        errorval = 15.0
        a1, a2 = 0.05, 0.005
        period1, period2 = 1.5, 24.4
        quad, cube = 1.5, 0.1
        mean = 3400.0
        
        np.random.seed(40)
        time_vals = np.arange(10, 10 + 10 * np.pi, 0.0008)
        
        # Define the flux with various components
        flux = (np.random.normal(0, errorval, time_vals.shape[0]) +
                mean + 
                a1 * mean * np.sin(period1 * time_vals + 1.) +
                a2 * mean * np.sin(period2 * time_vals) +
                quad * (time_vals - 25)**2 -
                cube * (time_vals - 25)**3)
        
        # Add a gap in the data by creating a masked array
        mask = (time_vals < 30) | (time_vals > 40)
        time_vals = time_vals[mask]
        flux = flux[mask]
        
        # Add big and long flare
        l = 66
        flux[5280:5280 + l] = flux[5280:5280 + l] + np.linspace(1000, 250, l)
        
        # Add tiny flare
        l = 3
        flux[15280:15280 + l] = flux[15280:15280 + l] + np.linspace(100, 60, l)
        
        # Add intermediate flare
        l, s = 15, 25280
        flux[s:s + l] = flux[s:s + l] + np.linspace(200, 60, l)
        
        # Flux error (typically underestimated)
        err = np.full_like(time_vals, errorval / 3 * 2)
        
        # Create FlareLightCurve with proper metadata
        time = Time(time_vals, format='jd')
        meta = {
            'TARGETID': 999888777,
            'MISSION': 'Kepler',
            'QUARTER': 10,
            'TIMEDEL': np.median(np.diff(time_vals)),
        }
        
        return FlareLightCurve(
            time=time, 
            flux=flux * (u.electron / u.s), 
            flux_err=err * (u.electron / u.s),
            meta=meta
        )
    
    # ========== Test Property Preservation ==========
    
    def test_custom_detrending_preserves_time(self, simple_flc):
        """Test that detrending preserves time array"""
        original_time = simple_flc.time.value.copy()
        
        flc_detrended = custom_detrending(simple_flc)
        
        assert np.allclose(flc_detrended.time.value, original_time)
        assert len(flc_detrended.time) == len(simple_flc.time)
    
    def test_custom_detrending_preserves_flux_length(self, simple_flc):
        """Test that detrending preserves flux array length"""
        original_length = len(simple_flc.flux)
        
        flc_detrended = custom_detrending(simple_flc)
        
        assert len(flc_detrended.flux) == original_length
        assert len(flc_detrended.detrended_flux) == original_length
    
    def test_custom_detrending_preserves_metadata(self, simple_flc):
        """Test that detrending preserves metadata"""
        original_targetid = simple_flc.targetid
        original_mission = simple_flc.meta.get('mission', simple_flc.meta.get('MISSION'))
        
        flc_detrended = custom_detrending(simple_flc)
        
        assert flc_detrended.targetid == original_targetid
        # Mission might be lowercase after conversion
        detrended_mission = flc_detrended.meta.get('mission', flc_detrended.meta.get('MISSION'))
        assert detrended_mission.lower() == original_mission.lower()
    
    def test_custom_detrending_creates_required_attributes(self, simple_flc):
        """Test that detrending creates all required attributes"""
        flc_detrended = custom_detrending(simple_flc)
        
        # Check required attributes exist
        assert hasattr(flc_detrended, 'detrended_flux')
        assert hasattr(flc_detrended, 'it_med')
        assert hasattr(flc_detrended, 'gaps')
        
        # Check they have correct length
        assert len(flc_detrended.detrended_flux) == len(simple_flc.flux)
        assert len(flc_detrended.it_med) == len(simple_flc.flux)
    
    def test_custom_detrending_does_not_modify_original(self, simple_flc):
        """Test that detrending doesn't modify the original lightcurve"""
        original_flux = simple_flc.flux.value.copy()
        
        flc_detrended = custom_detrending(simple_flc)
        
        # Original should be unchanged
        assert np.allclose(simple_flc.flux.value, original_flux, equal_nan=True)
        assert np.allclose(flc_detrended.flux.value, original_flux, equal_nan=True)
        # Detrended should be different
        assert not np.allclose(flc_detrended.detrended_flux.value, original_flux, equal_nan=True)
    
    # ========== Test Detrending Quality ==========
    
    def test_custom_detrending_removes_variability(self, flc_with_variability):
        """Test that detrending reduces variability"""
        original_std = np.nanstd(flc_with_variability.flux.value)
        
        flc_detrended = custom_detrending(flc_with_variability, spline_coarseness=8)
        
        detrended_std = np.nanstd(flc_detrended.detrended_flux)
        
        # Detrended flux should have lower standard deviation
        assert detrended_std < original_std
        # But not zero (there's still noise)
        assert detrended_std > 0
    
    def test_custom_detrending_handles_nans(self, flc_with_variability):
        """Test that detrending handles data gaps correctly"""
        flc_detrended = custom_detrending(flc_with_variability, spline_coarseness=8)
       
        # Detrended flux should have NaNs where original has NaNs
        original_nans = np.isnan(flc_with_variability.flux.value)
        detrended_nans = np.isnan(flc_detrended.detrended_flux)
        
        assert np.array_equal(original_nans, detrended_nans)
    
    # ========== Test estimate_detrended_noise ==========
    
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
    
    # ========== Test measure_flare ==========
    
    def test_measure_flare_basic(self, simple_flc):
        """Test basic flare measurement functionality"""
        # Detrend first
        flc_detrended = custom_detrending(simple_flc)
        
        # Add a synthetic flare
        flc_detrended.detrended_flux[100:120] = flc_detrended.detrended_flux[100:120] + 500
        
        # Measure the flare
        measure_flare(flc_detrended, 100, 120)
        
        # Check that flare was added to table
        assert len(flc_detrended.flares) == 1
        assert flc_detrended.flares.iloc[0].istart == 100
        assert flc_detrended.flares.iloc[0].istop == 120
    
    def test_measure_flare_has_required_columns(self, simple_flc):
        """Test that measured flare has all required columns"""
        flc_detrended = custom_detrending(simple_flc)
        flc_detrended.detrended_flux[100:120] = flc_detrended.detrended_flux[100:120] + 500
        
        measure_flare(flc_detrended, 100, 120)
        
        flare = flc_detrended.flares.iloc[0]
        
        # Check required columns exist
        required_cols = ['ed_rec', 'ed_rec_err', 'ampl_rec', 'istart', 'istop',
                        'cstart', 'cstop', 'tstart', 'tstop', 'dur']
        for col in required_cols:
            assert col in flare.index
            assert pd.notna(flare[col])
    
    def test_measure_flare_calculates_duration(self, simple_flc):
        """Test that flare duration is calculated correctly"""
        flc_detrended = custom_detrending(simple_flc)
        flc_detrended.detrended_flux[100:120] = flc_detrended.detrended_flux[100:120] + 500
        
        measure_flare(flc_detrended, 100, 120)
        
        flare = flc_detrended.flares.iloc[0]
        
        # Duration should equal tstop - tstart
        assert flare.dur == pytest.approx(flare.tstop - flare.tstart)
        # Duration should be positive
        assert flare.dur > 0
    
    # ========== Integration Test ==========
    
    @pytest.mark.parametrize("a1,a2,period1,period2,quad,cube", [
        (0.05, 0.005, 1.5, 24.4, 1.5, 0.1),
        (0.1, 0.005, 1.5, 14.4, 1.5, 0.5),
    ])
    def test_full_detrending_pipeline(self, a1, a2, period1, period2, quad, cube):
        """Integration test for full detrending pipeline"""
        # Use the fixture's logic but with parametrization
        errorval = 15.0
        mean = 3400.0
        
        np.random.seed(40)
        time_vals = np.arange(10, 10 + 10 * np.pi, 0.0008)
        
        flux = (np.random.normal(0, errorval, time_vals.shape[0]) +
                mean + 
                a1 * mean * np.sin(period1 * time_vals + 1.) +
                a2 * mean * np.sin(period2 * time_vals))
        
        time = Time(time_vals, format='jd')
        meta = {'TARGETID': 123456, 'MISSION': 'Kepler', 'QUARTER': 5, 
                'TIMEDEL': np.median(np.diff(time_vals))}
        
        flc = FlareLightCurve(time=time, flux=flux * (u.electron / u.s), 
                             flux_err=np.full(len(flux), 10) * (u.electron / u.s),
                             meta=meta)
        
        # Run detrending pipeline
        flc_detrended = custom_detrending(flc, spline_coarseness=8)
        flc_with_err = estimate_detrended_noise(flc_detrended, 
                                               mask_pos_outliers_sigma=2.5, 
                                               std_window=100)
        
        # Verify properties preserved
        assert len(flc_with_err.time) == len(flc.time)
        assert flc_with_err.targetid == flc.targetid
        
        # Verify detrending worked
        assert hasattr(flc_with_err, 'detrended_flux')
        assert hasattr(flc_with_err, 'detrended_flux_err')
        assert hasattr(flc_with_err, 'it_med')

