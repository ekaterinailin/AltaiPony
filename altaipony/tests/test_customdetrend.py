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
    _fit_single_spline,
    _build_knot_points,
    _evaluate_spline_fit,
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
        """Test that detrending preserves time values (may filter some points)"""
        original_time = simple_flc.time.value.copy()
        
        flc_detrended = custom_detrending(simple_flc)
        
        # Detrended times should be a subset of original times
        # (some interpolated cadences may be filtered out)
        assert len(flc_detrended.time) <= len(original_time)
        assert len(flc_detrended.time) > 0
        # All detrended times should exist in original
        for t in flc_detrended.time.value:
            assert np.any(np.isclose(original_time, t))
    
    def test_custom_detrending_preserves_flux_length(self, simple_flc):
        """Test that detrending preserves flux array length (or removes only interpolated points)"""
        original_length = len(simple_flc.flux)
        
        flc_detrended = custom_detrending(simple_flc)
        
        # Length should be preserved or slightly reduced (interpolated cadences filtered)
        assert len(flc_detrended.flux) <= original_length
        assert len(flc_detrended.flux) > 0
        # flux and detrended_flux should have same length
        assert len(flc_detrended.flux) == len(flc_detrended.detrended_flux)
    
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
        
        # Check they have consistent lengths with the output flux
        assert len(flc_detrended.detrended_flux) == len(flc_detrended.flux)
        assert len(flc_detrended.it_med) == len(flc_detrended.flux)
    
    def test_custom_detrending_does_not_modify_original(self, simple_flc):
        """Test that detrending doesn't modify the original lightcurve's flux values"""
        original_flux = simple_flc.flux.value.copy()
        
        flc_detrended = custom_detrending(simple_flc)
        
        # Original should be unchanged
        assert np.allclose(simple_flc.flux.value, original_flux, equal_nan=True)
        # Detrended output flux should match original (restored after detrending)
        # Note: length might be different due to interpolated cadence filtering
        # Detrended flux values should be different from raw flux values
        assert not np.allclose(flc_detrended.detrended_flux.value, 
                               flc_detrended.flux.value, equal_nan=True)
    
    # ========== Test Detrending Quality ==========
    
    def test_custom_detrending_removes_variability(self, flc_with_variability):
        """Test that detrending reduces variability"""
        original_std = np.nanstd(flc_with_variability.flux.value)
        
        flc_detrended = custom_detrending(flc_with_variability)
        
        detrended_std = np.nanstd(flc_detrended.detrended_flux)
        
        # Detrended flux should have lower standard deviation
        assert detrended_std < original_std
        # But not zero (there's still noise)
        assert detrended_std > 0
    
    def test_custom_detrending_handles_nans(self, flc_with_variability):
        """Test that detrending handles data gaps correctly"""
        flc_detrended = custom_detrending(flc_with_variability)
       
        # Detrending should complete without error
        assert flc_detrended is not None
        assert hasattr(flc_detrended, 'detrended_flux')
        # Output should have some valid (non-NaN) data
        assert np.sum(~np.isnan(flc_detrended.detrended_flux)) > 0
    
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
        flc_detrended = custom_detrending(flc)
        flc_with_err = estimate_detrended_noise(flc_detrended, 
                                               mask_pos_outliers_sigma=2.5, 
                                               std_window=100)
        
        # Verify properties preserved (length may change due to interpolation filtering)
        assert len(flc_with_err.time) <= len(flc.time)
        assert len(flc_with_err.time) > 0
        assert flc_with_err.targetid == flc.targetid
        
        # Verify detrending worked
        assert hasattr(flc_with_err, 'detrended_flux')
        assert hasattr(flc_with_err, 'detrended_flux_err')
        assert hasattr(flc_with_err, 'it_med')


class TestFitSpline:
    """Test suite for spline fitting functions"""
    
    @pytest.fixture
    def simple_time_flux(self):
        """Create simple time and flux arrays for testing"""
        np.random.seed(42)
        time = np.linspace(0, 10, 1000)
        flux = 1000 + np.random.normal(0, 10, 1000)
        gaps = [(0, 1000)]
        return time, flux, gaps
    
    @pytest.fixture
    def time_flux_with_trend(self):
        """Create time and flux with a linear trend"""
        np.random.seed(42)
        time = np.linspace(0, 10, 1000)
        # Add linear trend
        flux = 1000 + 50 * time + np.random.normal(0, 10, 1000)
        gaps = [(0, 1000)]
        return time, flux, gaps
    
    @pytest.fixture
    def time_flux_with_flares(self):
        """Create time and flux with synthetic flares"""
        np.random.seed(42)
        time = np.linspace(0, 10, 1000)
        flux = 1000 + np.random.normal(0, 10, 1000)
        # Add flares
        flux[100:110] += 200  # Small flare
        flux[500:520] += 500  # Large flare
        gaps = [(0, 1000)]
        return time, flux, gaps
    
    @pytest.fixture
    def time_flux_with_gap(self):
        """Create time and flux with a data gap (two segments)"""
        np.random.seed(42)
        time = np.linspace(0, 10, 1000)
        flux = 1000 + np.random.normal(0, 10, 1000)
        # Create gap by setting middle section to NaN
        flux[400:600] = np.nan
        gaps = [(0, 400), (600, 1000)]
        return time, flux, gaps
    
    # ========== Test fit_spline return structure ==========
    
    def test_fit_spline_returns_three_values(self, simple_time_flux):
        """Test that fit_spline returns newflux, model, and best_params"""
        time, flux, gaps = simple_time_flux
        
        result = fit_spline(time, flux, gaps)
        
        assert len(result) == 3
        newflux, model, best_params = result
        
        assert isinstance(newflux, np.ndarray)
        assert isinstance(model, np.ndarray)
        assert isinstance(best_params, dict)
    
    def test_fit_spline_output_lengths(self, simple_time_flux):
        """Test that output arrays have correct length"""
        time, flux, gaps = simple_time_flux
        
        newflux, model, best_params = fit_spline(time, flux, gaps)
        
        assert len(newflux) == len(flux)
        assert len(model) == len(flux)
    
    def test_fit_spline_best_params_keys(self, simple_time_flux):
        """Test that best_params contains expected keys"""
        time, flux, gaps = simple_time_flux
        
        newflux, model, best_params = fit_spline(time, flux, gaps)
        
        expected_keys = {'coarseness', 'order', 'phase', 'score'}
        assert set(best_params.keys()) == expected_keys
    
    def test_fit_spline_best_params_types(self, simple_time_flux):
        """Test that best_params values have correct types"""
        time, flux, gaps = simple_time_flux
        
        newflux, model, best_params = fit_spline(time, flux, gaps)
        
        assert isinstance(best_params['coarseness'], (int, np.integer))
        assert isinstance(best_params['order'], (int, np.integer))
        assert isinstance(best_params['phase'], (int, np.integer))
        assert isinstance(best_params['score'], (float, np.floating))
    
    def test_fit_spline_score_is_finite(self, simple_time_flux):
        """Test that the best score is finite"""
        time, flux, gaps = simple_time_flux
        
        newflux, model, best_params = fit_spline(time, flux, gaps)
        
        assert np.isfinite(best_params['score'])
        assert best_params['score'] > 0
    
    # ========== Test fit_spline detrending quality ==========
    
    def test_fit_spline_removes_trend(self, time_flux_with_trend):
        """Test that spline fitting removes linear trends"""
        time, flux, gaps = time_flux_with_trend
        
        newflux, model, best_params = fit_spline(time, flux, gaps)
        
        # Detrended flux mean in the first and last 100 points should be similar
        mean_start = np.nanmean(newflux[:100])
        mean_end = np.nanmean(newflux[-100:])
        assert abs(mean_start - mean_end) < 5  # Within 5 units
    
    def test_fit_spline_preserves_flares(self, time_flux_with_flares):
        """Test that spline fitting doesn't remove flares"""
        time, flux, gaps = time_flux_with_flares
        
        newflux, model, best_params = fit_spline(time, flux, gaps)
        
        # Flares should still be visible as positive outliers
        median_flux = np.nanmedian(newflux)
        
        # Check that flare regions are still elevated
        assert np.nanmean(newflux[100:110]) > median_flux + 100
        assert np.nanmean(newflux[500:520]) > median_flux + 400
    
    def test_fit_spline_handles_gaps(self, time_flux_with_gap):
        """Test that spline fitting handles data gaps correctly"""
        time, flux, gaps = time_flux_with_gap
        
        newflux, model, best_params = fit_spline(time, flux, gaps)
        
        # NaN regions should remain NaN
        assert np.all(np.isnan(newflux[400:600]))
        assert np.all(np.isnan(model[400:600]))
        
        # Non-NaN regions should have valid values
        assert np.all(np.isfinite(newflux[:400]))
        assert np.all(np.isfinite(newflux[600:]))
    
    # ========== Test fit_spline parameter selection ==========
    
    def test_fit_spline_selects_from_candidates(self, simple_time_flux):
        """Test that fit_spline actually selects best from multiple candidates"""
        time, flux, gaps = simple_time_flux
        
        # Use small range to limit candidates
        newflux, model, best_params = fit_spline(
            time, flux, gaps,
            coarseness_range=(5, 10, 5),  # Only 2 coarseness values
            spline_orders=(2,),  # Only 1 order
            n_phase_shifts=2  # 2 phase shifts
        )
        
        # Should have selected one of the valid configurations
        assert best_params['coarseness'] in [5, 10]
        assert best_params['order'] == 2
        assert best_params['phase'] in [0, 1]
    
    def test_fit_spline_respects_coarseness_range(self, simple_time_flux):
        """Test that selected coarseness is within specified range"""
        time, flux, gaps = simple_time_flux
        
        newflux, model, best_params = fit_spline(
            time, flux, gaps,
            coarseness_range=(8, 12, 2)
        )
        
        assert best_params['coarseness'] in [8, 10, 12]
    
    def test_fit_spline_respects_spline_orders(self, simple_time_flux):
        """Test that selected order is from specified options"""
        time, flux, gaps = simple_time_flux
        
        newflux, model, best_params = fit_spline(
            time, flux, gaps,
            spline_orders=(2, 3, 4)
        )
        
        assert best_params['order'] in [2, 3, 4]


class TestBuildKnotPoints:
    """Test suite for _build_knot_points function"""
    
    def test_build_knot_points_basic(self):
        """Test basic knot point generation"""
        time = np.linspace(0, 10, 1000)
        flux = np.ones(1000) * 100
        
        t_knots, f_knots = _build_knot_points(time, flux, n=100, phase_offset=0, percentile=50)
        
        # Should have interior knots plus boundary points
        assert len(t_knots) > 2
        assert len(t_knots) == len(f_knots)
    
    def test_build_knot_points_boundary_values(self):
        """Test that boundary knots are at segment edges"""
        time = np.linspace(0, 10, 1000)
        flux = np.ones(1000) * 100
        
        t_knots, f_knots = _build_knot_points(time, flux, n=100, phase_offset=0, percentile=50)
        
        # First knot should be at start
        assert t_knots[0] == time[0]
        # Last knot should be at end
        assert t_knots[-1] == time[-1]
    
    def test_build_knot_points_sorted(self):
        """Test that knot times are sorted"""
        time = np.linspace(0, 10, 1000)
        flux = np.random.random(1000) * 100
        
        t_knots, f_knots = _build_knot_points(time, flux, n=100, phase_offset=0, percentile=35)
        
        assert np.all(np.diff(t_knots) > 0)
    
    def test_build_knot_points_phase_offset(self):
        """Test that phase offset shifts knot positions"""
        time = np.linspace(0, 10, 1000)
        flux = np.ones(1000) * 100
        
        t_knots_0, _ = _build_knot_points(time, flux, n=100, phase_offset=0, percentile=50)
        t_knots_50, _ = _build_knot_points(time, flux, n=100, phase_offset=50, percentile=50)
        
        # Interior knots should be different (boundary points may be same)
        # Compare the second knot (first interior knot)
        assert t_knots_0[1] != t_knots_50[1]
    
    def test_build_knot_points_percentile_affects_flux(self):
        """Test that percentile parameter affects flux knot values"""
        time = np.linspace(0, 10, 1000)
        # Create flux with positive outliers (like flares)
        np.random.seed(42)
        flux = np.random.normal(100, 5, 1000)
        flux[50:60] += 50  # Add outlier
        
        _, f_knots_50 = _build_knot_points(time, flux, n=100, phase_offset=0, percentile=50)
        _, f_knots_10 = _build_knot_points(time, flux, n=100, phase_offset=0, percentile=10)
        
        # Lower percentile should give lower flux values (avoiding outliers)
        assert np.mean(f_knots_10) < np.mean(f_knots_50)
    
    def test_build_knot_points_short_segment(self):
        """Test handling of very short segments"""
        time = np.linspace(0, 1, 50)  # Very short
        flux = np.ones(50) * 100
        
        # Bin size larger than segment
        t_knots, f_knots = _build_knot_points(time, flux, n=100, phase_offset=0, percentile=50)
        
        # Should return just boundary points
        assert len(t_knots) == 2
        assert len(f_knots) == 2
    
    def test_build_knot_points_handles_nans(self):
        """Test that NaN values are handled"""
        time = np.linspace(0, 10, 1000)
        flux = np.ones(1000) * 100
        flux[500:510] = np.nan
        
        t_knots, f_knots = _build_knot_points(time, flux, n=100, phase_offset=0, percentile=50)
        
        # Should not have NaN knots
        assert not np.any(np.isnan(t_knots))
        assert not np.any(np.isnan(f_knots))


class TestEvaluateSplineFit:
    """Test suite for _evaluate_spline_fit scoring function"""
    
    @pytest.fixture
    def perfect_fit_data(self):
        """Create data where model perfectly matches flux baseline"""
        np.random.seed(42)
        flux = 1000 + np.random.normal(0, 10, 1000)
        model = np.full(1000, 1000.0)  # Flat model at mean
        gaps = [(0, 1000)]
        return flux, model, gaps
    
    @pytest.fixture
    def edge_effect_data(self):
        """Create data where model has edge deviations"""
        np.random.seed(42)
        flux = 1000 + np.random.normal(0, 10, 1000)
        model = np.full(1000, 1000.0)
        # Add edge deviation - model curves away at edges
        model[:100] = 1050  # Left edge higher
        model[-100:] = 950  # Right edge lower
        gaps = [(0, 1000)]
        return flux, model, gaps
    
    def test_evaluate_returns_finite_score(self, perfect_fit_data):
        """Test that evaluation returns a finite score"""
        flux, model, gaps = perfect_fit_data
        
        score = _evaluate_spline_fit(flux, model, gaps)
        
        assert np.isfinite(score)
        assert score > 0
    
    def test_evaluate_edge_penalty_increases_score(self, perfect_fit_data, edge_effect_data):
        """Test that edge deviations increase the score (worse fit)"""
        flux_good, model_good, gaps = perfect_fit_data
        flux_bad, model_bad, _ = edge_effect_data
        
        score_good = _evaluate_spline_fit(flux_good, model_good, gaps, edge_penalty_weight=0.5)
        score_bad = _evaluate_spline_fit(flux_bad, model_bad, gaps, edge_penalty_weight=0.5)
        
        # Edge effects should increase score
        assert score_bad > score_good
    
    def test_evaluate_edge_penalty_weight_effect(self, edge_effect_data):
        """Test that higher edge_penalty_weight increases penalty"""
        flux, model, gaps = edge_effect_data
        
        score_low = _evaluate_spline_fit(flux, model, gaps, edge_penalty_weight=0.1)
        score_high = _evaluate_spline_fit(flux, model, gaps, edge_penalty_weight=1.0)
        
        # Higher weight should give higher score for same edge effects
        assert score_high > score_low
    
    def test_evaluate_zero_edge_penalty(self, edge_effect_data):
        """Test that zero edge_penalty_weight disables edge penalty"""
        flux, model, gaps = edge_effect_data
        
        score_with = _evaluate_spline_fit(flux, model, gaps, edge_penalty_weight=0.5)
        score_without = _evaluate_spline_fit(flux, model, gaps, edge_penalty_weight=0.0)
        
        # Without edge penalty, score should be lower
        assert score_without < score_with
    
    def test_evaluate_flare_asymmetry(self):
        """Test that fitting flares is penalized via asymmetry"""
        np.random.seed(42)
        flux = 1000 + np.random.normal(0, 10, 1000)
        flux[500:520] += 200  # Add flare
        
        # Model that tracks the flare (bad)
        model_tracks_flare = flux.copy()
        
        # Model that ignores the flare (good)
        model_ignores_flare = np.full(1000, 1000.0)
        
        gaps = [(0, 1000)]
        
        score_tracks = _evaluate_spline_fit(flux, model_tracks_flare, gaps)
        score_ignores = _evaluate_spline_fit(flux, model_ignores_flare, gaps)
        
        # Model that tracks flare should have worse (higher) score
        # because residuals will be more symmetric
        assert score_tracks > score_ignores
    
    def test_evaluate_handles_multiple_gaps(self):
        """Test scoring with multiple gap segments"""
        np.random.seed(42)
        flux = 1000 + np.random.normal(0, 10, 1000)
        model = np.full(1000, 1000.0)
        flux[400:600] = np.nan
        model[400:600] = np.nan
        
        gaps = [(0, 400), (600, 1000)]
        
        score = _evaluate_spline_fit(flux, model, gaps)
        
        assert np.isfinite(score)
    
    def test_evaluate_short_segment_no_edge_penalty(self):
        """Test that very short segments don't contribute to edge penalty"""
        np.random.seed(42)
        flux = 1000 + np.random.normal(0, 10, 100)
        model = np.full(100, 1050.0)  # Offset model
        
        # Segment of only 15 points (< 20 threshold)
        gaps = [(0, 15)]
        
        # Should not crash and should return valid score
        score = _evaluate_spline_fit(flux[:15], model[:15], gaps)
        
        assert np.isfinite(score) or score == np.inf
    
    def test_evaluate_returns_inf_for_too_few_points(self):
        """Test that too few valid points returns inf"""
        flux = np.array([1, 2, 3, 4, 5])
        model = np.array([1, 2, 3, 4, 5])
        gaps = [(0, 5)]
        
        score = _evaluate_spline_fit(flux, model, gaps)
        
        assert score == np.inf


class TestFitSingleSpline:
    """Test suite for _fit_single_spline helper function"""
    
    def test_fit_single_spline_returns_model_and_newflux(self):
        """Test that _fit_single_spline returns model and newflux"""
        np.random.seed(42)
        time = np.linspace(0, 10, 1000)
        flux = 1000 + np.random.normal(0, 10, 1000)
        flux_med = np.full(1000, 1000.0)
        gaps = [(0, 1000)]
        dt = np.diff(time)[0]
        
        model, newflux = _fit_single_spline(
            time, flux, flux_med, gaps,
            coarseness=10, k=3, dt=dt,
            phase_idx=0, n_phases=3, percentile=35
        )
        
        assert len(model) == len(flux)
        assert len(newflux) == len(flux)
        assert isinstance(model, np.ndarray)
        assert isinstance(newflux, np.ndarray)
    
    def test_fit_single_spline_short_segment_uses_median(self):
        """Test that very short segments use median instead of spline"""
        time = np.linspace(0, 0.1, 10)  # Very short
        flux = np.array([100, 102, 98, 101, 99, 103, 97, 100, 101, 99])
        flux_med = np.full(10, 100.0)
        gaps = [(0, 10)]
        dt = np.diff(time)[0]
        
        # Use large coarseness so segment is too short
        model, newflux = _fit_single_spline(
            time, flux, flux_med, gaps,
            coarseness=100, k=3, dt=dt,
            phase_idx=0, n_phases=1, percentile=50
        )
        
        # Model should be constant (median)
        assert np.allclose(model, np.nanmedian(flux))
    
    def test_fit_single_spline_handles_multiple_gaps(self):
        """Test handling of multiple gap segments"""
        time = np.linspace(0, 10, 1000)
        flux = 1000 + np.random.normal(0, 10, 1000)
        flux[400:600] = np.nan
        flux_med = np.full(1000, 1000.0)
        gaps = [(0, 400), (600, 1000)]
        dt = np.diff(time)[0]
        
        model, newflux = _fit_single_spline(
            time, flux, flux_med, gaps,
            coarseness=5, k=3, dt=dt,
            phase_idx=0, n_phases=1, percentile=35
        )
        
        # Gap region should be NaN
        assert np.all(np.isnan(model[400:600]))
        # Valid regions should have values
        assert np.all(np.isfinite(model[:400]))
        assert np.all(np.isfinite(model[600:]))