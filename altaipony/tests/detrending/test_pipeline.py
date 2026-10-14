"""Tests of the contract of ``custom_detrending``: the detrended flux is
relative flux with a baseline of 1, independent of how the input flux is
scaled, both with and without the GP step."""

import numpy as np
import pandas as pd
import pytest
import astropy.units as u
from astropy.time import Time

from altaipony.altai import measure_flare
from altaipony.detrending import custom_detrending, estimate_detrended_noise
from altaipony.flarelc import FlareLightCurve

pytestmark = pytest.mark.slow

NOISE = 1e-3
FLARES = [(2.3, 0.02, 0.01), (5.7, 0.04, 0.02)]   # (tpeak, fwhm, ampl), relative flux
SCALES = (1.0, 3400.0)                            # normalised, and e-/s-like
# star: (spot amplitude, use_gp). The GP runs for every star unless switched
# off, so "no_gp" covers the path without it.
STARS = {"periodic": (0.02, True), "flat": (0.0, True), "no_gp": (0.02, False)}


def _quiet(time, table, pad_before=0.05, pad_after=0.3):
    """Cadences well away from the injected flares."""
    quiet = np.ones(len(time), dtype=bool)
    for a, b in table[["t_start", "t_end"]].values:
        quiet &= ~((time >= a - pad_before) & (time <= b + pad_after))
    return quiet


def _values(x):
    return np.asarray(getattr(x, "value", x), dtype=float)


@pytest.fixture(scope="module")
def detrended(make_lc):
    """Each star (8 days of 2-min data, spots at P = 1.3 d or none, white
    noise, two flares) detrended at each flux scale."""
    out = {}
    for star, (amp, use_gp) in STARS.items():
        for scale in SCALES:
            lc, table = make_lc(modes=[dict(period=1.3, amplitude=amp)], flares=FLARES,
                                noise=NOISE, scale=scale)
            out[star, scale] = (custom_detrending(lc, use_gp=use_gp), table)
    return out


def test_stars_take_the_intended_path(detrended):
    """Every star gets a GP model, with or without spots, unless the GP is
    switched off."""
    for scale in SCALES:
        assert np.isfinite(_values(detrended["periodic", scale][0].gp_model)).any()
        assert np.isfinite(_values(detrended["flat", scale][0].gp_model)).any()
        assert np.isnan(_values(detrended["no_gp", scale][0].gp_model)).all()


def test_noise_is_estimated(detrended):
    """custom_detrending also returns the noise of the detrended flux, close
    to the injected white noise."""
    for (star, scale), (lc, _) in detrended.items():
        err = _values(lc.detrended_flux_err)
        assert np.isfinite(err).all()
        assert np.nanmedian(err) == pytest.approx(NOISE, rel=0.2)


def test_detrend_then_find_flares(make_lc):
    """The documented workflow: FlareLightCurve.detrend() with its default
    de-trending, then find_flares(), finds the injected flares."""
    lc, table = make_lc(modes=[dict(period=1.3, amplitude=0.02)], flares=FLARES,
                        noise=NOISE, scale=3400.0)
    assert lc.detrend() is lc
    assert lc.find_flares() is lc
    for tpeak in table.tpeak:
        assert ((lc.flares.tstart <= tpeak) & (lc.flares.tstop >= tpeak)).any()


@pytest.mark.parametrize("star", STARS)
@pytest.mark.parametrize("scale", SCALES)
def test_quiet_baseline_is_one(detrended, star, scale):
    """Away from flares, the detrended flux sits at 1 (relative flux)."""
    lc, table = detrended[star, scale]
    det = _values(lc.detrended_flux)
    quiet = _quiet(lc.time.value, table)
    assert abs(np.nanmedian(det[quiet]) - 1.0) < 0.05 * NOISE
    # it_med describes the final detrended flux
    assert abs(np.nanmedian(_values(lc.it_med)) - 1.0) < 0.05 * NOISE


@pytest.mark.parametrize("star", STARS)
def test_detrended_flux_is_scale_invariant(detrended, star):
    """Multiplying the input flux by a constant does not change the relative
    detrended flux."""
    lc1, _ = detrended[star, 1.0]
    lc2, _ = detrended[star, 3400.0]
    np.testing.assert_allclose(_values(lc2.detrended_flux), _values(lc1.detrended_flux),
                               rtol=0, atol=0.05 * NOISE)


@pytest.mark.parametrize("star", STARS)
def test_recovered_amplitude_is_relative(detrended, star):
    """find_flares reports relative amplitudes, the same at both scales and
    close to the injected ones (the template peaks at ~0.95 ampl)."""
    ampl = {}
    for scale in SCALES:
        lc = detrended[star, scale][0].copy()
        lc.detrended_flux = _values(lc.detrended_flux)
        lc = estimate_detrended_noise(lc)
        # merge_sigma=1 joins tail fragments of the longer flare
        flares = lc.find_flares(merge_sigma=1.).flares.sort_values("tstart")
        assert len(flares) == len(FLARES)
        ampl[scale] = flares.ampl_rec.to_numpy(dtype=float)
    expected = np.array([0.95 * a for _, _, a in FLARES])
    np.testing.assert_allclose(ampl[1.0], expected, rtol=0.25)
    np.testing.assert_allclose(ampl[3400.0], ampl[1.0], rtol=0.05)


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
        assert hasattr(flc_detrended, 'cont_windows')
        
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
