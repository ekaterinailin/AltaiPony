# test_fit_flares6.py

import numpy as np
import pandas as pd
import pytest
from altaipony.tests.test_flarelc import mock_flc as mock

from altaipony.fit_flares import (
    combined_model,
    stacked_flare_model,
    build_baseline,
    log_likelihood,
    log_prior,
    log_posterior,
    model_selection,
    extract_lc,
    group_peaks,
    fit_flares,
    fit_single_flare,
    ed_from_model,
    make_flare_table,
    plot_flare_fit,
    plot_all_fits,
)

# Simulated FlareLightCurve generator (adapted from your test_flarelc.py)
def mock_flc(detrended=True, ampl=1.0, dur=1):
    """Wrapper for AltaiPony's mock_flc that returns time, flux, and flux_err arrays."""
    flc = mock(detrended=detrended, ampl=ampl, dur=dur)
    time = flc.time.value
    flux = flc.detrended_flux.value if detrended else flc.flux.value
    flux_err = flc.detrended_flux_err.value if detrended else flc.flux_err.value
    return time, flux, flux_err




def test_combined_model():
    """Test that combined_model runs and raises correct errors."""
   
    # Valid case with one flare
    time, flux, _ = mock_flc(detrended=True)
    baseline = [np.median(flux), 0, 0, 0, 0]
    flare_params = [time[485], 0.01, 1.0]
    params = baseline + flare_params

    model = combined_model(time, *params)
    assert isinstance(model, np.ndarray)
    assert model.shape == time.shape

    # No flare case
    model_no_flare = combined_model(time, *baseline)
    assert np.allclose(model_no_flare, build_baseline(time, baseline))

    # t not a NumPy array
    with pytest.raises(TypeError, match="must be a NumPy array"):
        combined_model("not an array", *params)

    # Empty t array
    with pytest.raises(ValueError, match="must not be empty"):
        combined_model(np.array([]), *params)

    # Too few parameters
    with pytest.raises(ValueError, match="at least 5 parameters"):
        combined_model(time, 1, 2, 3)

    # Flare parameters not in groups of 3
    with pytest.raises(ValueError, match="in groups of 3"):
        combined_model(time, *(baseline + [0.5, 0.01]))
        
        

def test_stacked_flare_model():
    """Test that stacked_flare_model runs and handles input errors."""

    time, _, _ = mock_flc(detrended=True)

    # Normal flare
    model_single = stacked_flare_model(time, time[485], 0.01, 1.0)
    assert model_single.shape == time.shape
    assert np.max(model_single) > 0.5

    # Two flares
    model_double = stacked_flare_model(
        time,
        time[485], 0.01, 1.0,
        time[490], 0.01, 1.1
    )
    assert model_double.shape == time.shape
    assert np.all(model_double >= model_single)
    assert np.max(model_double) > np.max(model_single)

    # Invalid flare count
    with pytest.raises(ValueError, match="groups of 3"):
        stacked_flare_model(time, time[485], 0.01)  # Only 2 args

    # Non-array time input
    with pytest.raises(TypeError):
        stacked_flare_model("not an array", time[485], 0.01, 1.0)

    # Non-1D time input
    with pytest.raises(ValueError, match="1D"):
        stacked_flare_model(np.array([[1, 2]]), time[485], 0.01, 1.0)

    # Empty time array
    with pytest.raises(ValueError, match="must not be empty"):
        stacked_flare_model(np.array([]), time[485], 0.01, 1.0)

    # NaN param should be skipped
    model_nan = stacked_flare_model(time, time[485], 0.01, np.nan)
    assert np.allclose(model_nan, 0.0)

    # Param causes internal flare_model exception
    model_broken = stacked_flare_model(time, "bad", 0.01, 1.0)
    assert np.allclose(model_broken, 0.0)



    
def test_build_baseline():
    """Test that build_baseline runs and handles bad input correctly."""

    time, flux, _ = mock_flc(detrended=True)

    # Flat baseline
    coeffs_const = [1.0, 0, 0, 0, 0]
    baseline = build_baseline(time, coeffs_const)
    assert np.allclose(baseline, 1.0)

    # Linear slope
    coeffs_slope = [np.median(flux), 2.0, 0, 0, 0]
    baseline_slope = build_baseline(time, coeffs_slope)
    assert baseline_slope.shape == time.shape
    assert not np.allclose(baseline_slope, baseline)

    # Nonlinear curve
    coeffs_poly = [np.median(flux), 0.1, -0.3, 0.1, -0.05]
    baseline_poly = build_baseline(time, coeffs_poly)
    assert np.any(np.diff(baseline_poly) != 0)

    # Wrong length coeffs
    with pytest.raises(ValueError, match="exactly 5 elements"):
        build_baseline(time, [1.0, 0, 0, 0])  # too short

    # NaN in coeffs
    with pytest.raises(ValueError, match="must be finite"):
        build_baseline(time, [1.0, 0, 0, 0, np.nan])

    # Non-array time
    with pytest.raises(TypeError):
        build_baseline("not an array", coeffs_const)

    # Empty time
    with pytest.raises(ValueError):
        build_baseline(np.array([]), coeffs_const)

    # Non-1D time
    with pytest.raises(ValueError, match="1D"):
        build_baseline(np.array([[1, 2], [3, 4]]), coeffs_const)



    
def test_log_likelihood():
    """Test that log_likelihood works and raises input errors."""

    time, flux, flux_err = mock_flc(detrended=True)
    params = [np.median(flux), 0, 0, 0, 0, time[485], 0.01, 1.0]
    model = combined_model(time, *params)

    # Perfect model: should return finite likelihood
    ll_perfect = log_likelihood(params, time, model, flux_err)
    assert np.isfinite(ll_perfect)

    # Slightly noisy: likelihood should decrease
    ll_observed = log_likelihood(params, time, flux, flux_err)
    assert np.isfinite(ll_observed)
    assert ll_observed < ll_perfect

    # Mismatched array lengths
    with pytest.raises(ValueError, match="same length"):
        log_likelihood(params, time[:-1], flux, flux_err)

    # Empty input
    with pytest.raises(ValueError, match="must not be empty"):
        log_likelihood(params, np.array([]), np.array([]), np.array([]))

    # NaNs in flux
    flux_bad = flux.copy()
    flux_bad[10] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        log_likelihood(params, time, flux_bad, flux_err)

    # Model-flux shape mismatch
    with pytest.raises(ValueError, match="same length"):
        log_likelihood(params, time[:-1], flux, flux_err)



def test_log_prior():
    """Test that log_prior runs and enforces all parameter bounds."""

    time, _, _ = mock_flc(detrended=True)
    t_peak = time[485]
    params_valid = [1.0, 0, 0, 0, 0, t_peak, 0.01, 1.0]
    t_bounds = [t_peak - 0.01, t_peak + 0.01]
    amp_bounds = (0.5, 2.0)
    fwhm_bounds = (0.001, 0.1)

    # Valid case
    assert log_prior(params_valid, t_bounds, amp_bounds, fwhm_bounds) == 0.0

    # t_peak out of bounds
    bad_t = params_valid[:]
    bad_t[5] = t_peak + 0.02
    assert log_prior(bad_t, t_bounds, amp_bounds, fwhm_bounds) == -np.inf

    # fwhm out of bounds
    bad_fwhm = params_valid[:]
    bad_fwhm[6] = 0.0001
    assert log_prior(bad_fwhm, t_bounds, amp_bounds, fwhm_bounds) == -np.inf

    # amp out of bounds
    bad_amp = params_valid[:]
    bad_amp[7] = 10.0
    assert log_prior(bad_amp, t_bounds, amp_bounds, fwhm_bounds) == -np.inf

    # NaN in flare params
    bad_nan = params_valid[:]
    bad_nan[6] = np.nan
    assert log_prior(bad_nan, t_bounds, amp_bounds, fwhm_bounds) == -np.inf

    # Too few params
    with pytest.raises(ValueError, match="at least 5 baseline"):
        log_prior([1.0, 0], t_bounds, amp_bounds, fwhm_bounds)

    # Flare params not in groups of 3
    with pytest.raises(ValueError, match="groups of 3"):
        log_prior([1.0, 0, 0, 0, 0, t_peak, 0.01], t_bounds, amp_bounds, fwhm_bounds)

    # t_bounds wrong length
    with pytest.raises(ValueError, match="2 entries per flare"):
        log_prior(params_valid, [t_peak - 0.01], amp_bounds, fwhm_bounds)

    # amp_bounds contains NaN
    with pytest.raises(ValueError, match="must contain finite values"):
        log_prior(params_valid, t_bounds, (0.5, np.nan), fwhm_bounds)


    
def test_log_posterior():
    """Test that log_posterior runs and handles all invalid input cases."""

    time, flux, flux_err = mock_flc(detrended=True)
    t_peak = time[485]
    params = [np.median(flux), 0, 0, 0, 0, t_peak, 0.01, 1.0]
    t_bounds = [t_peak - 0.01, t_peak + 0.01]
    amp_bounds = (0.5, 2.0)
    fwhm_bounds = (0.001, 0.1)

    # Valid case
    logpost = log_posterior(params, time, flux, flux_err, t_bounds, amp_bounds, fwhm_bounds)
    assert np.isfinite(logpost)

    # Prior fails (amp too large)
    bad_amp = params[:]
    bad_amp[7] = 10.0
    assert log_posterior(bad_amp, time, flux, flux_err, t_bounds, amp_bounds, fwhm_bounds) == -np.inf

    # NaN in params
    bad_nan = params[:]
    bad_nan[6] = np.nan
    assert log_posterior(bad_nan, time, flux, flux_err, t_bounds, amp_bounds, fwhm_bounds) == -np.inf

    # Too few parameters
    with pytest.raises(ValueError, match="at least 5 baseline"):
        log_posterior([1.0, 0.0], time, flux, flux_err, t_bounds, amp_bounds, fwhm_bounds)

    # Mismatched input lengths
    with pytest.raises(ValueError, match="must all have the same length"):
        log_posterior(params, time, "not_an_array", flux_err, t_bounds, amp_bounds, fwhm_bounds)
    
    with pytest.raises(ValueError, match="same length"):
        log_posterior(params, time[:-1], flux, flux_err, t_bounds, amp_bounds, fwhm_bounds)

    # Trigger fallback error from log_likelihood (e.g. broken flux input)
    broken_flux = "not_an_array"
    with pytest.raises(ValueError, match="same length"):
        result = log_posterior(params, time, broken_flux, flux_err, t_bounds, amp_bounds, fwhm_bounds)


    
def test_model_selection():
    """Test that model_selection returns AIC/BIC and handles invalid input."""

    time, flux, flux_err = mock_flc(detrended=True)
    params = [np.median(flux), 0, 0, 0, 0, time[485], 0.01, 1.0]
    model = combined_model(time, *params)

    # Valid scores
    score_bic = model_selection(model, flux, flux_err, params, method="bic")
    score_aic = model_selection(model, flux, flux_err, params, method="aic")
    assert np.isfinite(score_bic)
    assert np.isfinite(score_aic)
    assert isinstance(score_bic, float)
    assert isinstance(score_aic, float)

    # NaN in model
    model_nan = model.copy()
    model_nan[10] = np.nan
    assert model_selection(model_nan, flux, flux_err, params) == np.inf

    # Mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        model_selection(model[:-1], flux, flux_err, params)

    # Empty model/data
    empty = np.array([])
    assert model_selection(empty, np.ones(10), flux_err, params) == np.inf
    assert model_selection(np.ones(10), empty, flux_err, params) == np.inf


def test_model_selection_invalid_method():
    """Test that model_selection rejects invalid method strings."""

    time, flux, flux_err = mock_flc(detrended=True)
    params = [np.median(flux), 0, 0, 0, 0, time[485], 0.01, 1.0]
    model = combined_model(time, *params)

    with pytest.raises(ValueError, match="method must be 'aic' or 'bic'"):
        model_selection(model, flux, flux_err, params, method="invalid")



def test_extract_lc():
    """Test that extract_lc returns valid windows and handles invalid input."""

    time, flux, flux_err = mock_flc(detrended=True)

    # Normal case
    t0, t1 = 0.45, 0.55
    t_seg, f_seg, fe_seg = extract_lc(time, flux, flux_err, t0, t1)
    assert isinstance(t_seg, np.ndarray)
    assert t_seg.shape == f_seg.shape == fe_seg.shape
    assert np.all(t_seg >= t0) and np.all(t_seg <= t1)
    assert np.allclose(f_seg, flux[(time >= t0) & (time <= t1)])

    # t0 == t1
    with pytest.raises(ValueError, match="strictly less"):
        extract_lc(time, flux, flux_err, 0.5, 0.5)

    # Window outside data range
    t0_far, t1_far = 999.0, 1001.0
    t_seg, f_seg, fe_seg = extract_lc(time, flux, flux_err, t0_far, t1_far)
    assert len(t_seg) == 0 and len(f_seg) == 0

    # NaN in flux
    flux_bad = flux.copy()
    flux_bad[10] = np.nan
    with pytest.raises(ValueError, match="NaNs or infs"):
        extract_lc(time, flux_bad, flux_err, 0.4, 0.5)

    # Unequal lengths
    with pytest.raises(ValueError, match="same length"):
        extract_lc(time[:-1], flux, flux_err, 0.4, 0.5)

    # t0 or t1 not finite
    with pytest.raises(ValueError, match="finite"):
        extract_lc(time, flux, flux_err, np.nan, 0.5)

    with pytest.raises(ValueError, match="finite"):
        extract_lc(time, flux, flux_err, 0.4, np.inf)

    # Empty time input
    with pytest.raises(ValueError):
        extract_lc(np.array([]), np.array([]), np.array([]), 0.4, 0.5)


    
def test_group_peaks():
    """Test that group_peaks merges close flares correctly."""

    tstarts = np.array([0.48, 0.495, 0.7])
    tstops = np.array([0.485, 0.505, 0.71])
    buffer = 0.01

    groups = group_peaks(tstarts, tstops, buffer)
    assert isinstance(groups, list)
    assert groups == [[0, 1], [2]]
    
def test_group_peaks_invalid_buffer():
    """Test that group_peaks handles invalid buffer"""
    
    tstarts = [0.1, 0.2]
    tstops = [0.15, 0.25]
    with pytest.raises(ValueError, match="buffer must be"):
        group_peaks(tstarts, tstops, buffer=np.nan)
    with pytest.raises(ValueError, match="buffer must be"):
        group_peaks(tstarts, tstops, buffer=-1)



def test_group_peaks_edge_cases():
    """Test that group_peaks handles bad input and edge cases."""

    # Empty input
    assert group_peaks([], []) == []

    # Buffer negative
    with pytest.raises(ValueError, match="non-negative"):
        group_peaks([0.1], [0.2], buffer=-0.01)

    # NaN in input
    with pytest.raises(ValueError, match="NaN or inf"):
        group_peaks([0.1, np.nan], [0.2, 0.3])

    # Infs in input
    with pytest.raises(ValueError, match="NaN or inf"):
        group_peaks([0.1, 0.2], [0.2, np.inf])

    # Non-matching lengths
    with pytest.raises(AssertionError, match="Mismatch"):
        group_peaks([0.1, 0.2], [0.15])

    # Non-1D input
    with pytest.raises(ValueError, match="1D arrays"):
        group_peaks(np.array([[0.1, 0.2]]), np.array([0.2, 0.3]))


        
def test_fit_flares_empty_flares():
    """Should raise if no tstarts/tstops provided."""
    
    time, flux, flux_err = mock_flc(detrended=True)
    with pytest.raises(ValueError, match="No flare intervals"):
        fit_flares(time, flux, flux_err, [], [])


def test_fit_flares_nan_in_flux():
    """Should raise if flux contains NaN."""
    
    time, flux, flux_err = mock_flc(detrended=True)
    flux[10] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        fit_flares(time, flux, flux_err, [0.4], [0.5])


def test_fit_flares_invalid_max_flares():
    """Should raise if max_flares < 1."""
    
    time, flux, flux_err = mock_flc(detrended=True)
    with pytest.raises(ValueError, match="max_flares must be >= 1"):
        fit_flares(time, flux, flux_err, [0.4], [0.5], max_flares=0)


def test_fit_flares_no_peaks():
    """Should skip regions where find_peaks returns nothing."""
    
    time, flux, flux_err = mock_flc(detrended=True)
    # Flatten the region so no peaks exist
    flux[:] = np.median(flux)
    tstarts = [0.4]
    tstops = [0.5]
    results = fit_flares(time, flux, flux_err, tstarts, tstops)
    assert results == []


def test_fit_flares_result_group_none(monkeypatch):
    """Should skip result if fit_single_flare returns None."""
    
    time, flux, flux_err = mock_flc(detrended=True)
    tstarts = [0.48]
    tstops = [0.49]

    # Monkeypatch fit_single_flare to simulate failure
    monkeypatch.setattr("altaipony.fit_flares.fit_single_flare", lambda *a, **k: None)

    results = fit_flares(time, flux, flux_err, tstarts, tstops)
    assert results == []


def test_fit_flares_out_of_bounds_group():
    """Test that fit_flares raises IndexError when group region is empty."""
    t = np.linspace(0, 1, 100)
    f = np.ones_like(t)
    fe = 0.01 * np.ones_like(t)
    tstarts = [10.0]
    tstops = [10.1]  # Outside the time range

    with pytest.raises(IndexError, match="index 0 is out of bounds"):
        fit_flares(t, f, fe, tstarts, tstops, max_flares=1)


        
def test_fit_single_flare_empty_input():
    """Should return None if input arrays are empty."""
    
    result = fit_single_flare([], [], [], [], [], [], [], [], method="curve_fit")
    assert result is None


def test_fit_single_flare_mismatched_lengths():
    """Should return None if input arrays are mismatched."""
    
    time = np.linspace(0, 1, 100)
    flux = np.ones(99)
    flux_err = np.ones(100)
    result = fit_single_flare(time, flux, flux_err, flare_guess_all=[0, 0.01, 1.0],
                              bounds_lower_all=[], bounds_upper_all=[],
                              tstarts=[0], tstops=[1])
    assert result is None


def test_fit_single_flare_nan_input():
    """Should return None if NaNs in time/flux/flux_err."""
    
    time, flux, flux_err = mock_flc(detrended=True)
    flux[10] = np.nan
    t_peak = time[485]
    result = fit_single_flare(
        time, flux, flux_err,
        [t_peak, 0.01, 1.0],
        [t_peak - 0.005, 0.001, 0.5],
        [t_peak + 0.005, 0.1, 2.0],
        [t_peak - 0.01], [t_peak + 0.01],
        method="curve_fit"
    )
    assert result is None


def test_fit_single_flare_with_fixed_baseline():
    """Should accept and use fixed baseline instead of guessing."""
    
    time, flux, flux_err = mock_flc(detrended=True)
    t_peak = time[485]
    fixed_baseline = [1.0, 0, 0, 0, 0]
    result = fit_single_flare(
        time, flux, flux_err,
        [t_peak, 0.01, 1.0],
        [t_peak - 0.005, 0.001, 0.5],
        [t_peak + 0.005, 0.1, 2.0],
        [t_peak - 0.01], [t_peak + 0.01],
        method="curve_fit",
        fixed_baseline=fixed_baseline
    )
    assert isinstance(result, dict)
    assert "params" in result


def test_fit_single_flare_debug_plot():
    """Smoke test that debug_plot=True does not crash."""
    
    time, flux, flux_err = mock_flc(detrended=True)
    t_peak = time[485]
    result = fit_single_flare(
        time, flux, flux_err,
        [t_peak, 0.01, 1.0],
        [t_peak - 0.005, 0.001, 0.5],
        [t_peak + 0.005, 0.1, 2.0],
        [t_peak - 0.01], [t_peak + 0.01],
        method="curve_fit",
        debug_plot=True
    )
    assert isinstance(result, dict)


def test_fit_single_flare_curve_fit_exception():
    """Should handle failure inside curve_fit and return None."""
    
    time, flux, flux_err = mock_flc(detrended=True)
    t_peak = time[485]
    # Intentionally bad bounds (upper < lower)
    result = fit_single_flare(
        time, flux, flux_err,
        [t_peak, 0.01, 1.0],
        [t_peak + 0.005, 0.01, 1.0],   # upper < lower
        [t_peak - 0.005, 0.001, 0.5],
        [t_peak - 0.01], [t_peak + 0.01],
        method="curve_fit"
    )
    assert result is None


def test_fit_single_flare_emcee_runs():
    """Should return a valid result with emcee method."""
    
    time, flux, flux_err = mock_flc(detrended=True)
    t_peak = time[485]
    result = fit_single_flare(
        time, flux, flux_err,
        flare_guess_all=[t_peak, 0.01, 1.0],
        bounds_lower_all=[t_peak - 0.005, 0.001, 0.5],
        bounds_upper_all=[t_peak + 0.005, 0.05, 2.0],
        tstarts=[t_peak - 0.01], tstops=[t_peak + 0.01],
        method="emcee",
        max_flares=1
    )
    assert isinstance(result, dict)
    assert "posterior_samples" in result


    
    
def test_ed_from_model():
    """Ensure numerical ED calculation handles scalars, arrays, and bad inputs."""
    
    time = np.linspace(0, 1, 1000)
    baseline = np.ones_like(time)
    flare = np.exp(-((time - 0.5) / 0.01) ** 2)
    model = baseline + flare

    # Scalar baseline
    ed = ed_from_model(time, model, 1.0)
    assert np.isfinite(ed)
    assert ed > 0

    # Array baseline
    ed2 = ed_from_model(time, model, baseline)
    assert np.isclose(ed, ed2)

    # NaN in model
    model_bad = model.copy()
    model_bad[100] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        ed_from_model(time, model_bad, 1.0)

    # Mismatched lengths
    with pytest.raises(ValueError):
        ed_from_model(time[:-1], model, 1.0)

    # NaN in array baseline
    baseline_bad = baseline.copy()
    baseline_bad[10] = np.nan
    with pytest.raises(ValueError):
        ed_from_model(time, model, baseline_bad)

    # Non-finite scalar baseline
    with pytest.raises(ValueError):
        ed_from_model(time, model, np.inf)

    # Too few time points
    with pytest.raises(ValueError):
        ed_from_model(np.array([0.1]), np.array([1.0]), 1.0)


        
    
def test_make_flare_table():
    """Test that make_flare_table handles group, single, empty, and posterior errors correctly."""

    # Setup single flare
    time, flux, flux_err = mock_flc(detrended=True)
    tstarts = np.array([0.475])
    tstops = np.array([0.495])
    results = fit_flares(time, flux, flux_err, tstarts, tstops, method="curve_fit", plot=False)

    # Generate normal table
    table = make_flare_table(results)
    assert isinstance(table, pd.DataFrame)
    assert {"t_peak", "fwhm", "amplitude", "ed_rec"}.issubset(table.columns)
    assert len(table) >= 1
    assert table["ed_rec"].notna().any()

    # Empty result input should yield empty table
    empty = make_flare_table([])
    assert isinstance(empty, pd.DataFrame)
    assert empty.empty

    # Confirm EDs are computed with and without posterior
    assert table["ed_rec"].notna().all()
    if "t_peak_err" in table.columns:
        assert np.all(np.isfinite(table["t_peak_err"].dropna()))
        assert np.all(np.isfinite(table["fwhm_err"].dropna()))
        assert np.all(np.isfinite(table["amplitude_err"].dropna()))
    

    
def test_plot_flare_fit():
    """Test plot_flare_fit works without crashing and model shape matches fit region."""
    
    t, f, fe = mock_flc(ampl=0.5)
    tstarts = np.array([0.475])
    tstops = np.array([0.495])

    results = fit_flares(
        t, f, fe,
        tstarts=tstarts,
        tstops=tstops,
        max_flares=1
    )
    result = results[0]
    model = result["model"]
    t_fit = result["time"]

    assert isinstance(model, np.ndarray)
    assert model.shape == t_fit.shape

    try:
        plot_flare_fit(t_fit, result["flux"], model,
                       result.get("t_peaks", []), result.get("params", []))
    except Exception as e:
        pytest.fail(f"plot_flare_fit raised an exception: {e}")


        
def test_plot_all_fits():
    """Test plot_flare_fit works without crashing"""
    
    time, flux, flux_err = mock_flc(detrended=True)

    tstarts = np.array([0.475])
    tstops = np.array([0.495])
    results = fit_flares(time, flux, flux_err, tstarts, tstops, method="curve_fit", plot=False)

    try:
        plot_all_fits(time, flux, results)
    except Exception as e:
        assert False, f"plot_all_fits raised an exception: {e}"
