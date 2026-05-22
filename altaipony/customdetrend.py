"""
UTF-8, Python 3

------------------
AltaiPony
------------------

Ekaterina Ilin, 2023, MIT License

This module contains custom detrending functions.
"""

import numpy as np
import pandas as pd

from .altai import _find_iterative_median, equivalent_duration
from .utils import sigma_clip


import matplotlib.pyplot as plt

import astropy.units as u

from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

from astropy.timeseries import LombScargle




def custom_detrending(lc, 
                      savgol1=6., savgol2=3., pad=3, max_sigma=2.5, 
                      longdecay=6, maxgap=10, debug_plot=False,
                      break_tolerance=10,
                      periodicity_fap_threshold=1e-3,
                      periodicity_amplitude_threshold=0.01,
                      period_min_days=0.1,
                      n_sine_harmonics=5,
                      refine_period_per_segment=True,
                      multisine_gap_fraction=0.3,
                      n_per=10,
                      max_prewhiten_iter=3):
    """Custom de-trending for TESS and Kepler 
    short cadence light curves, including TESS Cycle 3 20s
    cadence.
    
    Parameters:
    ------------
    lc : FlareLightCurve
        light curve that has at least time, flux and flux_err
    spline_coarseness : float
        time scale in hours for spline points. 
        See fit_spline for details.
    spline_order: int
        Spline order for the coarse spline fit.
        Default is cubic spline.
    savgol1 : float
        Window size for first Savitzky-Golay filter application.
        Unit is hours, defaults to 6 hours.
    savgol2 : float
        Window size for second Savitzky-Golay filter application.
        Unit is hours, defaults to 3 hours.
    pad : 3
        Outliers in Savitzky-Golay filter are padded with this
        number of data points. Defaults to 3.
    max_sigma : float
        Outlier rejection threshold in sigma. Defaults to 2.5.
    longdecay : int
        Long decay time for outlier rejection. Defaults to 6.
    maxgap : float
        Maximum gap size in days for spline fitting. Defaults to 10 x cadence size.
    debug_plot: bool
        If True will plot a figure with the flux after each of the detrending steps, 
        i.e., spline, and the two Sav-Gol iterations 
    break_tolerance: int
        If there are large gaps in time, flatten will split the flux into 
        several sub-lightcurves and apply savgol_filter to each individually. 
        A gap is defined as a period in time larger than break_tolerance times 
        the median gap. To disable this feature, set break_tolerance to None.
    periodicity_fap_threshold : float
        False-alarm probability threshold for the Lomb-Scargle test.
        If the peak FAP is below this value the light curve is considered
        strongly periodic.  Defaults to 1e-3.
    periodicity_amplitude_threshold : float
        Minimum semi-amplitude (as a fraction of the median flux) for the
        periodic signal to trigger the multi-sine path.  Defaults to 0.01
        (1 % of the median flux).
    period_min_days : float
        Shortest period searched by the Lomb-Scargle periodogram, in days.
        Defaults to 0.1 days.
    n_sine_harmonics : int
        Number of harmonics included in the multi-sine baseline model.
        Harmonics 1 … n_sine_harmonics of the dominant period are fitted
        simultaneously via least-squares, which captures non-sinusoidal
        (but strictly periodic) shapes.  Defaults to 5.
    refine_period_per_segment : bool
        If True, each gap segment independently refines the global peak
        period with a narrow Lomb-Scargle search, accommodating slightly
        drifting rotation periods.  Defaults to True.
    multisine_gap_fraction : float
        When a strong periodicity is detected, gaps that are shorter than
        this fraction of the dominant period are bridged rather than used
        as segment breaks.  E.g. the default of 0.3 means gaps ≤ 30 % of
        the rotation period are treated as continuous data for the multi-sine
        fit.  The original (tight) segmentation is still used for all
        downstream Savitzky-Golay steps.  Defaults to 0.3.
    n_per : int
        Maximum segment length for the multi-sine fit, expressed in units
        of the dominant period.  Any segment longer than ``n_per`` cycles
        is split in two at its midpoint before fitting, so that the linear
        trend term has a shorter lever arm and the per-segment amplitude
        is more locally representative.  Defaults to 10.
    max_prewhiten_iter : int
        Maximum number of additional prewhitening iterations after the
        initial multisine fit.  Each iteration runs a fresh Lomb-Scargle
        periodogram on the current residuals; if a significant peak at a
        new period is found, another multisine is fitted and subtracted.
        Set to 0 to disable prewhitening.  Defaults to 3.

    Return:
    -------
    FlareLightCurve with detrended_flux attribute
    """
    dt = np.mean(np.diff(lc.time.value))
    gaps = lc.find_gaps(maxgap=maxgap * dt).gaps
    # Store original flux as a column so it survives filtering operations
    lc["original_flux"] = lc.flux.copy()
    lc["original_flux_err"] = lc.flux_err.copy()


    plt.figure(figsize=(20, 5))
    plt.plot(lc.time.value, lc.flux, 'r.', markersize=10)
    lc = lc.interpolate_missing_cadences()
    plt.plot(lc.time.value, lc.flux, 'k.', markersize=1)
    time, flux = lc.time.value, lc.flux.value
    
    

    # --- Periodicity check ------------------------------------------------
    # Run a Lomb-Scargle periodogram on the raw flux.  If a strong periodic
    # signal is found (low FAP *and* large amplitude) use a multi-harmonic
    # sine model as the baseline instead of the spline, because a spline
    # will chase the periodic oscillations and corrupt flare detection.
    period_max_days = (time[-1] - time[0]) / 2.0

    is_periodic, dominant_period, rel_amplitude, fap = detect_strong_periodicity(
        time, flux,
        fap_threshold=periodicity_fap_threshold,
        amplitude_threshold=periodicity_amplitude_threshold,
        period_min=period_min_days,
        period_max=period_max_days,
    )

    if is_periodic:
        print(
            f"Strong periodicity detected: P = {dominant_period:.4f} d, "
            f"rel. amplitude = {rel_amplitude:.4f}, FAP = {fap:.2e}. "
            "Using multi-sine baseline fit."
        )
        flux_med = _find_iterative_median(flux, gaps, longdecay=longdecay)

        # Re-segment with a more lenient maxgap so that short gaps
        # (≤ multisine_gap_fraction × period) are bridged.  This avoids
        # breaking a continuous rotation cycle into tiny segments that each
        # get a poor amplitude estimate.  The original tight `gaps` are kept
        # for all downstream Savitzky-Golay steps.
        multisine_maxgap = multisine_gap_fraction * dominant_period
        multisine_gaps = lc.find_gaps(maxgap=multisine_maxgap).gaps

        # Split any segment longer than n_per cycles so the per-segment
        # linear trend and amplitude have a shorter, more locally valid
        # lever arm.
        multisine_gaps = _split_long_segments(
            multisine_gaps, time, dominant_period, n_per
        )
        for l, r in multisine_gaps:
            plt.axvline(time[l], color='cyan', linestyle='--', alpha=0.5)
            plt.axvline(time[r-1], color='cyan', linestyle='--', alpha=0.5)

        print(
            f"Multi-sine gap threshold: {multisine_maxgap:.4f} d, "
            f"max segment: {n_per} cycles — "
            f"{len(multisine_gaps)} segments "
            f"(tight segmentation had {len(gaps)})."
        )

        m2flux, _, best_params = fit_multisine(
            time, flux, flux_med, multisine_gaps,
            period=dominant_period,
            n_harmonics=n_sine_harmonics,
            refine_period=refine_period_per_segment,
        )
        plt.plot(time, m2flux, 'b.', markersize=1, label="after multisine fit")
        best_params["method"] = "multisine"
        best_params["dominant_period"] = dominant_period
        best_params["multisine_n_segments"] = len(multisine_gaps)

    else:
        # fit a spline to the general trends
        m2flux, _, best_params = fit_spline(time, flux, gaps, longdecay=longdecay)
        best_params["method"] = "spline"

    print("Baseline detrending params:", best_params)
    
    # choose a 6 hour window
    w1 = int((np.rint(savgol1 / 24. / dt) // 2) * 2 + 1)

    lc.flux = m2flux * u.electron / u.s
    lc.flux_err = lc.flux_err * u.electron / u.s

    # Snapshot of flux after baseline (spline or multisine) removal,
    # aligned to the full interpolated grid before any Savitzky-Golay pass.
    flux_after_baseline = m2flux.copy()

    if debug_plot == True:
        plt.figure(figsize=(8,4))
        plt.plot(lc.time.value, lc.flux.value + 5000, 'k.', markersize=1,
                 label="after spline fit")

    # use Savitzy-Golay to iron out the rest    
    lc3 = lc.detrend("savgol", w=w1, pad=pad,
                      max_sigma=max_sigma, longdecay=longdecay,
                      break_tolerance=break_tolerance)
    
    lc3.flux = lc3.detrended_flux 
 
    if debug_plot == True:
        plt.plot(lc3.time.value, lc3.flux.value, 'r.', 
                 markersize=1, label="after first Sav-Gol step")

    # choose a uneven window size
    w2 = int((np.rint(savgol2 / 24. / dt) // 2) * 2 + 1)

    # use Savitzy-Golay to iron out the rest
    lc4 = lc3.detrend("savgol", w=w2, pad=pad, 
                      max_sigma=max_sigma, longdecay=longdecay,
                      break_tolerance=break_tolerance)
    
    if debug_plot == True:
        plt.plot(lc4.time.value, lc4.detrended_flux.value, 'b.', 
                 markersize=1, label="after second Sav-Gol step")
        plt.xlabel("Time [BTJD or BKJD]")
        plt.ylabel("Flux [e-/s]")
        plt.legend()

    # Restore original flux from the column (now properly filtered to match lc4's length)
    lc4.flux = lc4["original_flux"] * u.electron / u.s
    
    # Clean up the temporary column
    lc4.remove_column("original_flux")
    lc.flux = lc["original_flux"] * u.electron / u.s
    lc.flux_err = lc["original_flux_err"] * u.electron / u.s
    
    # find median value
    lc4.find_iterative_median()


    # ------------------------------------------------------------------
    # Check whether each SG step actually modified the flux.  A step is
    # considered to have had no effect when the RMS of its change is below
    # the point-to-point noise floor of the input stage.
    # ------------------------------------------------------------------
    t4 = lc4.time.value
    t_interp = lc.time.value

    # Align flux_after_baseline to lc4's grid
    if len(flux_after_baseline) == len(t4):
        f_bl = flux_after_baseline
    else:
        idx = np.searchsorted(t_interp, t4)
        idx = np.clip(idx, 0, len(flux_after_baseline) - 1)
        f_bl = flux_after_baseline[idx]

    # Align savgol1 output to lc4's grid
    t3 = lc3.time.value
    if len(lc3.detrended_flux) == len(t4):
        f_sg1 = np.array(lc3.detrended_flux)
    else:
        idx3 = np.searchsorted(t3, t4)
        idx3 = np.clip(idx3, 0, len(lc3.detrended_flux) - 1)
        f_sg1 = np.array(lc3.detrended_flux)[idx3]

    f_sg2 = np.array(lc4.detrended_flux)

    def _touched(before, after):
        """True when the RMS change exceeds the point-to-point noise floor."""
        delta = after - before
        valid = ~np.isnan(delta)
        if valid.sum() < 10:
            return False
        rms_change = np.sqrt(np.mean(delta[valid] ** 2))
        f_v = before[~np.isnan(before)]
        noise = np.nanmedian(np.abs(np.diff(f_v))) * 1.4826 / np.sqrt(2)
        return bool(rms_change > noise)

    savgol1_touched = _touched(f_bl,  f_sg1)
    savgol2_touched = _touched(f_sg1, f_sg2)

    if not savgol1_touched:
        print("WARNING: savgol1 did not modify the light curve.")
    if not savgol2_touched:
        print("WARNING: savgol2 did not modify the light curve.")

    best_params["savgol1_touched"] = savgol1_touched
    best_params["savgol2_touched"] = savgol2_touched

    return lc4



def estimate_detrended_noise(flc, mask_pos_outliers_sigma=2.5, 
                             std_window=100, longdecay=6):
    """
    Estimate detrended flux uncertainties using rolling standard deviation.
    
    Parameters
    ----------
    flc : FlareLightCurve
        Light curve with detrended_flux attribute
    mask_pos_outliers_sigma : float
        Sigma threshold for masking positive outliers (likely flares)
    std_window : int
        Window size for rolling standard deviation calculation
    longdecay : int
        Long decay time for outlier rejection
    
    Returns
    -------
    flc : FlareLightCurve
        Input light curve with detrended_flux_err attribute updated
    """
    # Find gaps if not already done
    if flc.gaps is None:
        flc = flc.find_gaps()
    
    # Extract arrays we need (avoids repeated attribute access)
    detrended_flux = flc.detrended_flux
    n_points = len(detrended_flux)
    
    # Initialize output array
    detrended_flux_err = np.full(n_points, np.nan)
    
    # Process each gap segment
    for (le, ri) in flc.gaps:
        # Extract segment
        flux_segment = detrended_flux[le:ri].copy()  # Copy just this segment
        
        # First pass: mask outliers and compute initial error estimate
        mask = sigma_clip(flux_segment, max_sigma=mask_pos_outliers_sigma, 
                         longdecay=longdecay)
        
        # Set outliers to NaN for error calculation
        flux_segment_masked = flux_segment.copy()
        flux_segment_masked[~mask] = np.nan
        
        # Second pass: refine by finding iterative median
        it_med_segment = _find_iterative_median(
            flux_segment, 
            gaps=[(0, ri - le)]
        )
        
        # Subtract iterative median for better outlier detection
        flux_normalized = flux_segment - it_med_segment
        
        # Mask outliers again
        mask_refined = sigma_clip(flux_normalized, 
                                 max_sigma=mask_pos_outliers_sigma, 
                                 longdecay=2)
                
        # Set outliers to NaN
        flux_normalized_masked = flux_normalized.copy()
        flux_normalized_masked[~mask_refined] = np.nan
        
        # Compute final rolling std
        final_err = (pd.Series(flux_normalized_masked)
                    .rolling(std_window, center=True, min_periods=1)
                    .std()
                    .interpolate()
                    .values)

        # Store in output array
        detrended_flux_err[le:ri] = final_err
    
    # Set the result on the original lightcurve
    flc.detrended_flux_err = detrended_flux_err
    
    return flc


def detect_strong_periodicity(time, flux,
                              fap_threshold=1e-3,
                              amplitude_threshold=0.01,
                              period_min=0.1,
                              period_max=None,
                              flux_median=None):
    """Test whether the light curve contains a strong, high-amplitude periodic
    signal using the Lomb-Scargle periodogram.

    Both conditions must be satisfied simultaneously for the function to return
    ``True``:

    * The false-alarm probability of the highest peak is below
      ``fap_threshold``.
    * The semi-amplitude of a single-frequency sine fitted at the peak
      frequency exceeds ``amplitude_threshold`` × (median flux).

    Parameters
    ----------
    time : array_like
        Time values in days.
    flux : array_like
        Raw flux values.
    fap_threshold : float
        FAP threshold below which the signal is considered significant.
        Defaults to 1e-3.
    amplitude_threshold : float
        Minimum fractional semi-amplitude (relative to median flux) for the
        signal to be considered high-amplitude.  Defaults to 0.01.
    period_min : float
        Minimum period to search, in days.  Defaults to 0.1.
    period_max : float or None
        Maximum period to search, in days.  Defaults to half the time-span.

    flux_median : float or None
        Reference median used to express the semi-amplitude as a fraction.
        If None, the median of ``flux`` itself is used.  Pass the original
        raw-flux median when calling on residuals (which are centred near
        zero) to avoid division by a near-zero value.

    Returns
    -------
    is_periodic : bool
        True when both the FAP and amplitude criteria are met.
    peak_period : float
        Period of the highest LS peak in days.
    rel_amplitude : float
        Fractional semi-amplitude of the best-fit sine at the peak period
        relative to the reference median flux.
    fap : float
        False-alarm probability of the peak power under the
        Baluev (2008) analytic approximation.
    """
    valid = ~(np.isnan(time) | np.isnan(flux))
    t = time[valid]
    f = flux[valid]

    if len(t) < 20:
        return False, np.nan, np.nan, np.nan

    f_median = np.nanmedian(f)

    # Use caller-supplied reference median for amplitude normalisation if given.
    # This is essential when `flux` is a residual centred near zero, where the
    # local median would be ~0 and cause division-by-zero.
    ref_median = flux_median if flux_median is not None else f_median

    if ref_median == 0:
        return False, np.nan, np.nan, np.nan

    # Centre flux so LS is not confused by a DC offset
    f_centred = f - f_median

    if period_max is None:
        period_max = (t[-1] - t[0]) / 2.0

    # Guard against degenerate ranges
    if period_max <= period_min:
        period_max = period_min * 10.0

    freq_min = 1.0 / period_max
    freq_max = 1.0 / period_min

    ls = LombScargle(t, f_centred)
    frequency, power = ls.autopower(
        minimum_frequency=freq_min,
        maximum_frequency=freq_max,
        samples_per_peak=10,
    )

    if len(power) == 0:
        return False, np.nan, np.nan, np.nan

    peak_idx = np.argmax(power)
    peak_freq = frequency[peak_idx]
    peak_period = 1.0 / peak_freq
    peak_power = power[peak_idx]

    # False-alarm probability (Baluev 2008 analytic approximation)
    fap = ls.false_alarm_probability(peak_power, method="baluev")

    # Semi-amplitude from LS model coefficients at the peak frequency:
    # model = offset + a*cos(2π f t) + b*sin(2π f t)
    # semi-amplitude = sqrt(a² + b²)
    theta = ls.model_parameters(peak_freq)   # [offset, a_cos, b_sin]
    rel_amplitude = np.sqrt(theta[1] ** 2 + theta[2] ** 2) / abs(ref_median)

    is_periodic = bool((fap < fap_threshold) and (rel_amplitude > amplitude_threshold))

    return is_periodic, peak_period, rel_amplitude, fap


def _split_long_segments(gaps, time, period, n_per):
    """Bisect any gap segment longer than ``n_per`` cycles of ``period``.

    The split is applied repeatedly until every segment satisfies the length
    criterion, so segments that are e.g. 3× too long get split into quarters,
    not just halves.  The split point is always the index closest to the
    temporal midpoint of the segment so that the two halves are roughly equal
    in duration.

    Parameters
    ----------
    gaps : list of (int, int)
        Segment boundaries as produced by ``FlareLightCurve.find_gaps``.
    time : array_like
        Full time array (days).
    period : float
        Dominant period in days.
    n_per : int
        Maximum allowed segment length in units of ``period``.

    Returns
    -------
    list of (int, int)
        New gap list with long segments bisected.
    """
    max_span = n_per * period
    result = []
    queue = list(gaps)

    while queue:
        le, ri = queue.pop(0)
        span = time[ri - 1] - time[le]

        if span <= max_span:
            result.append((le, ri))
        else:
            # Find the index closest to the temporal midpoint
            t_mid = 0.5 * (time[le] + time[ri - 1])
            mid = le + np.argmin(np.abs(time[le:ri] - t_mid))

            # Guard: both halves must be non-empty
            if mid <= le:
                mid = le + 1
            if mid >= ri:
                mid = ri - 1

            # Push both halves back for further checking
            queue.insert(0, (mid, ri))
            queue.insert(0, (le, mid))

    return result


def fit_multisine(time, flux, flux_med, gaps,
                  period,
                  n_harmonics=5,
                  refine_period=True,
                  period_refine_window=0.05):
    """Fit a multi-harmonic sine baseline to the light curve.

    The model for each gap-segment is::

        f_model(t) = c₀ + c₁(t − t_mid)
                   + Σₖ₌₁ᴺ [ aₖ cos(2π k t / Pₛₑg) + bₖ sin(2π k t / Pₛₑg) ]

    where the coefficients are solved via ordinary least-squares.  The linear
    term ``c₁(t − t_mid)`` absorbs any slow baseline drift within the segment
    so that the harmonic amplitudes are not biased by it.  Time is centred on
    the segment midpoint ``t_mid`` to keep the offset ``c₀`` and the slope
    ``c₁`` numerically orthogonal.  Fitting
    per segment allows the *amplitude to evolve* naturally across the
    observation baseline.  Optionally, ``Pₛₑg`` is refined independently for
    each segment with a narrow Lomb-Scargle search around the global ``period``
    to accommodate *slightly varying periods* (e.g. differential rotation).

    The detrended flux follows the same convention used by ``fit_spline``:
    residuals are re-centred at the iterative-median baseline so that
    subsequent Savitzky-Golay passes work on a nearly zero-mean signal.

    Parameters
    ----------
    time : array_like
        Full time array in days.
    flux : array_like
        Raw flux values.
    flux_med : array_like
        Iterative-median baseline (output of ``_find_iterative_median``).
    gaps : list of (int, int)
        Segment boundaries ``(left_index, right_index)`` as produced by
        ``FlareLightCurve.find_gaps``.
    period : float
        Starting period in days (typically the Lomb-Scargle peak).
    n_harmonics : int
        Number of harmonics to include (1 = pure sine; higher values capture
        non-sinusoidal waveforms).  Defaults to 5.
    refine_period : bool
        If True, refine the period independently for each segment using a
        narrow Lomb-Scargle search.  Defaults to True.
    period_refine_window : float
        Half-width of the period search range expressed as a *fraction* of
        ``period``.  E.g. 0.05 searches ±5 % around ``period``.
        Defaults to 0.05.

    Returns
    -------
    newflux : ndarray
        Detrended flux re-centred at ``flux_med``.
    model : ndarray
        Best-fit multi-sine model evaluated on the full ``time`` array.
    best_params : dict
        Summary of fit parameters: dominant period and per-segment periods.
    """
    model   = np.full_like(flux, np.nan, dtype=float)
    newflux = np.full_like(flux, np.nan, dtype=float)

    n_cols = 2 * n_harmonics + 2  # [offset, t_linear, cos_1, sin_1, …, cos_N, sin_N]
    seg_periods = {}

    for le, ri in gaps:
        t_seg = time[le:ri]
        f_seg = flux[le:ri]
        fmed_seg = np.nanmedian(flux_med[le:ri])

        valid = ~(np.isnan(t_seg) | np.isnan(f_seg))
        n_valid = np.sum(valid)

        # Need at least as many valid points as free parameters
        if n_valid < n_cols + 1:
            newflux[le:ri] = f_seg
            model[le:ri]   = fmed_seg
            seg_periods[le] = period
            continue

        t_v = t_seg[valid]
        f_v = f_seg[valid]
        seg_len_days = t_v[-1] - t_v[0]

        # Centre time on the segment midpoint so the linear term is
        # orthogonal to the constant offset and numerically well-conditioned.
        t_mid = 0.5 * (t_v[0] + t_v[-1])
        t_v_c   = t_v   - t_mid
        t_seg_c = t_seg - t_mid

        # --- optional per-segment period refinement -----------------------
        seg_period = period

        if refine_period and seg_len_days > 2.0 * period:
            # Only worth refining when the segment covers multiple cycles
            freq_ctr   = 1.0 / period
            freq_delta = freq_ctr * period_refine_window
            freq_lo    = max(freq_ctr - freq_delta, 1.0 / (seg_len_days + 1e-6))
            freq_hi    = freq_ctr + freq_delta

            if freq_lo < freq_hi:
                refine_freqs = np.linspace(freq_lo, freq_hi, 400)
                f_centred    = f_v - np.nanmedian(f_v)
                ls_seg       = LombScargle(t_v, f_centred)
                seg_power    = ls_seg.power(refine_freqs)
                seg_period   = 1.0 / refine_freqs[np.argmax(seg_power)]

        seg_periods[le] = seg_period

        # --- build harmonic design matrix ---------------------------------
        # Columns: [1, t_c, cos(2π t/P), sin(2π t/P), …, cos(2πN t/P), sin(2πN t/P)]
        # The linear term (t_c) captures any slow baseline drift within the
        # segment, so the harmonic coefficients are not biased by it.
        A_valid = np.ones((n_valid, n_cols))
        A_full  = np.ones((ri - le, n_cols))

        # Column 1: linear trend (time centred on segment midpoint)
        A_valid[:, 1] = t_v_c
        A_full[:, 1]  = t_seg_c

        for k in range(1, n_harmonics + 1):
            phase_v = 2.0 * np.pi * k * t_v   / seg_period
            phase_f = 2.0 * np.pi * k * t_seg / seg_period
            A_valid[:, 2*k]     = np.cos(phase_v)
            A_valid[:, 2*k + 1] = np.sin(phase_v)
            A_full[:, 2*k]      = np.cos(phase_f)
            A_full[:, 2*k + 1]  = np.sin(phase_f)

        # --- ordinary least-squares solution ------------------------------
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_valid, f_v, rcond=None)
            model_seg = A_full @ coeffs
        except Exception:
            # Fallback: constant equal to segment median
            model_seg = np.full(ri - le, np.nanmedian(f_seg))

        # plt.plot(t_seg, f_seg, 'k.', markersize=1)
        plt.plot(t_seg, model_seg, 'b-', linewidth=2)

        # Centre the residual before storing.  An imperfect fit leaves a
        # DC offset in (f_seg - model_seg): the residual median drifts away
        # from zero, which then biases the sigma-clip threshold in the
        # downstream Savitzky-Golay step (everything ends up above or below
        # the median, causing the clip to treat the oscillation asymmetrically
        # and flag ~40 % of the LC as a single flare candidate).
        # Subtracting the residual median forces the output to be centred at
        # fmed_seg regardless of fit quality, without altering the oscillation
        # shape or the period/harmonic content.
        residual  = f_seg - model_seg
        # valid_res = residual[~np.isnan(f_seg)]
        # dc_offset = np.nanmedian(valid_res) if len(valid_res) > 0 else 0.0

        model[le:ri]   = model_seg# + dc_offset   # keep model consistent
        newflux[le:ri] = residual  + fmed_seg #- dc_offset
        plt.plot(t_seg, residual+fmed_seg, 'b-', linewidth=2)

    best_params = {
        "n_harmonics"   : n_harmonics,
        "global_period" : period,
        "seg_periods"   : seg_periods,
    }

    return newflux, model, best_params


def fit_spline(time, flux, gaps, 
               coarseness_range=(5, 15, 1),
               spline_orders=(2, 3),
               n_phase_shifts=3,
               percentile_anchor=25,
               edge_penalty_weight=1.,
               **kwargs):
    """Fit multiple splines and select the one that best approximates
    the underlying light curve shape while avoiding flare contamination.
    
    Parameters:
    -----------
    time : array
        Time values
    flux : array  
        Flux values
    gaps : list of tuples
        List of (start, end) indices for continuous segments
    coarseness_range : tuple
        (min, max, step) for spline coarseness in hours
    spline_orders : tuple
        Spline orders to try
    n_phase_shifts : int
        Number of phase shifts to try for bin sampling
    percentile_anchor : float
        Percentile to use for robust bin estimation (lower = more flare-resistant)
    edge_penalty_weight : float
        Weight for penalizing edge deviations in scoring (higher = stronger penalty)
    **kwargs : dict
        Additional arguments for _find_iterative_median
        
    Returns:
    --------
    newflux : array
        Detrended flux
    model : array
        Best spline model
    best_params : dict
        Parameters of the best fit
    """
    flux_med = _find_iterative_median(flux, gaps, **kwargs)
    
    coarseness_values = np.arange(
        coarseness_range[0], 
        coarseness_range[1] + 1, 
        coarseness_range[2]
    )
    
    dt = np.nanmin(np.diff(time))
    
    candidates = []
    
    # Generate all candidate fits
    for coarseness in coarseness_values:
        for k in spline_orders:
            for phase_idx in range(n_phase_shifts):
                model, newflux = _fit_single_spline(
                    time, flux, flux_med, gaps, 
                    coarseness, k, dt, 
                    phase_idx, n_phase_shifts,
                    percentile_anchor
                )
                
                score = _evaluate_spline_fit(flux, model, gaps, 
                                             edge_penalty_weight=edge_penalty_weight)
                
                candidates.append({
                    'model': model,
                    'newflux': newflux,
                    'score': score,
                    'coarseness': coarseness,
                    'order': k,
                    'phase': phase_idx
                })

    
    # Select best candidate
    best = min(candidates, key=lambda x: x['score'])
    
    best_params = {
        'coarseness': best['coarseness'],
        'order': best['order'],
        'phase': best['phase'],
        'score': best['score']
    }
    
    return best['newflux'], best['model'], best_params


def _fit_single_spline(time, flux, flux_med, gaps, coarseness, k, dt, 
                       phase_idx, n_phases, percentile):
    """Fit a single spline configuration."""
    n = int(np.rint(coarseness / 24 / dt))
    
    model = np.full_like(flux, np.nan)
    newflux = np.full_like(flux, np.nan)
    
    for le, ri in gaps:
        segment_len = ri - le
        
        # Calculate phase offset for this segment
        phase_offset = min((phase_idx * n) // max(n_phases, 1), segment_len - 1)
        
        if segment_len <= n:
            # Segment too short for binning
            newflux[le:ri] = flux[le:ri]
            model[le:ri] = np.nanmedian(flux[le:ri])
            continue
            
        # Build knot points with phase offset
        t_knots, f_knots = _build_knot_points(
            time[le:ri], flux[le:ri], n, phase_offset, percentile
        )
        
        if len(t_knots) <= k:
            # Too few knots, use linear fit
            valid = ~np.isnan(flux[le:ri])
            if np.sum(valid) > 1:
                p2 = np.polyfit(time[le:ri][valid], flux[le:ri][valid], 1)
                model[le:ri] = np.polyval(p2, time[le:ri])
                newflux[le:ri] = flux[le:ri] - model[le:ri] + flux_med[le:ri]
            else:
                newflux[le:ri] = flux[le:ri]
                model[le:ri] = flux_med[le:ri]
        else:
            # Fit spline
            try:
                spline = UnivariateSpline(t_knots, f_knots, k=k, s=0)
                model[le:ri] = spline(time[le:ri])
                newflux[le:ri] = flux[le:ri] - model[le:ri] + flux_med[le:ri]
            except Exception:
                # Fallback to linear
                p2 = np.polyfit(time[le:ri], flux[le:ri], 1)
                model[le:ri] = np.polyval(p2, time[le:ri])
                newflux[le:ri] = flux[le:ri] - model[le:ri] + flux_med[le:ri]
    
    return model, newflux


def _build_knot_points(time, flux, n, phase_offset, percentile):
    """Build knot points for spline fitting using robust statistics.
    
    Parameters:
    -----------
    time : array
        Time values for this segment
    flux : array
        Flux values for this segment
    n : int
        Bin size in cadences
    phase_offset : int
        Starting offset for binning
    percentile : float
        Percentile for robust flux estimation (lower = more flare-resistant)
    """
    segment_len = len(time)
    
    # Apply phase offset
    start = phase_offset
    usable_len = segment_len - start
    n_bins = usable_len // n
    
    if n_bins == 0:
        return np.array([time[0], time[-1]]), np.array([flux[0], flux[-1]])
    
    remainder = usable_len % n
    end_idx = segment_len - remainder if remainder > 0 else segment_len
    
    # Reshape into bins
    t_binned = time[start:end_idx].reshape(n_bins, n)
    f_binned = flux[start:end_idx].reshape(n_bins, n)
    
    # Use mean for time, robust percentile for flux (avoids flare bias)
    t_knots = np.nanmean(t_binned, axis=1)
    f_knots = np.nanpercentile(f_binned, percentile, axis=1)
    
    # Add boundary points
    t_knots = np.concatenate([[time[0]], t_knots, [time[-1]]])
    f_knots = np.concatenate([[flux[0]], f_knots, [flux[-1]]])
    
    # Remove any NaN knots
    valid = ~(np.isnan(t_knots) | np.isnan(f_knots))
    
    return t_knots[valid], f_knots[valid]


def _evaluate_spline_fit(flux, model, gaps, edge_fraction=0.1, edge_penalty_weight=.5):
    """Evaluate spline fit quality, penalizing flare contamination and edge effects.
    
    A good baseline should have:
    1. Low scatter in residuals (captured by MAD)
    2. Symmetric negative residuals (noise-like)
    3. Positive outliers should be clearly separated (flares not fit)
    4. Model values at segment edges should not deviate strongly from segment mean
    
    Parameters:
    -----------
    flux : array
        Original flux values
    model : array
        Spline model values
    gaps : list of tuples
        Segment boundaries
    edge_fraction : float
        Fraction of segment to consider as "edge" (default 10%)
    edge_penalty_weight : float
        Weight for edge deviation penalty (default 0.5)
    """
    residuals = []
    edge_deviations = []
    
    for le, ri in gaps:
        seg_len = ri - le
        valid = ~(np.isnan(model[le:ri]) | np.isnan(flux[le:ri]))
        
        if np.sum(valid) > 0:
            residuals.extend(flux[le:ri][valid] - model[le:ri][valid])
        
        # Calculate edge deviation penalty for this segment
        if seg_len > 20:  # Only for segments long enough to have meaningful edges
            edge_size = max(int(seg_len * edge_fraction), 5)
            
            # Get segment median (robust estimate of typical level)
            seg_flux = flux[le:ri]
            seg_model = model[le:ri]
            seg_median = np.nanmedian(seg_flux)
            
            # Check model deviation from median at left edge
            left_model = np.nanmean(seg_model[:edge_size])
            left_dev = abs(left_model - seg_median)
            
            # Check model deviation from median at right edge
            right_model = np.nanmean(seg_model[-edge_size:])
            right_dev = abs(right_model - seg_median)
            
            edge_deviations.extend([left_dev, right_dev])
    
    residuals = np.array(residuals)
    
    if len(residuals) < 10:
        return np.inf
    
    median_res = np.median(residuals)
    mad = np.median(np.abs(residuals - median_res))
    
    if mad < 1e-10:
        return np.inf
    
    # Analyze residual distribution asymmetry
    # Lower residuals should behave like Gaussian noise
    # Upper residuals will include flares
    lower_res = residuals[residuals <= median_res]
    upper_res = residuals[residuals > median_res]
    
    if len(lower_res) < 5 or len(upper_res) < 5:
        return mad
    
    # For a good fit, the lower tail should be symmetric around median
    # Measure: how Gaussian-like is the lower distribution?
    lower_std = np.std(lower_res)
    lower_mad = np.median(np.abs(lower_res - np.median(lower_res)))
    
    # Ratio close to 1.4826 indicates Gaussian-like distribution
    # (for Gaussian: std/MAD ≈ 1.4826)
    gaussian_ratio = 1.4826
    lower_gaussianity = abs(lower_std / (lower_mad + 1e-10) - gaussian_ratio)
    
    # Penalize if model is tracking flares (upper spread much larger than lower)
    upper_spread = np.percentile(upper_res, 90) - median_res
    lower_spread = median_res - np.percentile(lower_res, 10)
    
    # Asymmetry ratio - for clean baseline, expect upper >> lower due to flares
    # If upper ≈ lower, model may be tracking flares
    if lower_spread > 1e-10:
        asymmetry = upper_spread / lower_spread
        # We want asymmetry > 1 (positive outliers = flares not being fit)
        # Penalize if asymmetry is too close to 1
        asymmetry_penalty = max(0, 2.0 - asymmetry) * 0.3
    else:
        asymmetry_penalty = 0
    
    # Edge deviation penalty: penalize if model deviates from segment mean at edges
    # Normalize by MAD so it's scale-independent
    if len(edge_deviations) > 0:
        mean_edge_dev = np.mean(edge_deviations)
        # Express edge deviation in units of MAD
        edge_penalty = edge_penalty_weight * (mean_edge_dev / mad)
    else:
        edge_penalty = 0
    
    # Combined score: lower is better
    score = mad * (1 + asymmetry_penalty + 0.1 * lower_gaussianity + edge_penalty)
    
    return score



def measure_flare(flc, sta, sto):
    """Give start and stop indices into a de-trended
    light curve, calculate flare properties assuming that
    what's inbetween is a flares, and add the result
    to FlareLightCurve.flares.
    
    Parameters:
    -------------
    flc : FlareLightCurve
        de-trended light curve
    sta : int
        start index of flare
    sto : int
        stop index of flare
    """
    # get ED
    ed_rec, ed_rec_err = equivalent_duration(flc, sta, sto, err=True)
    
    # get amplitude
    ampl_rec = np.max(flc.detrended_flux.value[sta:sto]) / flc.it_med.value[sta] - 1. 
    
    # get cadence numbers
    cstart = flc.cadenceno.value[sta]
    cstop = flc.cadenceno.value[sto]
    
    # get time stamps 
    tstart = flc.time.value[sta]
    tstop = flc.time.value[sto]
    
    # add result to flare table
    newline = pd.Series(
                        {'ed_rec': ed_rec,
                        'ed_rec_err': ed_rec_err,
                        'ampl_rec': ampl_rec,
                        'istart': sta,
                        'istop': sto,
                        'cstart': cstart,
                        'cstop': cstop,
                        'tstart': tstart,
                        'tstop': tstop,
                        'dur': tstop - tstart,
                        'total_n_valid_data_points': flc.flux.value.shape[0]
                        })
    
    flc.flares = pd.concat([flc.flares, newline.to_frame().T], ignore_index=True)

    return