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




def custom_detrending(lc, 
                      savgol1=6., savgol2=3., pad=3, max_sigma=2.5, 
                      longdecay=6, maxgap=10, debug_plot=False,
                      break_tolerance=10):
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

        
    Return:
    -------
    FlareLightCurve with detrended_flux attribute
    """
    dt = np.mean(np.diff(lc.time.value))
    gaps = lc.find_gaps(maxgap=maxgap * dt).gaps

    lc = lc.interpolate_missing_cadences()

    time, flux = lc.time.value, lc.flux.value
    
    # Store original flux as a column so it survives filtering operations
    lc["original_flux"] = lc.flux.copy()
    lc["orginal_flux_err"] = lc.flux_err.copy()

    # fit a spline to the general trends
    m2flux, _, best_params = fit_spline(time, flux, gaps, longdecay=longdecay)

    print("Spline detrending params:", best_params)
    
    # choose a 6 hour window
    w1 = int((np.rint(savgol1 / 24. / dt) // 2) * 2 + 1)

    lc.flux = m2flux * u.electron / u.s
    lc.flux_err = lc.flux_err * u.electron / u.s

    if debug_plot == True:
        plt.figure(figsize=(8,4))
        plt.plot(lc.time.value, lc.flux.value + 5000, 'k.', markersize=1,
                 label="after spline fit")

    # use Savitzy-Golay to iron out the rest    
    lc3 = lc.detrend("savgol", w=w1, pad=pad,
                      max_sigma=max_sigma, longdecay=longdecay,
                      break_tolerance=break_tolerance)
    
    lc3.flux = lc3.detrended_flux * u.electron / u.s
 
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
    lc.flux_err = lc["orginal_flux_err"] * u.electron / u.s
    
    
    # find median value
    lc4.find_iterative_median()

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