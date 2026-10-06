"""Spline baseline through robust knots, chosen from a grid of candidates."""

import numpy as np

from scipy.interpolate import UnivariateSpline

from ...altai import _find_iterative_median
from ...utils import MAD_TO_STD


def fit_spline(
    time,
    flux,
    gaps,
    coarseness_range=(5, 15, 1),
    spline_orders=(2, 3),
    n_phase_shifts=3,
    percentile_anchor=25,
    edge_penalty_weight=1.0,
    smoothing=0.0,
    **kwargs,
):
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
    smoothing : float
        Optional smoothing strength for the weighted spline fit, in units of
        the per-knot scatter: the spline may miss the knots by roughly
        ``sqrt(smoothing)`` standard deviations.  ``smoothing=0`` (the default)
        interpolates the knots exactly; the fit is already stabilised by the
        robust endpoint anchoring and knot sanitising in ``_build_knot_points``
        regardless of this value.  Raise it (e.g. 0.5-2) only if a particular
        light curve still shows knot-to-knot overshoot -- note that larger
        values trade overshoot suppression for a mild bias against real
        short-timescale variability.  Defaults to 0.0.
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
        coarseness_range[0], coarseness_range[1] + 1, coarseness_range[2]
    )

    dt = np.nanmin(np.diff(time))

    # Try every candidate and keep the best-scoring one (the first on ties).
    best = None
    for coarseness in coarseness_values:
        for k in spline_orders:
            for phase_idx in range(n_phase_shifts):
                model, newflux = _fit_single_spline(
                    time,
                    flux,
                    flux_med,
                    gaps,
                    coarseness,
                    k,
                    dt,
                    phase_idx,
                    n_phase_shifts,
                    percentile_anchor,
                    smoothing,
                )
                score = _evaluate_spline_fit(
                    flux, model, gaps, edge_penalty_weight=edge_penalty_weight
                )
                if best is None or score < best["score"]:
                    best = {
                        "model": model,
                        "newflux": newflux,
                        "score": score,
                        "coarseness": coarseness,
                        "order": k,
                        "phase": phase_idx,
                    }

    best_params = {
        "coarseness": best["coarseness"],
        "order": best["order"],
        "phase": best["phase"],
        "score": best["score"],
        "smoothing": smoothing,
    }

    return best["newflux"], best["model"], best_params


def _fit_single_spline(
    time,
    flux,
    flux_med,
    gaps,
    coarseness,
    k,
    dt,
    phase_idx,
    n_phases,
    percentile,
    smoothing,
):
    """Fit a single spline configuration.

    Uses a *weighted smoothing* spline through robust knots.  Each knot is
    weighted by the inverse of its intra-bin scatter and the total smoothing
    budget is ``smoothing x n_knots``.  With ``smoothing=0`` the (robust) knots
    are interpolated exactly; a positive value lets the spline approximate them
    to roughly their noise level, which suppresses knot-to-knot overshoot.
    Stability at the segment edges comes primarily from the robust endpoint
    anchoring in ``_build_knot_points`` and applies at any smoothing level.
    """
    n = int(np.rint(coarseness / 24 / dt))

    model = np.full_like(flux, np.nan)
    newflux = np.full_like(flux, np.nan)

    for le, ri in gaps:
        segment_len = ri - le
        t_s, f_s = time[le:ri], flux[le:ri]

        # Calculate phase offset for this segment
        phase_offset = min((phase_idx * n) // max(n_phases, 1), segment_len - 1)

        if segment_len <= n:
            # Segment too short for binning
            newflux[le:ri] = f_s
            model[le:ri] = np.nanmedian(f_s)
            continue

        # Build knot points (with per-knot scatter) using robust statistics
        t_knots, f_knots, s_knots = _build_knot_points(
            t_s, f_s, n, phase_offset, percentile
        )

        # Weighted smoothing spline.  Weights are 1/scatter (the convention
        # expected by UnivariateSpline) and the smoothing budget
        # s = smoothing × n_knots targets a reduced χ² ≈ smoothing.
        seg_model = None
        if len(t_knots) > k:
            try:
                spline = UnivariateSpline(
                    t_knots, f_knots, w=1.0 / s_knots, k=k, s=smoothing * len(t_knots)
                )
                seg_model = spline(t_s)
            except Exception:
                pass

        if seg_model is None:
            # Too few knots, or the spline failed: linear fit
            valid = ~np.isnan(f_s)
            if valid.sum() < 2:
                newflux[le:ri] = f_s
                model[le:ri] = flux_med[le:ri]
                continue
            seg_model = np.polyval(np.polyfit(t_s[valid], f_s[valid], 1), t_s)

        model[le:ri] = seg_model
        newflux[le:ri] = f_s - seg_model + flux_med[le:ri]

    return model, newflux


def _sanitize_knots(t_knots, f_knots, s_knots):
    """Clean knot arrays so they are safe to hand to ``UnivariateSpline``.

    Drops NaN knots, sorts by time, enforces *strictly increasing* knot times
    (duplicate or out-of-order times would otherwise raise and force a
    whole-segment linear fallback), and floors the per-knot scatter to a small
    positive value so the fit weights (1/scatter) stay finite.

    Returns
    -------
    t_knots, f_knots, s_knots : ndarray
        Cleaned, strictly time-ordered knot times, fluxes, and scatters.
    """
    valid = ~(np.isnan(t_knots) | np.isnan(f_knots))
    t_knots, f_knots, s_knots = t_knots[valid], f_knots[valid], s_knots[valid]

    if len(t_knots) == 0:
        return t_knots, f_knots, s_knots

    order = np.argsort(t_knots, kind="stable")
    t_knots, f_knots, s_knots = t_knots[order], f_knots[order], s_knots[order]

    # Keep only strictly increasing times (drop duplicates / reversals).
    keep = np.concatenate([[True], np.diff(t_knots) > 0])
    t_knots, f_knots, s_knots = t_knots[keep], f_knots[keep], s_knots[keep]

    # Floor the scatter to a small fraction of the knot spread so that flat
    # bins (scatter 0 or NaN) do not produce infinite weights.
    spread = np.nanmax(f_knots) - np.nanmin(f_knots) if len(f_knots) else 0.0
    floor = 1e-3 * (spread + 1e-30)
    s_knots = np.where(np.isfinite(s_knots) & (s_knots > floor), s_knots, floor)

    return t_knots, f_knots, s_knots


def _build_knot_points(time, flux, n, phase_offset, percentile):
    """Build knot points (with per-knot scatter) for spline fitting using
    robust statistics.

    Interior knots use the bin-mean time and a low percentile of the bin flux
    (flare-resistant).  Segment endpoints are anchored with the *same robust
    percentile* over the first/last bin rather than a single raw sample, so a
    noisy or flaring edge cadence cannot drag the baseline and cause the
    edge-swing that destabilises the fit.  The per-knot scatter (intra-bin
    standard deviation) is returned so the spline can be weighted.

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

    Returns
    -------
    t_knots, f_knots, s_knots : ndarray
        Strictly increasing knot times, robust knot fluxes, and per-knot
        scatter estimates.
    """
    segment_len = len(time)

    # Apply phase offset
    start = phase_offset
    usable_len = segment_len - start
    n_bins = usable_len // n

    if n_bins == 0:
        # Too short to bin: anchor both endpoints on a robust estimate over
        # the whole (short) segment rather than raw first/last samples.
        f_edge = np.nanpercentile(flux, percentile)
        s_edge = np.nanstd(flux)
        t_knots = np.array([time[0], time[-1]])
        f_knots = np.array([f_edge, f_edge])
        s_knots = np.array([s_edge, s_edge])
        return _sanitize_knots(t_knots, f_knots, s_knots)

    remainder = usable_len % n
    end_idx = segment_len - remainder if remainder > 0 else segment_len

    # Reshape into bins
    t_binned = time[start:end_idx].reshape(n_bins, n)
    f_binned = flux[start:end_idx].reshape(n_bins, n)

    # Use mean for time, robust percentile for flux (avoids flare bias),
    # intra-bin std for the per-knot scatter used to weight the fit.
    t_knots = np.nanmean(t_binned, axis=1)
    f_knots = np.nanpercentile(f_binned, percentile, axis=1)
    s_knots = np.nanstd(f_binned, axis=1)

    # Robust boundary anchors: percentile over the first/last bin placed at
    # the true segment edges, so the spline spans the whole segment without
    # being pinned to a single (possibly flaring) raw edge sample.
    left_f = np.nanpercentile(f_binned[0], percentile)
    right_f = np.nanpercentile(f_binned[-1], percentile)
    left_s = np.nanstd(f_binned[0])
    right_s = np.nanstd(f_binned[-1])

    t_knots = np.concatenate([[time[0]], t_knots, [time[-1]]])
    f_knots = np.concatenate([[left_f], f_knots, [right_f]])
    s_knots = np.concatenate([[left_s], s_knots, [right_s]])

    return _sanitize_knots(t_knots, f_knots, s_knots)


def _evaluate_spline_fit(flux, model, gaps, edge_fraction=0.1, edge_penalty_weight=0.5):
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

        residuals.append(flux[le:ri][valid] - model[le:ri][valid])

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

    residuals = np.concatenate(residuals) if residuals else np.array([])

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

    # Ratio close to MAD_TO_STD indicates a Gaussian-like distribution
    # (for a Gaussian: std/MAD ≈ MAD_TO_STD)
    gaussian_ratio = MAD_TO_STD
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
