"""Segmented-polynomial baseline.

The functions in this module fit segmented polynomial baselines and build
flare masks and noise estimates.
``polynomial_flare_mask`` returns the flare mask that seeds the GP in
``custom_detrending``.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from ...utils import MAD_TO_STD, robust_sigma, sigma_clipped_std

logger = logging.getLogger(__name__)

@dataclass
class DetrendConfig:
    """Settings used by the detrending pipeline.

    Attributes
    ----------
    window_sizes : tuple of float
        Window sizes, in days, used for segmented polynomial fits.
    poly_deg : int
        Polynomial degree used for each segment fit.
    n_edge : int
        Number of fitted points at each segment edge to up-weight.
    edge_weight : float
        Multiplicative weight applied to edge points.
    min_fit_pts : int
        Minimum number of cadences required for a segment fit.
    flare_mask_sigma : float
        Sigma threshold used for the preliminary flare mask.
    flare_mask_window_d : float
        Rough detrending window, in days, used for the preliminary flare mask.
    final_mask_sigma : float
        Sigma threshold of the final flare mask.  The final mask is the GP's
        starting flare mask, so it is kept to the brightest
        outliers (5 sigma): with 3 sigma, the maxima of fast, strong
        modulation were flagged as flares on low-noise stars, the GP never saw
        them and undershot there (62 vs. 29 false positives in 1000 synthetic
        light curves, recovery unchanged).  The matched filter masks the
        remaining flares on the GP's residual.
    final_flare_max_mask_fraction : float
        Maximum allowed fraction of cadences in the final flare mask.  If the
        candidate mask exceeds this fraction, the code falls back to the
        preliminary mask to prevent final-mask explosions.
    rolling_window_pts : int
        Rolling window size, in cadences, for local noise estimates.
    smooth_sigma_cadences : int
        Gaussian smoothing width, in cadences, for the combined trend.
    """
    window_sizes: tuple[float, ...] = (0.6, 0.9, 1.2)
    poly_deg: int = 4
    n_edge: int = 4
    edge_weight: float = 8.0
    min_fit_pts: int = 15
    flare_mask_sigma: float = 5.0
    flare_mask_window_d: float = 2.0
    final_mask_sigma: float = 5.0
    final_flare_max_mask_fraction: float = 0.05
    rolling_window_pts: int = 100
    smooth_sigma_cadences: int = 5

def compute_rolling_local_sigma(
    residuals: np.ndarray,
    window_pts: int = 100,
    flare_mask: np.ndarray | None = None,
    min_periods: int = 20,
) -> np.ndarray:
    """Estimate the local noise at each cadence with a rolling MAD.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residual flux values.
    window_pts : int, optional
        Rolling window length, in cadences.
    flare_mask : numpy.ndarray or None, optional
        Boolean mask of cadences to exclude from the rolling estimate.
    min_periods : int, optional
        Minimum number of values required in each rolling window.

    Returns
    -------
    numpy.ndarray
        Per-cadence local sigma estimate. Missing rolling estimates are filled with
        the nearest valid estimate or a sigma-clipped global fallback.
    """
    r = residuals.copy().astype(float)
    if flare_mask is not None:
        r[flare_mask] = np.nan
    s = pd.Series(r)
    rolling_med = s.rolling(window_pts, center=True, min_periods=min_periods).median()
    abs_dev = (s - rolling_med).abs()
    if flare_mask is not None:
        abs_dev.iloc[np.where(flare_mask)[0]] = np.nan
    rolling_mad = abs_dev.rolling(window_pts, center=True, min_periods=min_periods).median()
    sigma_local = (MAD_TO_STD * rolling_mad).to_numpy(dtype=float)
    sigma_local = pd.Series(sigma_local).ffill().bfill().to_numpy(dtype=float)
    finite = sigma_local[np.isfinite(sigma_local)]
    fallback = float(np.nanmedian(finite)) if len(finite) else sigma_clipped_std(residuals)
    return np.where(np.isfinite(sigma_local), sigma_local, fallback)

def preliminary_flare_mask(
    time, flux, flux_err,
    sigma_thresh=5, rough_window_days=2.0, poly_deg=3, rolling_pts=100,
) -> np.ndarray:
    """Build a preliminary flare mask with a single rough detrend.

    Parameters
    ----------
    time : array-like
        Time values, in days.
    flux : array-like
        Flux values.
    flux_err : array-like
        Flux-error values, used only to weight the fits.
    sigma_thresh : float, optional
        Positive-residual threshold in units of the local noise: the rolling
        MAD of the residual, at least its clipped standard deviation
        (``utils.sigma_clipped_std``, which includes the rough fit's misfit,
        so that modulation crests it misses are not flagged).  (Adding
        ``flux_err`` in quadrature counted
        the white noise twice and, with overestimated errors, raised the cut
        far above 5 sigma on quiet stars; removed 2026-10-15.)
    rough_window_days : float, optional
        Rough polynomial-fit segment length, in days.
    poly_deg : int, optional
        Polynomial degree used in each rough segment.
    rolling_pts : int, optional
        Rolling window length, in cadences, for the local sigma estimate.
    Returns
    -------
    numpy.ndarray
        Boolean array that is True for likely flare cadences.
    """
    seg_id   = np.floor((time - time.min()) / rough_window_days).astype(int)
    baseline = np.full(len(time), np.nan)

    for sid in np.unique(seg_id):
        idx = np.where(seg_id == sid)[0]
        if len(idx) < poly_deg + 2:
            continue
        t_loc = time[idx] - time[idx].mean()
        coeffs = np.polyfit(t_loc, flux[idx], deg=poly_deg, w=np.sqrt(1.0 / flux_err[idx]**2))
        baseline[idx] = np.polyval(coeffs, t_loc)

    residuals = flux - baseline
    finite    = np.isfinite(residuals)
    res_clean = np.where(finite, residuals, 0.0)

    sigma_local  = compute_rolling_local_sigma(res_clean, window_pts=rolling_pts)
    sc_floor     = sigma_clipped_std(res_clean[finite]) if finite.sum() > 10 else 0.0
    eff_sigma    = np.maximum(sigma_local, sc_floor)

    mask = (res_clean > sigma_thresh * eff_sigma) & finite
    logger.debug(f"Preliminary flare mask: {mask.sum():,} cadences flagged "
                 f"({100 * mask.sum() / len(time):.2f}% of {len(time):,})")
    return mask


def fit_window_grid(
    time, flux, flux_err, flare_mask,
    window_size, shift=0.0,
    poly_deg=3, n_edge=4, edge_weight=8.0, min_fit_pts=15,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a segmented polynomial baseline for one grid.

    Parameters
    ----------
    time : array-like
        Time values, in days.
    flux : array-like
        Flux values to fit.
    flux_err : array-like
        Positive flux-error values used as fit weights.
    flare_mask : array-like
        Boolean mask of cadences excluded from the preferred segment fit.
    window_size : float
        Segment width, in days.
    shift : float, optional
        Time offset for the window grid start, in days. Use ``0.0`` for normal
        windows and ``0.5 * window_size`` for shifted windows.
    poly_deg : int, optional
        Polynomial degree used in each segment.
    n_edge : int, optional
        Number of fitted points at each segment edge to up-weight.
    edge_weight : float, optional
        Multiplicative edge-point weight.
    min_fit_pts : int, optional
        Minimum number of unmasked cadences required before falling back to all
        points in the segment.

    Returns
    -------
    baseline : numpy.ndarray
        Fitted trend at every cadence. Cadences in unfitted segments are NaN.
    seg_id : numpy.ndarray
        Integer segment index at every cadence.
    metrics : dict
        Per-segment fit-quality statistics used by ``score_segment``.
    """
    # Build segment grid starting from time.min() + shift
    t_offset = time - (time.min() + shift)
    seg_id   = np.floor(t_offset / window_size).astype(int)

    baseline = np.full(len(time), np.nan)
    metrics  = {}

    for sid in np.unique(seg_id):
        idx = np.where(seg_id == sid)[0]
        if len(idx) < poly_deg + 2:
            continue

        t_seg, f_seg = time[idx], flux[idx]
        fe_seg = flux_err[idx]
        fm_seg = flare_mask[idx]

        t_local   = t_seg - t_seg.mean()
        n_total   = len(idx)
        frac_mask = fm_seg.sum() / n_total

        fit_idx = np.where(~fm_seg)[0]
        if len(fit_idx) < min_fit_pts:
            fit_idx = np.arange(n_total)   # fall back to all points

        t_fit  = t_local[fit_idx]
        f_fit  = f_seg[fit_idx]
        fe_fit = fe_seg[fit_idx]

        weights = 1.0 / fe_fit**2
        n_e = min(n_edge, len(t_fit) // 2)
        if n_e > 0:
            weights[:n_e]  *= edge_weight
            weights[-n_e:] *= edge_weight

        coeffs = np.polyfit(t_fit, f_fit, deg=poly_deg, w=np.sqrt(weights))

        baseline[idx] = np.polyval(coeffs, t_local)

        # Fit-quality statistics on non-flare points
        r_q  = f_fit - np.polyval(coeffs, t_fit)
        n_q  = len(r_q)
        k    = poly_deg + 1
        dof  = max(n_q - k, 1)

        ss_res = float(np.sum(r_q**2))
        ss_tot = float(np.sum((f_fit - f_fit.mean())**2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else np.nan
        r2_adj = (1.0 - (1.0 - r2) * (n_q - 1) / dof) if np.isfinite(r2) and dof > 0 else np.nan

        red_chi2 = float(np.sum((r_q / fe_fit)**2)) / dof

        metrics[sid] = {
            "r2_adj":      r2_adj,
            "red_chi2":    red_chi2,
            "n_fit":       len(fit_idx),
            "frac_masked": frac_mask,
        }

    return baseline, seg_id, metrics


def run_all_windows(time, flux, flux_err, flare_mask, window_sizes,
                    poly_deg=3, n_edge=4, edge_weight=8.0, min_fit_pts=15):
    """Fit all window-size and grid-shift combinations.

    For each window size, two grids are evaluated: a normal grid beginning at
    ``time.min()`` and a shifted grid beginning at ``time.min() + 0.5 * window``.

    Parameters
    ----------
    time : array-like
        Time values, in days.
    flux : array-like
        Flux values to fit.
    flux_err : array-like
        Positive flux-error values used as fit weights.
    flare_mask : array-like
        Boolean mask of cadences excluded from preferred segment fits.
    window_sizes : iterable of float
        Segment widths, in days.
    poly_deg : int, optional
        Polynomial degree used in each segment.
    n_edge : int, optional
        Number of fitted points at each segment edge to up-weight.
    edge_weight : float, optional
        Multiplicative edge-point weight.
    min_fit_pts : int, optional
        Minimum number of unmasked cadences required before fallback.

    Returns
    -------
    dict
        Dictionary keyed by ``(window_size, shift)`` with baselines, segment ids
        and metrics.
    """
    results = {}
    for w in window_sizes:
        for shift_frac in (0.0, 0.5):
            shift = shift_frac * w
            bl, sids, mets = fit_window_grid(
                time, flux, flux_err, flare_mask,
                window_size=w, shift=shift,
                poly_deg=poly_deg, n_edge=n_edge,
                edge_weight=edge_weight, min_fit_pts=min_fit_pts,
            )
            results[(w, shift)] = {"baseline": bl, "seg_ids": sids, "metrics": mets}

    return results


def score_segment(
    m: dict,
    min_fit_pts: int = 15,
) -> float:
    """Compute a scalar quality score for one fitted segment.

    The score combines adjusted R², reduced chi-square distance from one,
    fraction of masked flare points, and the number of fitted points.

    Parameters
    ----------
    m : dict
        Segment metric dictionary.
    min_fit_pts : int, optional
        Minimum number of fitted points required for a finite score.

    Returns
    -------
    float
        Segment score. Returns ``-numpy.inf`` for disqualified segments.
    """
    r2_adj, red_chi2 = m["r2_adj"], m["red_chi2"]
    n_fit, frac_mask = m["n_fit"], m["frac_masked"]

    # Hard disqualifiers
    if not np.isfinite(r2_adj):
        return -np.inf
    if n_fit < min_fit_pts:
        return -np.inf
    if frac_mask > 0.85:
        return -np.inf

    rc = np.clip(red_chi2, 1e-6, 1e6) if np.isfinite(red_chi2) else 1e6

    score = (
        r2_adj                             # [0, 1]: higher is better
        - 0.15 * abs(np.log(rc))           # penalise chi² far from 1
        - 0.30 * frac_mask                 # penalise high flare fraction
        + 0.05 * np.log1p(n_fit / 50.0)    # small bonus for more data
    )

    return float(score)


def assemble_combined_trend(
    time: np.ndarray,
    all_results: dict,
    smooth_sigma_cadences: int = 5,
    min_fit_pts: int = 15,
) -> np.ndarray:
    """Build a blended per-cadence baseline from all window grids.

    At each cadence, valid segment fits are weighted by ``exp(score)`` using scores
    from ``score_segment``. This softmax-style weighting smooths transitions between
    competing fits.

    Parameters
    ----------
    time : numpy.ndarray
        Time values corresponding to the fitted baselines.
    all_results : dict
        Output from ``run_all_windows``.
    smooth_sigma_cadences : int, optional
        Gaussian smoothing width, in cadences. Set to zero to skip smoothing.
    min_fit_pts : int, optional
        Minimum number of fitted points required for segment scoring.

    Returns
    -------
    numpy.ndarray
        Smoothed blended baseline.

    Raises
    ------
    RuntimeError
        If no finite trend can be built from the fitted window grids.
    """
    n         = len(time)
    keys      = list(all_results.keys())
    n_combos  = len(keys)

    bl_stack = np.full((n_combos, n), np.nan)
    sc_stack = np.full((n_combos, n), -np.inf)

    for ki, key in enumerate(keys):
        res  = all_results[key]
        bl   = res["baseline"]
        sids = res["seg_ids"]
        mets = res["metrics"]

        # Precompute segment scores
        seg_score_map = {sid: score_segment(m, min_fit_pts) for sid, m in mets.items()}

        bl_stack[ki] = bl
        for i in range(n):
            sid = int(sids[i])
            sc  = seg_score_map.get(sid, -np.inf)
            sc_stack[ki, i] = sc if np.isfinite(bl[i]) else -np.inf

    # Softmax-style weights: shift by per-cadence max before exponentiation
    valid  = np.isfinite(bl_stack) & (sc_stack > -1e5)
    sc_max = np.where(valid, sc_stack, -np.inf).max(axis=0, keepdims=True)
    sc_sh  = np.where(valid, sc_stack - sc_max, -np.inf)
    weights = np.where(sc_sh > -50, np.exp(sc_sh), 0.0)
    weights = np.where(valid, weights, 0.0)

    w_sum = weights.sum(axis=0)
    trend = np.where(w_sum > 0,
                     np.nansum(bl_stack * weights, axis=0) / w_sum,
                     np.nan)

    if not np.isfinite(trend).any():
        raise RuntimeError("No valid trend could be built from the fitted window grids.")

    # Fill residual NaNs by linear interpolation
    fin = np.isfinite(trend)
    if fin.sum() > 2 and (~fin).any():
        trend = np.interp(np.arange(n), np.where(fin)[0], trend[fin])

    # Light Gaussian smoothing to suppress window-boundary ringing
    if smooth_sigma_cadences > 0 and np.all(np.isfinite(trend)):
        trend = gaussian_filter1d(trend.astype(float), sigma=smooth_sigma_cadences)

    return trend



def build_safe_final_flare_mask(
    residuals: np.ndarray,
    sigma_thresh: float,
    previous_mask: np.ndarray,
    max_mask_fraction: float = 0.05,
) -> np.ndarray:
    """Mask the residual's positive outliers, with a guard against mask explosion.

    A cadence is masked if its residual exceeds the median by more than
    ``sigma_thresh`` robust sigmas.  Median and sigma are estimated from the
    cadences outside ``previous_mask`` (from all cadences if fewer than 20
    remain).  If more than ``max_mask_fraction`` of the cadences would be
    masked, ``previous_mask`` is returned instead: flares should not be almost
    the whole light curve.

    Parameters
    ----------
    residuals : numpy.ndarray
        Final residual flux values.
    sigma_thresh : float
        Positive-outlier threshold in robust sigma units.
    previous_mask : numpy.ndarray
        Previous flare mask: the fallback, and the cadences left out of the
        noise estimate.
    max_mask_fraction : float, optional
        Maximum fraction of cadences in the new mask.

    Returns
    -------
    numpy.ndarray
        Boolean flare mask.
    """
    q = ~previous_mask
    if q.sum() < 20:
        q = np.ones(len(residuals), dtype=bool)
    center = np.nanmedian(residuals[q])
    sigma = robust_sigma(residuals[q])
    candidate = residuals > center + sigma_thresh * sigma
    if candidate.mean() > max_mask_fraction:
        return previous_mask.copy()
    return candidate


def run_detrending(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    config: DetrendConfig | None = None,
) -> np.ndarray:
    """Fit the segmented-polynomial baseline and mask the residual's flares.

    A preliminary flare mask from a rough fit, then the blended fit of all
    window grids without it, then the final flare mask on that fit's residual.
    (A second fit with a mask from the first one changed nothing in
    injection-recovery and was removed on 2026-10-15.)

    Parameters
    ----------
    time : numpy.ndarray
        Time values, in days.
    flux : numpy.ndarray
        Flux values.
    flux_err : numpy.ndarray
        Positive finite flux-error values.
    config : DetrendConfig or None, optional
        Pipeline configuration. If None, the default ``DetrendConfig`` is used.

    Returns
    -------
    numpy.ndarray
        Boolean mask of the residual's positive outliers.

    Raises
    ------
    RuntimeError
        If no valid combined trend can be built.
    """
    cfg = config or DetrendConfig()

    flare_mask_prelim = preliminary_flare_mask(
        time, flux, flux_err,
        sigma_thresh=cfg.flare_mask_sigma,
        rough_window_days=cfg.flare_mask_window_d,
        poly_deg=cfg.poly_deg,
        rolling_pts=cfg.rolling_window_pts,
    )

    all_results = run_all_windows(
        time, flux, flux_err, flare_mask_prelim, cfg.window_sizes,
        poly_deg=cfg.poly_deg, n_edge=cfg.n_edge, edge_weight=cfg.edge_weight,
        min_fit_pts=cfg.min_fit_pts,
    )
    trend = assemble_combined_trend(
        time, all_results, smooth_sigma_cadences=cfg.smooth_sigma_cadences,
        min_fit_pts=cfg.min_fit_pts,
    )
    final_residual = flux - trend

    # Mask the residual's positive outliers (above its own median). This
    # releases points the rough preliminary fit flagged only because it
    # missed broad crests of the variability, while sharp positive outliers
    # remain. The helper guards against final-mask explosions that would
    # mark nearly the full light curve.
    return build_safe_final_flare_mask(
        final_residual,
        sigma_thresh=cfg.final_mask_sigma,
        previous_mask=flare_mask_prelim,
        max_mask_fraction=cfg.final_flare_max_mask_fraction,
    )


def polynomial_flare_mask(time, flux, flux_err, config=None):
    """Flare mask from the segmented-polynomial baseline: the GP's initial
    flare mask in ``custom_detrending``.

    Runs :func:`run_detrending` on the finite cadences and returns its flare
    mask on the full grid.

    Notes
    -----
    * ``run_detrending`` requires finite inputs, so the fit runs on the
      finite (non-NaN) subset of ``time``/``flux``/``flux_err`` and the results
      are mapped back onto the full grid (NaN elsewhere).
    * ``flux_err`` is forced strictly positive and finite (non-finite or
      non-positive values are replaced by the median positive error).

    Parameters
    ----------
    time, flux, flux_err : ndarray
        Full-grid arrays in days / flux units (may contain NaNs).
    config : DetrendConfig or None
        Optional configuration forwarded to ``run_detrending``.

    Returns
    -------
    ndarray
        Boolean flare mask on the full grid (False where input was NaN).
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    finite = np.isfinite(time) & np.isfinite(flux)
    if finite.sum() < 20:
        raise ValueError("Too few finite cadences for the polynomial baseline.")

    # Force strictly positive, finite errors on the finite subset.
    err = flux_err[finite]
    good_err = np.isfinite(err) & (err > 0)
    fill = np.median(err[good_err]) if good_err.any() else 1.0
    err = np.where(good_err, err, fill)

    final_mask = np.zeros(len(flux), dtype=bool)
    final_mask[finite] = run_detrending(time[finite], flux[finite], err, config)
    return final_mask
