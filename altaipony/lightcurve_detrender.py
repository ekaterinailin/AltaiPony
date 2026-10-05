"""Light-curve detrending utilities.

The functions in this module fit segmented polynomial baselines, optionally
remove sinusoidal residuals, and build flare masks and noise estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .utils import MAD_TO_STD, robust_sigma

try:
    from .periodogram import lomb_scargle
    _ASTROPY_LS = True
except ImportError:
    _ASTROPY_LS = False

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
    second_pass_sigma : float
        Sigma threshold used for second-pass and final flare masks.
    final_flare_max_mask_fraction : float
        Maximum allowed fraction of cadences in the recomputed final flare mask.
        If the candidate mask exceeds this fraction, the code falls back to the
        previous/provisional mask to prevent final-mask explosions.
    rolling_window_pts : int
        Rolling window size, in cadences, for local noise estimates.
    smooth_sigma_cadences : int
        Gaussian smoothing width, in cadences, for the combined trend.
    center_final_residual : bool
        If True, subtract the moving-average median from the final residual.
    centering_ma_window : int
        Moving-average window used for residual centering diagnostics.
    centering_clip_abs : float or None
        Optional absolute clipping limit applied when estimating the centering offset.
        The default is None, so no pre-centering clipping is performed.
    """
    window_sizes: tuple[float, ...] = (0.4, 0.6, 0.8)
    poly_deg: int = 4
    n_edge: int = 4
    edge_weight: float = 8.0
    min_fit_pts: int = 15
    flare_mask_sigma: float = 5.0
    flare_mask_window_d: float = 2.0
    second_pass_sigma: float = 3.0
    final_flare_max_mask_fraction: float = 0.05
    rolling_window_pts: int = 100
    smooth_sigma_cadences: int = 5
    center_final_residual: bool = True
    centering_ma_window: int = 10
    centering_clip_abs: float | None = None

    # Optional long-period correction applied to the second-pass residuals,
    # removing rotation-like residual modulation.
    apply_rotation_sinusoid_correction: bool = True
    rotation_sinusoid_min_period_hr: float = 6.0
    rotation_sinusoid_max_period_hr: float = 36.0
    rotation_sinusoid_n_components: int = 3
    rotation_sinusoid_min_improvement: float = 0.02
    rotation_amp_limit_fraction: float = 3.0

@dataclass
class DetrendResult:
    """Outputs produced by the detrending pipeline.

    Attributes
    ----------
    final_df : pandas.DataFrame
        Time-sorted table containing flux, trends, residuals, masks, and diagnostics.
    seg_stats_p1 : pandas.DataFrame
        First-pass per-segment fit-quality table.
    seg_stats_p2 : pandas.DataFrame
        Second-pass per-segment fit-quality table.
    summary : dict
        Scalar run summary and diagnostic values.
    """
    final_df: pd.DataFrame
    seg_stats_p1: pd.DataFrame
    seg_stats_p2: pd.DataFrame
    summary: dict

def sigma_clipped_std(arr: np.ndarray, n_sigma: float = 3.0, max_iter: int = 20) -> float:
    """Compute an iterative sigma-clipped standard deviation.

    Parameters
    ----------
    arr : numpy.ndarray
        Array containing values to summarize. Non-finite values are ignored.
    n_sigma : float, optional
        Sigma threshold used at each clipping iteration.
    max_iter : int, optional
        Maximum number of clipping iterations.

    Returns
    -------
    float
        Standard deviation after clipping. Returns ``numpy.nan`` if fewer than two
        finite points are available.
    """
    r = arr[np.isfinite(arr)].copy()
    if len(r) < 2:
        return np.nan
    for _ in range(max_iter):
        med, std = np.median(r), np.std(r)
        if not np.isfinite(std) or std == 0:
            break
        keep = np.abs(r - med) <= n_sigma * std
        if keep.sum() == len(r) or keep.sum() < 2:
            break
        r = r[keep]
    return float(np.std(r))

def _weighted_lstsq(X: np.ndarray, y: np.ndarray, flux_err=None) -> np.ndarray:
    """Solve a linear least-squares problem, weighted by ``1 / flux_err**2``.

    Falls back to an unweighted fit when ``flux_err`` is None or contains
    non-finite or non-positive values.
    """
    if flux_err is not None:
        fe = np.asarray(flux_err, dtype=float)
        if np.all(np.isfinite(fe)) and np.all(fe > 0):
            w = 1.0 / fe**2
            return np.linalg.lstsq((X.T * w) @ X, (X.T * w) @ y, rcond=None)[0]
    return np.linalg.lstsq(X, y, rcond=None)[0]

def _empty_sinusoid_stats(**extra) -> dict:
    """Return the stats dict of a periodic correction that was not applied."""
    stats = {
        "applied": False,
        "n_components": 0,
        "periods_hr": [],
        "improvement": 0.0,
        "std_before": np.nan,
        "std_after": np.nan,
        "boundary_hit": False,
        "reject_reason": "",
    }
    stats.update(extra)
    return stats

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

def multi_sinusoid_correction(
    times,
    residuals,
    flux_err=None,
    flare_mask=None,
    sigma_local=None,
    n_components=3,
    min_period_hr=6.0,
    max_period_hr=36.0,
    amp_limit_frac=3.0,
    min_improvement=0.02,
    label="Sinusoidal correction",
):
    """Remove dominant sinusoidal components from residuals.

    Components are accepted greedily and only kept when they produce a minimum
    additional scatter improvement. This prevents the function from always
    removing exactly ``n_components`` weak or alias-like sinusoids.

    Returns
    -------
    corrected : numpy.ndarray
        Residuals minus the accepted sinusoids (the input if none is accepted).
    stats : dict
        See ``_empty_sinusoid_stats``; ``reject_reason`` says why nothing was
        applied.
    """
    times = np.asarray(times, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    def reject(reason, **extra):
        return residuals, _empty_sinusoid_stats(reject_reason=reason, **extra)

    if flare_mask is None:
        flare_mask = np.zeros(len(times), dtype=bool)
    q_mask = ~np.asarray(flare_mask, dtype=bool) & np.isfinite(times) & np.isfinite(residuals)
    t_q, r_q = times[q_mask], residuals[q_mask]
    if len(t_q) < 30:
        return reject("too few points")

    if sigma_local is not None:
        sigma_arr = np.asarray(sigma_local, dtype=float)
        sigma_med = float(np.nanmedian(sigma_arr[np.isfinite(sigma_arr)]))
    else:
        sigma_med = robust_sigma(r_q)
    if not np.isfinite(sigma_med) or sigma_med <= 0:
        sigma_med = float(np.nanstd(r_q))
    amp_cap = amp_limit_frac * sigma_med if np.isfinite(sigma_med) else np.inf

    try:
        _, freqs, power = lomb_scargle(
            t_q, r_q, min_frequency=24.0 / max_period_hr, max_frequency=24.0 / min_period_hr
        )
    except Exception as exc:
        warnings.warn(f"{label} skipped: Lomb-Scargle fitting failed ({exc}).", RuntimeWarning)
        return reject("Lomb-Scargle failed")
    if len(freqs) == 0:
        return reject("period range invalid for duration")

    std_before = float(np.nanstd(residuals[q_mask]))
    if not np.isfinite(std_before) or std_before <= 0:
        return reject("invalid initial scatter")

    fe_q = None if flux_err is None else np.asarray(flux_err, dtype=float)[q_mask]

    def corrected_for(selected_freqs):
        """Fit offset + sinusoids at ``selected_freqs`` and subtract them, with
        each amplitude capped at ``amp_cap``. Returns None if the fit fails."""
        cols = [np.ones(len(t_q))]
        for f in selected_freqs:
            cols += [np.sin(2 * np.pi * f * t_q), np.cos(2 * np.pi * f * t_q)]
        try:
            coeffs = _weighted_lstsq(np.column_stack(cols), r_q, fe_q)
        except np.linalg.LinAlgError:
            return None
        model = np.full(len(times), float(coeffs[0]))
        for k, f in enumerate(selected_freqs):
            A, B = float(coeffs[1 + 2*k]), float(coeffs[2 + 2*k])
            amp = np.sqrt(A**2 + B**2)
            if amp > amp_cap > 0:
                A *= amp_cap / amp
                B *= amp_cap / amp
            model += A * np.sin(2 * np.pi * f * times) + B * np.cos(2 * np.pi * f * times)
        return residuals - model

    selected = []
    best_corrected = residuals.copy()
    best_std = std_before
    max_candidates = min(len(freqs), max(50, 25 * int(max(n_components, 1))))

    for idx in np.argsort(power)[::-1][:max_candidates]:
        f = float(freqs[idx])
        # Avoid selecting nearly duplicate frequencies.
        if any(abs(f - sf) < 0.10 * max(f, sf) for sf in selected):
            continue
        if len(selected) >= n_components:
            break
        trial_corrected = corrected_for(selected + [f])
        if trial_corrected is None:
            continue
        trial_std = float(np.nanstd(trial_corrected[q_mask]))
        if not np.isfinite(trial_std):
            continue
        # Each component must lower the scatter by at least min_improvement
        # (relative to the initial scatter), so weak or alias-like peaks are
        # not removed just because n_components allows it.
        if (best_std - trial_std) / std_before >= min_improvement and trial_std < best_std:
            selected.append(f)
            best_corrected, best_std = trial_corrected, trial_std

    improvement = (std_before - best_std) / std_before
    if not selected or improvement < min_improvement:
        logger.debug(f"  {label} discarded (improvement too small: {100 * max(improvement, 0.0):.2f}%).")
        return reject("improvement too small", std_before=std_before, std_after=best_std,
                      improvement=float(max(improvement, 0.0)))

    periods = [24.0 / f for f in selected]
    boundary_hit = any(p <= min_period_hr * 1.02 or p >= max_period_hr * 0.98 for p in periods)
    logger.debug(
        f"  {label}: removed {len(selected)} component(s) at {[round(p, 3) for p in periods]} hr; "
        f"scatter improvement {100 * improvement:.2f}%{' [boundary hit]' if boundary_hit else ''}"
    )
    return best_corrected, _empty_sinusoid_stats(
        applied=True,
        n_components=len(selected),
        periods_hr=[float(p) for p in periods],
        improvement=float(improvement),
        std_before=float(std_before),
        std_after=float(best_std),
        boundary_hit=bool(boundary_hit),
    )


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
        Flux-error values.
    sigma_thresh : float, optional
        Positive-residual threshold in units of effective sigma.
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
        w_    = 1.0 / np.maximum(flux_err[idx], 1e-15)**2
        try:
            coeffs = np.polyfit(t_loc, flux[idx], deg=poly_deg, w=np.sqrt(w_))
            baseline[idx] = np.polyval(coeffs, t_loc)
        except np.linalg.LinAlgError:
            continue

    residuals = flux - baseline
    finite    = np.isfinite(residuals)
    res_clean = np.where(finite, residuals, 0.0)

    sigma_local  = compute_rolling_local_sigma(res_clean, window_pts=rolling_pts)
    sigma_total  = np.sqrt(sigma_local**2 + flux_err**2)
    sc_floor     = sigma_clipped_std(res_clean[finite]) if finite.sum() > 10 else 0.0
    eff_sigma    = np.maximum(sigma_total, sc_floor)

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
        Per-segment fit-quality statistics.
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
        fe_seg = np.maximum(flux_err[idx], 1e-15)
        fm_seg = flare_mask[idx]

        t_local   = t_seg - t_seg.mean()
        n_total   = len(idx)
        n_masked  = int(fm_seg.sum())
        frac_mask = n_masked / n_total

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

        try:
            coeffs = np.polyfit(t_fit, f_fit, deg=poly_deg, w=np.sqrt(weights))
        except (np.linalg.LinAlgError, ValueError):
            continue

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

        chi2     = float(np.sum((r_q / fe_fit)**2))
        red_chi2 = chi2 / dof
        rsd      = float(np.std(r_q))

        rss = max(ss_res, 1e-300)
        aic = n_q * np.log(rss / max(n_q, 1)) + 2.0 * k
        bic = n_q * np.log(rss / max(n_q, 1)) + k * np.log(max(n_q, 1))

        metrics[sid] = {
            "r2_adj":     r2_adj,
            "red_chi2":   red_chi2,
            "rsd":        rsd,
            "aic":        float(aic),
            "bic":        float(bic),
            "n_fit":      len(fit_idx),
            "n_total":    n_total,
            "n_masked":   n_masked,
            "frac_masked": frac_mask,
            "seg_start":  float(t_seg.min()),
            "seg_end":    float(t_seg.max()),
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
        Dictionary keyed by ``(window_size, shift)`` with baselines, segment ids,
        metrics, and grid labels.
    """
    results = {}
    for w in window_sizes:
        for shift_frac, label in [(0.0, "normal"), (0.5, "shifted")]:
            shift = shift_frac * w
            bl, sids, mets = fit_window_grid(
                time, flux, flux_err, flare_mask,
                window_size=w, shift=shift,
                poly_deg=poly_deg, n_edge=n_edge,
                edge_weight=edge_weight, min_fit_pts=min_fit_pts,
            )
            results[(w, shift)] = {
                "baseline":    bl,
                "seg_ids":     sids,
                "metrics":     mets,
                "window_size": w,
                "shift":       shift,
                "label":       label,
            }

            if not mets:
                warnings.warn(
                    f"No valid segments were fitted for window {w:.2f} d ({label} grid).",
                    RuntimeWarning,
                )

    return results


def build_segment_stats_df(all_results: dict) -> pd.DataFrame:
    """Flatten per-segment metrics into one table.

    Parameters
    ----------
    all_results : dict
        Output from ``run_all_windows``.

    Returns
    -------
    pandas.DataFrame
        Table with one row per fitted segment and window grid.
    """
    rows = []
    for (w, shift), res in all_results.items():
        for sid, m in res["metrics"].items():
            rows.append({
                "window_size":  w,
                "shift":        shift,
                "window_label": res["label"],
                "seg_id":       sid,
                "seg_start":    m["seg_start"],
                "seg_end":      m["seg_end"],
                "r2_adj":       m["r2_adj"],
                "red_chi2":     m["red_chi2"],
                "rsd":          m["rsd"],
                "aic":          m["aic"],
                "bic":          m["bic"],
                "n_fit":        m["n_fit"],
                "n_total":      m["n_total"],
                "n_masked":     m["n_masked"],
                "frac_masked":  m["frac_masked"],
            })
    return pd.DataFrame(rows)


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
    r2_adj     = m.get("r2_adj",     np.nan)
    red_chi2   = m.get("red_chi2",   np.nan)
    n_fit      = m.get("n_fit",      0)
    frac_mask  = m.get("frac_masked", 1.0)

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
) -> tuple:
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
    trend : numpy.ndarray
        Smoothed blended baseline.
    best_window_size : numpy.ndarray
        Window size of the highest-scoring candidate per cadence.
    best_shift : numpy.ndarray
        Grid shift of the highest-scoring candidate per cadence.
    best_seg_id : numpy.ndarray
        Segment id of the highest-scoring candidate per cadence.
    best_score : numpy.ndarray
        Score of the highest-scoring candidate per cadence.

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

    # Track the single best candidate at each cadence (for diagnostics)
    best_ki           = np.argmax(sc_stack, axis=0)
    best_window_size  = np.array([keys[best_ki[i]][0] for i in range(n)])
    best_shift        = np.array([keys[best_ki[i]][1] for i in range(n)])
    best_seg_id       = np.array([int(all_results[keys[best_ki[i]]]["seg_ids"][i]) for i in range(n)])
    best_score        = sc_stack[best_ki, np.arange(n)]

    return trend, best_window_size, best_shift, best_seg_id, best_score



def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Return a centred moving average.

    Parameters
    ----------
    arr : numpy.ndarray
        Values to smooth.
    window : int
        Rolling window length, in cadences.

    Returns
    -------
    numpy.ndarray
        Centred moving-average values.

    Raises
    ------
    ValueError
        If ``window`` is smaller than one.
    """
    if window < 1:
        raise ValueError("window must be at least 1.")
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().to_numpy()


def center_residual_by_moving_average(
    residuals: np.ndarray,
    ma_window: int = 10,
    clip_abs: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Center residuals by subtracting the moving-average median.

    Because moving averages are linear, subtracting this offset from the
    residual shifts every recomputed moving average by the same amount.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residual flux values to center.
    ma_window : int, optional
        Moving-average window length, in cadences.
    clip_abs : float or None, optional
        Optional absolute clipping limit used when estimating the median offset. If
        None or non-positive, no absolute clipping is applied.

    Returns
    -------
    centered : numpy.ndarray
        Centered residual values.
    diagnostics : dict
        Whether centering was applied and the subtracted offset.
    """
    residuals = np.asarray(residuals, dtype=float)
    ma = moving_average(residuals, ma_window)

    # With the default clip_abs=None the centering sample is all finite
    # moving-average values.
    center_mask = np.isfinite(ma)
    if clip_abs is not None and np.isfinite(clip_abs) and clip_abs > 0 and center_mask.any():
        center_mask &= np.abs(ma) < clip_abs

    if not center_mask.any():
        offset = 0.0
        centered = residuals.copy()
    else:
        offset = float(np.nanmedian(ma[center_mask]))
        centered = residuals - offset

    diagnostics = {
        "centering_applied": bool(center_mask.any()),
        "residual_offset_subtracted": float(offset),
    }
    return centered, diagnostics


def build_safe_final_flare_mask(
    residuals: np.ndarray,
    sigma_thresh: float = 3.0,
    previous_mask: np.ndarray | None = None,
    max_mask_fraction: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """Build a positive-outlier flare mask with a guard against mask explosion.

    The final flare mask is recomputed after the periodic corrections and optional
    residual centering. In pathological cases a one-sided threshold can mark nearly
    the entire light curve as flaring, which then corrupts the later quiescent-noise
    estimate. This helper rejects such masks and falls back to the previous mask.

    Parameters
    ----------
    residuals : numpy.ndarray
        Final residual flux values.
    sigma_thresh : float, optional
        Positive-outlier threshold in robust sigma units.
    previous_mask : numpy.ndarray or None, optional
        Previous/provisional flare mask to use as the fallback and as the first
        quiescent sample for estimating the robust noise.
    max_mask_fraction : float, optional
        Maximum allowed fraction of finite cadences in the candidate final mask.
        If the candidate exceeds this fraction, ``previous_mask`` is returned.

    Returns
    -------
    final_mask : numpy.ndarray
        Safe final flare mask.
    diagnostics : dict
        Diagnostics describing whether the recomputed mask was accepted or rejected.
    """
    residuals = np.asarray(residuals, dtype=float)
    finite = np.isfinite(residuals)

    if previous_mask is None:
        previous_mask = np.zeros(len(residuals), dtype=bool)
    else:
        previous_mask = np.asarray(previous_mask, dtype=bool)
        if len(previous_mask) != len(residuals):
            raise ValueError("previous_mask must have the same length as residuals.")

    n_finite = int(finite.sum())
    if n_finite == 0:
        return previous_mask.copy(), {
            "final_mask_recomputed": False,
            "final_mask_reject_reason": "no finite residuals",
            "final_mask_candidate_fraction": np.nan,
            "final_mask_fraction": float(previous_mask.mean()) if len(previous_mask) else np.nan,
            "final_mask_center": np.nan,
            "final_mask_sigma": np.nan,
            "final_mask_candidate_count": 0,
        }

    # Estimate the noise from points not already suspected as flares. If too few
    # remain, use all finite residuals rather than estimating from a tiny sample.
    q = finite & ~previous_mask
    if q.sum() < 20:
        q = finite

    center = float(np.nanmedian(residuals[q]))
    sigma = robust_sigma(residuals[q])

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = sigma_clipped_std(residuals[q])

    previous_fraction = float((previous_mask & finite).sum() / n_finite)
    if not np.isfinite(sigma) or sigma <= 0:
        return previous_mask.copy(), {
            "final_mask_recomputed": False,
            "final_mask_reject_reason": "invalid sigma",
            "final_mask_candidate_fraction": np.nan,
            "final_mask_fraction": previous_fraction,
            "final_mask_center": center,
            "final_mask_sigma": sigma,
            "final_mask_candidate_count": 0,
        }

    candidate = finite & (residuals > center + sigma_thresh * sigma)
    candidate_count = int(candidate.sum())
    candidate_fraction = float(candidate_count / n_finite)

    # Safety guard: flares should not be almost the whole light curve. If this
    # happens, keep the previous/provisional mask so the downstream noise estimate
    # still has a meaningful quiescent sample.
    max_mask_fraction = float(max_mask_fraction)
    if np.isfinite(max_mask_fraction) and max_mask_fraction > 0 and candidate_fraction > max_mask_fraction:
        return previous_mask.copy(), {
            "final_mask_recomputed": False,
            "final_mask_reject_reason": f"candidate mask fraction too high: {candidate_fraction:.4f}",
            "final_mask_candidate_fraction": candidate_fraction,
            "final_mask_fraction": previous_fraction,
            "final_mask_center": center,
            "final_mask_sigma": sigma,
            "final_mask_candidate_count": candidate_count,
        }

    return candidate, {
        "final_mask_recomputed": True,
        "final_mask_reject_reason": "",
        "final_mask_candidate_fraction": candidate_fraction,
        "final_mask_fraction": candidate_fraction,
        "final_mask_center": center,
        "final_mask_sigma": sigma,
        "final_mask_candidate_count": candidate_count,
    }

def run_detrending(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    config: DetrendConfig | None = None,
    external_flare_mask: np.ndarray | None = None,
) -> DetrendResult:
    """Run the full two-pass detrending pipeline.

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
    external_flare_mask : numpy.ndarray or None, optional
        Optional externally supplied boolean mask. True values are included in all
        flare masks used for fitting/noise estimates, but the original behavior is
        unchanged when this is None.

    Returns
    -------
    DetrendResult
        Final detrended table, segment statistics, and summary diagnostics.

    Raises
    ------
    ValueError
        If array lengths differ, there are too few cadences, non-finite values are
        present in required inputs, flux errors are not positive and finite, window
        sizes are invalid, or the polynomial degree is negative.
    RuntimeError
        If no valid combined trend can be built during either pass.
    """
    cfg = config or DetrendConfig()
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    if not (len(time) == len(flux) == len(flux_err)):
        raise ValueError("time, flux, and flux_err must have the same length.")
    if len(time) < max(cfg.min_fit_pts, cfg.poly_deg + 2):
        raise ValueError("Not enough cadences to detrend.")
    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values.")
    if not np.all(np.isfinite(flux)):
        raise ValueError("flux must contain only finite values.")
    if not np.all(np.isfinite(flux_err)) or np.any(flux_err <= 0):
        raise ValueError("flux_err must contain only positive finite values.")
    if len(cfg.window_sizes) == 0 or any(w <= 0 for w in cfg.window_sizes):
        raise ValueError("window_sizes must contain at least one positive value.")
    if cfg.poly_deg < 0:
        raise ValueError("poly_deg must be zero or greater.")
    if cfg.min_fit_pts < cfg.poly_deg + 2:
        warnings.warn(
            "min_fit_pts is small for the polynomial degree; some fits may be unstable.",
            RuntimeWarning,
        )
    if not _ASTROPY_LS and cfg.apply_rotation_sinusoid_correction:
        warnings.warn(
            "Astropy LombScargle is unavailable, so the rotation sinusoid correction will be skipped.",
            RuntimeWarning,
        )

    if external_flare_mask is None:
        external_flare_mask_arr = np.zeros(len(time), dtype=bool)
    else:
        external_flare_mask_arr = np.asarray(external_flare_mask, dtype=bool)
        if len(external_flare_mask_arr) != len(time):
            raise ValueError("external_flare_mask must have the same length as time, flux, and flux_err.")

    preliminary_flare_mask_internal = preliminary_flare_mask(
        time, flux, flux_err,
        sigma_thresh=cfg.flare_mask_sigma,
        rough_window_days=cfg.flare_mask_window_d,
        poly_deg=cfg.poly_deg,
        rolling_pts=cfg.rolling_window_pts,
    )
    flare_mask_prelim = preliminary_flare_mask_internal | external_flare_mask_arr

    all_results_p1 = run_all_windows(
        time, flux, flux_err, flare_mask_prelim, cfg.window_sizes,
        poly_deg=cfg.poly_deg, n_edge=cfg.n_edge, edge_weight=cfg.edge_weight,
        min_fit_pts=cfg.min_fit_pts,
    )
    seg_stats_p1 = build_segment_stats_df(all_results_p1)
    fp_trend, *_ = assemble_combined_trend(
        time, all_results_p1, smooth_sigma_cadences=cfg.smooth_sigma_cadences,
        min_fit_pts=cfg.min_fit_pts,
    )
    fp_residuals = flux - fp_trend

    fp_global_std = sigma_clipped_std(fp_residuals[~flare_mask_prelim & np.isfinite(fp_residuals)])
    flare_mask_p2 = flare_mask_prelim | (
        np.isfinite(fp_residuals) & (fp_residuals > cfg.second_pass_sigma * fp_global_std)
    )
    all_results_p2 = run_all_windows(
        time, flux, flux_err, flare_mask_p2, cfg.window_sizes,
        poly_deg=cfg.poly_deg, n_edge=cfg.n_edge, edge_weight=cfg.edge_weight,
        min_fit_pts=cfg.min_fit_pts,
    )
    seg_stats_p2 = build_segment_stats_df(all_results_p2)
    sp_trend, sp_best_window, sp_best_shift, sp_best_seg, sp_best_score = assemble_combined_trend(
        time, all_results_p2, smooth_sigma_cadences=cfg.smooth_sigma_cadences,
        min_fit_pts=cfg.min_fit_pts,
    )
    sp_residuals = flux - sp_trend

    if cfg.apply_rotation_sinusoid_correction and _ASTROPY_LS:
        sigma_rotation = compute_rolling_local_sigma(
            sp_residuals, window_pts=cfg.rolling_window_pts, flare_mask=flare_mask_p2
        )

        rotation_residuals_corr, rotation_sinusoid_stats = multi_sinusoid_correction(
            time, sp_residuals, flux_err=flux_err, flare_mask=flare_mask_p2,
            sigma_local=sigma_rotation, n_components=cfg.rotation_sinusoid_n_components,
            min_period_hr=cfg.rotation_sinusoid_min_period_hr,
            max_period_hr=cfg.rotation_sinusoid_max_period_hr,
            amp_limit_frac=cfg.rotation_amp_limit_fraction,
            min_improvement=cfg.rotation_sinusoid_min_improvement,
            label="Long-period residual sinusoid correction",
        )
        rotation_sinusoid_stats["method"] = "generic_periodogram"
        rotation_sinusoid_model = sp_residuals - rotation_residuals_corr
        rotation_sinusoid_applied = bool(rotation_sinusoid_stats.get("applied", False))
    else:
        rotation_residuals_corr = sp_residuals.copy()
        rotation_sinusoid_model = np.zeros(len(time), dtype=float)
        rotation_sinusoid_stats = _empty_sinusoid_stats(method="disabled", reject_reason="disabled")
        rotation_sinusoid_applied = False

    final_residual = rotation_residuals_corr.copy()

    if cfg.center_final_residual:
        final_residual, centering_diagnostics = center_residual_by_moving_average(
            final_residual,
            ma_window=cfg.centering_ma_window,
            clip_abs=cfg.centering_clip_abs,
        )
    else:
        centering_diagnostics = {"centering_applied": False, "residual_offset_subtracted": 0.0}

    # Rebuild the final flare mask after the final periodic correction and
    # centering. This releases points that were provisionally flagged only
    # because they sat on broad sinusoid crests, while true sharp positive
    # outliers should remain. The helper guards against final-mask explosions
    # that would otherwise mark nearly the full light curve and corrupt the
    # downstream noise estimate.
    final_flare_mask, final_mask_diagnostics = build_safe_final_flare_mask(
        final_residual,
        sigma_thresh=cfg.second_pass_sigma,
        previous_mask=flare_mask_p2,
        max_mask_fraction=cfg.final_flare_max_mask_fraction,
    )
    final_flare_mask = final_flare_mask | external_flare_mask_arr

    sp_sigma_local = compute_rolling_local_sigma(
        final_residual, window_pts=cfg.rolling_window_pts, flare_mask=final_flare_mask
    )
    sp_sigma_total = np.sqrt(sp_sigma_local**2 + flux_err**2)
    q_sp = final_residual[np.isfinite(final_residual) & ~final_flare_mask]
    sp_sc_std = sigma_clipped_std(q_sp) if len(q_sp) > 10 else np.nan

    sort_idx = np.argsort(time)
    final_df = pd.DataFrame({
        "time": time[sort_idx],
        "flux": flux[sort_idx],
        "flux_err": flux_err[sort_idx],
        "first_pass_trend": fp_trend[sort_idx],
        "first_pass_residual": fp_residuals[sort_idx],
        "second_pass_trend": sp_trend[sort_idx],
        "second_pass_residual": sp_residuals[sort_idx],
        "rotation_sinusoid_model": rotation_sinusoid_model[sort_idx],
        "rotation_residual_corr": rotation_residuals_corr[sort_idx],
        "final_residual": final_residual[sort_idx],
        "local_sigma": sp_sigma_local[sort_idx],
        "total_sigma": sp_sigma_total[sort_idx],
        "external_flare_mask": external_flare_mask_arr[sort_idx],
        "flare_mask_prelim_internal": preliminary_flare_mask_internal[sort_idx],
        "flare_mask_prelim": flare_mask_prelim[sort_idx],
        "provisional_flare_mask": flare_mask_p2[sort_idx],
        "final_flare_mask": final_flare_mask[sort_idx],
        "selected_window_size": sp_best_window[sort_idx],
        "selected_shift": sp_best_shift[sort_idx],
        "selected_segment_id": sp_best_seg[sort_idx],
        "selected_segment_score": sp_best_score[sort_idx],
    })

    cadence_days = float(np.nanmedian(np.diff(time)))
    summary = {
        "span_days": float(time.max() - time.min()),
        "n_cadences": int(len(time)),
        "cadence_min": cadence_days * 24 * 60,
        "window_sizes": tuple(cfg.window_sizes),
        "poly_deg": cfg.poly_deg,
        "rotation_sinusoid_applied": rotation_sinusoid_applied,
        "rotation_sinusoid_n_components": int(rotation_sinusoid_stats.get("n_components", 0)),
        "rotation_sinusoid_improvement": float(rotation_sinusoid_stats.get("improvement", 0.0)),
        "rotation_sinusoid_periods_hr": tuple(rotation_sinusoid_stats.get("periods_hr", [])),
        "rotation_sinusoid_boundary_hit": bool(rotation_sinusoid_stats.get("boundary_hit", False)),
        "rotation_sinusoid_reject_reason": str(rotation_sinusoid_stats.get("reject_reason", "")),
        "rotation_correction_method": str(rotation_sinusoid_stats.get("method", "")),
        "n_external_flare_masked": int(external_flare_mask_arr.sum()),
        "n_prelim_internal_masked": int(preliminary_flare_mask_internal.sum()),
        "n_prelim_masked": int(flare_mask_prelim.sum()),
        "n_provisional_masked": int(flare_mask_p2.sum()),
        "n_final_masked": int(final_flare_mask.sum()),
        "median_local_sigma": float(np.nanmedian(sp_sigma_local)),
        "sigma_clipped_global_std": float(sp_sc_std),
        "median_total_sigma": float(np.nanmedian(sp_sigma_total)),
        **final_mask_diagnostics,
        **centering_diagnostics,
    }
    return DetrendResult(
        final_df=final_df,
        seg_stats_p1=seg_stats_p1,
        seg_stats_p2=seg_stats_p2,
        summary=summary,
    )
