"""
UTF-8, Python 3

------------------
AltaiPony
------------------

Ekaterina Ilin, 2023, MIT License

This module contains custom detrending functions.
"""

import logging

import numpy as np
import pandas as pd

import astropy.units as u

from scipy.interpolate import UnivariateSpline
from scipy.ndimage import binary_dilation
from scipy.optimize import minimize

from .altai import _find_iterative_median, equivalent_duration
from .lightcurve_detrender import run_detrending as _ld_run_detrending
from .matchedfilter import matched_filter_flare_mask as _matched_filter_mask
from .periodogram import lomb_scargle
from .utils import MAD_TO_STD, upper_outlier_threshold

try:
    import celerite2
    from celerite2 import terms as celerite2_terms

    _CELERITE2_AVAILABLE = True
except ImportError:
    _CELERITE2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Common baseline that every detrended light curve is normalised to, so that
#: light curves from different stars / sectors share the same flux zero-point
#: (a flat, flare-free baseline sits at exactly this value).
NORMALIZED_BASELINE = 1.0


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GP helper functions
# ---------------------------------------------------------------------------


def _segment_bounds(t, gap_factor=10):
    """Return ``(start, stop)`` index pairs of the segments of ``t`` separated
    by time steps longer than ``gap_factor`` x the median cadence."""
    n = len(t)
    if n < 2:
        return [(0, n)]
    dt = np.diff(t)
    breaks = np.where(dt > gap_factor * np.nanmedian(dt))[0] + 1
    bounds = np.concatenate([[0], breaks, [n]]).astype(int)
    return list(zip(bounds[:-1], bounds[1:]))


def _identify_flare_mask(
    detrended_residuals,
    time=None,
    sigma_threshold=3.0,
    expand_cadences=10,
    segment_gap_factor=10,
):
    """Flag cadences that are likely large flares in the detrended residuals.

    Uses a one-sided MAD threshold to find positive outliers above
    ``sigma_threshold``.  Each flagged run is then dilated by
    ``expand_cadences`` on each side so that the flare decay tail does not
    contaminate the GP training set.

    Only the positive tail is clipped because flares are always upward
    excursions; clipping the negative tail would incorrectly mask spot minima.

    **The threshold is computed per real segment, not globally.**  A single
    whole-light-curve median/MAD lets one segment that sits offset in the
    detrended flux (e.g. a large-amplitude segment left with a small DC offset
    by the baseline fit) read as one enormous flare, so the entire segment is
    masked and excluded from downstream GP training.  Anchoring the median and
    MAD to each segment makes that structurally impossible: a segment cannot be
    an outlier relative to its own median.  Segments are split wherever the
    time step jumps by more than ``segment_gap_factor`` × the median cadence.
    If ``time`` is None the old single-threshold behaviour is used.

    Parameters
    ----------
    detrended_residuals : ndarray
        Near-zero-mean residual flux (e.g. ``m2flux − flux_med``).
    time : ndarray or None
        Time array used only to locate segment boundaries.  If None, a single
        global threshold is applied (legacy behaviour).
    sigma_threshold : float
        Positive-side threshold in units of ``MAD_TO_STD`` × MAD.  Cadences
        above ``median + sigma_threshold × MAD × MAD_TO_STD`` are flagged.
        Higher values mask fewer, larger events.  Defaults to 3.0.
    expand_cadences : int
        Dilation half-width in cadences.  Defaults to 10.
    segment_gap_factor : int
        A time step larger than this multiple of the median cadence starts a
        new segment for the per-segment threshold.  Defaults to 10.

    Returns
    -------
    mask : bool ndarray
        True where a cadence is considered a flare (exclude from GP training).
    """
    resid = np.asarray(detrended_residuals)
    n = len(resid)

    # Segment boundaries from time gaps (one segment if no time given).
    if time is None:
        segments = [(0, n)]
    else:
        segments = _segment_bounds(np.asarray(time), segment_gap_factor)

    flagged = np.zeros(n, dtype=bool)
    for a, b in segments:
        seg = resid[a:b]
        valid = ~np.isnan(seg)
        if valid.sum() < 3:
            continue
        flagged[a:b] = seg > upper_outlier_threshold(seg, sigma_threshold)

    # Dilate to capture decay tails
    if expand_cadences > 0:
        struct = np.ones(2 * expand_cadences + 1, dtype=bool)
        flagged = binary_dilation(flagged, structure=struct)

    return flagged


def _segment_edge_mask(t, valid_mask=None, flux_err=None, segment_gap_factor=10):
    """Flag the first and last *usable* cadence of each segment.

    Segments are split where the time step exceeds ``segment_gap_factor`` ×
    the median cadence.  With ``valid_mask`` given, only valid cadences (and,
    for an array ``flux_err``, those with a finite error) are eligible to be an
    anchor; otherwise every cadence is eligible.  Used both to force segment
    edges into the GP training set and to protect them from the post-fit clip.
    """
    t = np.asarray(t)
    n = len(t)
    mask = np.zeros(n, dtype=bool)
    if valid_mask is None:
        eligible = np.ones(n, dtype=bool)
    else:
        eligible = np.asarray(valid_mask, dtype=bool).copy()
        if flux_err is not None and not np.isscalar(flux_err):
            eligible &= np.isfinite(np.asarray(flux_err))
    for a, b in _segment_bounds(t, segment_gap_factor):
        idx = np.where(eligible[a:b])[0]
        if len(idx):
            mask[a + idx[0]] = True
            mask[a + idx[-1]] = True
    return mask


def _bin_training_per_segment(t, f, e, bin_factor, segment_gap_factor=10):
    """Bin GP training points within each segment, with anchored edges.

    The training arrays are split into segments wherever the time step jumps by
    more than ``segment_gap_factor`` × the median cadence, and each segment is
    binned independently.  This avoids two edge-thinning problems of a plain
    global reshape-and-average:

    * **No tail drop.**  A global ``t[:n_bins*bin_factor]`` discards the final
      ``len % bin_factor`` cadences — always the right end of the last segment —
      leaving that edge weakly anchored.  Here the leftover points of every
      segment are kept as a final (smaller) bin.
    * **Full-resolution edges.**  A bin sits at its members' *mean* time, so a
      single raw anchor pins the edge value but leaves the cadences between it
      and the first bin mean (~half a bin inside) constrained only by that
      smoothed mean — enough for a high-amplitude edge to sag.  Instead the
      outermost ``bin_factor`` cadences of each segment are kept UNBINNED, so
      the edge and its immediate neighbourhood are at full resolution and the
      GP is pinned through the whole edge region, not just its endpoint.

    Segments too short to leave a binnable interior are kept unbinned.  Bin
    errors follow the same ``sqrt(Σσᵢ²) / n`` convention as the caller.

    Parameters
    ----------
    t, f, e : ndarray
        Time-ordered training time, flux, and error arrays.
    bin_factor : int
        Target number of cadences per bin (> 1).
    segment_gap_factor : int
        Time-gap multiple (of the median cadence) that starts a new segment.

    Returns
    -------
    tb, fb, eb : ndarray
        Binned, edge-anchored, strictly time-ordered training arrays.
    """
    n = len(t)
    if n < 2 or bin_factor <= 1:
        return t, f, e

    tb_parts, fb_parts, eb_parts = [], [], []
    edge_keep = bin_factor  # cadences at each segment end kept at full resolution
    for a, b in _segment_bounds(t, segment_gap_factor):
        st, sf, se = t[a:b], f[a:b], e[a:b]
        m = b - a
        if m <= 2 * edge_keep + bin_factor:
            # Too short to leave a worthwhile binnable interior once both edges
            # are kept raw: keep the whole segment at full resolution.
            tb_parts.append(st)
            fb_parts.append(sf)
            eb_parts.append(se)
            continue

        # Keep the outermost `edge_keep` cadences of each end UNBINNED, so the
        # edge and its immediate neighbourhood are at full resolution — a single
        # raw anchor pins the edge value but leaves cadences 1..bin_factor
        # constrained only by a smoothed bin mean ~half a bin inside, which lets
        # a high-amplitude edge sag.  Only the interior is binned.
        int_t, int_f, int_e = (
            st[edge_keep:-edge_keep],
            sf[edge_keep:-edge_keep],
            se[edge_keep:-edge_keep],
        )
        mi = len(int_t)
        n_full = mi // bin_factor
        n_keep = n_full * bin_factor
        seg_t, seg_f, seg_e = [], [], []
        if n_full > 0:
            seg_t.append(int_t[:n_keep].reshape(n_full, bin_factor).mean(axis=1))
            seg_f.append(int_f[:n_keep].reshape(n_full, bin_factor).mean(axis=1))
            seg_e.append(
                np.sqrt((int_e[:n_keep].reshape(n_full, bin_factor) ** 2).sum(axis=1))
                / bin_factor
            )
        # Remainder of the interior kept as one final (smaller) bin — no drop.
        if n_keep < mi:
            rem = mi - n_keep
            seg_t.append([int_t[n_keep:].mean()])
            seg_f.append([int_f[n_keep:].mean()])
            seg_e.append([np.sqrt((int_e[n_keep:] ** 2).sum()) / rem])

        interior_t = np.concatenate(seg_t) if seg_t else np.empty(0)
        interior_f = np.concatenate(seg_f) if seg_f else np.empty(0)
        interior_e = np.concatenate(seg_e) if seg_e else np.empty(0)

        # Full-resolution left edge + binned interior + full-resolution right edge.
        bt = np.concatenate([st[:edge_keep], interior_t, st[-edge_keep:]])
        bf = np.concatenate([sf[:edge_keep], interior_f, sf[-edge_keep:]])
        be = np.concatenate([se[:edge_keep], interior_e, se[-edge_keep:]])

        tb_parts.append(bt)
        fb_parts.append(bf)
        eb_parts.append(be)

    tb = np.concatenate(tb_parts)
    fb = np.concatenate(fb_parts)
    eb = np.concatenate(eb_parts)
    order = np.argsort(tb, kind="stable")
    return tb[order], fb[order], eb[order]


def fit_gp_rotation(
    time,
    flux,
    flux_err,
    period,
    flare_mask,
    optimize_hyperparams=True,
    initial_sigma=None,
    initial_Q0=1.0,
    initial_dQ=0.5,
    initial_f=0.5,
    period_tolerance=0.05,
    bin_factor=10,
    gp_clip_sigma=3.0,
    gp_clip_iters=3,
    anchor_edges=True,
    return_std=False,
):
    """Fit a celerite2 quasi-periodic GP to flare-masked flux.

    Uses ``celerite2.terms.RotationTerm`` — a mixture of two SHO terms at
    the rotation period ``P`` and its first harmonic ``P/2`` — which is the
    standard kernel for stellar spot-driven variability.  White-noise jitter
    (excess noise beyond the formal ``flux_err``) is handled via the ``diag``
    argument to ``gp.compute()`` rather than as a kernel term, which is the
    correct celerite2 v2 API.

    The GP is **trained only on unmasked cadences** (no NaNs, no flagged
    flares).  It is **predicted at every cadence** in ``time``, so it
    smoothly bridges masked windows rather than leaving gaps.  This means
    the GP model is a continuous, physically motivated version of the
    stellar variability that can be directly subtracted.

    Hyperparameter optimisation
    ---------------------------
    When ``optimize_hyperparams=True``, all six parameters are optimised
    jointly in log-space via L-BFGS-B:

    =========  ============================================================
    sigma      Overall amplitude of the variability (initialised from the
               standard deviation of the unmasked flux).
    period     Rotation period (constrained to
               ``period × [1 − tolerance, 1 + tolerance]``).
    Q0         Quality factor of the primary (P) SHO term.  Higher Q →
               more coherent oscillation.
    dQ         Additional quality for the secondary (P/2) term.
    f          Fractional energy in the secondary term (0 < f < 1).
    jitter     White-noise amplitude added in quadrature to ``flux_err``.
    =========  ============================================================

    Parameters
    ----------
    time : ndarray
        Full time array in days (may contain NaNs from interpolated gaps).
    flux : ndarray
        Original (non-detrended) flux.
    flux_err : ndarray or float
        Formal flux uncertainties.  A scalar is broadcast to all cadences.
    period : float
        Rotation period in days (from the Lomb-Scargle periodogram).
    flare_mask : bool ndarray
        True where cadences are excluded from GP training (large flares).
    optimize_hyperparams : bool
        Optimise hyperparameters by log-likelihood maximisation.
        Defaults to True.
    initial_sigma : float or None
        Starting amplitude.  If None, estimated from ``std(flux[train])``.
    initial_Q0, initial_dQ, initial_f : float
        Starting quality factors and energy fraction.  Defaults: 1.0, 0.5, 0.5.
    period_tolerance : float
        Fractional half-width of the period search box.  Defaults to 0.05.
    bin_factor : int
        Bin the training data by averaging every ``bin_factor`` consecutive
        cadences before fitting the GP.  Reduces the number of training
        points by this factor, which speeds up both ``gp.compute()`` (O(N))
        and ``gp.log_likelihood()`` (called hundreds of times during
        optimisation).  The GP is still *predicted* at the original full
        cadence, so the model and residuals are at full resolution.
        Binning is done per segment with the raw first/last cadence of each
        segment kept as anchor points and the trailing partial bin retained,
        so the segment edges are not thinned.  Errors on binned means are
        propagated as ``sqrt(Σσᵢ²) / n``.  Set to 1 to disable binning.
        Defaults to 10.
    gp_clip_sigma : float
        After the initial optimisation, residuals at training points that
        exceed ``gp_clip_sigma × MAD_TO_STD × MAD`` *above* the GP prediction
        are treated as unmasked flares or decay tails and removed from the
        training set.  The GP is then recomputed (hyperparameters frozen)
        and the process repeats for up to ``gp_clip_iters`` passes.  Only
        the positive tail is clipped — negative residuals are spot troughs.
        Defaults to 3.0.
    gp_clip_iters : int
        Maximum number of post-fit residual clipping iterations.  In
        practice convergence (no new points removed) happens in 1–2 passes.
        Set to 0 to disable.  Defaults to 3.
    anchor_edges : bool
        If True (default), force the first/last usable cadence of every real
        segment into the training set (overriding the flare mask there) and
        protect those points from the post-fit clip, so the GP is pinned
        *through* the segment edges instead of extrapolating past them.
        Prevents the large edge residuals / spurious edge flares seen on
        low-noise, high-amplitude rotators.  Defaults to True.
    return_std : bool
        Also compute the GP predictive standard deviation.  celerite2 builds
        dense (training x prediction) matrices for it, which takes several GB
        on long light curves, so it is off by default.

    Returns
    -------
    gp_model : ndarray
        GP predictive mean at every cadence in ``time``
        (NaN where ``time`` is NaN).
    gp_model_std : ndarray or None
        GP predictive standard deviation (same shape), or None unless
        ``return_std`` is True.
    params : dict
        Fitted hyperparameters plus ``converged`` and ``nll`` keys.
    """
    if not _CELERITE2_AVAILABLE:
        raise ImportError(
            "celerite2 is required for the GP step.  "
            "Install with: pip install celerite2"
        )

    # ── training set: unmasked, non-NaN cadences ──────────────────────────
    valid = ~np.isnan(flux) & ~np.isnan(time)

    # Force the first/last usable cadence of every real segment into the
    # training set, overriding the flare mask there.  The GP is otherwise
    # unconstrained at segment edges (the detrender's mask can flag an edge
    # cadence, and binning/clipping thin them), so it extrapolates and
    # mean-reverts — which on low-noise, high-amplitude rotators shows up as
    # large edge residuals and spurious flares.  Pinning the raw edges forces
    # the GP *through* the edges instead of past them.  These anchors are also
    # protected from the post-fit clip below.
    if anchor_edges:
        edge_anchor = _segment_edge_mask(time, valid_mask=valid, flux_err=flux_err)
    else:
        edge_anchor = np.zeros(len(time), dtype=bool)

    train_mask = (valid & ~flare_mask) | edge_anchor

    if train_mask.sum() < 20:
        raise ValueError(
            f"Only {train_mask.sum()} training points after flare masking — "
            "too few for a reliable GP fit."
        )

    t_train = time[train_mask]
    f_train = flux[train_mask]

    if np.isscalar(flux_err):
        err_train = np.full(len(t_train), float(flux_err))
    else:
        err_train = np.asarray(flux_err)[train_mask]
    err_train = np.clip(np.abs(err_train), 1e-10, None)

    # ── optional binning ──────────────────────────────────────────────────
    # Bin the training arrays before fitting so the GP kernel matrix is
    # smaller.  Binning is done *per segment* with the raw first/last cadence
    # of each segment kept as anchors (see _bin_training_per_segment), so the
    # edges are not thinned — a plain global reshape drops the final partial
    # bin (right-edge tail) and insets every edge by ~half a bin, which the
    # one-sided post-fit clip can then unravel on low-noise, high-amplitude
    # light curves.  Prediction always runs at full cadence regardless.
    if bin_factor > 1:
        t_train, f_train, err_train = _bin_training_per_segment(
            t_train, f_train, err_train, bin_factor
        )

    # ── initial hyperparameters ───────────────────────────────────────────
    f_std = float(np.std(f_train))
    f_mean = float(np.median(f_train))
    if initial_sigma is None:
        initial_sigma = f_std

    # ── build kernel and GP ───────────────────────────────────────────────
    # RotationTerm = two damped SHO modes at P and P/2, ideal for starspot LC.
    # In celerite2 v2, jitter is folded into yerr in quadrature:
    #   yerr_total = sqrt(yerr² + jitter²)
    # passing both yerr= and diag= simultaneously raises a ValueError.
    def _make_kernel(sigma, per, Q0, dQ, f):
        return celerite2_terms.RotationTerm(sigma=sigma, period=per, Q0=Q0, dQ=dQ, f=f)

    def _compute(jitter_sq):
        gp.compute(t_train, yerr=np.sqrt(err_train**2 + jitter_sq), quiet=True)

    initial_jitter = f_std * 0.1

    kernel = _make_kernel(initial_sigma, period, initial_Q0, initial_dQ, initial_f)
    gp = celerite2.GaussianProcess(kernel, mean=f_mean)
    _compute(initial_jitter**2)

    # ── optimise hyperparameters ──────────────────────────────────────────
    log_P = np.log(period)

    if optimize_hyperparams:

        def _neg_log_like(log_params):
            ls, lp, lQ0, ldQ, lf, lj = log_params
            try:
                gp.kernel = _make_kernel(*(np.exp(x) for x in (ls, lp, lQ0, ldQ, lf)))
                _compute(np.exp(2 * lj))
                return -gp.log_likelihood(f_train)
            except Exception:
                return 1e15

        x0 = np.array(
            [
                np.log(initial_sigma),
                log_P,
                np.log(initial_Q0),
                np.log(initial_dQ),
                np.log(initial_f),
                np.log(initial_jitter),
            ]
        )
        bounds = [
            (np.log(1e-4 * f_std), np.log(1e3 * f_std)),  # sigma
            (log_P - period_tolerance, log_P + period_tolerance),  # period
            (np.log(1.0), np.log(200.0)),  # Q0
            (np.log(1.0), np.log(200.0)),  # dQ
            (np.log(0.01), np.log(0.99)),  # f
            (np.log(0.05 * f_std), np.log(2.0 * f_std)),  # jitter
        ]
        result = minimize(
            _neg_log_like,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        ls, lp, lQ0, ldQ, lf, lj = result.x
        fitted = dict(
            sigma=np.exp(ls),
            period=np.exp(lp),
            Q0=np.exp(lQ0),
            dQ=np.exp(ldQ),
            f=np.exp(lf),
            jitter=np.exp(lj),
            converged=result.success,
            nll=result.fun,
        )
        # Re-apply the fitted kernel and jitter at the optimum.
        gp.kernel = _make_kernel(*(fitted[k] for k in ("sigma", "period", "Q0", "dQ", "f")))
        _compute(fitted["jitter"] ** 2)
    else:
        fitted = dict(
            sigma=initial_sigma,
            period=period,
            Q0=initial_Q0,
            dQ=initial_dQ,
            f=initial_f,
            jitter=initial_jitter,
            converged=None,
            nll=None,
        )

    # ── iterative one-sided residual clipping ─────────────────────────────
    # The initial flare mask (from _identify_flare_mask) only catches the
    # largest events.  Smaller flares and decay tails that survived the mask
    # still appear as positive residuals after the GP fit.  Iteratively
    # removing them and recomputing — with hyperparameters *frozen* at the
    # optimised values — makes the model progressively less sensitive to
    # residual flare contamination without the cost of re-running the full
    # optimisation.
    #
    # Only the positive tail is clipped: negative residuals are genuine spot
    # troughs, not flares, and removing them would bias the model high.
    # MAD is computed over the current training residuals so the threshold
    # stays anchored to the noise floor rather than the flare population.
    if gp_clip_iters > 0:
        for _clip in range(gp_clip_iters):
            mu_tr = gp.predict(f_train, t=t_train, return_var=False)
            resid = f_train - mu_tr
            thr = upper_outlier_threshold(resid, gp_clip_sigma)
            keep = resid <= thr  # one-sided: positive only
            if anchor_edges:
                # Never clip the first/last point of a segment: removing an
                # edge anchor re-opens the extrapolation this whole scheme is
                # meant to prevent.
                keep = keep | _segment_edge_mask(t_train)
            n_removed = int((~keep).sum())
            if n_removed == 0:
                break  # converged
            t_train = t_train[keep]
            f_train = f_train[keep]
            err_train = err_train[keep]
            if len(t_train) < 20:
                break
            _compute(fitted["jitter"] ** 2)

    # ── predict at all valid cadences ─────────────────────────────────────
    # The GP predicts smoothly over masked windows because the kernel is a
    # continuous function of time separation — not data-point-to-data-point.
    t_pred = time[valid]
    gp_model = np.full_like(flux, np.nan, dtype=float)
    gp_model_std = None
    if return_std:
        mu, var = gp.predict(f_train, t=t_pred, return_var=True)
        gp_model_std = np.full_like(flux, np.nan, dtype=float)
        gp_model_std[valid] = np.sqrt(np.abs(var))
    else:
        mu = gp.predict(f_train, t=t_pred)
    gp_model[valid] = mu

    return gp_model, gp_model_std, fitted


def fit_lightcurve_detrender(time, flux, flux_err, gaps, config=None):
    """Baseline fit via the external ``lightcurve_detrender`` pipeline.

    Adapter that runs :func:`lightcurve_detrender.run_detrending` and returns
    the same ``(newflux, model, best_params)`` contract as ``fit_multisine`` /
    ``fit_spline``, so it drops straight into the baseline slot of
    ``custom_detrending``.  The external pipeline fits a segmented-polynomial
    baseline plus optional sinusoid corrections and carries its own robust,
    fraction-capped flare masking.

    Notes
    -----
    * ``run_detrending`` requires finite inputs, so the fit runs on the
      finite (non-NaN) subset of ``time``/``flux``/``flux_err`` and the results
      are mapped back onto the full grid (NaN elsewhere).
    * ``flux_err`` is forced strictly positive and finite (non-finite or
      non-positive values are replaced by the median positive error).
    * The pipeline's ``final_residual`` is centred near zero; it is rebased to
      the flux level (``+ median(flux)``) so the downstream Savitzky-Golay
      passes see a signal at the original level, matching the convention of the
      other baseline fitters.
    * ``gaps`` is accepted for signature compatibility but not used directly —
      the external pipeline detects its own continuous blocks from ``time``.

    Parameters
    ----------
    time, flux, flux_err : ndarray
        Full-grid arrays in days / flux units (may contain NaNs).
    gaps : list of (int, int)
        Segment boundaries (unused; see Notes).
    config : lightcurve_detrender.DetrendConfig or None
        Optional configuration forwarded to ``run_detrending``.

    Returns
    -------
    newflux : ndarray
        Detrended flux at the original flux level (NaN where input was NaN).
    model : ndarray
        Baseline (``second_pass_trend``) on the full grid.
    best_params : dict
        ``method='lightcurve_detrender'`` plus selected run-summary fields and
        the pipeline's ``final_flare_mask`` (full grid) under
        ``'ld_final_flare_mask'``.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if np.isscalar(flux_err):
        flux_err = np.full(len(time), float(flux_err))
    flux_err = np.asarray(flux_err, dtype=float)

    finite = np.isfinite(time) & np.isfinite(flux)
    if finite.sum() < 20:
        raise ValueError("Too few finite cadences for the lightcurve detrender.")

    # Force strictly positive, finite errors on the finite subset.
    err = flux_err[finite]
    good_err = np.isfinite(err) & (err > 0)
    fill = np.median(err[good_err]) if good_err.any() else 1.0
    err = np.where(good_err, err, fill)

    # Run the external pipeline on the finite subset.
    result = _ld_run_detrending(time[finite], flux[finite], err, config)

    df = result.final_df
    # run_detrending returns rows time-sorted and 1:1 with its input; the finite
    # subset is already time-ordered, so rows map back by position.
    trend_sub = df["second_pass_trend"].to_numpy(dtype=float)
    resid_sub = df["final_residual"].to_numpy(dtype=float)
    mask_sub = df["final_flare_mask"].to_numpy(dtype=bool)

    flux_level = np.nanmedian(flux[finite])

    model = np.full_like(flux, np.nan, dtype=float)
    newflux = np.full_like(flux, np.nan, dtype=float)
    final_mask = np.zeros(len(flux), dtype=bool)
    model[finite] = trend_sub
    newflux[finite] = resid_sub + flux_level
    final_mask[finite] = mask_sub

    best_params = {
        "method": "lightcurve_detrender",
        "ld_final_flare_mask": final_mask,
        "ld_n_final_masked": int(final_mask.sum()),
        "ld_median_local_sigma": float(
            result.summary.get("median_local_sigma", np.nan)
        ),
        "ld_window_sizes": result.summary.get("window_sizes"),
        "ld_poly_deg": result.summary.get("poly_deg"),
        "ld_rotation_sinusoid_applied": result.summary.get("rotation_sinusoid_applied"),
    }
    return newflux, model, best_params


def custom_detrending(
    lc,
    savgol1=6.0,
    savgol2=3.0,
    use_savgol=True,
    pad=3,
    max_sigma=2.5,
    longdecay=6,
    maxgap=10,
    break_tolerance=10,
    periodicity_fap_threshold=1e-3,
    periodicity_amplitude_threshold=0.01,
    period_min_days=0.1,
    n_sine_harmonics=5,
    refine_period_per_segment=True,
    n_per=10,
    clip_sigma=3.0,
    max_clip_iter=5,
    multisine_amp_degree=1,
    baseline_method="auto",
    detrender_config=None,
    use_gp=True,
    gp_flare_sigma=5.0,
    gp_flare_expand_cadences=10,
    gp_optimize=True,
    gp_period_tolerance=0.05,
    gp_bin_factor=10,
    gp_clip_sigma=3.0,
    gp_clip_iters=3,
    gp_anchor_edges=True,
    matched_filter_flares=True,
    matched_filter_snr=5.0,
    matched_filter_fwhm_grid=(0.02, 0.05, 0.1, 0.15),
):
    """Custom de-trending for TESS and Kepler
    short cadence light curves, including TESS Cycle 3 20s
    cadence.

    Parameters:
    ------------
    lc : FlareLightCurve
        light curve that has at least time, flux and flux_err
    savgol1 : float
        Window size for first Savitzky-Golay filter application.
        Unit is hours, defaults to 6 hours.
    savgol2 : float
        Window size for second Savitzky-Golay filter application.
        Unit is hours, defaults to 3 hours.
    use_savgol : bool
        If True (default), run the two Savitzky-Golay passes between the
        baseline fit and the GP step.  If False, skip them entirely: the
        baseline output is fed straight to the GP, and — when the baseline is
        ``lightcurve_detrender`` — that stage's flare mask is used as the GP's
        initial mask.  The intended detrender pipeline is therefore
        ``baseline_method="detrender", use_savgol=False``.  Defaults to True.
    pad : 3
        Outliers in Savitzky-Golay filter are padded with this
        number of data points. Defaults to 3.
    max_sigma : float
        Outlier rejection threshold in sigma. Defaults to 2.5.
    longdecay : int
        Long decay time for outlier rejection. Defaults to 6.
    maxgap : float
        Maximum gap size in days for spline fitting. Defaults to 10 x cadence size.
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
    clip_sigma : float
        Positive-residual clipping threshold used inside the per-segment
        multi-sine fit (see ``fit_multisine``).  Residuals more than
        ``clip_sigma`` × MAD above the model are treated as flare
        contamination and excluded from subsequent iterations.  Defaults
        to 3.0.  Raise this value to be more permissive (fewer points
        removed); lower it to be more aggressive.
    max_clip_iter : int
        Maximum number of one-sided clipping iterations inside the
        per-segment multi-sine solver.  In practice convergence is
        reached in 2–3 passes; 5 is a safe upper bound.  Defaults to 5.
    multisine_amp_degree : int
        Degree of the polynomial-in-time that modulates each harmonic's
        amplitude/phase within a multi-sine segment (see ``fit_multisine``).
        ``0`` gives the classic fixed-amplitude model; ``1`` (the default)
        lets the amplitude drift linearly across a segment, which is needed
        for long multi-cycle segments where spot growth/decay or differential
        rotation change the modulation amplitude.  Higher values allow faster
        evolution at the cost of more free parameters.  Defaults to 1.
    baseline_method : str
        Which baseline model to use.  ``"auto"`` (default) keeps the built-in
        behaviour (multi-sine for strong sub-5-day periods, spline otherwise).
        ``"detrender"`` replaces *both* the multi-sine and the spline with the
        external ``lightcurve_detrender`` pipeline (segmented polynomial plus
        sinusoid corrections and its own fraction-capped flare masking);
        requires ``lightcurve_detrender.py`` to be importable.  ``"multisine"``
        or ``"spline"`` force that single method.  Defaults to ``"auto"``.
    detrender_config : lightcurve_detrender.DetrendConfig or None
        Configuration forwarded to the external pipeline when
        ``baseline_method="detrender"``.  If None, its default config is used.
    use_gp : bool
        If True (and ``is_periodic`` is True and celerite2 is installed),
        follow the initial multisine fit with a Gaussian Process step.
        Large flares are identified in the multisine residuals, masked,
        and a celerite2 ``RotationTerm`` GP is fitted to the masked
        *original* flux.  The GP model smoothly bridges the masked
        windows and is then subtracted to produce the final detrended
        flux.  The Savitzky-Golay passes are effectively bypassed because
        the GP already removes all periodic structure.  Defaults to True.
    gp_flare_sigma : float
        One-sided threshold (in units of 1.4826 × MAD) above which a
        cadence is considered a large flare and excluded from GP training.
        Higher values mask fewer cadences (only the brightest flares).
        The recovery rate of flares above this threshold is high in
        typical injection-recovery experiments.  Defaults to 5.0.
    gp_flare_expand_cadences : int
        Number of cadences to expand each masked flare window on each
        side so that the flare decay tail does not contaminate the GP
        training data.  Defaults to 10 (~20 min at 2-min cadence).
    gp_optimize : bool
        If True, optimise the GP hyperparameters (sigma, period within
        ±``gp_period_tolerance``, Q0, dQ, f, jitter) by maximising the
        log-likelihood via L-BFGS-B.  If False, use the initial values
        derived from the data.  Defaults to True.
    gp_period_tolerance : float
        Fractional half-width of the period search range during GP
        optimisation.  E.g. 0.05 allows the period to drift ±5 % from
        the LS peak value.  Defaults to 0.05.
    gp_bin_factor : int
        Bin the GP training data by this factor before fitting.  A value
        of 5 reduces a typical 18 000-point TESS segment to ~3 600 points,
        cutting ``gp.compute()`` and each log-likelihood evaluation to ~1/5
        of their unbinned cost.  Prediction still runs at full cadence.
        Defaults to 5.
    gp_clip_sigma : float
        One-sided residual clipping threshold applied after the GP
        optimisation.  Training points whose residual exceeds
        ``gp_clip_sigma × MAD_TO_STD × MAD`` above the GP prediction are
        removed and the GP is recomputed (hyperparameters frozen).
        Lowers sensitivity to unmasked small flares and decay tails.
        Defaults to 3.0.
    gp_clip_iters : int
        Maximum number of post-fit residual clipping passes.  Set to 0
        to disable.  Defaults to 3.
    gp_anchor_edges : bool
        If True (default), pin the first/last cadence of every segment into
        the GP training set and shield them from the clip, so the GP is
        forced through the segment edges rather than extrapolating past them.
        Fixes large edge residuals / spurious edge flares on low-noise,
        high-amplitude rotators.  Defaults to True.
    matched_filter_flares : bool
        If True (default), run a matched-filter pass (``matchedfilter`` module)
        on the baseline-subtracted residual and union its detections into the
        GP's initial flare mask.  This catches wide, moderate-SNR flares that
        per-cadence masks miss, so the GP does not train on and remove them.
        Defaults to True.
    matched_filter_snr : float
        Matched-filter detection threshold, in filter sigma.  Defaults to 5.0.
    matched_filter_fwhm_grid : sequence of float
        Flare FWHMs (days) the matched filter templates against.  Defaults to
        ``(0.02, 0.05, 0.1, 0.15)``.

    Return:
    -------
    FlareLightCurve with detrended_flux attribute
    """
    dt = np.mean(np.diff(lc.time.value))
    gaps = lc.find_gaps(maxgap=maxgap * dt).gaps
    # Store original flux as a column so it survives filtering operations
    lc["original_flux"] = lc.flux.copy()
    lc["original_flux_err"] = lc.flux_err.copy()

    lc = lc.interpolate_missing_cadences()
    time, flux = lc.time.value, lc.flux.value

    # --- Periodicity check ------------------------------------------------
    # Run a Lomb-Scargle periodogram on the raw flux.  If a strong periodic
    # signal is found (low FAP *and* large amplitude) use a multi-harmonic
    # sine model as the baseline instead of the spline, because a spline
    # will chase the periodic oscillations and corrupt flare detection.
    period_max_days = (time[-1] - time[0]) / 2.0

    is_periodic, dominant_period, rel_amplitude, fap = detect_strong_periodicity(
        time,
        flux,
        fap_threshold=periodicity_fap_threshold,
        amplitude_threshold=periodicity_amplitude_threshold,
        period_min=period_min_days,
        period_max=period_max_days,
    )

    # The multi-sine baseline is only used for *short-period* strong rotators.
    # A strong but long (>= 5 d) period is better handled by the spline, which
    # tracks slow trends without needing many harmonics — so it falls through
    # to the spline branch below along with the non-periodic case.
    use_multisine = is_periodic and dominant_period < 5

    if baseline_method == "detrender":
        # Replace both the multi-sine and spline baselines with the external
        # lightcurve_detrender pipeline.
        m2flux, _, best_params = fit_lightcurve_detrender(
            time, flux, lc.flux_err.value, gaps, config=detrender_config
        )
        if is_periodic:
            best_params["dominant_period"] = dominant_period

    elif baseline_method in ("auto", "multisine") and (
        use_multisine or baseline_method == "multisine"
    ):
        flux_med = _find_iterative_median(flux, gaps, longdecay=longdecay)

        # Divide each real segment into equal subsegments of at most n_per
        # cycles.  Because the split is computed upfront for the whole segment
        # and the pieces are equal, there are no leftover slivers at the edges.
        multisine_gaps = _segment_gaps(gaps, time, dominant_period, n_per)

        m2flux, _, best_params = fit_multisine(
            time,
            flux,
            flux_med,
            multisine_gaps,
            period=dominant_period,
            n_harmonics=n_sine_harmonics,
            refine_period=refine_period_per_segment,
            clip_sigma=clip_sigma,
            max_clip_iter=max_clip_iter,
            amp_degree=multisine_amp_degree,
        )
        best_params["method"] = "multisine"
        best_params["dominant_period"] = dominant_period
        best_params["multisine_n_segments"] = len(multisine_gaps)

    else:
        # Non-periodic, or strong but long-period: fit a spline to the general
        # trends.
        m2flux, _, best_params = fit_spline(time, flux, gaps, longdecay=longdecay)
        best_params["method"] = "spline"

    # Flare mask carried from the baseline stage (currently only the external
    # lightcurve_detrender produces one).  When present it is used as the GP's
    # initial flare mask instead of one rebuilt from a savgol residual.
    baseline_flare_mask = best_params.pop("ld_final_flare_mask", None)

    if use_savgol:
        # choose a 6 hour window
        w1 = int((np.rint(savgol1 / 24.0 / dt) // 2) * 2 + 1)

        lc.flux = m2flux * u.electron / u.s
        lc.flux_err = lc.flux_err * u.electron / u.s

        # use Savitzy-Golay to iron out the rest
        lc3 = lc.detrend(
            "savgol",
            w=w1,
            pad=pad,
            max_sigma=max_sigma,
            longdecay=longdecay,
            break_tolerance=break_tolerance,
        )

        lc3.flux = lc3.detrended_flux

        # choose a uneven window size
        w2 = int((np.rint(savgol2 / 24.0 / dt) // 2) * 2 + 1)

        # use Savitzy-Golay to iron out the rest
        lc4 = lc3.detrend(
            "savgol",
            w=w2,
            pad=pad,
            max_sigma=max_sigma,
            longdecay=longdecay,
            break_tolerance=break_tolerance,
        )

        # Restore original flux from the column (now filtered to lc4's length)
        lc4.flux = lc4["original_flux"] * u.electron / u.s

        # Clean up the temporary column
        lc4.remove_column("original_flux")
        lc.flux = lc["original_flux"] * u.electron / u.s
        lc.flux_err = lc["original_flux_err"] * u.electron / u.s

        # find median value
        lc4.find_iterative_median()
    else:
        # No Savitzky-Golay: the baseline output IS the detrended flux fed to
        # the GP step.  Flares were already flagged by the baseline stage
        # (``baseline_flare_mask``), so no savgol residual is needed to build a
        # mask.  Keeping the grid untouched also means that mask stays aligned
        # with the GP's cadences.
        lc4 = lc
        lc4.detrended_flux = m2flux * u.electron / u.s
        lc4.flux = lc4["original_flux"] * u.electron / u.s
        lc4.flux_err = lc4["original_flux_err"] * u.electron / u.s
        lc4.remove_column("original_flux")
        lc4.find_iterative_median()
        best_params["savgol_skipped"] = True

    # Optional final Gaussian-Process refinement (in place on lc4/best_params).
    _apply_gp_step(
        lc4,
        best_params,
        use_gp=use_gp,
        is_periodic=is_periodic,
        dominant_period=dominant_period,
        gp_flare_sigma=gp_flare_sigma,
        gp_flare_expand_cadences=gp_flare_expand_cadences,
        gp_optimize=gp_optimize,
        gp_period_tolerance=gp_period_tolerance,
        gp_bin_factor=gp_bin_factor,
        gp_clip_sigma=gp_clip_sigma,
        gp_clip_iters=gp_clip_iters,
        external_flare_mask=baseline_flare_mask,
        gp_anchor_edges=gp_anchor_edges,
        matched_filter_flares=matched_filter_flares,
        matched_filter_snr=matched_filter_snr,
        matched_filter_fwhm_grid=matched_filter_fwhm_grid,
    )

    return lc4


def _apply_gp_step(
    lc4,
    best_params,
    *,
    use_gp,
    is_periodic,
    dominant_period,
    gp_flare_sigma,
    gp_flare_expand_cadences,
    gp_optimize,
    gp_period_tolerance,
    gp_bin_factor,
    gp_clip_sigma,
    gp_clip_iters,
    external_flare_mask=None,
    gp_anchor_edges=True,
    matched_filter_flares=True,
    matched_filter_snr=5.0,
    matched_filter_fwhm_grid=(0.02, 0.05, 0.1, 0.15),
):
    """Apply the final Gaussian-Process detrending step, in place.

    The GP is trained on ``lc4.flux`` (= the original, undetrended flux) so it
    learns the full stellar variability shape.  It predicts smoothly across
    masked windows.  Subtracting the GP model yields the final detrended flux.

    The initial flare mask (cadences excluded from GP training) comes from one
    of two sources:

    * ``external_flare_mask`` when provided — e.g. the flare mask produced by
      the ``lightcurve_detrender`` baseline.  This is used as-is, so the GP is
      built directly on the baseline stage's flare flags with no intermediate
      savgol pass.
    * otherwise, a mask rebuilt from ``lc4.detrended_flux`` (the savgol
      residual) via :func:`_identify_flare_mask`.

    Either way, the post-fit residual clipping still catches flares the initial
    mask missed.  Runs only when ``use_gp`` and ``is_periodic`` are True and
    celerite2 is installed; otherwise ``lc4`` is left untouched.  ``lc4``
    (``detrended_flux``, ``gp_model``) and ``best_params`` are modified in
    place.
    """
    if not (use_gp and is_periodic and _CELERITE2_AVAILABLE):
        if use_gp and not _CELERITE2_AVAILABLE:
            logger.warning("GP requested but celerite2 is not installed (pip install celerite2) — skipping.")
        return

    t4_gp = lc4.time.value
    f4_gp = lc4.flux.value  # original flux on lc4's filtered grid
    fe4_gp = lc4.flux_err.value  # formal errors on same grid
    det4 = np.asarray(lc4.detrended_flux)  # baseline-subtracted residual

    if external_flare_mask is not None:
        flare_mask = np.asarray(external_flare_mask, dtype=bool)
    else:
        flare_mask = _identify_flare_mask(
            det4,
            time=t4_gp,
            sigma_threshold=gp_flare_sigma,
            expand_cadences=gp_flare_expand_cadences,
        )

    # Augment the per-cadence mask with a matched-filter pass on the
    # baseline-subtracted residual.  Per-cadence masks catch narrow flares but
    # miss wide, moderate-SNR ones (each cadence sits under the cut while the
    # integrated flux is large); the matched filter's coherent sum recovers
    # them, so the GP does not train on and absorb them.
    if matched_filter_flares:
        mf_mask = _matched_filter_mask(
            t4_gp,
            det4,
            noise=None,
            snr_threshold=matched_filter_snr,
            fwhm_grid=matched_filter_fwhm_grid,
        )
        flare_mask = flare_mask | mf_mask

    n_masked = int(flare_mask.sum())

    try:
        gp_model, _, gp_params = fit_gp_rotation(
            t4_gp,
            f4_gp,
            fe4_gp,
            period=dominant_period,
            flare_mask=flare_mask,
            optimize_hyperparams=gp_optimize,
            period_tolerance=gp_period_tolerance,
            bin_factor=gp_bin_factor,
            gp_clip_sigma=gp_clip_sigma,
            gp_clip_iters=gp_clip_iters,
            anchor_edges=gp_anchor_edges,
        )

        unmasked_valid = ~flare_mask & ~np.isnan(f4_gp) & ~np.isnan(gp_model)
        gp_offset = np.nanmedian(f4_gp[unmasked_valid]) - np.nanmedian(
            gp_model[unmasked_valid]
        )

        # Add NORMALIZED_BASELINE so every detrended light curve shares the
        # same flux zero-point (a flat, flare-free baseline sits at exactly
        # NORMALIZED_BASELINE), which keeps light curves from different
        # stars/sectors directly comparable downstream.
        gp_detrended = f4_gp - gp_model + gp_offset + NORMALIZED_BASELINE
        lc4.detrended_flux = gp_detrended * u.electron / u.s
        lc4.gp_model = gp_model + gp_offset

        best_params["method"] = f"{best_params.get('method', 'baseline')}+gp"
        best_params["gp"] = gp_params
        best_params["n_gp_masked"] = n_masked

    except Exception as exc:
        logger.warning(f"  GP fit failed ({exc!r}); keeping pre-GP result.")
        best_params["gp_error"] = str(exc)


def estimate_detrended_noise(
    flc, mask_pos_outliers_sigma=2.5, std_window=100, longdecay=6
):
    """
    Estimate detrended flux uncertainties using rolling standard deviation.

    Flare masking is intentionally **one-sided**: only data points whose flux
    exceeds the segment median by more than ``mask_pos_outliers_sigma`` × MAD
    are excluded.  Clipping negative deviations (genuine troughs from spot
    modulation or imperfect detrending) would artificially inflate the noise
    estimate at those phases, so they are left in.  MAD is used instead of
    std so that the threshold itself is not inflated by any surviving flare
    signal.

    Parameters
    ----------
    flc : FlareLightCurve
        Light curve with detrended_flux attribute
    mask_pos_outliers_sigma : float
        One-sided sigma threshold: points with residual > median +
        ``mask_pos_outliers_sigma`` × 1.4826 × MAD are treated as flare
        contamination and set to NaN before the rolling std is computed.
        Defaults to 2.5.
    std_window : int
        Window size for the rolling standard deviation (in cadences).
        Defaults to 100.
    longdecay : int
        Long-decay parameter forwarded to ``_find_iterative_median``.
        Defaults to 6.

    Returns
    -------
    flc : FlareLightCurve
        Input light curve with detrended_flux_err attribute updated.
    """
    # Find gaps if not already done
    if flc.gaps is None:
        flc = flc.find_gaps()

    # Extract arrays we need (avoids repeated attribute access)
    detrended_flux = np.asarray(
        getattr(flc.detrended_flux, "value", flc.detrended_flux), dtype=float
    )
    n_points = len(detrended_flux)

    # Initialize output array
    detrended_flux_err = np.full(n_points, np.nan)

    # Process each gap segment
    for le, ri in flc.gaps:
        # Extract segment
        flux_segment = detrended_flux[le:ri].copy()

        # --- first pass: one-sided positive clip on the raw segment ----------
        # Only mask upward outliers (flares).  Clipping the lower tail would
        # bias the noise estimate high at spot minima.
        upper_threshold1 = upper_outlier_threshold(flux_segment, mask_pos_outliers_sigma)
        mask = flux_segment <= upper_threshold1  # True = keep

        flux_segment_masked = flux_segment.copy()
        flux_segment_masked[~mask] = np.nan

        # --- second pass: refine on iterative-median-subtracted residuals ----
        # The iterative median tracks slow baseline drifts; subtracting it
        # centres the segment near zero so the MAD threshold is not biased
        # by any residual low-frequency trend.
        it_med_segment = _find_iterative_median(
            flux_segment,
            gaps=[(0, ri - le)],
            longdecay=longdecay,
        )
        flux_normalized = flux_segment - it_med_segment

        # One-sided clip on the centred residuals.  Same logic: positive-only.
        upper_threshold2 = upper_outlier_threshold(flux_normalized, mask_pos_outliers_sigma)
        mask_refined = flux_normalized <= upper_threshold2

        flux_normalized_masked = flux_normalized.copy()
        flux_normalized_masked[~mask_refined] = np.nan

        # Compute rolling std on the cleaned, centred residuals
        final_err = (
            pd.Series(flux_normalized_masked)
            .rolling(std_window, center=True, min_periods=1)
            .std()
            .interpolate()
            .values
        )

        # Store in output array
        detrended_flux_err[le:ri] = final_err

    # Set the result on the original lightcurve
    flc.detrended_flux_err = detrended_flux_err

    return flc


def detect_strong_periodicity(
    time,
    flux,
    fap_threshold=1e-3,
    amplitude_threshold=0.01,
    period_min=0.1,
    period_max=None,
    flux_median=None,
):
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

    ls, frequency, power = lomb_scargle(
        t, f_centred, min_frequency=1.0 / period_max, max_frequency=1.0 / period_min
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
    theta = ls.model_parameters(peak_freq)  # [offset, a_cos, b_sin]
    rel_amplitude = np.sqrt(theta[1] ** 2 + theta[2] ** 2) / abs(ref_median)

    is_periodic = bool((fap < fap_threshold) and (rel_amplitude > amplitude_threshold))

    return is_periodic, peak_period, rel_amplitude, fap


def _segment_gaps(gaps, time, period, n_per):
    """Divide each real gap segment into evenly-sized subsegments.

    For each segment in ``gaps`` (which represent real data gaps, not
    artificial cuts), compute how many equal pieces are needed so that no
    piece exceeds ``n_per`` cycles, then split the segment evenly by index.
    Segments already shorter than ``n_per`` cycles are left untouched.

    Because the number of pieces is chosen upfront and the segment is divided
    evenly, all subsegments have the same length — there are no leftover slivers
    that would give the harmonic solver poor phase coverage at the edges.

    Parameters
    ----------
    gaps : list of (int, int)
        Real segment boundaries from ``FlareLightCurve.find_gaps``.
    time : array_like
        Full time array (days).
    period : float
        Dominant period in days.
    n_per : float
        Maximum allowed subsegment length in units of ``period``.

    Returns
    -------
    list of (int, int)
        New segment list where every subsegment is ≤ ``n_per`` cycles long
        and all subsegments within a real gap are equal in length.
    """
    result = []
    max_span = n_per * period

    for le, ri in gaps:
        span = time[ri - 1] - time[le]
        n_pieces = int(np.ceil(span / max_span)) if span > max_span else 1
        # Divide the index range into n_pieces equal slices
        indices = np.linspace(le, ri, n_pieces + 1, dtype=int)
        for k in range(n_pieces):
            result.append((int(indices[k]), int(indices[k + 1])))

    return result


def fit_multisine(
    time,
    flux,
    flux_med,
    gaps,
    period,
    n_harmonics=5,
    refine_period=True,
    period_refine_window=0.05,
    clip_sigma=3.0,
    max_clip_iter=5,
    amp_degree=1,
):
    """Fit a multi-harmonic sine baseline to the light curve.

    The model for each gap-segment is::

        f_model(t) = c₀ + c₁(t − t_mid)
                   + Σₖ₌₁ᴺ [ Aₖ(t) cos(2π k t / Pₛₑg)
                            + Bₖ(t) sin(2π k t / Pₛₑg) ]

    where each ``Aₖ(t)``, ``Bₖ(t)`` is a degree-``amp_degree`` polynomial in
    (mid-centred, scaled) time — so the harmonic amplitude *and* phase are
    allowed to drift across the segment.  With ``amp_degree=0`` these reduce to
    constants ``aₖ, bₖ`` and the model is the classic fixed-amplitude
    multi-sine.  Letting them evolve lets a single long (multi-cycle) segment
    follow slow spot growth/decay and differential rotation instead of forcing
    the whole segment onto one amplitude.

    The coefficients are solved via **iterative one-sided least-squares**.
    The linear term ``c₁(t − t_mid)`` absorbs any slow baseline drift within
    the segment so that the harmonic amplitudes are not biased by it.  Time is
    centred on the segment midpoint ``t_mid`` to keep the offset ``c₀`` and the
    slope ``c₁`` numerically orthogonal.  Fitting per segment (and, with
    ``amp_degree>0``, within a segment) lets the amplitude evolve across the
    observation baseline.  Optionally, ``Pₛₑg`` is refined independently for
    each segment with a narrow Lomb-Scargle search around the global ``period``
    to accommodate slightly varying periods (e.g. differential rotation).

    **Flare suppression** — after each least-squares solve, residuals are
    computed at *all* valid points (including those excluded in the previous
    iteration).  Points whose residual exceeds ``clip_sigma`` × 1.4826 × MAD
    *above* the model are excluded from the next iteration.  The clip is
    strictly one-sided (positive residuals only) so that genuine spot troughs —
    which produce negative residuals — are never removed; only flare-driven
    upward spikes are suppressed.  MAD is evaluated on the current in-mask
    residuals so the threshold is not inflated by any surviving flare signal.
    Convergence is declared when no new points are clipped; this typically
    happens within 2–3 passes.

    The per-segment period refinement step also uses a preliminary one-sided
    clip before running the Lomb-Scargle search so that large flares do not
    shift the peak frequency.

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
        narrow Lomb-Scargle search on flare-clipped data.  Defaults to True.
    period_refine_window : float
        Half-width of the period search range expressed as a *fraction* of
        ``period``.  E.g. 0.05 searches ±5 % around ``period``.
        Defaults to 0.05.
    clip_sigma : float
        One-sided clipping threshold in units of 1.4826 × MAD.  Residuals
        above this threshold are considered flare-contaminated and excluded
        from subsequent iterations.  Defaults to 3.0.
    max_clip_iter : int
        Maximum number of clipping iterations per segment.  In practice
        convergence is reached in 2–3 passes; 5 is a safe upper bound.
        Defaults to 5.

    Returns
    -------
    newflux : ndarray
        Detrended flux re-centred at ``flux_med``.
    model : ndarray
        Best-fit multi-sine model evaluated on the full ``time`` array.
    best_params : dict
        Summary of fit parameters: dominant period, per-segment periods, and
        per-segment fundamental amplitudes sqrt(a₁² + b₁²).  The amplitude
        entries use the segment left-index as key, matching ``seg_periods``.
    """
    model = np.full_like(flux, np.nan, dtype=float)
    newflux = np.full_like(flux, np.nan, dtype=float)

    # Each harmonic contributes 2 * (amp_degree + 1) columns: its cos/sin
    # terms multiplied by 1, τ, τ², … so the harmonic *amplitude and phase*
    # can drift as a polynomial in time across the segment (amp_degree=0
    # recovers the classic constant-amplitude model).
    n_cols = 2 + 2 * n_harmonics * (amp_degree + 1)
    seg_periods = {}
    seg_amplitudes = {}  # mid-segment fundamental amplitude sqrt(a₁² + b₁²)

    for le, ri in gaps:
        t_seg = time[le:ri]
        f_seg = flux[le:ri]
        fmed_seg = np.nanmedian(flux_med[le:ri])

        valid = ~(np.isnan(t_seg) | np.isnan(f_seg))
        n_valid = np.sum(valid)

        # Need at least as many valid points as free parameters
        if n_valid < n_cols + 1:
            newflux[le:ri] = f_seg
            model[le:ri] = fmed_seg
            seg_periods[le] = period
            seg_amplitudes[le] = np.nan  # too few points for a reliable fit
            continue

        t_v = t_seg[valid]
        f_v = f_seg[valid]
        seg_len_days = t_v[-1] - t_v[0]

        # Centre time on the segment midpoint so the linear term is
        # orthogonal to the constant offset and numerically well-conditioned.
        t_mid = 0.5 * (t_v[0] + t_v[-1])
        t_seg_c = t_seg - t_mid

        # --- optional per-segment period refinement -----------------------
        seg_period = period

        if refine_period and seg_len_days > 2.0 * period:
            # Only worth refining when the segment covers multiple cycles.
            #
            # Use a preliminary one-sided clip before the LS search so that
            # large flares do not shift the peak frequency.  A single pass
            # (no iteration) is sufficient here because we only need the peak
            # period to within the ±period_refine_window window, not a precise
            # amplitude estimate.
            prelim_mask = f_v <= upper_outlier_threshold(f_v, clip_sigma)

            freq_ctr = 1.0 / period
            freq_delta = freq_ctr * period_refine_window
            freq_lo = max(freq_ctr - freq_delta, 1.0 / (seg_len_days + 1e-6))
            freq_hi = freq_ctr + freq_delta

            if freq_lo < freq_hi and prelim_mask.sum() > n_cols + 1:
                refine_freqs = np.linspace(freq_lo, freq_hi, 400)
                # Centre the clipped flux so LS is not confused by a DC offset
                f_clipped_c = f_v[prelim_mask] - np.nanmedian(f_v[prelim_mask])
                _, _, seg_power = lomb_scargle(
                    t_v[prelim_mask], f_clipped_c, frequency=refine_freqs
                )
                seg_period = 1.0 / refine_freqs[np.argmax(seg_power)]

        seg_periods[le] = seg_period

        # --- build harmonic design matrix ---------------------------------
        # Columns: [1, t_c] followed, for each harmonic k, by its cos/sin
        # multiplied by τ_norm^p for p = 0..amp_degree, where τ_norm is time
        # centred on the segment midpoint and scaled to ≈[-1, 1].  A harmonic's
        # effective amplitude and phase are then degree-``amp_degree``
        # polynomials in time, so slow spot growth/decay and differential
        # rotation across a long (multi-cycle) segment are captured instead of
        # being forced onto a single fixed amplitude.  The linear term (t_c)
        # still absorbs slow baseline drift.  Column order keeps the k=1, p=0
        # cos/sin at indices 2 and 3, so the mid-segment fundamental amplitude
        # below is unchanged.
        A_full = np.ones((ri - le, n_cols))

        # Column 1: linear trend (time centred on segment midpoint)
        A_full[:, 1] = t_seg_c

        # Normalised time for the amplitude-modulation polynomial (bounded so
        # the higher powers stay well-conditioned).
        half_span = 0.5 * seg_len_days if seg_len_days > 0 else 1.0
        tau_f = t_seg_c / half_span

        col = 2
        for k in range(1, n_harmonics + 1):
            cos_f = np.cos(2.0 * np.pi * k * t_seg / seg_period)
            sin_f = np.sin(2.0 * np.pi * k * t_seg / seg_period)
            pow_f = np.ones_like(tau_f)
            for _p in range(amp_degree + 1):
                A_full[:, col] = cos_f * pow_f
                A_full[:, col + 1] = sin_f * pow_f
                col += 2
                pow_f = pow_f * tau_f
        A_valid = A_full[valid]

        # --- iterative one-sided least-squares solution -------------------
        #
        # Plain OLS is sensitive to flares: a single large spike shifts all
        # harmonic coefficients to reduce its squared residual, biasing both
        # amplitude and phase.  We suppress this by clipping only positive
        # residuals (one-sided), iterating until convergence.
        #
        # Why one-sided?  The sinusoidal model has already absorbed the real
        # peaks, so their residuals sit near zero (noise-level).  Flares
        # appear as large *positive* spikes in the residuals because they
        # are not captured by the sinusoidal model.  Clipping the negative
        # tail would remove genuine spot troughs, biasing the amplitude high.
        #
        # MAD (not std) is used for the threshold so that any flare signal
        # surviving the previous iteration does not inflate the scale estimate
        # and push the threshold up, hiding the very outliers we want to clip.
        #
        # Residuals are evaluated at *all* valid points each iteration
        # (not just the in-mask subset), so a point excluded in one pass
        # can still be re-evaluated and re-included if the fit improves.
        coeffs = None
        clip_mask = np.ones(n_valid, dtype=bool)  # True = included in fit

        try:
            for _iter in range(max_clip_iter):
                n_clipped = clip_mask.sum()
                if n_clipped < n_cols + 1:
                    # Safety: too few points remain; stop before the solve
                    # and keep the coefficients from the previous iteration.
                    break

                coeffs_iter, _, _, _ = np.linalg.lstsq(
                    A_valid[clip_mask], f_v[clip_mask], rcond=None
                )

                # Residuals at ALL valid points so flares masked in a prior
                # iteration are still visible and cannot quietly re-enter.
                residuals = f_v - A_valid @ coeffs_iter

                # MAD over the current in-mask residuals only, so the scale
                # is not inflated by surviving flare signal outside the mask.
                # One-sided threshold: clip only above the model
                threshold = upper_outlier_threshold(
                    residuals[clip_mask], clip_sigma, center=0.0
                )
                new_mask = clip_mask & (residuals < threshold)

                coeffs = coeffs_iter  # commit this iteration's solution
                if np.array_equal(new_mask, clip_mask):
                    break  # converged: no new points removed
                clip_mask = new_mask

            if coeffs is None:
                # Fallback if the very first iteration had too few points
                raise ValueError("clip_mask exhausted before first solve")

            model_seg = A_full @ coeffs

        except Exception:
            # Fallback: constant equal to segment median
            model_seg = np.full(ri - le, np.nanmedian(f_seg))
            coeffs = None

        # --- per-segment fundamental amplitude ----------------------------
        # sqrt(a₁² + b₁²) from columns 2 and 3 — the k=1, p=0 (constant) cos/sin
        # coefficients.  With amplitude evolution these are the τ=0 terms, so
        # this is the fundamental amplitude *at the segment midpoint*.  Stored
        # in best_params; plot against segment midpoint time to diagnose
        # amplitude modulation across segments.
        if coeffs is not None and len(coeffs) >= 4:
            seg_amplitudes[le] = float(np.sqrt(coeffs[2] ** 2 + coeffs[3] ** 2))
        else:
            seg_amplitudes[le] = np.nan

        # Store the residual re-centred on the iterative-median baseline
        # ``fmed_seg`` so that the downstream Savitzky-Golay step operates on a
        # near-baseline signal rather than a near-zero one.
        residual = f_seg - model_seg

        model[le:ri] = model_seg
        newflux[le:ri] = residual + fmed_seg

    best_params = {
        "n_harmonics": n_harmonics,
        "amp_degree": amp_degree,
        "global_period": period,
        "seg_periods": seg_periods,
        # Fundamental amplitude (sqrt(a₁²+b₁²)) per segment, keyed by left
        # index.  Plot these against segment midpoint times to diagnose
        # amplitude modulation on timescales of a few rotation periods.
        "seg_amplitudes": seg_amplitudes,
    }

    return newflux, model, best_params


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
    ampl_rec = np.max(flc.detrended_flux.value[sta:sto]) / flc.it_med.value[sta] - 1.0

    # get cadence numbers
    cstart = flc.cadenceno.value[sta]
    cstop = flc.cadenceno.value[sto]

    # get time stamps
    tstart = flc.time.value[sta]
    tstop = flc.time.value[sto]

    # add result to flare table
    newline = pd.Series(
        {
            "ed_rec": ed_rec,
            "ed_rec_err": ed_rec_err,
            "ampl_rec": ampl_rec,
            "istart": sta,
            "istop": sto,
            "cstart": cstart,
            "cstop": cstop,
            "tstart": tstart,
            "tstop": tstop,
            "dur": tstop - tstart,
            "total_n_valid_data_points": flc.flux.value.shape[0],
        }
    )

    flc.flares = pd.concat([flc.flares, newline.to_frame().T], ignore_index=True)