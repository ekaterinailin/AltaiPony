"""Gaussian-process model of rotational modulation (celerite2 RotationTerm),
trained on flare-masked flux."""

import numpy as np

from scipy.ndimage import binary_dilation
from scipy.optimize import minimize

from ..utils import upper_outlier_threshold

try:
    import celerite2
    from celerite2 import terms as celerite2_terms

    _CELERITE2_AVAILABLE = True
except ImportError:
    _CELERITE2_AVAILABLE = False


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
