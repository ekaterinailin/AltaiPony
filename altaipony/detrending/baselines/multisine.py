"""Multi-harmonic sine baseline for short-period rotators."""

import numpy as np

from ...utils import upper_outlier_threshold
from ..periodicity import lomb_scargle


def _split_cont_windows(cont_windows, time, period, n_per):
    """Divide each continuous observing window into evenly-sized subsegments.

    For each window in ``cont_windows`` (bounded by real data gaps, not
    artificial cuts), compute how many equal pieces are needed so that no
    piece exceeds ``n_per`` cycles, then split the window evenly by index.
    Segments already shorter than ``n_per`` cycles are left untouched.

    Because the number of pieces is chosen upfront and the segment is divided
    evenly, all subsegments have the same length — there are no leftover slivers
    that would give the harmonic solver poor phase coverage at the edges.

    Parameters
    ----------
    cont_windows : list of (int, int)
        Real segment boundaries from ``FlareLightCurve.find_cont_windows``.
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

    for le, ri in cont_windows:
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
    cont_windows,
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
    cont_windows : list of (int, int)
        Segment boundaries ``(left_index, right_index)`` as produced by
        ``FlareLightCurve.find_cont_windows``.
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

    for le, ri in cont_windows:
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
