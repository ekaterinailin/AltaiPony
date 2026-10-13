"""
UTF-8, Python 3

------------------
AltaiPony
------------------

Ekaterina Ilin, 2023, MIT License

The detrending pipeline: polynomial baseline and GP model.
"""

import logging

import numpy as np
import astropy.units as u

from ..altai import _find_iterative_median
from .baselines.polynomial import fit_lightcurve_detrender
from .gp import _CELERITE2_AVAILABLE, fit_gp_rotation
from .matchedfilter import matched_filter_flare_mask as _matched_filter_mask
from .periodicity import detect_strong_periodicity, lomb_scargle

#: Common baseline that every detrended light curve is normalised to, so that
#: light curves from different stars / sectors share the same flux zero-point
#: (a flat, flare-free baseline sits at exactly this value).
NORMALIZED_BASELINE = 1.0

logger = logging.getLogger(__name__)


def custom_detrending(
    lc,
    maxgap=10,
    periodicity_fap_threshold=1e-3,
    period_min_days=0.1,
    detrender_config=None,
    use_gp=True,
    gp_optimize=True,
    gp_period_tolerance=0.05,
    gp_bin_factor=10,
    gp_clip_sigma=3.0,
    gp_clip_iters=3,
    gp_anchor_edges=True,
    gp_second_rotation=True,
    gp_residual_term=True,
    matched_filter_flares=True,
    matched_filter_snr=5.0,
    matched_filter_iters=3,
    matched_filter_fwhm_grid=(0.02, 0.05, 0.1, 0.2),
):
    """Custom de-trending for TESS and Kepler
    short cadence light curves, including TESS Cycle 3 20s
    cadence.

    Parameters:
    ------------
    lc : FlareLightCurve
        light curve that has at least time, flux and flux_err
    maxgap : float
        Gaps longer than ``maxgap`` times the mean cadence split the light
        curve into continuous windows.  Defaults to 10.
    periodicity_fap_threshold : float
        False-alarm probability threshold for the periodograms.  The GP always
        starts from the highest peak of the periodogram of the flux without
        the baseline's flare mask, significant or not; the threshold applies
        to the candidates of the second-period search
        (``gp_second_rotation``).  Defaults to 1e-3.
    period_min_days : float
        Shortest period searched by the Lomb-Scargle periodogram, in days.
        Defaults to 0.1 days.
    detrender_config : baselines.polynomial.DetrendConfig or None
        Configuration of the baseline: degree-4 polynomial fits on grids of
        0.6–1.2-day windows (plain and shifted by half a window, blended) and
        its own fraction-capped flare mask.  If None, its defaults are used.
    use_gp : bool
        If True (and celerite2 is installed), follow the baseline fit with a
        Gaussian Process step, for every star: in injection-recovery tests it
        recovered more flares than the polynomial baseline alone even on stars
        without spots.  Flares are masked (the baseline stage's flare mask),
        and a celerite2 ``RotationTerm`` GP is fitted to the masked *original*
        flux.  The GP model smoothly bridges the masked windows and is then
        subtracted to produce the final detrended flux, so the baseline
        result only shapes the flare mask.  Defaults to True.
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
        of 10 reduces a typical 18 000-point TESS segment to ~1 800 points
        (20-min bins at 2-min cadence), cutting ``gp.compute()`` and each
        log-likelihood evaluation to ~1/10 of their unbinned cost.
        Prediction still runs at full cadence.  Binning also makes the GP
        less flexible on short timescales: with a factor of 2, injection-
        recovery lost 31 of 658 recovered flares (weak ones, S/N 2.6–6.5).
        Defaults to 10.
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
    gp_second_rotation : bool
        If True, search the flux's periodogram for up to three candidate
        second periods, ignoring the GP's rotation period and its first
        harmonic, with the same FAP threshold as the initial periodicity
        check.  The GP is refitted with a second rotation term at
        each, and the fit with the lowest residual scatter is kept if it beats
        the single term.  For stars with two unrelated periodic signals.
        Defaults to True.
    gp_residual_term : bool
        If True (default), search the residual of the GP fit for a
        significant periodic signal (FAP below ``periodicity_fap_threshold``)
        and, if there is one, refit with an extra rotation term at its period,
        kept if it lowers the residual scatter.  This covers variability the
        kernel cannot follow, e.g. a fast, large modulation that changes faster
        than the terms' minimum damping time allows.
    matched_filter_flares : bool
        If True (default), fit the GP twice: run a matched-filter pass
        (``matchedfilter`` module) on the residual of the first GP fit, union
        its detections into the flare mask, and refit.  This catches wide,
        moderate-SNR flares that per-cadence masks miss, so the GP does not
        train on and remove them.  The filter runs on the GP residual rather
        than the baseline residual, because rotational modulation the
        baseline could not follow would otherwise be masked as flares and
        hidden from the GP.  Defaults to True.
    matched_filter_snr : float
        Matched-filter detection threshold, in filter sigma.  Defaults to 5.0.
    matched_filter_iters : int
        Number of matched-filter passes.  Each pass runs the matched filter on
        the residual of the latest GP fit, adds its detections to the flare
        mask and refits; it stops early when a pass adds no new cadences.
        Defaults to 3 (+5 flares over a single pass in injection-recovery,
        no extra false positives).
    matched_filter_fwhm_grid : sequence of float
        Flare FWHMs (days) the matched filter templates against.  Defaults to
        ``(0.02, 0.05, 0.1, 0.2)``.

    Return:
    -------
    FlareLightCurve with
        detrended_flux : relative flux whose flat, flare-free baseline is
            ``NORMALIZED_BASELINE`` (1.0), independent of the units of the
            input flux (still labelled e-/s for now).
        it_med : iterative median of the final detrended flux.
        gp_model : the GP baseline in input flux units (NaN if no GP was
            fitted).
    """
    dt = np.mean(np.diff(lc.time.value))
    cont_windows = lc.find_cont_windows(maxgap=maxgap * dt).cont_windows
    # Store original flux as a column so it survives filtering operations
    lc["original_flux"] = lc.flux.copy()
    lc["original_flux_err"] = lc.flux_err.copy()

    lc = lc.interpolate_missing_cadences()
    time, flux = lc.time.value, lc.flux.value

    # --- Baseline: segmented polynomial ------------------------------------
    m2flux, _, best_params = fit_lightcurve_detrender(
        time, flux, lc.flux_err.value, cont_windows, config=detrender_config
    )

    # The baseline stage's flare mask is the GP's initial flare mask.
    baseline_flare_mask = best_params.pop("ld_final_flare_mask")

    # --- Periodicity check ------------------------------------------------
    # Lomb-Scargle periodogram of the raw flux without the cadences in the
    # baseline's flare mask (flares alone can produce a highly significant
    # peak, at roughly their spacing).  The GP's rotation period starts from
    # the highest peak, whether it is significant or not.
    period_max_days = (time[-1] - time[0]) / 2.0
    quiet = ~baseline_flare_mask
    is_periodic, dominant_period, fap = detect_strong_periodicity(
        time[quiet],
        flux[quiet],
        fap_threshold=periodicity_fap_threshold,
        period_min=period_min_days,
        period_max=period_max_days,
    )
    best_params["dominant_period"] = dominant_period
    best_params["periodogram_fap"] = fap
    best_params["periodic"] = is_periodic

    # The baseline output is the detrended flux, at the input flux level; the
    # GP step replaces it.  ``flux`` and ``flux_err``
    # stay the original ones, which the GP is trained on.
    lc4 = lc
    lc4.detrended_flux = m2flux * u.electron / u.s
    lc4.flux = lc4["original_flux"] * u.electron / u.s
    lc4.flux_err = lc4["original_flux_err"] * u.electron / u.s
    lc4.remove_column("original_flux")

    # Optional final Gaussian-Process refinement (in place on lc4/best_params).
    _apply_gp_step(
        lc4,
        best_params,
        use_gp=use_gp,
        dominant_period=dominant_period,
        flare_mask=baseline_flare_mask,
        gp_optimize=gp_optimize,
        gp_period_tolerance=gp_period_tolerance,
        gp_bin_factor=gp_bin_factor,
        gp_clip_sigma=gp_clip_sigma,
        gp_clip_iters=gp_clip_iters,
        gp_anchor_edges=gp_anchor_edges,
        gp_second_rotation=gp_second_rotation,
        gp_residual_term=gp_residual_term,
        periodicity_fap_threshold=periodicity_fap_threshold,
        matched_filter_flares=matched_filter_flares,
        matched_filter_snr=matched_filter_snr,
        matched_filter_iters=matched_filter_iters,
        matched_filter_fwhm_grid=matched_filter_fwhm_grid,
    )

    if "gp" not in best_params:
        # No GP model (GP disabled, no usable period, or GP fit failed): the
        # detrended flux is still at the input flux level.  Express it as
        # relative flux too, scaled by its global iterative median.
        det = np.asarray(getattr(lc4.detrended_flux, "value", lc4.detrended_flux), dtype=float)
        level = _find_iterative_median(det, None)[0]
        lc4.detrended_flux = (NORMALIZED_BASELINE * det / level) * u.electron / u.s

    # Drop the cadences that interpolate_missing_cadences filled in: they
    # only served to give the fits a regular grid.
    if "interpolated" in lc4.colnames:
        lc4 = lc4[np.asarray(lc4["interpolated"]) == 0]

    # Iterative median of the final detrended flux.
    lc4.find_iterative_median()

    return lc4


def _second_period_candidates(time, flux, periods, fap_threshold,
                              n_candidates=3, period_min=0.1, harmonic_tolerance=0.1):
    """Candidate periods for a second periodic signal in ``flux``, strongest
    first.

    ``flux`` is relative flux (median ~1).  The Lomb-Scargle periodogram is
    searched from ``period_min`` to half the time span, ignoring frequencies
    within ``harmonic_tolerance`` of ``1 / P`` and ``2 / P`` for each ``P`` in
    ``periods`` (the rotation period from the periodogram and as fitted by the
    GP): a rotation term at ``P`` already covers both.  Up to ``n_candidates``
    peaks are returned, each more than ``harmonic_tolerance`` away in
    frequency from the stronger ones and passing the same FAP threshold as
    the initial periodicity check
    (``detect_strong_periodicity``).  Several candidates are needed because a
    strong peak can be a sidelobe of the rotation signal (e.g. from its
    changing amplitude) rather than a second signal.
    """
    finite = np.isfinite(time) & np.isfinite(flux)
    t, f = time[finite], flux[finite]
    if len(t) < 20:
        return []
    ls, frequency, power = lomb_scargle(
        t, f - np.median(f), min_frequency=2.0 / (t[-1] - t[0]), max_frequency=1.0 / period_min
    )
    excluded = np.zeros(len(frequency), dtype=bool)
    for period in periods:
        for harmonic in (1.0 / period, 2.0 / period):
            excluded |= np.abs(frequency / harmonic - 1) < harmonic_tolerance

    candidates = []
    while len(candidates) < n_candidates and not excluded.all():
        peak = np.argmax(np.where(excluded, -np.inf, power))
        fap = ls.false_alarm_probability(power[peak], method="baluev")
        if not fap < fap_threshold:
            break  # all remaining peaks are weaker
        candidates.append(1.0 / frequency[peak])
        excluded |= np.abs(frequency / frequency[peak] - 1) < harmonic_tolerance
    return candidates


def _residual_period(time, residual, fap_threshold, period_min=0.1):
    """Period of the highest Lomb-Scargle peak of a GP residual if it is
    significant (FAP below ``fap_threshold``), else None.  The periodogram is
    searched from ``period_min`` to half the time span."""
    finite = np.isfinite(time) & np.isfinite(residual)
    t, r = time[finite], residual[finite]
    if len(t) < 20:
        return None
    ls, frequency, power = lomb_scargle(
        t, r - np.median(r), min_frequency=2.0 / (t[-1] - t[0]), max_frequency=1.0 / period_min
    )
    if len(power) == 0:
        return None
    peak = np.argmax(power)
    if ls.false_alarm_probability(power[peak], method="baluev") < fap_threshold:
        return 1.0 / frequency[peak]
    return None


def _apply_gp_step(
    lc4,
    best_params,
    *,
    use_gp,
    dominant_period,
    flare_mask,
    gp_optimize,
    gp_period_tolerance,
    gp_bin_factor,
    gp_clip_sigma,
    gp_clip_iters,
    gp_anchor_edges=True,
    gp_second_rotation=True,
    gp_residual_term=True,
    periodicity_fap_threshold=1e-3,
    matched_filter_flares=True,
    matched_filter_snr=5.0,
    matched_filter_iters=3,
    matched_filter_fwhm_grid=(0.02, 0.05, 0.1, 0.2),
):
    """Apply the final Gaussian-Process detrending step, in place.

    The GP is trained on ``lc4.flux`` (= the original, undetrended flux) so it
    learns the full stellar variability shape.  It predicts smoothly across
    masked windows.  Subtracting the GP model yields the final detrended flux.

    ``flare_mask`` (cadences excluded from GP training) is the polynomial
    baseline's flare mask, aligned with ``lc4``.  With
    ``matched_filter_flares``, the GP is fitted twice: first with this mask,
    then with the detections of a matched filter run on the first fit's
    residual added to it.  With ``gp_second_rotation``, a second rotation
    term at a second period from the flux's periodogram is added to all fits
    after the first, if it lowers the residual scatter.  Either way, the
    post-fit residual clipping still catches flares the mask missed.  Runs
    when ``use_gp`` is True, ``dominant_period`` is finite and celerite2 is
    installed; otherwise ``lc4`` is left untouched.  ``lc4`` (``detrended_flux``, ``gp_model``) and
    ``best_params`` are modified in place.
    """
    if not (use_gp and np.isfinite(dominant_period) and _CELERITE2_AVAILABLE):
        if use_gp and not _CELERITE2_AVAILABLE:
            logger.warning("GP requested but celerite2 is not installed (pip install celerite2) — skipping.")
        return

    t4_gp = lc4.time.value
    f4_gp = lc4.flux.value  # original flux on lc4's filtered grid
    fe4_gp = lc4.flux_err.value  # formal errors on same grid
    flare_mask = np.asarray(flare_mask, dtype=bool)

    try:
        # Fit the GP to flux normalised by its median, so that the fit (and the
        # optimiser's convergence) does not depend on the units of the flux.
        # The fitted sigma and jitter are therefore relative to the median.
        norm = np.nanmedian(f4_gp)

        def _fit(mask, extra_periods=()):
            model, _, params = fit_gp_rotation(
                t4_gp,
                f4_gp / norm,
                fe4_gp / norm,
                period=dominant_period,
                flare_mask=mask,
                optimize_hyperparams=gp_optimize,
                period_tolerance=gp_period_tolerance,
                bin_factor=gp_bin_factor,
                gp_clip_sigma=gp_clip_sigma,
                gp_clip_iters=gp_clip_iters,
                anchor_edges=gp_anchor_edges,
                extra_periods=extra_periods,
            )
            return model, params

        gp_model, gp_params = _fit(flare_mask)

        # A second periodic signal at a period unrelated to the first (e.g.
        # from a second star) cannot be fitted by the rotation term.  Its
        # residual after the first fit is small and non-sinusoidal (the GP
        # follows most of it), so search for it in the periodogram of the
        # flux instead, refit with a second rotation term at each candidate
        # period, and keep the fit with the lowest scatter of the unmasked
        # residuals, if it is lower than with one term: a candidate can be a
        # sidelobe, and two terms can also compete and fit worse than one.
        def _scatter(model):
            return np.nanstd((f4_gp / norm - model)[~flare_mask])

        extra = []
        if gp_second_rotation:
            best = _scatter(gp_model)
            for candidate in _second_period_candidates(
                t4_gp[~flare_mask],
                (f4_gp / norm)[~flare_mask],
                (dominant_period, gp_params["period"]),
                fap_threshold=periodicity_fap_threshold,
            ):
                model2, params2 = _fit(flare_mask, [candidate])
                if _scatter(model2) < best:
                    best = _scatter(model2)
                    gp_model, gp_params, extra = model2, params2, [candidate]

        # What the kernel still cannot follow (e.g. a fast, large modulation
        # that changes faster than the terms' damping time allows) stays in
        # the residual as a periodic signal.  If the residual periodogram has
        # a significant peak, add a rotation term at its period, and keep it
        # if it lowers the residual scatter.
        if gp_residual_term:
            residual_period = _residual_period(
                t4_gp[~flare_mask],
                (f4_gp / norm - gp_model)[~flare_mask],
                fap_threshold=periodicity_fap_threshold,
            )
            if residual_period is not None:
                model2, params2 = _fit(flare_mask, extra + [residual_period])
                if _scatter(model2) < _scatter(gp_model):
                    gp_model, gp_params = model2, params2
                    extra = extra + [residual_period]

        # Second pass: per-cadence masks catch narrow flares but miss wide,
        # moderate-SNR ones (each cadence sits under the cut while the
        # integrated flux is large); the matched filter's coherent sum
        # recovers them, so the GP does not train on and absorb them.  It
        # runs on the first GP's residual, not on the baseline residual:
        # there, rotation the baseline could not follow would be masked as
        # flares, and the GP would never see the maxima it has to fit.
        # Each further pass runs on the residual of the refit, in which the
        # flares masked so far are no longer absorbed, so they stand out more
        # and their masks grow further.
        for _ in range(matched_filter_iters if matched_filter_flares else 0):
            mf_mask = _matched_filter_mask(
                t4_gp,
                f4_gp / norm - gp_model,
                noise=None,
                snr_threshold=matched_filter_snr,
                fwhm_grid=matched_filter_fwhm_grid,
            )
            if not (mf_mask & ~flare_mask).any():
                break
            flare_mask = flare_mask | mf_mask
            gp_model, gp_params = _fit(flare_mask, extra)

        n_masked = int(flare_mask.sum())
        gp_model = gp_model * norm

        # Baseline model in input flux units: the GP shifted so that the
        # residuals of the unmasked (quiescent) cadences have zero median.
        resid = f4_gp - gp_model
        quiet = ~flare_mask & np.isfinite(resid)
        model = gp_model + np.nanmedian(resid[quiet])

        # Relative flux, 1 + residual / quiescent level: the flat, flare-free
        # baseline sits at exactly NORMALIZED_BASELINE and flare amplitudes
        # are relative, whatever the units of the input flux.
        level = np.nanmedian(model[quiet])
        lc4.detrended_flux = (NORMALIZED_BASELINE + (f4_gp - model) / level) * u.electron / u.s
        lc4.gp_model = model

        best_params["method"] = f"{best_params.get('method', 'baseline')}+gp"
        best_params["gp"] = gp_params
        best_params["n_gp_masked"] = n_masked

    except Exception as exc:
        logger.warning(f"  GP fit failed ({exc!r}); keeping pre-GP result.")
        best_params["gp_error"] = str(exc)
