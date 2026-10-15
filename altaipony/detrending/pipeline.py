"""
UTF-8, Python 3

------------------
AltaiPony
------------------

Ekaterina Ilin, 2026, MIT License

The detrending pipeline: polynomial baseline and GP model.
"""

import numpy as np
import astropy.units as u

from .baselines.polynomial import polynomial_flare_mask
from .gp import _bin_training_per_segment, fit_gp_rotation
from .matchedfilter import matched_filter_flare_mask as _matched_filter_mask
from .noise import estimate_detrended_noise
from .periodicity import detect_strong_periodicity, lomb_scargle
from ..utils import robust_sigma

# Common baseline in detrended light curves 
NORMALIZED_BASELINE = 1.0


def custom_detrending(
    lc,
    maxgap=10,
    periodicity_fap_threshold=1e-3,
    period_min_days=0.1,
    detrender_config=None,
    gp_optimize=True,
    gp_period_tolerance=0.05,
    gp_bin_factor=10,
    gp_bin_max_bias=0.5,
    gp_jitter_only=True,
    gp_jitter_floor=3.0,
    gp_clip_sigma=3.0,
    gp_clip_iters=3,
    gp_anchor_edges=True,
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
        to the GP residual's periodogram (``gp_residual_term``).  Defaults to
        1e-3.
    period_min_days : float
        Shortest period searched by the Lomb-Scargle periodogram, in days.
        Defaults to 0.1 days.
    detrender_config : baselines.polynomial.DetrendConfig or None
        Configuration of the baseline: degree-4 polynomial fits on grids of
        0.6–1.2-day windows (plain and shifted by half a window, blended) and
        its own fraction-capped flare mask.  If None, its defaults are used.
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
    gp_bin_max_bias : float or None
        If set, lower the bin factor where binning distorts the modulation:
        after the first GP fit, take the largest factor (at most
        ``gp_bin_factor``) at which binning the fitted model itself shifts
        the bins by less than this many binned noise units (99th
        percentile), and refit with it (see ``_bin_factor_for_curvature``).
        On fast, quiet rotators binning by 10 otherwise flattens the
        modulation's sharp peaks, and the GP undercuts them.  None always
        uses ``gp_bin_factor``.  Defaults to 0.5.
    gp_jitter_only : bool
        Weight the GP's training points by a fitted white-noise jitter alone,
        ignoring ``flux_err`` (see ``fit_gp_rotation(jitter_only=...)``).
        Defaults to True.
    gp_jitter_floor : float
        With ``gp_jitter_only``, the jitter's lower bound in units of the
        estimated white noise (see ``fit_gp_rotation``).  Defaults to 3.
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
            input flux (dimensionless).
        it_med : iterative median of the final detrended flux.
        detrended_flux_err : noise of the detrended flux, estimated from
            its flare-masked rolling scatter (``estimate_detrended_noise``).
        gp_model : the GP baseline in input flux units.

    Raises
    ------
    ValueError
        If the light curve is too short for a periodogram (fewer than 20
        unmasked cadences, or spanning less than ``2 * period_min_days``).
    """
    # work on a copy
    lc = lc.copy()

    # time dt
    dt = np.mean(np.diff(lc.time.value))

    # inplace: find continuous stretches of time series
    lc.find_cont_windows(maxgap=maxgap * dt)

    
    time, flux = lc.time.value, lc.flux.value

    # --- Baseline: segmented polynomial ------------------------------------
    # It only provides the GP's initial flare mask.
    baseline_flare_mask = polynomial_flare_mask(
        time, flux, lc.flux_err.value, config=detrender_config
    )

    # --- Periodicity check ------------------------------------------------
    # Lomb-Scargle periodogram of the raw flux without the cadences in the
    # baseline's flare mask (flares alone can produce a highly significant
    # peak, at roughly their spacing).  The GP's rotation period starts from
    # the highest peak, whether it is significant or not.
    period_max_days = (time[-1] - time[0]) / 2.0
    quiet = ~baseline_flare_mask
    _, dominant_period, _ = detect_strong_periodicity(
        time[quiet],
        flux[quiet],
        fap_threshold=periodicity_fap_threshold,
        period_min=period_min_days,
        period_max=period_max_days,
    )
    if not np.isfinite(dominant_period):
        raise ValueError("No periodogram: the light curve has fewer than 20 unmasked "
                         f"cadences or spans less than 2 x period_min_days ({period_min_days} d).")

    # ``flux`` and ``flux_err`` are the original ones, which the GP is
    # trained on.
    lc4 = lc

    # --- Gaussian process (in place on lc4) --------------------------------
    _apply_gp_step(
        lc4,
        dominant_period=dominant_period,
        flare_mask=baseline_flare_mask,
        gp_optimize=gp_optimize,
        gp_period_tolerance=gp_period_tolerance,
        gp_bin_factor=gp_bin_factor,
        gp_bin_max_bias=gp_bin_max_bias,
        gp_jitter_only=gp_jitter_only,
        gp_jitter_floor=gp_jitter_floor,
        gp_clip_sigma=gp_clip_sigma,
        gp_clip_iters=gp_clip_iters,
        gp_anchor_edges=gp_anchor_edges,
        gp_residual_term=gp_residual_term,
        periodicity_fap_threshold=periodicity_fap_threshold,
        matched_filter_flares=matched_filter_flares,
        matched_filter_snr=matched_filter_snr,
        matched_filter_iters=matched_filter_iters,
        matched_filter_fwhm_grid=matched_filter_fwhm_grid,
    )

    # Iterative median and noise of the final detrended flux.
    lc4.find_iterative_median()
    lc4 = estimate_detrended_noise(lc4)

    return lc4


def _bin_factor_for_curvature(time, flux, model, flare_mask, bin_factor, max_bias):
    """Largest bin factor of ``bin_factor``, 5, 3, 2, 1 (at most
    ``bin_factor``) at which binning shifts the modulation by less than
    ``max_bias`` binned noise units.

    A bin averages the modulation over its span, which differs from the
    modulation at the bin's mean time where the modulation curves; the
    binned error shrinks as 1/sqrt(k), this shift does not.  It is measured
    by binning ``model`` (the first GP fit) over the unmasked cadences, as
    the GP's training data are binned, and comparing with ``model`` at the
    bins' times; its 99th percentile must stay below ``max_bias`` times
    sigma/sqrt(k), with sigma the robust scatter of ``flux - model`` (not
    the formal errors)."""
    ok = np.isfinite(model) & np.isfinite(flux)
    train = ok & ~flare_mask
    resid = (flux - model)[train]
    sigma = robust_sigma(resid)
    for k in sorted({bin_factor, 5, 3, 2, 1}, reverse=True):
        if k > bin_factor:
            continue
        if k == 1:
            return 1
        tb, mb, _ = _bin_training_per_segment(time[train], model[train],
                                              np.ones(train.sum()), k)
        shift = mb - np.interp(tb, time[ok], model[ok])
        if np.percentile(np.abs(shift), 99) < max_bias * sigma / np.sqrt(k):
            return k


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
    *,
    dominant_period,
    flare_mask,
    gp_optimize,
    gp_period_tolerance,
    gp_bin_factor,
    gp_bin_max_bias=0.5,
    gp_jitter_only=True,
    gp_jitter_floor=3.0,
    gp_clip_sigma,
    gp_clip_iters,
    gp_anchor_edges=True,
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
    residual added to it.  With ``gp_residual_term``, a rotation term at a
    significant period of the first fit's residual is added to all later
    fits, if it lowers the residual scatter.  Either way, the post-fit
    residual clipping still catches flares the mask missed.
    ``lc4`` (``detrended_flux``, ``gp_model``) is modified in place.
    """

    t4_gp = lc4.time.value
    f4_gp = lc4.flux.value  # original flux on lc4's filtered grid
    fe4_gp = lc4.flux_err.value  # formal errors on same grid
    flare_mask = np.asarray(flare_mask, dtype=bool)

    # Fit the GP to flux normalised by its median, so that the fit (and the
    # optimiser's convergence) does not depend on the units of the flux.
    # The fitted sigma and jitter are therefore relative to the median.
    norm = np.nanmedian(f4_gp)

    def _fit(mask, extra_periods=()):
        model, _, _ = fit_gp_rotation(
            t4_gp,
            f4_gp / norm,
            fe4_gp / norm,
            period=dominant_period,
            flare_mask=mask,
            optimize_hyperparams=gp_optimize,
            period_tolerance=gp_period_tolerance,
            bin_factor=bin_factor,
            gp_clip_sigma=gp_clip_sigma,
            gp_clip_iters=gp_clip_iters,
            anchor_edges=gp_anchor_edges,
            extra_periods=extra_periods,
            jitter_only=gp_jitter_only,
            jitter_floor=gp_jitter_floor,
        )
        return model

    bin_factor = gp_bin_factor
    gp_model = _fit(flare_mask)
    if gp_bin_max_bias is not None and gp_bin_factor > 1:
        k = _bin_factor_for_curvature(t4_gp, f4_gp / norm, gp_model, flare_mask,
                                      gp_bin_factor, gp_bin_max_bias)
        if k != bin_factor:
            bin_factor = k
            gp_model = _fit(flare_mask)

    def _scatter(model):
        return np.nanstd((f4_gp / norm - model)[~flare_mask])

    extra = []

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
            model2 = _fit(flare_mask, extra + [residual_period])
            if _scatter(model2) < _scatter(gp_model):
                gp_model = model2
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
        gp_model = _fit(flare_mask, extra)

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
    lc4.detrended_flux = (NORMALIZED_BASELINE + (f4_gp - model) / level) * u.dimensionless_unscaled
    lc4.gp_model = model
