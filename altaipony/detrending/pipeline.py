"""
UTF-8, Python 3

------------------
AltaiPony
------------------

Ekaterina Ilin, 2023, MIT License

The detrending pipeline: polynomial baseline, Savitzky-Golay passes, and GP
model.
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
    detrender_config=None,
    use_gp=True,
    gp_optimize=True,
    gp_period_tolerance=0.05,
    gp_bin_factor=10,
    gp_clip_sigma=3.0,
    gp_clip_iters=3,
    gp_anchor_edges=True,
    gp_second_rotation=True,
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
        baseline output is fed straight to the GP.  Either way, the baseline
        stage's flare mask is the GP's initial mask.  Defaults to True.
    pad : 3
        Outliers in Savitzky-Golay filter are padded with this
        number of data points. Defaults to 3.
    max_sigma : float
        Outlier rejection threshold in sigma. Defaults to 2.5.
    longdecay : int
        Long decay time for outlier rejection. Defaults to 6.
    maxgap : float
        Gaps longer than ``maxgap`` times the mean cadence split the light
        curve into continuous windows.  Defaults to 10.
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
        light curve to count as strongly periodic, which enables the GP step.
        Defaults to 0.01 (1 % of the median flux).
    period_min_days : float
        Shortest period searched by the Lomb-Scargle periodogram, in days.
        Defaults to 0.1 days.
    detrender_config : baselines.polynomial.DetrendConfig or None
        Configuration of the baseline: cubic fits on grids of 0.4–0.8-day
        windows (plain and shifted by half a window, blended), a 6–36 h
        sinusoid correction, and its own fraction-capped flare mask.  If None,
        its defaults are used.
    use_gp : bool
        If True (and ``is_periodic`` is True and celerite2 is installed),
        follow the baseline fit with a Gaussian Process step.  Flares are
        masked (the baseline stage's flare mask), and a celerite2
        ``RotationTerm`` GP is fitted to the masked *original* flux.  The GP model smoothly bridges the masked
        windows and is then subtracted to produce the final detrended flux,
        so the baseline and Savitzky-Golay results only shape the flare mask.
        Defaults to True.
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
    gp_second_rotation : bool
        If True, search the flux's periodogram for up to three candidate
        second periods, ignoring the GP's rotation period and its first
        harmonic, with the same FAP and amplitude thresholds as the initial
        periodicity check.  The GP is refitted with a second rotation term at
        each, and the fit with the lowest residual scatter is kept if it beats
        the single term.  For stars with two unrelated periodic signals.
        Defaults to True.
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
    matched_filter_fwhm_grid : sequence of float
        Flare FWHMs (days) the matched filter templates against.  Defaults to
        ``(0.02, 0.05, 0.1, 0.15)``.

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

    # --- Periodicity check ------------------------------------------------
    # Run a Lomb-Scargle periodogram on the raw flux.  If a strong periodic
    # signal is found (low FAP *and* large amplitude), the GP step models the
    # variability; its period starts from the periodogram peak.
    period_max_days = (time[-1] - time[0]) / 2.0

    is_periodic, dominant_period, rel_amplitude, fap = detect_strong_periodicity(
        time,
        flux,
        fap_threshold=periodicity_fap_threshold,
        amplitude_threshold=periodicity_amplitude_threshold,
        period_min=period_min_days,
        period_max=period_max_days,
    )

    # --- Baseline: segmented polynomial ------------------------------------
    m2flux, _, best_params = fit_lightcurve_detrender(
        time, flux, lc.flux_err.value, cont_windows, config=detrender_config
    )
    if is_periodic:
        best_params["dominant_period"] = dominant_period

    # The baseline stage's flare mask is the GP's initial flare mask.
    baseline_flare_mask = best_params.pop("ld_final_flare_mask")

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

        # Restore original flux and errors from the columns (now filtered to
        # lc4's length).  The Savitzky-Golay passes divide flux_err by their
        # trend, so the GP would otherwise get flux in input units but errors
        # in relative units.
        lc4.flux = lc4["original_flux"] * u.electron / u.s
        lc4.flux_err = lc4["original_flux_err"] * u.electron / u.s

        # Clean up the temporary column
        lc4.remove_column("original_flux")
        lc.flux = lc["original_flux"] * u.electron / u.s
        lc.flux_err = lc["original_flux_err"] * u.electron / u.s

        # find median value
        lc4.find_iterative_median()
    else:
        # No Savitzky-Golay: the baseline output IS the detrended flux fed to
        # the GP step.  Keeping the grid untouched also means the baseline's
        # flare mask stays aligned with the GP's cadences.
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
        flare_mask=baseline_flare_mask,
        gp_optimize=gp_optimize,
        gp_period_tolerance=gp_period_tolerance,
        gp_bin_factor=gp_bin_factor,
        gp_clip_sigma=gp_clip_sigma,
        gp_clip_iters=gp_clip_iters,
        gp_anchor_edges=gp_anchor_edges,
        gp_second_rotation=gp_second_rotation,
        periodicity_fap_threshold=periodicity_fap_threshold,
        periodicity_amplitude_threshold=periodicity_amplitude_threshold,
        matched_filter_flares=matched_filter_flares,
        matched_filter_snr=matched_filter_snr,
        matched_filter_fwhm_grid=matched_filter_fwhm_grid,
    )

    if "gp" not in best_params:
        # No GP model (non-periodic star, GP disabled, or GP fit failed): the
        # detrended flux is still at the input flux level.  Express it as
        # relative flux too, scaled by its global iterative median.
        det = np.asarray(getattr(lc4.detrended_flux, "value", lc4.detrended_flux), dtype=float)
        level = _find_iterative_median(det, None)[0]
        lc4.detrended_flux = (NORMALIZED_BASELINE * det / level) * u.electron / u.s

    # Iterative median of the final detrended flux.
    lc4.find_iterative_median()

    return lc4


def _second_period_candidates(time, flux, periods, fap_threshold, amplitude_threshold,
                              n_candidates=3, period_min=0.1, harmonic_tolerance=0.1):
    """Candidate periods for a second periodic signal in ``flux``, strongest
    first.

    ``flux`` is relative flux (median ~1).  The Lomb-Scargle periodogram is
    searched from ``period_min`` to half the time span, ignoring frequencies
    within ``harmonic_tolerance`` of ``1 / P`` and ``2 / P`` for each ``P`` in
    ``periods`` (the rotation period from the periodogram and as fitted by the
    GP): a rotation term at ``P`` already covers both.  Up to ``n_candidates``
    peaks are returned, each more than ``harmonic_tolerance`` away in
    frequency from the stronger ones and passing the same FAP and
    semi-amplitude thresholds as the initial periodicity check
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
        theta = ls.model_parameters(frequency[peak])
        if np.hypot(theta[1], theta[2]) / abs(np.median(f)) > amplitude_threshold:
            candidates.append(1.0 / frequency[peak])
        excluded |= np.abs(frequency / frequency[peak] - 1) < harmonic_tolerance
    return candidates


def _apply_gp_step(
    lc4,
    best_params,
    *,
    use_gp,
    is_periodic,
    dominant_period,
    flare_mask,
    gp_optimize,
    gp_period_tolerance,
    gp_bin_factor,
    gp_clip_sigma,
    gp_clip_iters,
    gp_anchor_edges=True,
    gp_second_rotation=True,
    periodicity_fap_threshold=1e-3,
    periodicity_amplitude_threshold=0.01,
    matched_filter_flares=True,
    matched_filter_snr=5.0,
    matched_filter_fwhm_grid=(0.02, 0.05, 0.1, 0.15),
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
    only when ``use_gp`` and
    ``is_periodic`` are True and celerite2 is installed; otherwise ``lc4`` is
    left untouched.  ``lc4`` (``detrended_flux``, ``gp_model``) and
    ``best_params`` are modified in place.
    """
    if not (use_gp and is_periodic and _CELERITE2_AVAILABLE):
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

        def _fit(mask, second_period=None):
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
                second_period=second_period,
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
        second_period = None
        if gp_second_rotation:
            def _scatter(model):
                return np.nanstd((f4_gp / norm - model)[~flare_mask])

            best = _scatter(gp_model)
            for candidate in _second_period_candidates(
                t4_gp[~flare_mask],
                (f4_gp / norm)[~flare_mask],
                (dominant_period, gp_params["period"]),
                fap_threshold=periodicity_fap_threshold,
                amplitude_threshold=periodicity_amplitude_threshold,
            ):
                model2, params2 = _fit(flare_mask, candidate)
                if _scatter(model2) < best:
                    best = _scatter(model2)
                    gp_model, gp_params, second_period = model2, params2, candidate

        # Second pass: per-cadence masks catch narrow flares but miss wide,
        # moderate-SNR ones (each cadence sits under the cut while the
        # integrated flux is large); the matched filter's coherent sum
        # recovers them, so the GP does not train on and absorb them.  It
        # runs on the first GP's residual, not on the baseline residual:
        # there, rotation the baseline could not follow would be masked as
        # flares, and the GP would never see the maxima it has to fit.
        if matched_filter_flares:
            mf_mask = _matched_filter_mask(
                t4_gp,
                f4_gp / norm - gp_model,
                noise=None,
                snr_threshold=matched_filter_snr,
                fwhm_grid=matched_filter_fwhm_grid,
            )
            if (mf_mask & ~flare_mask).any():
                flare_mask = flare_mask | mf_mask
                gp_model, gp_params = _fit(flare_mask, second_period)

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
