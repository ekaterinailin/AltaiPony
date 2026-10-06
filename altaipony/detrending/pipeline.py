"""
UTF-8, Python 3

------------------
AltaiPony
------------------

Ekaterina Ilin, 2023, MIT License

The detrending pipeline: baseline fit, Savitzky-Golay passes, and GP model.
"""

import logging

import numpy as np
import astropy.units as u

from ..altai import _find_iterative_median
from .baselines.multisine import _segment_gaps, fit_multisine
from .baselines.polynomial import fit_lightcurve_detrender
from .baselines.spline import fit_spline
from .gp import _CELERITE2_AVAILABLE, _identify_flare_mask, fit_gp_rotation
from .matchedfilter import matched_filter_flare_mask as _matched_filter_mask
from .periodicity import detect_strong_periodicity

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
