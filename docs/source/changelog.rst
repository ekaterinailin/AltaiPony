Changelog (starting 04-2023)
=======================================

15-10-2026
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``custom_detrending`` always fits the GP; the ``use_gp`` option is removed. The polynomial baseline only provides the GP's initial flare mask, and it fits once instead of twice (the second fit made no difference in injection-recovery). Light curves too short for a periodogram raise a ``ValueError``, and a failed GP fit raises its error instead of silently returning the polynomial baseline. In ``DetrendConfig``, ``second_pass_sigma`` is now ``final_mask_sigma``.
* The polynomial baseline's preliminary flare mask thresholds at 5 times the residual's measured noise. It used to add ``flux_err`` in quadrature, which counted the white noise twice (an effective 7 sigma with correct errors, far more with overestimated ones). In injection-recovery on 1000 synthetic light curves: 7170 instead of 7165 flares recovered, 13 instead of 14 false positives outside injected flares.
* ``utils.robust_sigma`` (with optional clipping, ``n_clip``), ``utils.clipped_stats`` (median and robust sigma after ``sigma_clip``) and ``utils.sigma_clipped_std`` are the shared noise estimates; the copies in the GP, the matched filter and the polynomial baseline use them.

14-10-2026
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The GP in ``custom_detrending`` weights its training data by a white-noise jitter fitted to the data, and ignores ``flux_err``. The jitter is estimated from the scatter between neighbouring cadences and kept at least 3 times that estimate (``gp_jitter_floor=3``). Before, the jitter was added to ``flux_err`` and could not fall below 5 % of the flux's standard deviation, modulation included, so on quiet but strongly spotted stars the GP assumed ~10 times the real noise, sagged at segment edges and produced spurious flares there. In injection-recovery on 1000 synthetic light curves: 19 instead of 29 false positives outside injected flares, 7172 instead of 7166 of 7542 flares recovered. ``gp_jitter_only=False`` restores the previous noise model.
* The GP in ``custom_detrending`` bins its training data less where binning by 10 would distort the modulation (``gp_bin_max_bias=0.5``): after the first fit, it uses the largest bin factor of 10, 5, 3, 2 or 1 at which binning the fitted model shifts the bins by less than half their noise. On fast, quiet rotators 20-min bins flatten the modulation's sharp peaks, and the GP undercut them. This applies to about 2 % of the synthetic light curves: 13 instead of 19 false positives outside injected flares, 7165 instead of 7172 flares recovered. ``gp_bin_max_bias=None`` always bins by ``gp_bin_factor``.
* Together, compared with 13-10-2026: 13 instead of 29 false positives outside injected flares and 13 instead of 16 on the tails of bright flares, in 1000 synthetic light curves with 7542 injected flares; 7165 instead of 7166 recovered.
* Removed the experimental options ``gp_clip_two_sided``, ``gp_edge_pad`` and ``matched_filter_edge_days`` of ``custom_detrending``, which did not improve injection-recovery.
* ``custom_detrending`` returns ``detrended_flux`` and ``detrended_flux_err`` as dimensionless relative flux (baseline 1.0), like lightkurve's ``normalize()``. They were labelled e-/s, so adding or comparing plain numbers failed.

13-10-2026
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``find_flares`` uses a new default detection rule. Instead of requiring 3 consecutive points above 3 sigma, it flags windows of 3 to 10 consecutive points whose combined probability of being noise is below that of 3 points at 3 sigma, corrected for residuals that are correlated over minutes to hours. Fragments of one flare are merged (``merge_sigma=0.25``) and detections are extended over adjacent points above 1 sigma (``extend_sigma=1.0``). Pass ``max_window=None, merge_sigma=None, extend_sigma=None`` for the previous behaviour.
* ``custom_detrending`` is a new pipeline: a polynomial baseline (which masks only points above 5 sigma before the GP, so that more of the modulation's peaks stay in the GP's training data), then a Gaussian process with a rotation kernel, iteratively masking flares with a matched filter. It works on the actual time stamps (missing cadences are not interpolated) and leaves its input light curve unchanged. It also estimates the noise of the detrended flux (``detrended_flux_err``), so that ``flc.detrend()`` can be followed directly by ``flc.find_flares()``.
* ``FlareLightCurve.detrend()`` uses ``custom_detrending`` by default and passes keyword arguments on to it (e.g. ``flc.detrend(gp_bin_factor=5)``); pass ``func=`` for your own function. The ``mode`` argument is deprecated: ``detrend("custom", func=...)`` still works with a warning, and ``sample_flare_recovery(mode=...)`` is ignored.
* Removed the Savitzky-Golay de-trending mode (``detrend("savgol")``).
* Removed ``FlareLightCurve.get_saturation`` and the remaining K2SC dependencies (``k2sc``, ``george``). ``celerite2`` (for the GP de-trending) and ``tqdm`` are now dependencies.
* ``sample_flare_recovery`` shows a ``tqdm`` progress bar with the number of fake flares injected and recovered so far (``progress=False`` turns it off), replacing ``progressbar2``.
* ``sample_flare_recovery`` counts an injected flare as recovered if its peak lies inside a detection (``fakeflares.match_flares``), and its results have a ``recovered`` column.
* ``FlareLightCurve`` methods that analyse the light curve now change it in place and return it: ``find_flares`` fills ``flc.flares``, ``detrend`` adds ``detrended_flux`` and the other columns of the detrending function, and ``find_cont_windows``, ``mark_flagged_flares``, ``characterize_flares``, ``sample_flare_recovery`` and ``fit_flares`` update the light curve's attributes. ``flc = flc.find_flares()`` keeps working. ``inject_fake_flares`` still returns a new light curve.
* Removed ``FlareLightCurve.interpolate_missing_cadences``; missing cadences are no longer filled in.
* ``FlareLightCurve.find_flares`` always searches anew and replaces the flare table; before, it silently returned the previous result when flares had already been found. New options ``maxgap`` and ``median_iterations``.
* ``to_flare_lightcurve`` returns a converted copy and leaves the input light curve unchanged. It no longer adds empty ``detrended_flux`` columns.
* Reading ``detrended_flux``, ``detrended_flux_err``, ``gp_model`` or ``it_med`` before they are computed returns ``None`` instead of creating a column of NaNs; ``find_flares`` and ``find_iterative_median`` then say to call ``detrend()`` first. ``flares``, ``fake_flares`` and ``cont_windows`` are empty or ``None`` until filled, without changing the light curve.
* ``read_from_fits`` reads BTJD and BKJD time columns correctly (they were read as JD), and only warns if ``TARGETID`` is missing.
* Importing AltaiPony no longer configures the root logger, which made other libraries print their INFO messages (e.g. arviz's "arviz_base not installed"). AltaiPony's own messages still appear; silence them with ``logging.getLogger("altaipony").setLevel(logging.WARNING)``.
* Fixed: ``fit_flares`` failed (``t0 must be strictly less than t1``) when the flare table was not in time order, e.g. after sorting by energy; flares are now grouped in time order, and its results are in time order.
* Fixed: ``repr`` of a ``FlareLightCurve`` without a known cadence, and the file names used by ``sample_flare_recovery(save=True, save_lc_to_file=True)``.

25-08-2023
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Removed K2SC tests for reading files, as it is not maintained anymore. The functionality is kept to keep a version of the code alive, but the tests are removed.
* Updated requirements for astropy to be version 5.2.2. (latest version before 5.3, which currently breaks some lightkurve functionality, in particular periodograms). 


15-06-2023
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Added more documentation on how to use keyword arguments in the ``custom_detrending`` function.
* Removed tests from for K2SC de-trending, as it is not maintainted anymore. The functionality is kept to keep a version of the code alive, but the tests are removed.

03-05-2023 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Added how to fix disabled plotting when AltaiPony is imported to the FAQs.

28-04-2023 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Discontinued support for Python 3.7, now we only support 3.8 and 3.9
