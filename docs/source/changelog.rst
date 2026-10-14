Changelog (starting 04-2023)
=======================================

13-10-2026
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``find_flares`` uses a new default detection rule. Instead of requiring 3 consecutive points above 3 sigma, it flags windows of 3 to 10 consecutive points whose combined probability of being noise is below that of 3 points at 3 sigma, corrected for residuals that are correlated over minutes to hours. Fragments of one flare are merged (``merge_sigma=0.25``) and detections are extended over adjacent points above 1 sigma (``extend_sigma=1.0``). Pass ``max_window=None, merge_sigma=None, extend_sigma=None`` for the previous behaviour.
* ``custom_detrending`` is a new pipeline: a polynomial baseline, then a Gaussian process with a rotation kernel, iteratively masking flares with a matched filter. It works on the actual time stamps (missing cadences are not interpolated) and leaves its input light curve unchanged. It also estimates the noise of the detrended flux (``detrended_flux_err``), so that ``flc.detrend()`` can be followed directly by ``flc.find_flares()``.
* ``FlareLightCurve.detrend()`` uses ``custom_detrending`` by default and passes keyword arguments on to it (``flc.detrend(use_gp=False)``); pass ``func=`` for your own function. The ``mode`` argument is deprecated: ``detrend("custom", func=...)`` still works with a warning, and ``sample_flare_recovery(mode=...)`` is ignored.
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
