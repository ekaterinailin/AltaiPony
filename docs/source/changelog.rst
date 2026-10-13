Changelog (starting 04-2023)
=======================================

13-10-2026
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``find_flares`` uses a new default detection rule. Instead of requiring 3 consecutive points above 3 sigma, it flags windows of 3 to 10 consecutive points whose combined probability of being noise is below that of 3 points at 3 sigma, corrected for residuals that are correlated over minutes to hours. Fragments of one flare are merged (``merge_sigma=0.25``) and detections are extended over adjacent points above 1 sigma (``extend_sigma=1.0``). Pass ``max_window=None, merge_sigma=None, extend_sigma=None`` for the previous behaviour.
* ``custom_detrending`` is a new pipeline: a polynomial baseline, then a Gaussian process with a rotation kernel, iteratively masking flares with a matched filter.
* Removed the Savitzky-Golay de-trending mode (``detrend("savgol")``); use ``detrend("custom", func=custom_detrending)``.
* ``sample_flare_recovery`` counts an injected flare as recovered if its peak lies inside a detection (``fakeflares.match_flares``), and its results have a ``recovered`` column.

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
