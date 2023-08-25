Changelog (starting 04-2023)
=======================================

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
