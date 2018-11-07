.. AltaiPony documentation master file, created by
   sphinx-quickstart on Wed Sep 26 15:22:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======================================================
Find and analyse flares in K2 light curves
======================================================

**AltaiPony** replaces the original Appaloosa_. **AltaiPony** finds flares in Kepler and K2 photometry. It is optimised to work with K2SC de-trended light curves but fetches ``KeplerTargetPixelFiles`` for ancillary data or if you want to remove systematic and periodic variability (vulgo *de-trending*) yourself. You can use most functionalities from lightkurve_'s ``KeplerLightCurve`` class. **AltaiPony** also runs **k2sc** for de-trending and flare characterization.

.. _user-docs:

.. toctree::
	:caption: Contents
	:maxdepth: 1

	install
   	api/index
	
	
Problems?
^^^^^^^^^

 Often, when something does not work in **AltaiPony** and this documentation is useless, troubleshooting can be done by diving into the extensive **lightkurve** docs_. Otherwise, you can always shoot Ekaterina an email_ or open an issue on GitHub_. Many foreseeable problems will be due to **AltaiPony** itself.
 
 
.. _Appaloosa: https://github.com/jradavenport/appaloosa
.. _lightkurve: https://github.com/KeplerGO/lightkurve
.. _docs: http://docs.lightkurve.org/index.html
.. _email: eilin@aip.de
.. _GitHub: https://github.com/ekaterinailin/AltaiPony

