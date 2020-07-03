.. AltaiPony documentation master file, created by
   sphinx-quickstart on Wed Sep 26 15:22:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======================================================
Flare science in Kepler, K2 and TESS light curves
======================================================

**AltaiPony** finds flares in Kepler, K2, and TESS photometry. You can create a ``FlareLightCurve``, remove systematics and photometric variability, and search it for flares. 

K2 light curves are best de-trended using **k2sc_**. For TESS and Kepler we employ a Savitzky-Golay filter. You can use most methods from lightkurve_'s ``KeplerLightCurve`` and ``TessLightCurve`` classes. **AltaiPony** can be used to inject and recover synthetic flare signatures. Jump to the core class is here_. **AltaiPony** replaces the original Appaloosa_ (Davenport 2016). 

.. _user-docs:

.. toctree::
	:caption: AltaiPony
	:maxdepth: 2

	install
        tutorials/index
   	api/index
	
	
Problems?
^^^^^^^^^

 Often, when something does not work in **AltaiPony** and this documentation is useless, troubleshooting can be done by diving into the extensive **lightkurve** docs_. Otherwise, you can always shoot Ekaterina an email_ or open an issue on GitHub_. Many foreseeable problems will be due to **AltaiPony** itself.
 
.. _k2sc: https://github.com/OxES/k2sc 
.. _Appaloosa: https://github.com/jradavenport/appaloosa
.. _lightkurve: https://github.com/KeplerGO/lightkurve
.. _docs: http://docs.lightkurve.org/index.html
.. _email: eilin@aip.de
.. _GitHub: https://github.com/ekaterinailin/AltaiPony
.. _here: https://altaipony.readthedocs.io/en/latest/api/altaipony.flarelc.FlareLightCurve.html
