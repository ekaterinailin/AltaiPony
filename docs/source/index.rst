.. AltaiPony documentation master file, created by
   sphinx-quickstart on Wed Sep 26 15:22:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======================================================
Flare science in Kepler, K2 and TESS light curves
======================================================

**AltaiPony** is a toolbox for statistical flare studies in photometric time series from Kepler, K2, and TESS, including flare search and characterization, injection/recovery diagnostics, and statistical analysis of flare frequency distributions along with extensive documentation and tutorials.

Jump to `Getting Started`_ to get an overview!

.. K2 light curves are best de-trended using **K2SC** [5]_. For TESS and Kepler we employ a Savitzky-Golay filter. You can use most methods from lightkurve_'s ``KeplerLightCurve`` and ``TessLightCurve`` classes. **AltaiPony** can be used to inject and recover synthetic flare signatures. Jump to the core class here_. **AltaiPony** features the flare candidate definition from **Appaloosa** [b]_. 

.. _user-docs:

.. toctree::
	:caption: AltaiPony
	:maxdepth: 2

	install
        tutorials/index
   	api/index
	
	
Problems?
^^^^^^^^^

 Often, when something does not work in **AltaiPony**, and this documentation is useless, troubleshooting can be done by diving into the extensive **lightkurve** docs_. Otherwise, you can always shoot Ekaterina an email_ or directly open an issue on GitHub_. Many foreseeable problems will be due to bugs in **AltaiPony** or bad instructions on this website.


How to cite this work
^^^^^^^^^^^^^^^^^^^^^

If you end up using this package for your science, please cite Ilin et al. (2020) [a]_ and Davenport (2016) [b]_.

Please also cite **lightkurve** as indicated in their docs [1]_. 

Depending on the methods you use, you may also want to cite 

  - Maschberger and Kroupa (2009) [2]_ (MMLE power law fit)
  - Wheatland (2004) [3]_ (MCMC power law fit)
  - Aigrain et al. (2016) [4]_ and their software [5]_ (**K2SC** de-trending)


.. [a] Ekaterina Ilin, Sarah J. Schmidt, Katja Poppenhäger, James R. A. Davenport, Martti H. Kristiansen, Mark Omohundro (2020). "Flares in Open Clusters with K2. II. Pleiades, Hyades, Praesepe, Ruprecht 147, and M67" https://arxiv.org/abs/2010.05576

.. [b] James R. A. Davenport "The Kepler Catalog of Stellar Flares" The Astrophysical Journal, Volume 829, Issue 1, article id. 23, 12 pp. (2016). https://doi.org/10.3847/0004-637X/829/1/23

.. [1] https://docs.lightkurve.org/about/citing.html

.. [2] Thomas Maschberger, Pavel Kroupa, "Estimators for the exponent and upper limit, and goodness-of-fit tests for (truncated) power-law distributions" Monthly Notices of the Royal Astronomical Society, Volume 395, Issue 2, May 2009, Pages 931–942, https://doi.org/10.1111/j.1365-2966.2009.14577.x

.. [3] Wheatland, Michael S. "A Bayesian approach to solar flare prediction." The Astrophysical Journal 609.2 (2004): 1134. https://doi.org/10.1086/421261

.. [4] Aigrain, Suzanne; Parviainen, Hannu; Pope, Benjamin "K2SC: flexible systematics correction and detrending of K2 light curves using Gaussian process regression" Monthly Notices of the Royal Astronomical Society, Volume 459, Issue 3, p.2408-2419 https://doi.org/10.1093/mnras/stw706

.. [5] Aigrain, Suzanne; Parviainen, Hannu; Pope, Benjamin "K2SC: K2 Systematics Correction." Astrophysics Source Code Library, record ascl:1605.012 https://ui.adsabs.harvard.edu/abs/2016ascl.soft05012A/abstract


.. _k2sc: https://github.com/OxES/k2sc 
.. _Appaloosa: https://github.com/jradavenport/appaloosa
.. _lightkurve: https://github.com/KeplerGO/lightkurve
.. _docs: http://docs.lightkurve.org/index.html
.. _email: eilin@aip.de
.. _GitHub: https://github.com/ekaterinailin/AltaiPony
.. _here: https://altaipony.readthedocs.io/en/latest/api/altaipony.flarelc.FlareLightCurve.html
.. _Quickstart: https://altaipony.readthedocs.io/en/latest/install.html
.. _Getting Started: https://altaipony.readthedocs.io/en/latest/install.html#getting-started
