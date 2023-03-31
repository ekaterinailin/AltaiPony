|docs-badge| |license-badge| |joss-badge| |zenodo-badge|


.. |joss-badge| image:: https://joss.theoj.org/papers/10.21105/joss.02845/status.svg
   :target: https://doi.org/10.21105/joss.02845

..  |zenodo-badge| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5040830.svg
                    :target: https://doi.org/10.5281/zenodo.5040830

.. |docs-badge| image:: https://readthedocs.org/projects/altaipony/badge/?version=latest
	      :target: https://altaipony.readthedocs.io/en/latest/?badge=latest
	      :alt: Documentation Status
	      
.. |license-badge|  image:: https://img.shields.io/github/license/mashape/apistatus.svg   
		    :target: https://github.com/ekaterinailin/AltaiPony/blob/master/LICENSE 
		    :alt: GitHub	

.. image:: logo.png
   :height: 100px
   :width: 100px
   :alt: Logo credit: Elizaveta Ilin, 2018

AltaiPony
=========

De-trend light curves from Kepler, K2, and TESS missions, and search them for flares. Inject and recover synthetic flares to account for de-trending and noise loss in flare energy and determine energy-dependent recovery probability for every flare candidate. Uses the ``K2SC`` and ``lightkurve`` under the cover, as well as ``pandas``, ``numpy``, ``pytest``, ``astropy`` and more.

Find the documentation at altaipony.readthedocs.io_

Installation
^^^^^^^^^^^^^

Use pip to install AltaiPony

>>> pip install altaipony


Or install directly from the repository:

>>> git clone https://github.com/ekaterinailin/AltaiPony.git
>>> cd AltaiPony
>>> python setup.py install



Getting Started
^^^^^^^^^^^^^^^^

See this notebook_ for an easy introduction, also docs_.


Problems?
^^^^^^^^^

 Often, when something does not work in **AltaiPony**, and this documentation is useless, troubleshooting can be done by diving into the extensive **lightkurve** docs_. Otherwise, you can always shoot Ekaterina an email_ or directly open an issue on GitHub_. Many foreseeable problems will be due to bugs in **AltaiPony** or bad instructions on this website.


Contribute to AltaiPony
^^^^^^^^^^^^^^^^^^^^^^^

**AltaiPony** is under active development on Github_. If you use **AltaiPony** in your research and find yourself missing a functionality, I recommend opening an issue on GitHub_ or shooting Ekaterina an email_. Please do either of the two before you open a pull request. This may save you a lot of development time.

How to cite this work
^^^^^^^^^^^^^^^^^^^^^

If you end up using this package for your science, please cite Ilin et al. (2021) [a]_ and Davenport (2016) [b]_.

Please also cite `lightkurve` as indicated in their docs [1]_. 

Depending on the methods you use, you may also want to cite 

  - Maschberger and Kroupa (2009) [2]_ (MMLE power law fit)
  - Wheatland (2004) [3]_ (MCMC power law fit)
  - Aigrain et al. (2016) [4]_ and their softwar [5]_ (K2SC de-trending)


.. [a] Ekaterina Ilin, Sarah J. Schmidt, Katja Poppenhäger, James R. A. Davenport, Martti H. Kristiansen, Mark Omohundro (2021). "Flares in Open Clusters with K2. II. Pleiades, Hyades, Praesepe, Ruprecht 147, and M67" Astronomy & Astrophysics, Volume 645, id.A42, 25 pp.  	https://doi.org/10.1051/0004-6361/202039198 

.. [b] James R. A. Davenport "The Kepler Catalog of Stellar Flares" The Astrophysical Journal, Volume 829, Issue 1, article id. 23, 12 pp. (2016). https://doi.org/10.3847/0004-637X/829/1/23

.. [1] https://docs.lightkurve.org/about/citing.html

.. [2] Thomas Maschberger, Pavel Kroupa, "Estimators for the exponent and upper limit, and goodness-of-fit tests for (truncated) power-law distributions" Monthly Notices of the Royal Astronomical Society, Volume 395, Issue 2, May 2009, Pages 931–942, https://doi.org/10.1111/j.1365-2966.2009.14577.x

.. [3] Wheatland, Michael S. "A Bayesian approach to solar flare prediction." The Astrophysical Journal 609.2 (2004): 1134. https://doi.org/10.1086/421261

.. [4] Aigrain, Suzanne; Parviainen, Hannu; Pope, Benjamin "K2SC: flexible systematics correction and detrending of K2 light curves using Gaussian process regression" Monthly Notices of the Royal Astronomical Society, Volume 459, Issue 3, p.2408-2419 https://doi.org/10.1093/mnras/stw706

.. [5] Aigrain, Suzanne; Parviainen, Hannu; Pope, Benjamin "K2SC: K2 Systematics Correction." Astrophysics Source Code Library, record ascl:1605.012 https://ui.adsabs.harvard.edu/abs/2016ascl.soft05012A/abstract


.. _Appaloosa: https://github.com/jradavenport/appaloosa/
.. _altaipony.readthedocs.io: https://altaipony.readthedocs.io/en/latest/
.. _notebook: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/Getting_Started.ipynb
.. _docs: https://altaipony.readthedocs.io/en/latest/
.. _Github: https://github.com/ekaterinailin/AltaiPony/issues/new
.. _email: eilin@aip.de
