|ci-badge| |docs-badge| |license-badge| |requirements-badge|


.. |ci-badge| image:: https://travis-ci.org/ekaterinailin/AltaiPony.svg?branch=master
              :target: https://travis-ci.org/ekaterinailin/AltaiPony

.. |docs-badge| image:: https://readthedocs.org/projects/altaipony/badge/?version=latest
	      :target: https://altaipony.readthedocs.io/en/latest/?badge=latest
	      :alt: Documentation Status
	      
.. |license-badge|  image:: https://img.shields.io/github/license/mashape/apistatus.svg   
		    :target: https://github.com/ekaterinailin/AltaiPony/blob/master/LICENSE 
		    :alt: GitHub	
.. |requirements-badge| image:: https://requires.io/github/ekaterinailin/AltaiPony/requirements.svg?branch=master
                       :target: https://requires.io/github/ekaterinailin/AltaiPony/requirements/?branch=master
                       :alt: Requirements Status


.. image:: logo.png
   :height: 100px
   :width: 100px
   :alt: Credit: Elizaveta Ilin, 2018

AltaiPony
=========

De-trend light curves from Kepler, K2, and TESS missions, and search them for flares. Inject and recover synthetic flares to account for de-trending and noise loss in flare energy and determine energy-dependent recovery probability for every flare candidate. Uses the ``K2SC`` and ``lightkurve`` under the cover, as well as ``pandas``, ``numpy``, ``pytest``, ``astropy`` and more.

Find the documentation at altaipony.readthedocs.io_

Installation
^^^^^^^^^^^^^

You need to install a fork of ``K2SC`` first, then clone this repository and install the package:

>>> git clone https://github.com/ekaterinailin/k2sc.git
>>> cd k2sc
>>> python3 setup.py install
>>> cd ..
>>> git clone https://github.com/ekaterinailin/AltaiPony.git
>>> cd AltaiPony
>>> python3 setup.py install


Getting Started
^^^^^^^^^^^^^^^^

See this notebook_ for an easy introduction, also docs_.

.. _Appaloosa: https://github.com/jradavenport/appaloosa/
.. _altaipony.readthedocs.io: https://altaipony.readthedocs.io/en/latest/
.. _notebook: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/Getting_Started.ipynb
.. _docs: https://altaipony.readthedocs.io/en/latest/
