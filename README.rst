|ci-badge| |docs-badge| |license-badge|


.. |ci-badge| image:: https://travis-ci.org/ekaterinailin/AltaiPony.svg?branch=master
              :target: https://travis-ci.org/ekaterinailin/AltaiPony

.. |docs-badge| image:: https://readthedocs.org/projects/altaipony/badge/?version=latest
	      :target: https://altaipony.readthedocs.io/en/latest/?badge=latest
	      :alt: Documentation Status
	      
	      
.. |license-badge|  image:: https://img.shields.io/github/license/mashape/apistatus.svg   
		    :target: https://github.com/ekaterinailin/AltaiPony/blob/master/LICENSE 
		    :alt: GitHub	      

AltaiPony
=========

An improved and lean version of Appaloosa_ w/o extensive I/O, but with de-trending but using ``K2SC`` and ``lightkurve``.
Find the documentation at altaipony.readthedocs.io_

Installation
^^^^^^^^^^^^^
::
    
    git clone https://github.com/ekaterinailin/AltaiPony.git
    cd AltaiPony
    python setup.py install


Getting Started
^^^^^^^^^^^^^^^^

See this notebook_ for an easy introduction.

Structure 
^^^^^^^^^^

``flarelc.py``

Contains the core class - ``FlareLightCurve``. Flare finding, characterization and many utils, such as finding gaps are implemented as methods.

``lcio.py`` and ``mast.py``

Everything related to reading in ``KeplerTargetPixelFiles``, ``KeplerLightCurveFiles``, and ``K2SC`` light curves is dealt with here. All these data formats can be fetched from MAST. 

``altai.py``

All the core flare finding functions live here.

``fakeflares.py``

Includes:

- semi-empirical flare model
- injection/recovery procedure for synthetic flares.
- flare characterization functions that use the results from injection/recovery

``utils.py``

Internal helper function dump.

.. _Appaloosa: https://github.com/jradavenport/appaloosa/
.. _altaipony.readthedocs.io: https://altaipony.readthedocs.io/en/latest/
.. _notebook: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/Getting_Started.ipynb
