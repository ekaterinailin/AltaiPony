|ci-badge| |docs-badge|


.. |ci-badge| image:: https://travis-ci.org/ekaterinailin/AltaiPony.svg?branch=master
              :target: https://travis-ci.org/ekaterinailin/AltaiPony

.. |docs-badge| image:: https://readthedocs.org/projects/altaipony/badge/?version=latest
	      :target: https://altaipony.readthedocs.io/en/latest/?badge=latest
	      :alt: Documentation Status

AltaiPony
=========

An improved and lean version of Appaloosa_ w/o extensive I/O, but with de-trending but using ``K2SC`` and ``lightkurve``.
The documenation (work in progress) are at altaipony.readthedocs.io_

Minimum How-To
^^^^^^^^^^^^^
::
    
    git clone https://github.com/ekaterinailin/AltaiPony.git
    cd AltaiPony
    python setup.py install


Structure 
^^^^^^^^^^

``flarelc.py``

A lightcurve class with its constructor that inherits from ```k2sc``` and/or ```lightkurve```.
If raw LC is read in - run detrend.py 
Convenience function: check if K2SC has de-trended LC already available.

``detrend.py``

Do K2SC detrending is stitched in here: use ``standalone.py``  

``findflares.py``

Split LC into continuous observation chunks.
Apply thresholds to detect candidates.

``fakeflares.py``

Includes:

- semi-empirical flare model
- injection/recovery procedure for synthetic flares.

``analysis.py``

- calculates ED, duration, amplitude, uncertainties, observation times
- possibly other stats about the original flares 
- (correlations with other astrophysical photometric varibility)
- flare energy correction factor
- flare recovery probability

``altai.py``

Main wrapper that

- 1a. takes a K2 (or TESS) ID or a path to a .fits or TPF.gz file
- 2a. creates a light curve using lightkurve.
- (2b. de-trends light curve using K2SC
- 3a. find flare candidates
- (3b. runs fake flare injection/recovery)
- (3c. Calculates flare parameters, corrects ED and returns recovery probability)

.. _Appaloosa: https://github.com/jradavenport/appaloosa/
.. _altaipony.readthedocs.io: https://altaipony.readthedocs.io/en/latest/
