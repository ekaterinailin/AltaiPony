Quickstart
=======================================

Installation
^^^^^^^^^^^^

::
    
    git clone https://github.com/ekaterinailin/AltaiPony.git
    cd AltaiPony
    python setup.py install

You will also need to install ***k2sc* (Aigrain et al. 2016) from this fork_ of the original version.


Getting Started
^^^^^^^^^^^^^^^^

Working with Kepler and TESS light curves is similar, K2 light curves need extra love, so we treat them separately. For each mission, there is a notebook that will guide you through the basic applications.

Kepler Light Curves
...................

If you are working with KeplerLightCurve objects from the original Kepler mission, try this_ notebook. This shows how to fetch a light curve from MAST, de-trend it with a Savitzky-Golay filter, and find some flares.

TESS Light Curves
...................

If you are working with TessLightCurve objects from the TESS mission, you may want to try this other_ notebook instead. In this one, you can also test the injection recovery feature of **AltaPony**, and obtain recovery probability and a corrected equivalent duration (ED, aka luminosity independent flare energy).

K2 Light Curves
...................

If you are working with KeplerLightCurve objects from the K2 mission, this last notebook_ is for you. It will run **k2sc**'s GP de-trending and find flares in a typical K2 long cadence light curve despite its heavy instrumental artifacts.


.. _fork: https://github.com/ekaterinailin/k2sc
.. _notebook: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/Getting_Started.ipynb
.. _this: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/Kepler_Light_Curves_With_Flares.ipynb
.. _other: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/TESS_Light_Curves_With_Flares.ipynb
