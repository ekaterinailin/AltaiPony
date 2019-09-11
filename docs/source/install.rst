Quickstart
=======================================

Installation
^^^^^^^^^^^^

::
    
    git clone https://github.com/ekaterinailin/AltaiPony.git
    cd AltaiPony
    python setup.py install

You will also need to install K2SC (Aigrain et al. 2016), ideally from this fork_ of the original version.


Getting Started
^^^^^^^^^^^^^^^^

Working with Kepler and TESS light curves is similar, K2 light curves need extra love, so we treat them separately. For each mission, there is a notebook that will guide you through the basic applications.

Kepler Light Curves
...................

If you are working with KeplerLightCurve objects from the original Kepler mission, you may want to try this_ notebook. This shows how to fetch a light curve from MAST, de-trend it with a Savitzky-Golay filter, and find some flares.

TESS Light Curves
...................

If you are working with TessLightCurve objects from the TESS mission, you may want to try this other_ notebook. In this one, you can also try out the injection recovery feature of **AltaPony**, and obtain recovery probability and a corrected equivalent duration (aka luminosity independent flare energy ED).

K2 Light Curves
...................

If you are working with KeplerLightCurve objects from the K2 mission, you may want to try this last notebook_. It will run GP de-trending and find flares in a typical K2 long cadence light curve with heavy instrumental artifacts.


.. _fork: https://github.com/ekaterinailin/k2sc
.. _notebook: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/Getting_Started.ipynb
.. _this: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/Kepler_Light_Curves_With_Flares.ipynb
.. _other: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/TESS_Light_Curves_With_Flares.ipynb
