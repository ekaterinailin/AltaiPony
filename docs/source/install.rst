Quickstart
=======================================

Installation
^^^^^^^^^^^^

::
    
    git clone https://github.com/ekaterinailin/AltaiPony.git
    cd AltaiPony
    python setup.py install

You will also need to install **k2sc** (`Aigrain et al. 2016`_) from this fork_ of the original version:

:: 
   
   git clone https://github.com/ekaterinailin/k2sc.git
   cd k2sc
   python3 setup.py install
   

Getting Started
^^^^^^^^^^^^^^^^

Working with Kepler and TESS light curves is similar, K2 light curves need extra love, so we treat them separately. For each mission, there is a notebook that will guide you through the basic applications.

Kepler Light Curves
...................

If you are working with ``KeplerLightCurve`` objects, i.e. light curves from the original Kepler mission, try this_ notebook. This shows how to fetch a light curve from MAST, de-trend it with a Savitzky-Golay_ filter from scipy_, and find some flares.

TESS Light Curves
...................

If you are working with ``TessLightCurve`` objects, i.e. light curves from the TESS mission, you may want to try this other_ notebook instead. In this one, you can also test the injection recovery feature of **AltaiPony**, and obtain recovery probability and a corrected equivalent duration (*ED*, aka luminosity independent flare energy).

K2 Light Curves
...................

If you are working with ``KeplerLightCurve`` objects, i.e. light curves from the K2 mission, this last notebook_ is for you. It will run **k2sc**'s GP de-trending and find flares in a typical K2 long cadence light curve despite its heavy instrumental artifacts.


.. _Aigrain et al. 2016: http://ascl.net/1605.012
.. _fork: https://github.com/ekaterinailin/k2sc
.. _notebook: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/Getting_Started.ipynb
.. _this: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/Kepler_Light_Curves_With_Flares.ipynb
.. _Savitzky-Golay: http://www.statistics4u.info/fundstat_eng/cc_filter_savgolay.html
.. _scipy: https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.savgol_filter.html
.. _other: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/TESS_Light_Curves_With_Flares.ipynb
