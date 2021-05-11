Quickstart
=======================================

Installation
^^^^^^^^^^^^


Use pip to install AltaiPony

::
	
    pip install altaipony


Or install directly from the repository:

::
    
    git clone https://github.com/ekaterinailin/AltaiPony.git
    cd AltaiPony
    python setup.py install

This package depends, on `lightkurve`, `k2sc`, `numpy`, `pandas` and some other packages, most of which will be installed automatically. Have a look at `requirements.txt` in the repository to see a more extensive list.
   

Getting Started
^^^^^^^^^^^^^^^^

Working with Kepler and TESS light curves is similar, K2 light curves need extra attention, so we treat them separately. For each mission, there is a notebook that will guide you through the basic applications. We recommend taking a look at the `Finding Data`_ tutorial.

Kepler Light Curves
...................

If you are working with ``KeplerLightCurve`` objects, i.e. light curves from the original Kepler mission, try this_ notebook. This shows how to fetch a light curve from MAST, de-trend it with a Savitzky-Golay_ filter from scipy_, and find some flares.

TESS Light Curves
...................

If you are working with ``TessLightCurve`` objects, i.e. light curves from the TESS mission, you may want to try this other_ notebook instead. In this one, you can also test the injection recovery feature of **AltaiPony**, and obtain recovery probability and a corrected equivalent duration (*ED*, aka luminosity independent flare energy).

K2 Light Curves
...................

If you are working with ``KeplerLightCurve`` objects, i.e. light curves from the K2 mission, this last notebook_ is for you. It will run **k2sc**'s GP de-trending and find flares in a typical K2 long cadence light curve despite its heavy instrumental artifacts.


Next Steps
^^^^^^^^^^^

Define your own flare finding
.............................

Once you have tried basic **AltaiPony** on your light curves, you can start to adjust the flare finding parameters to your application, as explained in the `Finding Flares`_ tutorial.


Test the performance of your flare finding algorithm
.....................................................

You may then want to test the perfomance of your chosen flare finding setup by injecting and recoving synthetic flares into your light curves. **AltaiPony** provides a framework to do so, explained in the `Synthetic Flare Injection and Recovery`_ tutorial. Check out the `visualization`_ notebook for nice plots.

Analyze flare frequency distributions
......................................

For a statistical analysis of your flares, **AltaiPony** also features a set of tools for the analysis of flare frequency distributions, including visualization, and different methods for power law fitting. For starters, check out the tutorial on `Flare Frequency Distributions and Power Laws`_. If you want to go hands on, start with the `beginner`_ notebook. For more advanced applications, like working with samples of multiple stars and their flares, go to the `advanced`_ notebook. 


.. _Aigrain et al. 2016: http://ascl.net/1605.012
.. _fork: https://github.com/ekaterinailin/k2sc
.. _notebook: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/01_Getting_Started.ipynb
.. _this: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/03_Kepler_Light_Curves_With_Flares.ipynb
.. _Savitzky-Golay: http://www.statistics4u.info/fundstat_eng/cc_filter_savgolay.html
.. _scipy: https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.savgol_filter.html
.. _other: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/04_TESS_Light_Curves_With_Flares.ipynb
.. _in this tutorial: https://altaipony.readthedocs.io/en/latest/tutorials/altai.html
.. _Finding Flares: https://altaipony.readthedocs.io/en/latest/tutorials/altai.html
.. _Finding Data: https://altaipony.readthedocs.io/en/latest/tutorials/lcio.html
.. _Synthetic Flare Injection and Recovery: https://altaipony.readthedocs.io/en/latest/tutorials/fakeflares.html
.. _visualization: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/05_Visualize_Injection_Recovery.ipynb
.. _beginner: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/02_Beginner_Flare_Frequency_Distributions_and_Power_Laws.ipynb
.. _advanced: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/06_Advanced_Flare_Frequency_Distributions_and_Power_Laws.ipynb
.. _Flare Frequency Distributions and Power Laws: https://altaipony.readthedocs.io/en/latest/tutorials/ffds.html
