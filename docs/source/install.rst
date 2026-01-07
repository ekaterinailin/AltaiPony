Quickstart
=======================================

ALTAIPONY IS CURRENTLY UNDERGOING MAJOR RENOVATIONS. PLEASE STAND BY. [Jan 7, 2026]

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

This package depends, on `lightkurve`, `numpy`, `pandas` and some other packages, most of which will be installed automatically. Have a look at `requirements.txt` in the repository to see a more extensive list. It will run stably for Python 3.10-3.13.
   

Getting Started
^^^^^^^^^^^^^^^^

We recommend taking a look at the `Quickstart`_ tutorial.


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
.. _Quickstart: https://altaipony.readthedocs.io/en/latest/tutorials/01_Getting_Started.html
.. _Synthetic Flare Injection and Recovery: https://altaipony.readthedocs.io/en/latest/tutorials/fakeflares.html
.. _visualization: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/05_Visualize_Injection_Recovery.ipynb
.. _beginner: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/02_Beginner_Flare_Frequency_Distributions_and_Power_Laws.ipynb
.. _advanced: https://github.com/ekaterinailin/AltaiPony/blob/master/notebooks/06_Advanced_Flare_Frequency_Distributions_and_Power_Laws.ipynb
.. _Flare Frequency Distributions and Power Laws: https://altaipony.readthedocs.io/en/latest/tutorials/ffds.html
