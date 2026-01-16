Quickstart
=======================================

ALTAIPONY IS CURRENTLY UNDERGOING MAJOR RENOVATIONS. 

PLEASE STAND BY. [Jan 7, 2026]

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
   

Getting started
^^^^^^^^^^^^^^^^

The core features of AltaiPony are 

	- detrending light curves,
	- finding flares in those light curves,
	- characterizing these flares, and
	- running sample statistics on flare frequency distributions.

We recommend taking a look at the `Quickstart`_ tutorial for the first three steps on this list. For the flare statistics, take a look at the tutorial on `Flare Frequency Distributions and Power Laws`_.


Deep dives (under construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Define your own detrending pipeline
...................................

Follow the `Detrending`_ tutorial to understand how AltaiPony de-trends flare light curves, including how to add your own de-trending procedure to your analysis pipeline.


Play with the flare finding parameters
.......................................

**TBD** Flare finding tutorial



Test the performance of your flare finding algorithm
.....................................................

**TBD** Test the perfomance of your chosen flare finding setup by injecting and recoving synthetic flares into your light curves. **AltaiPony** provides a framework to do so, explained in the `Synthetic Flare Injection and Recovery`_ tutorial. 


.. _Quickstart: https://altaipony.readthedocs.io/en/latest/tutorials/01_Getting_Started.html
.. _Detrending: https://altaipony.readthedocs.io/en/latest/tutorials/02_Detrending.html
.. _Synthetic Flare Injection and Recovery: https://altaipony.readthedocs.io/en/latest/tutorials/fakeflares.html
.. _Flare Frequency Distributions and Power Laws: https://altaipony.readthedocs.io/en/latest/tutorials/ffds.html
