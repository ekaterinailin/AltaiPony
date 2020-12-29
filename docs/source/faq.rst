Frequently Asked Questions
=======================================


What is detected as a flare and not a flare? 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A flare in a Kepler or TESS light curve is a series of data points that fullfils the flare definition criteria. In AltaiPony, flares are positive excursions from the de-trended light curve above a certain noise threshold. You can play around with a number of parameters - details are explained in the section on `Defining Flare Candidates`_.

What are the default settings for flare detection? 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default settings for flare detection are explained in the section on `Defining Flare Candidates`_.

How does **AltaiPony** handle the ramp ups at the end of TESS orbits? 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**AltaiPony** currently does not explicitly handle these ramp ups, so they can cause false positive detections. Using PDCSAP_FLUX instead of SAP_FLUX flux avoid these problems for the most part. However, if your algorithm can deal with these, you can pass a custom de-trending function to 

::

    FlareLightCurve.detrend("custom", func=<your custom function>) 


``func`` should *take* a ``FlareLightCurve`` as an argument, and can have arbitrary numbers of keyword arguments. It should also *return* a ``FlareLightCurve``.


What is defined as a the start and stop time of a flare? 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The start and stop times of a flare candidate mark the first and last+1 timestamp that fullfil all flare definition criteria. Accordingly the start and stop cadences and indices can be used to mask flares. An example:

::

    import numpy as np
    [...]
    # Take table of flare candidates in a FlareLightCurve (``flc``) to find all indices:
    flareindices = [list(np.arange(row.istart, row.istop)) for i, row in flc.flares.iterrows()]
    # Flatten the list:
    l = [i for sublist in l for i in sublist]
    # Mask the flares:
    flc.detrended_flux[l] = np.nan

.. _Defining Flare Candidates: https://altaipony.readthedocs.io/en/latest/tutorials/altai.html#defining-flare-candidates

