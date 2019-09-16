Synthetic Flare Injection and Recovery
=====

These are the functions that inject synthetic flare events and manage the output. To run a series of iterations with synthetic flares, call the ``sample_flare_recovery()`` method on a ``FlareLightCurve``.

::

    from altaipony.lcio import from_mast
    flc = from_mast("29780677", mode="LC", c=2, mission="TESS")
    flc = flc.detrend("savgol")
    flc, fake_flc = flc.sample_flare_recovery(inject_before_detrending=True, mode="savgol", 
                                              iterations=50, fakefreq=1, ampl=[1e-4, 0.5], 
                                              dur=[.001/6., 0.1/6.])

``flc`` is the original light curve with a new attribute ``fake_flares``, which is a DataFrame_ that includes the following columns:

* ``amplitude``: the synthetic flare's relative amplitude
* ``duration_d`` : the synthetic flare's duration in days [1]_
* ``ed_inj``: injected equivalent duration in seconds
* ``peak_time``: time at which the synthetic flare flux peaks 	
* all columns that appear in the `flares` attribute of `FlareLightCurve`, see here_. If the respective row has a value, the synthetic flare was recovered with some results, otherwise **AltaiPony** could not re-discover this flare at all.

``fake_flc`` is just like the original one, but without the ``fake_flares`` attribute. Instead, its flux contains synthetic flares from the last iteration run by ``sample_flare_recovery``. We return it because it is often useful to see what one actually injects.

Flare characterization
-----------------------

What can we do with all these synthetic flares? We can use them to characterize the flare candidates in the original light curve. To do this, call the ``characterize_flares`` method on your ``FlareLightCurve``:

>>> flc = flc.characterize_flares(ampl_bins=20, dur_bins=30)

This method will tile up your sample of fake flares into amplitude and duration bins twice. First, it will tile up the sample into a matrix based on the *recovered* amplitude and durations. Second, it will do the same with the *injected* properties, and so include also those injected flares that were not recovered. 

The first matrix can be used to map each flare candidate's recovered equivalent duration to a value that accounts for losses dealt to the ED by photometric noise, and introduced by the de-trending procedure (if you chose ``inject_before_detrending=True`` above). The typical injected amplitude and duration of flares in that tile of the matrix can then be used by the second matrix to derive the candidate's recovery probability from the ratio of lost to recovered injected flares.

The results from this mapping are stored in the ``flares`` attribute, which now contains the following additional columns in the table:

* ``dur``: ``= tstop - tstart``


* ``ed_ratio``: ratio of recovered ED to injected ED in the synthetic flares in the matrix tile that contains flares with measured properties that are most similar to the candidate flare.
* ``ed_ratio_count``: number of synthetic flares in the tile
* ``ed_ratio_std``: standard deviation of ED ratios in the tile
* ``ed_corr``: ``= rec_err / ed_ratio``
* ``ed_corr_err``: quadratically propagated uncertainties, including ``ed_rec_err`` and ``ed_ratio_std``


As in ``ed_ratio`` but with amplitude:
* ``amplitude_ratio``
* ``amplitude_ratio_count``
* ``amplitude_ratio_std``
* ``amplitude_corr``
* ``amplitude_corr_err`` : uncertainty propagated from ``amplitude_ratio_std``


As in ``amplitude_ratio`` but with duration in days:
* ``duration_ratio``
* ``duration_ratio_count``
* ``duration_ratio_std``
* ``duration_corr``
* ``duration_corr_err``

As in the columns but now for recovery probability:
* ``recovery_probability``: float between 0 and 1
* ``recovery_probability_count``:
* ``recovery_probability_std``:

"Properties" always refers to amplitude and duration.


.. rubric:: Footnotes

.. [1] At the moment this is not a very meaningful quantity because the decay of the flare goes on to infitiny! We may define full width at 1% of the fluxe or something as an approximation but that is for later and I am getting distracted. But we need it to map between injected and recovered flares, that is why it's hanging around in that table.

.. _DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
.. _here: https://altaipony.readthedocs.io/en/latest/api/altai.html
