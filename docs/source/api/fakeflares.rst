Synthetic Flare Injection and Recovery
=====

These are the functions that inject synthetic flare events and manage the output. To run a series of iterations with synthetic flares, call the `sample_flare_recovery()` method on a `FlareLightCurve`.

>>> from altaipony.lcio import from_mast
flc = from_mast("29780677", mode="LC", c=2, mission="TESS")
flc = flc.detrend("savgol")
flc, fake_flc = flc.sample_flare_recovery(inject_before_detrending=True, mode="savgol", iterations=50, fakefreq=1, ampl=[1e-4, 0.5], dur=[.001/6., 0.1/6.])

`flc` is the original light curve with a new attribute `fake_flares`, which is a DataFrame_ that includes the following columns

* `amplitude`: relative amplitude of the synthetic flare
* `duration_d` : duration of the synthetic flare in days (At the moment this is not a very meaningful quantity because the decay of the flare goes on to infitiny! We may define full width at 1% of the fluxe or something as an approximation but that is for later and I am getting distracted. But we need it to map between injected and recovered flares, that is why it's hanging around in that table.)
* `ed_inj`: injected equivalent duration in seconds
* `peak_time`: time at which the synthetic flare flux peaks 	
* all columns that appear in the `flares` attribute of `FlareLightCurve`, see here_. If the respective row has a value, the synthetic flare was recovered with some results, otherwise **AltaiPony** could not re-discover this flare at all.

`fake_flc` is just like the original one, but without the `fake_flares` attribute. Instead, its flux contains synthetic flares from the last iteration run by `sample_flare_recovery`. It is nice to see what one actually injects from time to time.

.. _DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
.. _here: https://altaipony.readthedocs.io/en/latest/api/altai.html
