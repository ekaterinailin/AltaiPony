Finding Flares
=====

First you'll need a de-trended light curve. If you have a raw `FlareLightCurve` we can call `rawflc`, for Kepler and TESS light curves use:

>>> flc = rawflc.detrend("savgol")

K2 is more difficult, and computationally intense, but doable with:

>>> flc - rawflc.detrend("k2sc")

Now you have a de-trended light curve `flc`, and you can search it for flares:

>>> flc = flc.find_flares()

This will return the initial light curve with a new attribute - `flares`. This is a DataFrame_ with the following columns:

* `ampl_rec` - recovered amplitude measured relative to the quiescent stellar flux
* `ed_rec` - recovered equivalent duration of the flare, that is, the are under the light curve with quiescent flux subtracted.
* `ed_rec_err` - the minimum uncertainty on equivalent duration derived from the uncertainty on the flare flux values (see Davenport (2016)_ for details, Eq. (2)).
* `cstart, cstop, istart, istop, tstart, tstop` - start and stop of flare candidates in units of cadence, array index and actual time in days.



.. _DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
.. _Davenport (2016): https://iopscience.iop.org/article/10.3847/0004-637X/829/1/23
