De-trending Light Curves
========================

AltaiPony currently features three different de-trending aproaches that aim at removing astrophysical and instrumental trends in the time series while preserving the flares.

Call the de-trender on a ``FlareLightCurve`` with 

::
   
     detrended_flc = flc.detrend(<detrending method>, **kwargs)


For <detrending method> you can pass one the following:

- "k2sc" - if you wish to de-trend a K2 light curve, this is the best way to deal with, among other issues in K2, the Kepler spacecraft roll (see [1]_, [2]_ and [3]_ for details)
- "savgol" - applies a Savitky-Golay [4]_ filter to the light curve. This method is quick and gives good results in both Kepler and TESS light curves, but you may have to play with the "window_length" keyword to get optimal results
- "custom" - you can use your own de-trending pipeline. Pass a function that takes a ``FlareLightCurve`` and returns a FlareLightCurve to `flc.detrend("custom", func=<your custom de-trending function>)` [5]_. 




.. [1] Ekaterina Ilin, Sarah J. Schmidt, Katja Poppenhäger, James R. A. Davenport, Martti H. Kristiansen, Mark Omohundro (2021). "Flares in Open Clusters with K2. II. Pleiades, Hyades, Praesepe, Ruprecht 147, and M67" Astronomy & Astrophysics, Volume 645, id.A42, 25 pp.  	https://doi.org/10.1051/0004-6361/202039198 

.. [2] Aigrain, Suzanne; Parviainen, Hannu; Pope, Benjamin (2016). "K2SC: flexible systematics correction and detrending of K2 light curves using Gaussian process regression" Monthly Notices of the Royal Astronomical Society, Volume 459, Issue 3, p.2408-2419 https://doi.org/10.1093/mnras/stw706

.. [3] Aigrain, Suzanne; Parviainen, Hannu; Pope, Benjamin "K2SC: K2 Systematics Correction." Astrophysics Source Code Library, record ascl:1605.012 https://ui.adsabs.harvard.edu/abs/2016ascl.soft05012A/abstract

.. [4] Savitzky, Abraham. and Golay, M. J. E. (1964). "Smoothing and Differentiation of Data by Simplified Least Squares Procedures.". Analytical Chemistry, Volume 36, Issue 8 https://doi.org/10.1021/ac60214a047

.. [5] One such function, used to de-trend Kepler and TESS short (1 min, 2 min and 20s) cadence light curve is currently accessible via ``altaipony.customdetrend.custom_detrending``, and will be documented and tested in detail soon (Ilin+2021 in prep.). 
