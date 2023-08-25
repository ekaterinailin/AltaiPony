De-trending Light Curves
========================

AltaiPony currently features three different de-trending aproaches that aim at removing astrophysical and instrumental trends in the time series while preserving the flares.

Call the de-trender on a ``FlareLightCurve`` with 

::
   
     detrended_flc = flc.detrend(<detrending method>, **kwargs)


For ``<detrending method>`` you can pass one the following:


* "k2sc" - DEPRECATED AND NO LONGER MAINTAINED, WILL BE REMOVED IN A FUTURE VERSION! If you wish to de-trend a K2 light curve, this is the best way to deal with, among other issues in K2, the Kepler spacecraft roll (see [1]_, [2]_ and [3]_ for details)
* "savgol" - applies a Savitky-Golay [4]_ filter to the light curve. This method is quick and gives good results in both Kepler and TESS light curves, but you may have to play with the ``window_length`` keyword to get optimal results.
* "custom" - you can use your own de-trending pipeline. Pass a function that takes a ``FlareLightCurve`` and returns a FlareLightCurve to ``flc.detrend("custom", func=<your custom de-trending function>)`` . 


You can use the custom de-trending method detailed in [6]_ that can be used to de-trend Kepler and TESS light curves as follows with AltaiPony

::
 

     from funcs.customdetrend import custom_detrending
                                 
     flcd = flc.detrend("custom", func=custom_detrending)


The ``custom_detrending`` function takes the following keyword arguments:


* *spline_coarseness* (default=30) - time in hours to average over for the spline fit
* *spline_order* (default=3) - the order of the spline fit 
* *savgol1* (default=6) - window length of the first Savitzky-Golay filter
* *savgol2* (default=3) - the window length of the second Savitzky-Golay filter
* *max_sigma* (default=2.5) - the number of standard deviations above the median to clip the light curve at for de-trending (**not** for flare searching!) 
* *longdecay* (default=6) - the number of consecutive clipped points times this parameter is additionally clipped at the end of the clipped points to avoid smoothing away flare decay 
* *pad* (default=3) - outliers in Savitzky-Golay filter are padded with this number of data points to avoid cutting flares **after** applying the longdecay expansion. The padding is applied before and after the outliers.


.. [1] Ekaterina Ilin, Sarah J. Schmidt, Katja Poppenhäger, James R. A. Davenport, Martti H. Kristiansen, Mark Omohundro (2021). "Flares in Open Clusters with K2. II. Pleiades, Hyades, Praesepe, Ruprecht 147, and M67" Astronomy & Astrophysics, Volume 645, id.A42, 25 pp.  	https://doi.org/10.1051/0004-6361/202039198 

.. [2] Aigrain, Suzanne; Parviainen, Hannu; Pope, Benjamin (2016). "K2SC: flexible systematics correction and detrending of K2 light curves using Gaussian process regression" Monthly Notices of the Royal Astronomical Society, Volume 459, Issue 3, p.2408-2419 https://doi.org/10.1093/mnras/stw706

.. [3] Aigrain, Suzanne; Parviainen, Hannu; Pope, Benjamin "K2SC: K2 Systematics Correction." Astrophysics Source Code Library, record ascl:1605.012 https://ui.adsabs.harvard.edu/abs/2016ascl.soft05012A/abstract

.. [4] Savitzky, Abraham. and Golay, M. J. E. (1964). "Smoothing and Differentiation of Data by Simplified Least Squares Procedures.". Analytical Chemistry, Volume 36, Issue 8 https://doi.org/10.1021/ac60214a047

.. [6] Ekaterina Ilin and Katja Poppenhäger (2022). "Searching for flaring star-planet interactions in AU Mic TESS observations" Monthly Notices of the Royal Astronomical Society, Volume 513, Issue 3, p.4579-4586 https://ui.adsabs.harvard.edu/abs/2022MNRAS.513.4579I/abstract
