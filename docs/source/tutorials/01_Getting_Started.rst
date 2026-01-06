Quickstart AltaiPony: De-trend and find flares
==============================================

``AltaiPony`` works off of the ``lightkurve`` package, with a
``FlareLightCurve`` being a subclass of the more general ``LightCurve``.
So, to get started with ``AltaiPony``, you first want to get a light
curve via ``lightkurve``.

.. code:: ipython3

    import lightkurve as lk
    
    from altaipony.customdetrend import custom_detrending
    from altaipony.lcio import to_flare_lightcurve
    
    import matplotlib.pyplot as plt


.. parsed-literal::

    arviz_base not installed
    arviz_stats not installed
    arviz_plots not installed


Let’s start by searching for AU Mic, a young, active star, observed with
TESS with ``search_lightkurve``. (Consult the docs of ``lightkurve`` for
the various parameters.)

.. code:: ipython3

    # get a table of light curves from MAST
    lc_table = lk.search_lightcurve('AU Mic', mission='TESS', author='SPOC')
    
    lc_table




.. raw:: html

    SearchResult containing 5 data products.
    
    <table id="table133927996322880">
    <thead><tr><th>#</th><th>mission</th><th>year</th><th>author</th><th>exptime</th><th>target_name</th><th>distance</th></tr></thead>
    <thead><tr><th></th><th></th><th></th><th></th><th>s</th><th></th><th>arcsec</th></tr></thead>
    <tr><td>0</td><td>TESS Sector 01</td><td>2018</td><td><a href='https://heasarc.gsfc.nasa.gov/docs/tess/pipeline.html'>SPOC</a></td><td>120</td><td>441420236</td><td>0.0</td></tr>
    <tr><td>1</td><td>TESS Sector 27</td><td>2020</td><td><a href='https://heasarc.gsfc.nasa.gov/docs/tess/pipeline.html'>SPOC</a></td><td>20</td><td>441420236</td><td>0.0</td></tr>
    <tr><td>2</td><td>TESS Sector 27</td><td>2020</td><td><a href='https://heasarc.gsfc.nasa.gov/docs/tess/pipeline.html'>SPOC</a></td><td>120</td><td>441420236</td><td>0.0</td></tr>
    <tr><td>3</td><td>TESS Sector 95</td><td>2025</td><td><a href='https://heasarc.gsfc.nasa.gov/docs/tess/pipeline.html'>SPOC</a></td><td>20</td><td>441420236</td><td>0.0</td></tr>
    <tr><td>4</td><td>TESS Sector 95</td><td>2025</td><td><a href='https://heasarc.gsfc.nasa.gov/docs/tess/pipeline.html'>SPOC</a></td><td>120</td><td>441420236</td><td>0.0</td></tr>
    </table>



Let take the first on the list, download it and plot it:

.. code:: ipython3

    %matplotlib inline
    lc = lc_table[0].download()
    lc.plot();


.. parsed-literal::

    5% (982/19261) of the cadences will be ignored due to the quality mask (quality_bitmask=175).
    5% (982/19261) of the cadences will be ignored due to the quality mask (quality_bitmask=175).



.. image:: 01_Getting_Started_files/01_Getting_Started_5_1.png


We can immediately see that there are some flares (bursty positive
excursions). Now let’s make ``lc`` a FlareLightCurve object using the
``to_flarelightcurve`` function:

.. code:: ipython3

    flc = to_flare_lightcurve(lc)
    flc.plot();



.. image:: 01_Getting_Started_files/01_Getting_Started_7_0.png


Note that this preserves the functions that you are used to from
``lightcurve``, such as ``plot()``. Nifty.

Let’s check what’s changed:

.. code:: ipython3

    flc




.. raw:: html

    <pre>FlareLightCurve(ID: 441420236 | Mission: TESS   | QCS:   1 | Cadence: 120 s</pre>



The above tells us that we have indeed created a ``FlareLightCurve``,
and it still has all the attributes, plus some more, like an empty flare
table, and an empty detrended_flux column.

This is the raw light curve. The is intrumental noise but also stellar
variability. Let’s remove it with K2SC:

.. code:: ipython3

    flcd = flc.detrend("custom", func=custom_detrending, spline_coarseness=8)



.. parsed-literal::

    /home/ilin/Documents/AltaiPony_dev/AltaiPony/altaipony/flarelc.py:465: UserWarning: Lightkurve doesn't allow columns or meta values to be created via a new attribute name.A new attribute is created. It will not be carried over when the object is copied. - see https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.html
      lc.gaps = split_gaps(gaps, splits)


.. code:: ipython3

    
    plt.figure(figsize=(10, 5))
    plt.plot(flc.time.value, flc.flux, label='Original Flux', alpha=0.5)
    
    plt.plot(flcd.time.value, flcd.detrended_flux, label='Detrended Flux', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.title('Custom Detrending of Flare Light Curve')
    plt.legend()
    # plt.ylim(282500,299500)
    # plt.xlim(2120,2122)




.. parsed-literal::

    <matplotlib.legend.Legend at 0x79ce8a2ba7e0>




.. image:: 01_Getting_Started_files/01_Getting_Started_14_1.png


.. code:: ipython3

    import astropy.units as u
    # flcd.flux = flcd.flux * u.electron / u.s 
    # flcd.detrended_flux = flcd.detrended_flux * u.electron / u.s
    flcd = flcd.find_flares()
    flcd.flares.sort_values(by="ed_rec", ascending=False)


.. parsed-literal::

    Found 41 candidate(s) in the (0,8949) gap.
    Found 28 candidate(s) in the (8949,14785) gap.
    Found 3 candidate(s) in the (14785,14973) gap.
    Found 12 candidate(s) in the (14973,17693) gap.
    /home/ilin/Documents/AltaiPony_dev/AltaiPony/altaipony/altai.py:210: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
      lc.flares = pd.concat([lc.flares, new], ignore_index=True)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>istart</th>
          <th>istop</th>
          <th>cstart</th>
          <th>cstop</th>
          <th>tstart</th>
          <th>tstop</th>
          <th>ed_rec</th>
          <th>ed_rec_err</th>
          <th>ampl_rec</th>
          <th>dur</th>
          <th>total_n_valid_data_points</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>72</th>
          <td>14974</td>
          <td>15060</td>
          <td>87456</td>
          <td>87732</td>
          <td>1348.927417</td>
          <td>1349.310738</td>
          <td>802.4740820079422</td>
          <td>0.7067276213899982</td>
          <td>0.031812071800231934</td>
          <td>0.383321</td>
          <td>17693.0</td>
        </tr>
        <tr>
          <th>68</th>
          <td>14631</td>
          <td>14784</td>
          <td>86616</td>
          <td>86794</td>
          <td>1347.760787</td>
          <td>1348.008002</td>
          <td>581.0991946404808</td>
          <td>0.3569859650722981</td>
          <td>0.029750585556030273</td>
          <td>0.247214</td>
          <td>17693.0</td>
        </tr>
        <tr>
          <th>30</th>
          <td>6517</td>
          <td>6634</td>
          <td>77508</td>
          <td>77625</td>
          <td>1335.111071</td>
          <td>1335.273570</td>
          <td>93.40855439063128</td>
          <td>0.1833644468114996</td>
          <td>0.032552480697631836</td>
          <td>0.162498</td>
          <td>17693.0</td>
        </tr>
        <tr>
          <th>65</th>
          <td>14426</td>
          <td>14604</td>
          <td>86350</td>
          <td>86568</td>
          <td>1347.391354</td>
          <td>1347.694123</td>
          <td>57.612051350846016</td>
          <td>0.3893842042881597</td>
          <td>0.005483508110046387</td>
          <td>0.302768</td>
          <td>17693.0</td>
        </tr>
        <tr>
          <th>0</th>
          <td>521</td>
          <td>614</td>
          <td>71430</td>
          <td>71525</td>
          <td>1326.669456</td>
          <td>1326.801400</td>
          <td>46.92867223049883</td>
          <td>0.20671797342301984</td>
          <td>0.01085352897644043</td>
          <td>0.131944</td>
          <td>17693.0</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>35</th>
          <td>7536</td>
          <td>7539</td>
          <td>78537</td>
          <td>78540</td>
          <td>1336.540223</td>
          <td>1336.544389</td>
          <td>0.3043378586891201</td>
          <td>0.04198446456723675</td>
          <td>0.0010994672775268555</td>
          <td>0.004167</td>
          <td>17693.0</td>
        </tr>
        <tr>
          <th>61</th>
          <td>14233</td>
          <td>14236</td>
          <td>86142</td>
          <td>86145</td>
          <td>1347.102475</td>
          <td>1347.106641</td>
          <td>0.29483364204473794</td>
          <td>0.04267687936134341</td>
          <td>0.0008928775787353516</td>
          <td>0.004167</td>
          <td>17693.0</td>
        </tr>
        <tr>
          <th>6</th>
          <td>1450</td>
          <td>1453</td>
          <td>72374</td>
          <td>72377</td>
          <td>1327.980563</td>
          <td>1327.984730</td>
          <td>0.2835979993528088</td>
          <td>0.04268989637507565</td>
          <td>0.0009565353393554688</td>
          <td>0.004167</td>
          <td>17693.0</td>
        </tr>
        <tr>
          <th>78</th>
          <td>15921</td>
          <td>15924</td>
          <td>88623</td>
          <td>88626</td>
          <td>1350.548196</td>
          <td>1350.552363</td>
          <td>0.2800417867487468</td>
          <td>0.042662375199455094</td>
          <td>0.0009180307388305664</td>
          <td>0.004167</td>
          <td>17693.0</td>
        </tr>
        <tr>
          <th>55</th>
          <td>13337</td>
          <td>13340</td>
          <td>85232</td>
          <td>85235</td>
          <td>1345.838624</td>
          <td>1345.842790</td>
          <td>0.2619046013077746</td>
          <td>0.04281527857414693</td>
          <td>0.0008718967437744141</td>
          <td>0.004167</td>
          <td>17693.0</td>
        </tr>
      </tbody>
    </table>
    <p>84 rows × 11 columns</p>
    </div>



.. code:: ipython3

    flcd.flares = flcd.flares.iloc[10:13]  # Keep only 3 medium sized flares to showcase fitting
    flcd.flares




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>istart</th>
          <th>istop</th>
          <th>cstart</th>
          <th>cstop</th>
          <th>tstart</th>
          <th>tstop</th>
          <th>ed_rec</th>
          <th>ed_rec_err</th>
          <th>ampl_rec</th>
          <th>dur</th>
          <th>total_n_valid_data_points</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>10</th>
          <td>1993</td>
          <td>2001</td>
          <td>72918</td>
          <td>72926</td>
          <td>1328.736116</td>
          <td>1328.747227</td>
          <td>2.1315681889181146</td>
          <td>0.05715828309010595</td>
          <td>0.006310462951660156</td>
          <td>0.011111</td>
          <td>17693.0</td>
        </tr>
        <tr>
          <th>11</th>
          <td>2410</td>
          <td>2417</td>
          <td>73340</td>
          <td>73347</td>
          <td>1329.322225</td>
          <td>1329.331947</td>
          <td>1.3534009177933761</td>
          <td>0.0636832318895726</td>
          <td>0.002482891082763672</td>
          <td>0.009722</td>
          <td>17693.0</td>
        </tr>
        <tr>
          <th>12</th>
          <td>2451</td>
          <td>2465</td>
          <td>73381</td>
          <td>73396</td>
          <td>1329.379169</td>
          <td>1329.400002</td>
          <td>5.192819408116435</td>
          <td>0.07724999319352319</td>
          <td>0.007713794708251953</td>
          <td>0.020833</td>
          <td>17693.0</td>
        </tr>
      </tbody>
    </table>
    </div>



This flare table gives a first impression of where the flares are found,
and what their properties are. To get a better estimate, you can fit a
flare template to each detection using ``fit_flares()``, which follows
the procedure described in `Guenther et
al. (2020) <https://ui.adsabs.harvard.edu/abs/2020AJ....159...60G>`__:

.. code:: ipython3

    flcd.fit_flares(max_flares=3, plot=True, n_steps=1500, discard=100)


.. parsed-literal::

    
     Fitting region from 1328.68612 to 1328.79723; Region [0](max. number of flares in group = 3 x 1)


.. parsed-literal::

      0%|          | 0/1500 [00:00<?, ?it/s]100%|██████████| 1500/1500 [00:49<00:00, 30.40it/s]
    100%|██████████| 1500/1500 [00:42<00:00, 35.19it/s]
    100%|██████████| 1500/1500 [00:38<00:00, 39.27it/s]



.. image:: 01_Getting_Started_files/01_Getting_Started_18_2.png


.. parsed-literal::

    
     Fitting region from 1329.27222 to 1329.45000; Region [1](max. number of flares in group = 3 x 2)


.. parsed-literal::

    100%|██████████| 1500/1500 [00:47<00:00, 31.43it/s]
    100%|██████████| 1500/1500 [00:50<00:00, 29.55it/s]
    100%|██████████| 1500/1500 [00:55<00:00, 26.90it/s]
    100%|██████████| 1500/1500 [00:51<00:00, 29.27it/s]
    100%|██████████| 1500/1500 [00:52<00:00, 28.52it/s]
    100%|██████████| 1500/1500 [00:56<00:00, 26.60it/s]



.. image:: 01_Getting_Started_files/01_Getting_Started_18_5.png


.. code:: ipython3

    flcd.flare_table()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>t_peak</th>
          <th>t_peak_err</th>
          <th>fwhm</th>
          <th>fwhm_err</th>
          <th>amplitude</th>
          <th>amplitude_err</th>
          <th>ed_rec</th>
          <th>fit_type</th>
          <th>group_index</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1328.736142</td>
          <td>0.000082</td>
          <td>0.004722</td>
          <td>0.000472</td>
          <td>1882.622828</td>
          <td>103.315017</td>
          <td>2.496575</td>
          <td>single</td>
          <td></td>
        </tr>
        <tr>
          <th>1</th>
          <td>1329.323987</td>
          <td>0.000098</td>
          <td>0.001450</td>
          <td>0.000970</td>
          <td>3420.591087</td>
          <td>3200.508176</td>
          <td>1.054790</td>
          <td>group_member</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1329.379935</td>
          <td>0.000131</td>
          <td>0.006053</td>
          <td>0.001044</td>
          <td>2899.350999</td>
          <td>870.772327</td>
          <td>4.651307</td>
          <td>group_member</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



