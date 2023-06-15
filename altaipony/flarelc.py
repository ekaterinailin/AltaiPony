import numpy as np
import pandas as pd
import os
import copy
import logging
import progressbar
import datetime



from lightkurve import KeplerLightCurve, TessLightCurve
from lightkurve.utils import KeplerQualityFlags


from .k2scmod import k2sc_lc
from .altai import (find_flares,
                    find_iterative_median, 
                    detrend_savgol)
from .fakeflares import (merge_fake_and_recovered_events,
                         generate_fake_flare_distribution,
                         mod_random,
                         flare_model,
                         )
from .injrecanalysis import wrap_characterization_of_flares, _heatmap
from .utils import split_gaps

import time
LOG = logging.getLogger(__name__)


FLARE_COLUMNS = ['istart', 'istop', 'cstart', 'cstop', 'tstart',
                 'tstop', 'ed_rec', 'ed_rec_err', 'ampl_rec', 
                  'total_n_valid_data_points', 'dur']

FAKE_FLARE_COLUMNS = ['duration_d', 'amplitude', 'ed_inj', 'peak_time']


class FlareLightCurve(KeplerLightCurve, TessLightCurve):
    """
    Flare light curve class that unifies properties of ``K2SC``-de-trended and
    Kepler's ``lightkurve.KeplerLightCurve``.

    Attributes
    -------------
    time : array-like
        Time measurements.
    flux : array-like
        Flux count for every time point.
    flux_err : array-like
        Uncertainty on each flux data point.
    pixel_flux : multi-dimensional array
        Flux in the target pixels from the KeplerTargetPixelFile.
    pixel_flux_err : multi-dimensional array
        Uncertainty on pixel_flux.
    pipeline_mask : multi-dimensional boolean array
        TargetPixelFile mask for aperture photometry.
    time_format : str
        String specifying how an instant of time is represented,
        e.g., 'bkjd' or ‘jd'.
    time_scale : str
        String that specifies how the time is measured, e.g.,
        tdb', ‘tt', ‘ut1', or 'utc'.
    time_unit : astropy.unit
        Astropy unit object defining unit of time.
    centroid_col : array-like
        Centroid column coordinates as a function of time.
    centroid_row : array-like
        Centroid row coordinates as a function of time.
    quality : array-like
        Kepler quality flags.
    quality_bitmask : str
        Can be 'none', 'default', 'hard' or 'hardest'.
    channel : int
        Channel number, where aperture is located on the CCD.
    campaign : int
        K2 campaign number.
    quarter : int
        Kepler Quarter number.
    sector : int
        TESS sector number
    mission : string
        Mission identifier, e.g., 'TESS', 'K2' or 'Kepler'.
    cadenceno : array-like
        Cadence number - unique identifier.
    object : str
        target ID.
    ra : float
        RA in deg.
    dec : float
        Declination in deg.
    label : string
        'EPIC xxxxxxxxx'.
    meta : dict
        Free-form metadata associated with the LightCurve. Not populated in
        general.
    detrended_flux : array-like
        K2SC detrend flux, same units as flux.
    detrended_flux_err : array-like
        K2SC detrend flux error, same units as flux.
    flux_trends : array-like
        Astrophysical variability as derived by K2SC.
    gaps : list of tuples of ints
        Each tuple contains the start and end indices of observation gaps. See
        ``find_gaps``.
    flares : DataFrame
        Table of flares, their start and stop time, recovered equivalent duration
        (ED), and, if applicable, recovery probability, ratio of recovered ED to
        injected synthetic ED. Also information about quality flags may be stored
        here.
    it_med : array-like
        Iterative median, see the ``find_iterative_median`` method.


    """

    @property
    def detrended_flux_err(self) -> np.array:
        try:
            return self["detrended_flux_err"]
        except KeyError:
            self["detrended_flux_err"] = np.full_like(self.time.value, np.nan)
            return self["detrended_flux_err"]

    @detrended_flux_err.setter
    def detrended_flux_err(self, detrended_flux_err):
        self["detrended_flux_err"] = detrended_flux_err


    @property
    def detrended_flux(self) -> np.array:
        try:
            return self["detrended_flux"]
        except KeyError:
            self["detrended_flux"] = np.full_like(self.time.value, np.nan)
            return self["detrended_flux"]

    @detrended_flux.setter
    def detrended_flux(self, detrended_flux):
        self["detrended_flux"] = detrended_flux 

    @property
    def cadenceno(self):
        try:
            return self["cadenceno"]
        except KeyError:
            self["cadenceno"] = np.full_like(self.time.value, np.nan)
            return self["cadenceno"]

    @cadenceno.setter
    def cadenceno(self, cadenceno):
        self["cadenceno"] = cadenceno

    @property
    def it_med(self):
        try:
            return self["it_med"]
        except KeyError:
            self["it_med"] = np.full_like(self.time.value, np.nan)
            return self["it_med"]

    @it_med.setter
    def it_med(self, it_med):
        self["it_med"] = it_med

    @property
    def origin(self):
        try:
            return self.meta["origin"]
        except KeyError:
            self.meta["origin"] = "FLC"
            return self.meta["origin"]

    @origin.setter
    def origin(self, origin):
        self.meta["origin"] = origin 



    
    @property
    def saturation(self):
        try:
            return self.meta["saturation"]
        except KeyError:
            self.meta["saturation"] = []
            return self.meta["saturation"]

    @saturation.setter
    def saturation(self, saturation):
        self.meta["saturation"] = saturation 

    @property
    def flares(self) -> pd.DataFrame:
        try:
            return self.meta["flares"]
        except KeyError:
            self.meta["flares"] = pd.DataFrame(columns=FLARE_COLUMNS)
            return self.meta["flares"]

    @flares.setter
    def flares(self, flares):
        self.meta["flares"] = flares 


    @property
    def fake_flares(self) -> pd.DataFrame:
        
        try:
            return self.meta["fake_flares"]
        except KeyError:
            self.meta["fake_flares"] = pd.DataFrame(columns=FAKE_FLARE_COLUMNS)
            return self.meta["fake_flares"]

    @fake_flares.setter
    def fake_flares(self, fake_flares):
        self.meta["fake_flares"] = fake_flares


    @property
    def gaps(self):
        try:
            return self.meta["gaps"]
        except KeyError:
            self.meta["gaps"] = None
            return self.meta["gaps"]

    @gaps.setter
    def gaps(self, gaps):
        self.meta["gaps"] = gaps 

    def _init_flare_table(self, flares=None, fake_flares=None):

        if flares is None:
            self.flares = pd.DataFrame(columns=FLARE_COLUMNS)
        else:
            self.flares = flares

        if fake_flares is None:
            
            self.fake_flares = pd.DataFrame(columns=FAKE_FLARE_COLUMNS)
        else:
            self.fake_flares = fake_flares

    def _add_tpf_columns(self, pixel_flux=None, pixel_flux_err=None, pipeline_mask=None):

        self.pixel_flux = pixel_flux
        self.pixel_flux_err = pixel_flux_err
        self.pipeline_mask = pipeline_mask

    def __repr__(self):
        return('FlareLightCurve(ID: {})'.format(self.targetid))



    def find_gaps(self, maxgap=0.09, minspan=10, splits=[]):
        '''
        Find gaps in light curve and stores them in the gaps attribute.
        If required, adds additional splits in an arbitrary number of places.
        Caution: passing splits values means that you override the minspan
        and maxgap.

        Parameters
        ------------
        time : numpy array with floats
            sorted array, in units of days
        maxgap : 0.09 or float
            maximum time gap between two datapoints in days,
            default equals approximately 2h
        minspan : 10 or int
            minimum number of datapoints in continuous observation,
            i.e., w/o gaps as defined by maxgap
        splits : list of floats or ints
            additional places in which to slice the time series

        Returns
        --------
        FlareLightCurve

        '''
        lc = copy.copy(self)
        dt = np.diff(lc.time.value)
        gap = np.where(np.append(0, dt) >= maxgap)[0]

        # add start/end of LC to loop over easily
        gap_out = np.append(0, np.append(gap, len(lc.time.value)))

        # left start, right end of data
        left, right = gap_out[:-1], gap_out[1:]

        # drop too short observation periods
        too_short = np.where(np.diff(gap_out) < 10)
        left, right = np.delete(left,too_short), np.delete(right,(too_short))

        # get the gaps
        gaps = list(zip(left, right))
        
        # split up the time series in additional place if needed
        lc.gaps = split_gaps(gaps, splits)

        return lc

    def detrend(self, mode, save=False,
                path='detrended_lc.fits', de_niter=30, max_sigma=3, 
                func=None,
                **kwargs):
        """
        De-trends a FlareLightCurve using ``K2SC``.
        Optionally saves the LightCurve in a fits file that can
        be read as K2SC file.

        Parameters:
        ----------
        mode : str
            "k2sc" or "savgol" or "custom"
        de_niter : int
            Differential Evolution global optimizer parameter. K2SC
            default is 150, here set to 30 as a safety net to avoid
            unintenional computational effort.
        max_sigma: int
            Default is 3, value is passed to iterative sigma clipping
            in K2SC
        save : False or bool
            If True, the light curve is saved as a fits file to a
            given folder.
        path : str
            Path to resulting fits file. 
            As a default, the fits file will be stored in the
            working directory.
        func : function
            custom detrending function
        kwargs : dict
            Keyword arguments to pass to k2sc, detrend_savgol, or custom
            method

        Returns
        --------
        FlareLightCurve
        """
        if mode == "savgol":
        
            new_lc = copy.deepcopy(self)
            new_lc =  detrend_savgol(new_lc, **kwargs)
            if save == True:
                new_lc.to_fits(path)
            return new_lc
        
        elif mode == "k2sc":
                
            #make sure there is no detrended_flux already
            if self.origin != 'TPF':
                err_str = ('Only KeplerTargetPixelFile derived FlareLightCurves can be'
                          ' passed to K2SC de-trending.')
                LOG.exception(err_str)
                raise ValueError(err_str)

            else:
                new_lc = copy.deepcopy(self)
                new_lc.meta["keplerid"] = self.targetid

                #K2SC MAGIC
                new_lc.__class__ = k2sc_lc
                try:
                    new_lc.k2sc(de_niter=de_niter, max_sigma=max_sigma, **kwargs)
                    new_lc["detrended_flux"] = (new_lc.corr_flux.value - new_lc.tr_time.value
                                             + np.nanmedian(new_lc.tr_time.value))
                    new_lc["detrended_flux_err"] = copy.copy(new_lc.flux_err) # does k2sc share their uncertainties somewhere?
                    new_lc["flux_trends"] = new_lc.tr_time.value
                    if new_lc.detrended_flux.shape != self.flux.value.shape:
                        LOG.error('De-detrending messed up the flux arrays.')
                    else:
                        LOG.info('De-trending successfully completed.')

                except np.linalg.linalg.LinAlgError as e:
                    LOG.error(e)
                    LOG.error('Detrending failed because probably Cholesky '
                              'decomposition failed. Try again, you shall succeed.')
                new_lc.__class__ = FlareLightCurve
                if save == True:
                    new_lc.to_fits(path)
                return new_lc
        
        elif mode=="custom":
            
            if func is None:
                LOG.error('If you wish to use a custom detrending function you'
                          ' must pass a callable to the "func" parameter.')
                raise ValueError

            new_lc = copy.deepcopy(self)
            
            new_lc = func(new_lc, **kwargs)            
            
            if (np.isnan(new_lc.detrended_flux).all() | np.isnan(new_lc.detrended_flux_err).all()):
                LOG.error('The custom de-trending function you passed does not'
                          ' return an detrended_flux or detrended_flux_err attri'
                          'bute.')
                raise AttributeError
            
            if save == True:
                new_lc.to_fits(path)
            
            return new_lc
        
        else:
            err_str = (f'\nDe-trending mode {mode} does not exist. Pass "savgol" (for a Savitzky-Golay filter based detrending) '
                       ' or "custom" (to pass a custom detrending function to func=).')
            LOG.exception(err_str)
            raise ValueError(err_str)


    def find_flares(self, minsep=3, fake=False, **kwargs):

        '''
        Find flares in a ``FlareLightCurve``.

        Parameters
        -------------
        minsep : 3 or int
            Minimum distance between two candidate start times in datapoints.
        kwargs : dict
            keyword arguments to pass to :func:`find_flares_in_cont_obs_period`
        
        Possible keyword arguments: 
   	    
        sigma : numpy array
            local scatter of the flux. Array should be the same length as the
            detrended flux array. 
            If sigma=None, error is used instead.
        N1 : int (default is 3)
            How many times above sigma is required.
        N2 : int (Default is 2)
            How many times above sigma and detrended_flux_err is required
        N3 : int (Default is 3)
            The number of consecutive points required to flag as a flare.
        
        Returns
        ----------
        FlareLightCurve
        '''
        if ((fake==False) & (self.flares.shape[0]>0)):
            return self
        else:
            lc = copy.deepcopy(self)
            #re-init flares
            columns = ['istart', 'istop', 'cstart', 'cstop', 'tstart',
                       'tstop', 'ed_rec', 'ed_rec_err', 'ampl_rec', 'dur']
            lc.flares = pd.DataFrame(columns=columns)
            #find continuous observing periods
            lc = lc.find_gaps()
            #find the true median value iteratively
            lc = find_iterative_median(lc)
            #find flares
            lc = find_flares(lc, minsep=minsep, **kwargs)

        return lc

    def sample_flare_recovery(self, iterations=2000, inject_before_detrending=False,
                              mode=None, func=None, save_lc_to_file=False, folder="", 
                              fakefreq=0.05, save=False, path=None, detrend_kwargs={},
                              **kwargs):
        """
        Runs a number of injection recovery cycles and characterizes the light
        curve by recovery probability and equivalent duration underestimation.
        Inject one flare per light curve.

        Parameters
        -----------
        iterations : 2000 or int
            Number of injection/recovery cycles
        inject_before_detrending : False or bool
            If True, fake flare are injected directly into raw data.
        mode : str
            "savgol" or "k2sc". Required if ``inject_before_detrending`` is True.
        fakefreq : 0.05 or float
            number of flares per day, but at least one per continuous observation period will be injected
        detrend_kwargs : dict
            Keyword arguments to pass to FlareLightCurve.detrend
        kwargs : dict
            Keyword arguments to pass to inject_fake_flares

        Returns
        -------
        lc : FlareLightCurve
            Detrended LC with all fake_flares listed in the attribute
        fake_lc : FlareLightCurve
            Light curve with the last iteration of synthetic flares injected.
        """
        injrecstr = {True : "before", False : "after"} # define string to identify fake flare analysis by file name
        
        lc = copy.deepcopy(self)
        if inject_before_detrending == True:
            lc = lc.detrend(mode, func=func, **detrend_kwargs)
        lc = lc.find_gaps()
        lc = lc.find_flares()
        lc = find_iterative_median(lc)
        
        lc_ = copy.deepcopy(lc)
        
        columns =  ['istart', 'istop', 'cstart', 'cstop', 'tstart', 'tstop',
                    'ed_rec', 'ed_rec_err', 'duration_d', 'amplitude', 'ed_inj',
                    'peak_time', 'ampl_rec', 'dur']
        

        widgets = [progressbar.Percentage(), progressbar.Bar()]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=iterations).start()
        for i in range(iterations):
            fake_lc = lc.inject_fake_flares(inject_before_detrending=inject_before_detrending,
                                                 fakefreq=fakefreq,
                                                 **kwargs)
            
            if save_lc_to_file == True:
                fake_lc.to_fits("{folder}before.fits")
                print(f"saved {self.targetit} LC before detrending")
                
            injs = fake_lc.fake_flares
           
            if inject_before_detrending == True:
                LOG.info('\nDetrending fake LC:\n')
                fake_lc = fake_lc.detrend(mode, func=func, **detrend_kwargs)
            
            fake_lc = fake_lc.find_flares(fake=True)
            recs = fake_lc.flares
          
            if save_lc_to_file == True:
                fake_lc.to_fits(f"{folder}after.fits")
                print(f"saved {self.targetit} LC after detrending")
                
            # The following line makes sure that column order is retained    
            injrec_results = pd.DataFrame(columns=columns)
            
            # Merge injected and recovered flares
            injrec_results = pd.concat([injrec_results, merge_fake_and_recovered_events(injs, recs)],
                                                   ignore_index=True)
            

            bar.update(i + 1)
            
            if save == True:
            
                #Define default path if needed
                if path is None:
                    path = (f'{iterations}_{lc.targetid}_inj_'
                            f'{injrecstr[inject_before_detrending]}_'
                            f'{lc.campaign}.csv')
                            
                # If it already exists append new injrec to the end       
                if os.path.exists(path):
                    with open(path, 'a') as f:
                        injrec_results.to_csv(f, index=False, header=False)
                # If it doesn't then write it out but keep the header
                else:
                    injrec_results.to_csv(path, index=False)
            
            # Add to previous runs of sample_flare_recovery on the same LC or create new table    
            if lc.fake_flares.shape[0] > 0:    
                lc.fake_flares = pd.concat([lc.fake_flares,injrec_results], ignore_index=True)
            else:
                lc.fake_flares = injrec_results
                
    
        if save == True:
            # Finally read in the result                    
            lc.fake_flares = pd.read_csv(path)  

        
        # End monitoring
        bar.finish()
        return lc, fake_lc

 
    def mark_flagged_flares(self, explain=False):
        """
        Mark all flares that coincide with K2 flagged cadences.
        Explain the flags if needed.

        Parameters
        -----------
        explain : False or bool
            If True, an ``explanation`` column will be added to the flares table
            explaining the flags that were raised during the flare duration.

        Returns
        --------
        FlareLightCurve with the flares table supplemented with an integer
        ``quality`` and, if applicable, a string ``explanation`` column.
        """
        lc = copy.copy(self)
        f = lc.flares
        if 'quality' not in f.columns:
            f['quality'] = 0
        f.quality = f.apply(lambda x: np.sum(lc.quality[x.istart:x.istop],
                                             dtype=int),
                            axis=1)
        if explain == True:
            g = lambda x: ', '.join(KeplerQualityFlags.decode(x.quality))
            f['explanation'] = f.apply(g, axis=1)
        lc.flares = f
        return lc

    def get_saturation(self, factor=10, return_level=False):
        """
        Goes back to the TPF and measures the maximum saturation level during a
        flare, averaged over the aperture mask.

        Parameters
        -----------
        factor : 10 or float
            Saturation level in full well depths.

        Returns
        -------
        FlareLightCurve with modified 'flares' attribute.
        """
        flc = copy.copy(self)
        well_depth = 10093

        def sat(flares, flc=flc, well_depth=10093, return_level=False):
            pfl = flc.pixel_flux[flares.istart:flares.istop]
            flare_aperture_pfl = pfl[:,flc.pipeline_mask]
            return sat_level(flare_aperture_pfl, well_depth, return_level)
            
        def sat_level(flare_aperture_pfl, well_depth, return_level):
            saturation_level = np.nanmean(flare_aperture_pfl, axis=1) / well_depth
            if return_level == False:
                return np.any(saturation_level > factor)
            else:
                return np.nanmax(saturation_level)
        
        colname = 'saturation_f{}'.format(factor)
        
        if np.isnan(flc.saturation).all():
            
            if flc.flares.shape[0] > 0:#do not attempt if no flares are detected
                flc.flares[colname] = flc.flares.apply(sat, axis=1,
                                                    return_level=return_level)
                
            elif flc.flares.shape[0] == 0: # calculate saturation for all times
                flare_aperture_pfl = flc.pixel_flux[:,flc.pipeline_mask]
                saturation_level = np.nanmax(flare_aperture_pfl, axis=tuple(np.arange(len(flare_aperture_pfl.shape)))[1:]) / well_depth
                if return_level == False:
                    flc.saturation = saturation_level > factor
                else:
                    flc.saturation = saturation_level
                                           

        else:
             if flc.flares.shape[0] > 0:#do not attempt if no flares are detected
                 if isinstance(flc.saturation[0], np.bool_) :
                     if return_level == False:
                        flc.flares[colname] = flc.flares.apply(lambda x: (flc.saturation[x.istart: x.istop] == True).any(),
                                                        axis=1)
                     elif return_level == True:
                         LOG.info("Warning: Saturation is given as boolean flag. \n Choose return_level=False.")
                         return flc.get_saturation(factor=factor, return_level=False)
                    
                 elif (isinstance(flc.saturation[0], np.float_) | isinstance(flc.saturation[0], np.float32)) :
                     if return_level == False:
                        flc.flares[colname] = flc.flares.apply(lambda x: (flc.saturation[x.istart: x.istop] > factor).any(),
                                                               axis=1)

                     elif return_level == True:
                        flc.flares[colname] = flc.flares.apply(lambda x: np.nanmax(flc.saturation[x.istart: x.istop]),
                                                        axis=1)

                                               

        return flc


    def inject_fake_flares(self, model="mendoza2022", gapwindow=0.1, fakefreq=.005,
                           inject_before_detrending=False, d=False, seed=None, 
                           **kwargs):
        '''
        Create a number of events, inject them in to data
        Use grid of amplitudes and durations, keep ampl in relative flux units
        Keep track of energy in Equiv Dur.
        Duration defined in minutes
        Amplitude defined multiples of the median error


        Parameters:
        -------------
        model : "mendoza2022" or "davenport2014"
            The flare model to use. Default is "mendoza2022".
        gapwindow : 0.1 or float

        fakefreq : .005 or float
            flares per day, but at least one per continuous observation period will be injected
        inject_before_detrending : True or bool
            By default, flares are injected before the light curve is detrended.
        d : boolean
            If True, a seed for random numbers will be set
        seed : int or None
            If d == True, seed will be set to this number 
        kwargs : dict
            Keyword arguments to pass to generate_fake_flare_distribution.

        Returns:
        ------------
        FlareLightCurve with fake flare signatures

        '''

        def _equivalent_duration(time, flux):
            '''
            Compute the Equivalent Duration of a fake flare.
            This is the area under the flare, in relative flux units.

            Parameters:
            -------------
            time : numpy array
                units of DAYS
            flux : numpy array
                relative flux units
            Return:
            ------------
            p : float
                equivalent duration of a single event in units of seconds
            '''
            x = time * 60.0 * 60.0 * 24.0
            integral = np.sum(np.diff(x) * flux[:-1])
            return integral
        
        
        fake_lc = copy.deepcopy(self)
        LOG.debug(str() + '{} FakeFlares started'.format(datetime.datetime.now()))
        
        # Either inject flares into the un-detrended light curve
        if inject_before_detrending == True:
            typ, typerr = 'flux', 'flux_err'
            LOG.debug('Injecting before detrending.')
            
        # ... or into the detrended one
        elif inject_before_detrending == False:
            typ, typerr = 'detrended_flux', 'detrended_flux_err'
            LOG.debug('Injecting after detrending.')
        
        # How many flares do you want to inject
        # At least one per gap
        # or as defined by the frequency
        nfakesum = max(len(fake_lc.gaps),
                       int(np.rint(fakefreq *
                           (fake_lc.time.value.max() - fake_lc.time.value.min()))
                           )
                       )
        
        # Use a light curve where you know the median flux
        fake_lc = find_iterative_median(fake_lc)
        
        # Init arrays for the synthetic flare parameters
        t0_fake = np.zeros(nfakesum, dtype='float') # peak times
        ed_fake = np.zeros(nfakesum, dtype='float') # ED
        dur_fake = np.zeros(nfakesum, dtype='float') # duration
        ampl_fake = np.zeros(nfakesum, dtype='float') # amplitude
        
        # Init the synthetic flare counter to allow to point to the right
        # places in the arrays above (XXX_fake etc.)
        ckm = 0
        
        # Iterate over continuous observing periods
        for (le,ri) in fake_lc.gaps:
            
            # Pick the observing period
            gap_fake_lc = fake_lc[le:ri]
            
            # Define the number of synthetic flares you want to inject
            # minimum of 1
            nfake = max(1, int(np.rint(fakefreq *
                                       (gap_fake_lc.time.value.max() -
                                        gap_fake_lc.time.value.min()
                                       )
                                      )
                              )
                        )

            LOG.debug(f'Inject {nfake} fake flares into a {ri-le} datapoint long array.')
            
            # Are there real flares to deal with in the gap?
            real_flares_in_gap = self.flares[(self.flares.istart >= le) &
                                             (self.flares.istop <= ri)]
                                             
            # Pick flux, time, and flux error arrays 
            error = gap_fake_lc[typerr]
            flux = gap_fake_lc[typ]

            # account for different data types for detrended and non-detrended data
            if typ == "flux":
                flux = flux.value
                error = error.value

            time = gap_fake_lc.time.value
            
            # generate the time constraints for the flares you want to inject
            mintime, maxtime = np.min(time), np.max(time)
            dtime = maxtime - mintime
            
            # generate a distribution of durations and amplitudes
            distribution  = generate_fake_flare_distribution(nfake, d=d,
                                                            seed=seed, **kwargs)
            # add the distribution for this observing period 
            # to the full list of injected flares
            dur_fake[ckm:ckm+nfake], ampl_fake[ckm:ckm+nfake] = distribution
            
            # loop over the numer of fake flares you want to generate
            for k in range(ckm, ckm+nfake):
                
                # generate random peak time, avoid known flares
                isok = False
                
                # keep picking new random peak times for your synthetic flares
                # until it does not overlap with a real one
                while isok is False:
                
                    # choose a random peak time
                    # if you set a seed you will get the same synthetic flares
                    # all the time
                    if isinstance(seed, int):
                        t0 = (mod_random(1, d=d, seed=seed * k) * dtime + mintime)[0]
                        
                    # if you do note set a seed, the synthetic flares will be
                    # randomly distributed
                    elif seed is None:
                        t0 = (mod_random(1, d=d) * dtime + mintime)[0]
                     
                    # Check if there are there any real flares to deal with
                    # at that peak time. Only relevant if there were any flares
                    # detected at all:
                    if real_flares_in_gap.tstart.shape[0] > 0:
                    
                        # Are there any real flares happening at peak time?
                        # Fake flares should not overlap with real ones.
                        b = (real_flares_in_gap[(t0 >= real_flares_in_gap.tstart) &
                                                (t0 <= real_flares_in_gap.tstop)].
                                                 shape[0] )
                                                 
                        # number of flares that overlap should be 0
                        if b == 0:
                            isok = True
                            
                    # No real flares, no trouble:
                    else:
                        isok = True
                        
                    # add the peak time to the list    
                    t0_fake[k] = t0
                    
                    # generate the flare flux from the Davenport 2014 model
                    fl_flux = flare_model(model,time, t0, dur_fake[k], ampl_fake[k])
                    
                    # calculate the injected ED
                    ed_fake[k] = _equivalent_duration(time, fl_flux)
                    
                # inject flare in to light curve by adding the flare flux
                if typ == "flux":
                    fake_lc[typ].value[le:ri] = (fake_lc[typ][le:ri].value +
                                                fl_flux * fake_lc.it_med[le:ri])
                elif typ == "detrended_flux":
                    fake_lc[typ][le:ri] = (fake_lc[typ][le:ri] +
                                                fl_flux * fake_lc.it_med[le:ri])
            # Increment the counter
            ckm += nfake
            
        # error minimum is a safety net for the spline function if mode=3
        fake_lc[typerr] = max( 1e-10, np.nanmedian( pd.Series(fake_lc[typ]).
                                                rolling(3, center=True).
                                                std() ) )*np.ones_like(fake_lc[typ])
        # Put the data together
        injected_events = {'duration_d' : dur_fake,
                           'amplitude' : ampl_fake,  
                           'ed_inj' : ed_fake,
                           'peak_time' : t0_fake}

        # the fake_flares attribute is a pandas DataFrame
        fake_lc.fake_flares = pd.DataFrame(injected_events)
        
        # Free up space
        del dur_fake
        del ampl_fake
        
        # Return the FLC with the injected flares
        return fake_lc

    def load_injrec_data(self, path, **kwargs):
        """Fetch the injection-recovery table
        from a given path, and append it to 
        any existing table.

        Parameters:
        -----------
        path : string
            path to file
        kwargs : dict
            keyword arguments to pass to
            `pandas.read_csv()`
        """

        df = pd.read_csv(path)
        if self.fake_flares.shape[0]>0:
            LOG.warning("The file is appended to an existing table.")
            self.fake_flares = pd.concat([self.fake_flares, df])
        else:
            self.fake_flares = df

    def plot_recovery_probability_heatmap(self, ampl_bins=None, 
                                          dur_bins=None, flares_per_bin=20, **kwargs):
        """Plot injected amplitude and injected
        FWHM vs. the fraction of recovered flares.
        
        Parameters:
        -----------
        ampl_bins : int or array
            bins for amplitudes
        dur_bins : int or array
            bins for FWHM
        flares_per_bin : int
            number of flares per bin, default is 20
        """

        flc = copy.deepcopy(self)
        return _heatmap(flc, "recovery_probability", 
                        ampl_bins, dur_bins, flares_per_bin, **kwargs)

    def plot_ed_ratio_heatmap(self, ampl_bins=None, dur_bins=None, flares_per_bin=20, **kwargs):
        """Plot recovered amplitude and recovered
        duration vs. the ratio of recovered ED to
        injected ED.
        
        Parameters:
        -----------
        ampl_bins : int or array
            bins for recovered amplitudes
        dur_bins : int or array
            bins for recovered duration
        flares_per_bin : int
            number of flares per bin, default is 20
        """
        flc = copy.deepcopy(self)
        return _heatmap(flc, "ed_ratio", 
                        ampl_bins, dur_bins, flares_per_bin, **kwargs)

    def characterize_flares(self, flares_per_bin=30, ampl_bins=None, dur_bins=None):
        """Use results from injection recovery to determine
        corrected flare characteristics.
        
        """
        flc = copy.deepcopy(self)
        flares = wrap_characterization_of_flares(flc.fake_flares, flc.flares,
                                                 flares_per_bin=flares_per_bin,
                                                 ampl_bins=ampl_bins,
                                                 dur_bins=dur_bins)
        flc.flares = flares
        return flc
    

    def to_fits(self, *args, **kwargs):
        if self.mission in ["Kepler", "K2"]:
            self.__class__ = KeplerLightCurve
            self.to_fits(*args, cadenceno=self.cadenceno, **kwargs)

        elif self.mission in ["TESS"]:
            self.__class__ = TessLightCurve
            self.to_fits(*args, cadenceno=self.cadenceno, **kwargs)



def generate_lightcurve(errorval,  a1, a2, period1, period2, quad, cube,
                        mean=3400.):
    
    """Generate wild light curves with variability on several
    timescales.
    
    Returns:
    ---------
    FlareLightCurve with time, flux, and flux_err attributes
    """
    time = np.arange(10, 10 + 10 * np.pi,.0008)

    # define the flux
    flux = (np.random.normal(0,errorval,time.shape[0]) +
            mean + 
            a1*mean*np.sin(period1*time +1.)  +
            a2*mean*np.sin(period2*time) +
            quad*(time-25)**2 -
            cube*(time-25)**3)

    # add a gap in the data
    flux[5600:7720] = np.nan

    # add big and long flare
    l = 66
    flux[5280:5280 + l] = flux[5280:5280 + l] + np.linspace(1000,250,l)

    # add tiny flare
    l = 3
    flux[15280:15280 + l] = flux[15280:15280 + l] + np.linspace(100,60,l)

    # add intermediate flare
    l, s = 15, 25280
    flux[s:s + l] = flux[s:s + l] + np.linspace(200,60,l)

    # typically Kepler and TESS underestimate the real noise
    err = np.full_like(time,errorval/3*2)

    # define FLC
    return FlareLightCurve(time=time, flux=flux, flux_err=err)



