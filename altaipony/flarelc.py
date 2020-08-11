import numpy as np
import pandas as pd
import os
import copy
import logging
import progressbar
import datetime


from lightkurve import KeplerLightCurve, KeplerTargetPixelFile, TessLightCurve
from lightkurve.utils import KeplerQualityFlags

from astropy.io import fits

from .k2scmod import k2sc_lc
from .altai import (find_flares, find_iterative_median, detrend_savgol)
from .fakeflares import (merge_fake_and_recovered_events,
                         generate_fake_flare_distribution,
                         mod_random,
                         aflare,
                         )
from .injrecanalysis import wrap_characterization_of_flares, _heatmap

import time
LOG = logging.getLogger(__name__)

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
    targetid : int
        EPIC ID number.
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
    def __init__(self, time=None, flux=None, flux_err=None, time_format=None,
                 time_scale=None, time_unit=None, centroid_col=None,
                 centroid_row=None, quality=None, quality_bitmask=None,
                 channel=None, campaign=None, quarter=None, sector=None, mission=None,
                 cadenceno=None, targetid=None, ra=None, dec=None, label=None,
                 meta={}, detrended_flux=None, detrended_flux_err=None,
                 flux_trends=None, gaps=None, flares=None, flux_unit = None,
                 primary_header=None, data_header=None, pos_corr1=None,
                 pos_corr2=None, origin='FLC', fake_flares=None, it_med=None,
                 pixel_flux=None, pixel_flux_err=None, pipeline_mask=None,
                 camera=None, ccd=None, saturation=None):

        if mission == 'TESS':
                TessLightCurve.__init__(self, time=time, flux=flux, flux_err=flux_err,
                                        time_format=time_format, time_scale=time_scale,
                                        centroid_col=centroid_col, centroid_row=centroid_row,
                                        quality=quality, quality_bitmask=quality_bitmask,
                                        camera=camera, cadenceno=cadenceno, targetid=targetid,
                                        ra=ra, dec=dec, label=label, meta=meta, sector=sector,
                                        )
                self.mission = mission
                self.campaign = None
                self.quarter = None
        else:
                KeplerLightCurve.__init__(self, time=time, flux=flux, flux_err=flux_err,
                                          time_format=time_format, time_scale=time_scale,
                                          centroid_col=centroid_col, centroid_row=centroid_row,
                                          quality=quality, quality_bitmask=quality_bitmask,
                                          channel=channel, campaign=campaign, quarter=quarter,
                                          mission=mission, cadenceno=cadenceno, targetid=targetid,
                                          ra=ra, dec=dec, label=label, meta=meta)
        
        self.flux_unit = flux_unit
        self.time_unit = time_unit
        self.gaps = gaps
        self.flux_trends = flux_trends
        self.primary_header = primary_header
        self.data_header = data_header
        self.pos_corr1 = pos_corr1
        self.pos_corr2 = pos_corr2
        self.origin = origin
        self.detrended_flux = detrended_flux
        self.detrended_flux_err = detrended_flux_err
        self.pixel_flux = pixel_flux
        self.pixel_flux_err = pixel_flux_err
        self.pipeline_mask = pipeline_mask
        self.it_med = it_med
        self.saturation = saturation

        

        columns = ['istart', 'istop', 'cstart', 'cstop', 'tstart',
                   'tstop', 'ed_rec', 'ed_rec_err', 'ampl_rec', 
                   'total_n_valid_data_points']

        if detrended_flux is None:
            self.detrended_flux = np.full_like(time, np.nan)
        else:
            self.detrended_flux = detrended_flux

        if detrended_flux_err is None:
            self.detrended_flux_err = np.full_like(time, np.nan)
        else:
            self.detrended_flux_err = detrended_flux_err
        
        if saturation is None:
            self.saturation = np.full_like(time, np.nan)
        else:
            self.saturation = saturation

        if flares is None:
            self.flares = pd.DataFrame(columns=columns)
        else:
            self.flares = flares

        if fake_flares is None:
            other_columns = ['duration_d', 'amplitude', 'ed_inj', 'peak_time']
            self.fake_flares = pd.DataFrame(columns=other_columns)
        else:
            self.fake_flares = fake_flares

    def __repr__(self):
        return('FlareLightCurve(ID: {})'.format(self.targetid))

    def __getitem__(self, key):
        """
        Override default indexing to cover all time-domain attributes.
        """
        copy_self = copy.copy(self)
        copy_self.time = self.time[key]
        copy_self.flux = self.flux[key]
        copy_self.flux_err = self.flux_err[key]
        if copy_self.cadenceno is not None:
            copy_self.cadenceno = self.cadenceno[key]
        if copy_self.pos_corr1 is not None:
            copy_self.pos_corr1 = self.pos_corr1[key]
            copy_self.pos_corr2 = self.pos_corr2[key]
        if copy_self.centroid_col is not None:
            copy_self.centroid_col = self.centroid_col[key]
        if copy_self.centroid_row is not None:
            copy_self.centroid_row = self.centroid_row[key]
        if copy_self.detrended_flux is not None:
            copy_self.detrended_flux = self.detrended_flux[key]
            copy_self.detrended_flux_err = self.detrended_flux_err[key]
        if copy_self.it_med is not None:
            copy_self.it_med = self.it_med[key]
        if copy_self.flux_trends is not None:
            copy_self.flux_trends = self.flux_trends[key]
        if copy_self.pixel_flux is not None:
            copy_self.pixel_flux = self.pixel_flux[key]
        if copy_self.pixel_flux_err is not None:
            copy_self.pixel_flux_err = self.pixel_flux_err[key]
        if copy_self.quality is not None:
            copy_self.quality = self.quality[key]
        if copy_self.saturation is not None:
            copy_self.saturation = self.saturation[key]
        return copy_self

    def find_gaps(self, maxgap=0.09, minspan=10):
        '''
        Find gaps in light curve and stores them in the gaps attribute.

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

        Returns
        --------
        FlareLightCurve

        '''
        lc = copy.copy(self)
        dt = np.diff(lc.time)
        gap = np.where(np.append(0, dt) >= maxgap)[0]
        # add start/end of LC to loop over easily
        gap_out = np.append(0, np.append(gap, len(lc.time)))

        # left start, right end of data
        left, right = gap_out[:-1], gap_out[1:]

        #drop too short observation periods
        too_short = np.where(np.diff(gap_out) < 10)
        left, right = np.delete(left,too_short), np.delete(right,(too_short))
        lc.gaps = list(zip(left, right))

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
                new_lc.keplerid = self.targetid

                #K2SC MAGIC
                new_lc.__class__ = k2sc_lc
                try:
                    new_lc.k2sc(de_niter=de_niter, max_sigma=max_sigma, **kwargs)
                    new_lc.detrended_flux = (new_lc.corr_flux - new_lc.tr_time
                                             + np.nanmedian(new_lc.tr_time))
                    new_lc.detrended_flux_err = copy.copy(new_lc.flux_err) # does k2sc share their uncertainties somewhere?
                    new_lc.flux_trends = new_lc.tr_time
                    if new_lc.detrended_flux.shape != self.flux.shape:
                        LOG.error('De-detrending messed up the flux arrays.')
                    else:
                        LOG.info('De-trending successfully completed.')

                except np.linalg.linalg.LinAlgError:
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
            err_str = (f'\nDe-trending mode {mode} does not exist. Pass "k2sc" (K2 LCs)'
                       ' or "savgol" (Kepler, TESS).')
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
                       'tstop', 'ed_rec', 'ed_rec_err', 'ampl_rec']
            lc.flares = pd.DataFrame(columns=columns)
            #find continuous observing periods
            lc = lc.find_gaps()
            #find the true median value iteratively
            lc = find_iterative_median(lc)
            #find flares
            lc = find_flares(lc, **kwargs)

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
                
            injrec_results = merge_fake_and_recovered_events(injs, recs)

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
                lc.fake_flares = lc.fake_flares.append(injrec_results, ignore_index=True)
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


    def inject_fake_flares(self, gapwindow=0.1, fakefreq=.005,
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
        mode : 'loglog', 'hawley2014' or 'rand'
            injection mode
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
                           (fake_lc.time.max() - fake_lc.time.min()))
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
                                       (gap_fake_lc.time.max() -
                                        gap_fake_lc.time.min()
                                       )
                                      )
                              )
                        )
            LOG.debug(f'Inject {nfake} fake flares into a {ri-le} datapoint long array.')
            
            # Are there real flares to deal with in the gap?
            real_flares_in_gap = self.flares[(self.flares.istart >= le) &
                                             (self.flares.istop <= ri)]
                                             
            # Pick flux, time, and flux error arrays 
            error = gap_fake_lc.__dict__[typerr]
            flux = gap_fake_lc.__dict__[typ]
            time = gap_fake_lc.time
            
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
                    fl_flux = aflare(time, t0, dur_fake[k], ampl_fake[k])
                    
                    # calculate the injected ED
                    ed_fake[k] = _equivalent_duration(time, fl_flux)
                    
                # inject flare in to light curve by adding the flare flux
                fake_lc.__dict__[typ][le:ri] = (fake_lc.__dict__[typ][le:ri] +
                                                fl_flux * fake_lc.it_med[le:ri])
                
            # Increment the counter
            ckm += nfake
            
        # error minimum is a safety net for the spline function if mode=3
        fake_lc.__dict__[typerr] = max( 1e-10, np.nanmedian( pd.Series(fake_lc.__dict__[typ]).
                                                rolling(3, center=True).
                                                std() ) )*np.ones_like(fake_lc.__dict__[typ])
        # Put the data together
        injected_events = {'duration_d' : dur_fake,
                           'amplitude' : ampl_fake,  
                           'ed_inj' : ed_fake,
                           'peak_time' : t0_fake}
        fake_lc.fake_flares = pd.DataFrame(injected_events)
        
        # Free up space
        del dur_fake
        del ampl_fake
        
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
	        self.fake_flares = self.fake_flares.append(df)
        else:
	        self.fake_flares = df

    def plot_recovery_probability_heatmap(self, ampl_bins=None, 
                                          dur_bins=None, flares_per_bin=20):
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
                        ampl_bins, dur_bins, flares_per_bin)

    def plot_ed_ratio_heatmap(self, ampl_bins=None, dur_bins=None, flares_per_bin=20):
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
                        ampl_bins, dur_bins, flares_per_bin)

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
    

    def to_fits(self, path):
        """Write FlareLightCurve to a .fits
        file. Read it in again using from_path().

        Parameters:
        ------------
        path : str
            Path to location.
        """
        flc = copy.deepcopy(self)
        bintab = [] # list for main table
        hdr = fits.Header() # empty header
        vals = flc.__dict__ # all attributes from light curve

        # Place attributes into header or main table depending on dtype:
        for key, val in vals.items():
            if type(val)==np.ndarray:
                if len(val.shape) == 1:
                    bintab.append(fits.Column(name=key, format='D', array=val))
                else:
                    LOG.warning("Did not save {} because fits files only accept 1D arrays.".format(key))
            elif (type(val) == str) | (type(val) == int) | (type(val) == float):
                hdr[key] = val
            elif type(val) == dict:
                for k, v in val:
                    LOG.debug("Extra column {} defined in header.".format(k))
                    hdr[k] = v
            else:
                LOG.debug("{} was not written to .fits file.".format(key))
                
        # Define columns
        cols = fits.ColDefs(bintab)

        # Define header and binary table
        hdu = fits.BinTableHDU.from_columns(cols)
        primary_hdu = fits.PrimaryHDU(header=hdr)

        # Stick header and main table together
        hdul = fits.HDUList([primary_hdu, hdu])
        hdul.writeto(path, overwrite=True)


