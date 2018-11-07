import numpy as np
import pandas as pd
import os
import copy
import logging
import progressbar

from k2sc.standalone import k2sc_lc
from lightkurve import KeplerLightCurve, KeplerTargetPixelFile
from lightkurve.utils import KeplerQualityFlags

from astropy.io import fits

from .altai import (find_flares, find_iterative_median,)
from .fakeflares import (inject_fake_flares,
                         merge_fake_and_recovered_events,
                         merge_complex_flares,
                         characterize_one_flare,
                         )


LOG = logging.getLogger(__name__)

class FlareLightCurve(KeplerLightCurve):
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
    mission : string
        Mission identifier, e.g., 'K2' or 'Kepler'.
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
                 time_scale=None, time_unit = None, centroid_col=None,
                 centroid_row=None, quality=None, quality_bitmask=None,
                 channel=None, campaign=None, quarter=None, mission=None,
                 cadenceno=None, targetid=None, ra=None, dec=None, label=None,
                 meta={}, detrended_flux=None, detrended_flux_err=None,
                 flux_trends=None, gaps=None, flares=None, flux_unit = None,
                 primary_header=None, data_header=None, pos_corr1=None,
                 pos_corr2=None, origin='FLC', fake_flares=None, it_med=None):

        super(FlareLightCurve, self).__init__(time=time, flux=flux, flux_err=flux_err, time_format=time_format, time_scale=time_scale,
                                              centroid_col=centroid_col, centroid_row=centroid_row, quality=quality,
                                              quality_bitmask=quality_bitmask, channel=channel, campaign=campaign, quarter=quarter,
                                              mission=mission, cadenceno=cadenceno, targetid=targetid, ra=ra, dec=dec, label=label,
                                              meta=meta)
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
        self.it_med = it_med

        columns = ['istart', 'istop', 'cstart', 'cstop', 'tstart',
                   'tstop', 'ed_rec', 'ed_rec_err']
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
        if copy_self.pos_corr1 is not None:
            copy_self.pos_corr1 = self.pos_corr1[key]
            copy_self.pos_corr2 = self.pos_corr2[key]
        if copy_self.detrended_flux is not None:
            copy_self.detrended_flux = self.detrended_flux[key]
            copy_self.detrended_flux_err = self.detrended_flux_err[key]
        if copy_self.it_med is not None:
            copy_self.it_med = self.it_med[key]
        if copy_self.flux_trends is not None:
            copy_self.flux_trends = self.flux_trends[key]
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

    def detrend(self):
        """
        De-trends a FlareLightCurve using ``K2SC``.

        Returns
        --------
        FlareLightCurve
        """
        #make sure there is no detrended_flux already
        if self.origin != 'TPF':
            err_str = ('Only KeplerTargetPixelFile derived FlareLightCurves can be'
                      ' passed to detrend().')
            LOG.exception(err_str)
            raise ValueError(err_str)

        else:
            new_lc = copy.copy(self)
            new_lc.keplerid = self.targetid

            #K2SC MAGIC
            new_lc.__class__ = k2sc_lc
            try:
                new_lc.k2sc(de_niter=3) #de_niter set low for testing purpose
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
            return new_lc

    def find_flares(self, minsep=3, fake=False):

        '''
        Find flares in a ``FlareLightCurve``.

        Parameters
        -------------
        minsep : 3 or int
            Minimum distance between two candidate start times in datapoints.

        Returns
        ----------
        FlareLightCurve
        '''
        if ((fake==False) & (self.flares.shape[0]>0)):
            return self
        else:
            lc = copy.copy(self)
            #find continuous observing periods
            lc = lc.find_gaps()
            #find the true median value iteratively
            if fake==False:
                lc = find_iterative_median(lc)
            #find flares
            lc = find_flares(lc)

        return lc

    def sample_flare_recovery(self, iterations=20,
                                    inject_before_detrending=False, **kwargs):
        """
        Runs a number of injection recovery cycles and characterizes the light
        curve by recovery probability and equivalent duration underestimation.

        Parameters
        -----------
        iterations : 20 or int
            Number of injection/recovery cycles
        inject_before_detrending : False or bool
            If True, fake flare are injected directly into raw data.
        kwargs : dict
            Keyword arguments to pass to inject_fake_flares

        Returns
        -------
        combined_irr : DataFrame
            All injected and recovered flares. Complex flare superpositions are
            collapsed.
        fake_lc : FlareLightCurve
            Light curve with the last iteration of synthetic flares injected.
        """
        lc = copy.copy(self)
        lc = lc.find_gaps()
        lc = find_iterative_median(lc)
        columns =  ['istart', 'istop', 'cstart', 'cstop', 'tstart', 'tstop',
                    'ed_rec', 'ed_rec_err', 'duration_d', 'amplitude', 'ed_inj',
                    'peak_time']
        combined_irr = pd.DataFrame(columns=columns)

        widgets = [progressbar.Percentage(), progressbar.Bar()]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=iterations).start()
        for i in range(iterations):
            fake_lc = inject_fake_flares(lc,
                                         inject_before_detrending=inject_before_detrending,
                                         **kwargs)
            injs = fake_lc.fake_flares
            if inject_before_detrending == True:
                fake_lc = fake_lc.detrend()
            fake_lc = fake_lc.find_flares(fake=True)
            recs = fake_lc.flares

            injection_recovery_results = merge_fake_and_recovered_events(injs, recs)
            irr_w_merged_complex_flares = merge_complex_flares(injection_recovery_results)
            combined_irr = combined_irr.append(irr_w_merged_complex_flares,
                                                      ignore_index=True,
                                                      sort=True)

            bar.update(i + 1)
        bar.finish()
        return combined_irr, fake_lc

    def characterize_flares(self, inject_before_detrending=False, **kwargs):
        """
        Add information about recovery probability and systematic energy
        correction for every flare in a light curve using injection/recovery
        sampling of synthetic flares.

        Parameters
        -----------
        inject_before_detrending : False or bool

        kwargs : dict
            Keyword arguments to pass to :py:func:`characterize_one_flare`.

        Return
        -------
        FlareLightCurve
            The flares attribute is modified, now containing `rec_prob`
            and `ed_rec_corr` columns.
        """
        flc = copy.copy(self)
        if ((flc.detrended_flux is None) & (inject_before_detrending==False)):
            LOG.error('Please de-trend light curve first or set '
                          'inject_before_detrending=True. The latter is advised.')
            raise AttributeError('detrended_flux attribute is missing.')
        elif ((flc.detrended_flux is None) & (inject_before_detrending==True)):
            flc = flc.detrend()
            flc = flc.find_flares()
            flc = find_iterative_median(flc)
        if flc.flares.shape[0]==0:
            flc = flc.find_flares()
        if flc.flares.shape[0]>0:
            f2 = pd.DataFrame(columns=flc.flares.columns)
            for i,f in flc.flares.iterrows():
                res = characterize_one_flare(flc,f,
                                             inject_before_detrending=inject_before_detrending,
                                             **kwargs)
                f2 = f2.append(res, ignore_index=True)
            flc.flares = f2
            return flc
        else:
            LOG.info('No flares to characterize.')
            return flc

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
