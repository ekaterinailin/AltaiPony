import numpy as np
import pandas as pd
import os
import copy
import logging

from k2sc.standalone import k2sc_lc
from lightkurve import KeplerLightCurve, KeplerTargetPixelFile
from astropy.io import fits

from .altai import find_flares
from .fakeflares import inject_fake_flares, merge_fake_and_recovered_events

LOG = logging.getLogger(__name__)

class FlareLightCurve(KeplerLightCurve):
    """
    Flare light curve class that unifies properties of ``K2SC``-de-trended and
    Kepler's ``lightkurve.KeplerLightCurve``.

    Attributes
    -------------
    time : array-like
        Time measurements
    flux : array-like
        Data flux for every time point
    flux_err : array-like
        Uncertainty on each flux data point
    time_format : str

    time_scale : str

    time_unit : astropy.unit
        Astropy unit object defining unit of time
    centroid_col : array-like

    centroid_row : array-like

    quality : array-like

    quality_bitmask : str
        Can be 'none', 'default', 'hard' or 'hardest'
    channel : int

    campaign : int
        K2 campaign number
    quarter : int
        Kepler Quarter number
    mission :

    cadenceno :

    targetid : int
        EPIC ID number
    ra : float

    dec : float

    label :

    meta : dict

    detrended_flux : array-like

    detrended_flux_err : array-like

    flux_trends : array-like

    gaps : list of tuples of ints
        Each tuple contains the start and end indices of observation gaps. See
        ``find_gaps``
    flares : DataFrame

    """
    def __init__(self, time=None, flux=None, flux_err=None, time_format=None,
                 time_scale=None, time_unit = None, centroid_col=None,
                 centroid_row=None, quality=None, quality_bitmask=None,
                 channel=None, campaign=None, quarter=None, mission=None,
                 cadenceno=None, targetid=None, ra=None, dec=None, label=None,
                 meta={}, detrended_flux=None, detrended_flux_err=None,
                 flux_trends=None, gaps=None, flares=None, flux_unit = None,
                 primary_header=None, data_header=None, pos_corr1=None,
                 pos_corr2=None, origin='FLC', fake_flares=None):

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
                if new_lc.detrended_flux.shape != self.flux.shape:
                    LOG.error('De-detrending messed up the flux arrays.')
                else:
                    LOG.info('De-trending successfully completed.')

            except np.linalg.linalg.LinAlgError:
                LOG.error('Detrending failed because probably Cholesky '
                          'decomposition failed. Try again, you shall succeed.')
            new_lc.__class__ = FlareLightCurve
            return new_lc

    def find_flares(self, minsep=3):

        '''
        Find flares in a FlareLightCurve.

        Parameters
        -------------
        minsep : 3 or int
            minimum distance between two candidate start times in datapoints

        Return
        ----------
        numpy arrays of start and stop cadence numbers of flare candidates
        '''
        lc = copy.copy(self)
        #find continuous observing periods
        lc = lc.find_gaps()
        #find flares
        lc.flares = find_flares(lc)

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

        Return
        -------
        ed_correction_factor : numpy 2D array
            Contains [recovered equivalent duration, systematic correction factor].
        recovery_probability : numpy 2D array
            Contains [corrected equivalent duration, recovery probability].
        """
        lc = copy.copy(self)
        lc = lc.find_gaps()
        columns =  ['istart', 'istop', 'cstart', 'cstop', 'tstart', 'tstop',
                    'ed_rec', 'ed_rec_err', 'duration_d', 'amplitude', 'ed_inj',
                    'peak_time']
        combined_irr = pd.DataFrame(columns=columns)
        for i in range(iterations):
            fake_lc = inject_fake_flares(lc,
                                         inject_before_detrending=inject_before_detrending,
                                         **kwargs)
            injs = fake_lc.fake_flares
            if inject_before_detrending == True:
                fake_lc = fake_lc.detrend()
            fake_lc = fake_lc.find_flares()
            recs = fake_lc.flares
            injection_recovery_results = merge_fake_and_recovered_events(injs, recs)
            combined_irr = combined_irr.append(injection_recovery_results,
                                                      ignore_index=True,
                                                      sort=True)
        return combined_irr

    def test_characterize_flare_recovery():
        #analysis function
        return
