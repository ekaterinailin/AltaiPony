import numpy as np
import os
import copy

from k2sc.standalone import k2sc_lc
from lightkurve import KeplerLightCurve, KeplerTargetPixelFile
from astropy.io import fits


class FlareLightCurve(KeplerLightCurve):
    """
    Flare light curve class that unifies properties of ``K2SC``-de-trended and
    Kepler's ``lightkurve.KeplerLightCurve``.

    Attributes:
    -------------
    time : array-like
        Time measurements
    flux : array-like
        Data flux for every time point
    flux_err : array-like
        Uncertainty on each flux data point
    targetid : int
        Kepler ID number
    """
    def __init__(self, time=None, flux=None, flux_err=None, time_format=None, time_scale=None,
                 centroid_col=None, centroid_row=None, quality=None, quality_bitmask=None,
                 channel=None, campaign=None, quarter=None, mission=None, cadenceno=None,
                 targetid=None, ra=None, dec=None, label=None, meta={}, detrended_flux=None,
                 flux_trends = None, gaps=None, flares=None):

        super(FlareLightCurve, self).__init__(time=time, flux=flux, flux_err=flux_err, time_format=time_format, time_scale=time_scale,
                                              centroid_col=centroid_col, centroid_row=centroid_row, quality=quality,
                                              quality_bitmask=quality_bitmask, channel=channel, campaign=campaign, quarter=quarter,
                                              mission=mission, cadenceno=cadenceno, targetid=targetid, ra=ra, dec=dec, label=label,
                                              meta=meta)
        self.gaps = gaps
        self.flares = flares #pd.DataFrame(columns=['istart','istop','cstart','cstop', 'ed'])
        self.detrended_flux = detrended_flux
        self.flux_trends = flux_trends

    def __repr__(self):
        return('FlareLightCurve(ID: {})'.format(self.targetid))

    def find_gaps(self, maxgap=0.09, minspan=10):
        '''
        Find gaps in light curve and stores them in the gaps attribute.

        Parameters:
        ----------
        time : numpy array with floats
            sorted array, in units of days
        maxgap : 0.09 or float
            maximum time gap between two datapoints in days,
            default equals approximately 2h
        minspan : 10 or int
            minimum number of datapoints in continuous observation,
            i.e., w/o gaps as defined by maxgap

        '''

        dt = np.diff(self.time)
        gap = np.where(np.append(0, dt) >= maxgap)[0]
        # add start/end of LC to loop over easily
        gap_out = np.append(0, np.append(gap, len(self.time)))

        # left start, right end of data
        left, right = gap_out[:-1], gap_out[1:]

        #drop too short observation periods
        too_short = np.where(np.diff(gap_out) < 10)
        left, right = np.delete(left,too_short), np.delete(right,(too_short))
        self.gaps = list(zip(left, right))

        return

    def detrend(self):
        #make sure there is no detrended_flux already
        # = make sure you only pass KeplerLightCurve derived FLCs
        tpf = KeplerTargetPixelFile.from_archive(self.targetid)
        new_lc = copy.copy(self)
        new_lc.keplerid = self.targetid
        new_lc.primary_header = tpf.hdu[0].header
        new_lc.data_header = tpf.hdu[1].header
        new_lc.pos_corr1 = tpf.hdu[1].data['POS_CORR1'][tpf.quality_mask]
        new_lc.pos_corr2 = tpf.hdu[1].data['POS_CORR2'][tpf.quality_mask]
        del tpf

        #K2SC MAGIC
        new_lc.__class__ = k2sc_lc
        new_lc.k2sc(de_niter=3) #de_niter set low for testing purpose
        # something like assert new_lc.time == self.time is needed here
        self.detrended_flux = (new_lc.corr_flux - new_lc.tr_time
                              + np.nanmedian(new_lc.tr_time))
        return
