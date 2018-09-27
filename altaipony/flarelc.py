from lightkurve import KeplerLightCurve
from astropy.io import fits
import numpy as np
import os

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
                 targetid=None, ra=None, dec=None, label=None, meta={},
                 gaps=None, flares=None):

        super(FlareLightCurve, self).__init__(time=time, flux=flux, flux_err=flux_err, time_format=time_format, time_scale=time_scale,
                                              centroid_col=centroid_col, centroid_row=centroid_row, quality=quality,
                                              quality_bitmask=quality_bitmask, channel=channel, campaign=campaign, quarter=quarter,
                                              mission=mission, cadenceno=cadenceno, targetid=targetid, ra=ra, dec=dec, label=label,
                                              meta=meta)
        self.gaps = gaps
        self.flares = flares #pd.DataFrame(columns=['istart','istop','cstart','cstop', 'ed'])

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
