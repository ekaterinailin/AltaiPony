from astropy.io import fits
import pandas as pd
import numpy as np
import logging
from lightkurve import KeplerLightCurve

LOG = logging.getLogger(__name__)

def find_flares_in_cont_obs_period(flux, error, N1=4, N2=4, N3=3):
    '''
    The algorithm for local changes due to flares defined by
    S. W. Chang et al. (2015), Eqn. 3a-d
    http://arxiv.org/abs/1510.01005

    Note: these equations were originally in magnitude units, i.e. smaller
    values are increases in brightness. The signs have been changed, but
    coefficients have not been adjusted to change from log(flux) to flux.

    Parameters:
    ----------
    flux : numpy array
        data to search over
    error : numpy array
        errors corresponding to data.
    N1 : int, optional
        Coefficient from original paper (Default is 3 in paper, 4 here)
        How many times above the stddev is required.
    N2 : int, optional
        Coefficient from original paper (Default is 1 in paper, 4 here)
        How many times above the stddev and uncertainty is required
    N3 : int, optional
        Coefficient from original paper (Default is 3)
        The number of consecutive points required to flag as a flare


    Return:
    ------------
    isflare : numpy array of booleans
        datapoints are flagged with 1 if they belong to a flare candidate
    '''

    median = np.nanmedian(flux)
    sigma = np.nanstd(flux)
    T0 = flux - median # excursion should be positive #"N0"
    T1 = np.abs(flux - median) / sigma #N1
    T2 = np.abs(flux - median - error) / sigma #N2
    # apply thresholds N0-N2:
    LOG.INFO('Factor above standard deviation: N1 = {},\n'
             'Factor above standard deviation + uncertainty N2 = {},\n'
             'Minimum number of consecutive data points for candidate N3 = {}')
    pass_thresholds = np.where((T0 > 0) & (T1 > N1) & (T2 > N2))
    #array of indices where thresholds are exceeded:
    is_pass_thresholds = np.zeros_like(flux)
    is_pass_thresholds[pass_thresholds] = 1

    # Need to find cumulative number of points that pass_thresholds
    # Counted in reverse!
    # Examples reverse_counts = [0 0 0 3 2 1 0 0 1 0 4 3 2 1 0 0 0 1 0 2 1 0]
    #                 isflare = [0 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0]

    reverse_counts = np.zeros_like(flux, dtype='int')
    for k in range(2, len(flux)):
        reverse_counts[-k] = (is_pass_thresholds[-k]
                             * (reverse_counts[-(k-1)]
                                + is_pass_thresholds[-k]))

    # find flare start where values in reverse_counts switch from 0 to >=N3
    istart_i = np.where((reverse_counts[1:] >= N3) &
                        (reverse_counts[:-1] - reverse_counts[1:] < 0))[0] + 1
    # use the value of reverse_counts to determine how many points away stop is
    istop_i = istart_i + (reverse_counts[istart_i])
    isflare = np.zeros_like(flux, dtype='int')
    for (l,r) in list(zip(istart_i,istop_i)):
        isflare[l:r+1] = 1
    return isflare

def find_flares(lc, minsep=3):
    '''
    Main wrapper to obtain and process a light curve.

    Parameters:
    -------------
    lc : light curve
        FlareLightCurve object
    minsep : 1 or int
        minimum distance between two candidate start times in datapoints

    Return:
    ----------
    numpy arrays of start and stop cadence numbers of flare candidates
    '''

    lc.flares = []
    istart = np.array([], dtype='int')
    istop = np.array([], dtype='int')
    #Now work on periods of continuous observation with no gaps
    for (le,ri) in lc.gaps:
        error = lc.flux_err[le:ri]
        flux = lc.flux[le:ri]
        flux_model_i = np.nanmedian(flux) * np.ones_like(flux)
        flux_diff = flux - flux_model_i
        # run final flare-find on DATA - MODEL
        isflare = find_flares_in_cont_obs_period(flux_diff, error)

        # now pick out final flare candidate indices
        candidates = np.where( isflare > 0)[0]

        if (len(candidates) < 1):#no candidates = no indices
            LOG.info('INFO: No candidates were found in the ({},{}) gap.'.format(le,ri))
            istart_gap = np.array([])
            istop_gap = np.array([])
        else:
            # find start and stop index, combine neighboring candidates
            # in to same events
            separated_candidates = np.where( (np.diff(candidates)) > minsep )[0]
            istart_gap = candidates[ np.append([0], separated_candidates + 1) ]
            istop_gap = candidates[ np.append(separated_candidates,
                                    [len(candidates) - 1]) ]

        #stitch indices back into the original light curve
        istart = np.array(np.append(istart, istart_gap + le), dtype='int')
        istop = np.array(np.append(istop, istop_gap + le), dtype='int')
    lc.flares += list(zip(lc.cadenceno[istart], lc.cadenceno[istop]))

    return lc.flares
