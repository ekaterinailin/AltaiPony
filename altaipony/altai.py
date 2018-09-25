from astropy.io import fits
import pandas as pd
import numpy as np
from lightkurve import KeplerLightCurve

def get_k2sc_lc(file):

    '''

    Read in light curve as a numpy recarray from a .fits K2SC light curve.

    Parameters:
    ----------
    file : str
        light curve file location of a K2SC fits file

    Return:
    -------
    lc : numpy recarray ()
        light curve with field names ['cadence','quality','x','y','flux',
                                      'error', 'mflags', 'trtime','trposi']
    '''

    hdu = fits.open(file)
    lc = hdu[2].data #'time',

    #critical: remove nans from time and flux
    lc = lc[np.where(np.isfinite(lc.time))]
    lc = lc[np.where(np.isfinite(lc.flux))]
    hdu.close()

    return lc

def find_gaps(time, maxgap=0.09, minspan=10):
    '''

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

    Return:
    -------
    list of tuples with left and right edges of sufficiently long periods
    of continuous observation

    '''

    dt = np.diff(time)
    gap = np.where(np.append(0, dt) >= maxgap)[0]
    # add start/end of LC to loop over easily
    gap_out = np.append(0, np.append(gap, len(time)))

    # left start, right end of data
    left, right = gap_out[:-1], gap_out[1:]

    #drop too short observation periods
    too_short = np.where(np.diff(gap_out) < 10)
    left, right = np.delete(left,too_short), np.delete(right,(too_short))

    return list(zip(left, right))

def find_flares(flux, error, N1=3, N2=1, N3=3):
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
        Coefficient from original paper (Default is 3)
        How many times above the stddev is required.
    N2 : int, optional
        Coefficient from original paper (Default is 1)
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

def wrapper(lc, gapwindow=0.1, minsep=3):
    '''
    Main wrapper to obtain and process a light curve.

    Parameters:
    -------------
    lc : light curve

    minsep : 1 or int
        minimum distance between two candidate start times in datapoints

    Return:
    ----------
    numpy arrays of start and stop cadence numbers of flare candidates
    '''

    #find continuous observing periods
    dlr = find_gaps(lc.time)

    istart = np.array([], dtype='int')
    istop = np.array([], dtype='int')
    #Now work on periods of continuous observation with no gaps
    for (le,ri) in dlr:
        lct = lc[le:ri]
        flux_model_i = np.nanmedian(lct.flux) * np.ones_like(lct.flux)
        flux_diff = lct.flux - flux_model_i
        # run final flare-find on DATA - MODEL
        isflare = find_flares(flux_diff, lct.error, N1=3, N2=4, N3=3)

        # now pick out final flare candidate indices
        candidates = np.where( isflare > 0)[0]

        if (len(candidates) < 1):#no candidates = no indices
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

    return lc.cadence[istart], lc.cadence[istop]

#lc = get_k2sc_lc('examples/hlsp_k2sc_k2_llc_211117077-c04_kepler_v2_lc.fits')
#lc = get_k2sc_lc('examples/hlsp_k2sc_k2_llc_210951703-c04_kepler_v2_lc.fits')
