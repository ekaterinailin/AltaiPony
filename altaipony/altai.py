from astropy.io import fits
import pandas as pd
import numpy as np
from lightkurve import KeplerLightCurve

#Ingest a K2SC light curve from file
def get_k2sc_lc(file):
    '''
    Parameters:
    ----------
    file : str
        light curve file location of a K2SC fits file

    Returns:
    -------
    lc : numpy recarray ()
        light curve with field names ['cadence','quality','x','y','flux',
                                      'error', 'mflags', 'trtime','trposi']

    # copy from k2sc/standalone.py
    tpf = KeplerTargetPixelFile.from_archive(212300977) # WASP-55
    lc = tpf.to_lightcurve() # load some data either as a tpf or just straight up as a lightcurve
    lc.primary_header = tpf.hdu[0].header
    lc.data_header = tpf.hdu[1].header
    lc.pos_corr1 = tpf.hdu[1].data['POS_CORR1'][tpf.quality_mask]
    lc.pos_corr2 = tpf.hdu[1].data['POS_CORR2'][tpf.quality_mask]

    # now the magic happens
    lc.__class__ = k2sc_lc
    lc.k2sc()
    '''

    hdu = fits.open(file)
    lc = hdu[2].data #'time',

    #remove nans
    lc = lc[np.where(np.isfinite(lc.time))]
    lc = lc[np.where(np.isfinite(lc.flux))]
    hdu.close()
    return lc

def find_gaps(time, maxgap=0.125, minspan=10):
    '''

    Parameters
    ----------
    time : numpy array with floats
        sorted array, in units of days
    maxgap : 0.125 or float
        maximum time gap between two datapoints in days
    minspan : 10 or int
        minimum number of datapoints in continuous observation,
        i.e., w/o gaps as defined by maxgap

    Returns
    -------
    list of tuples with left and right edges of sufficiently long periods
    of continuous observation

    '''
    dt = time[1:] - time[:-1]
    dt = np.append(0, dt)
    gap = np.where(dt >= maxgap)[0]

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

    Parameters
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
    '''

    median = np.nanmedian(flux)
    sigma = np.nanstd(flux) # just the stddev of the window
    ca = flux - median # excursion should be positive #"N0"
    cb = np.abs(flux - median) / sigma #N1
    cc = np.abs(flux - median - error) / sigma #N2
    # apply thresholds N0-N2:
    ctmp = np.where((ca > 0) & (cb > N1) & (cc > N2))
    #array of indices where thresholds are exceeded:
    cindx = np.zeros_like(flux)
    cindx[ctmp] = 1

    # Need to find cumulative number of points that pass "ctmp"
    # Count in reverse!
    ConM = np.zeros_like(flux)
    # this requires a full pass thru the data -> bottleneck
    for k in range(2, len(flux)):
        ConM[-k] = cindx[-k] * (ConM[-(k-1)] + cindx[-k])
    # these only defined between dl[i] and dr[i]
    # find flare start where values in ConM switch from 0 to >=N3
    istart_i = np.where((ConM[1:] >= N3) &
                        (ConM[:-1] - ConM[1:] < 0))[0] + 1
    # use the value of ConM to determine how many points away stop is
    istop_i = istart_i + (ConM[istart_i])

    istart_i = np.array(istart_i, dtype='int')
    istop_i = np.array(istop_i, dtype='int')


    bin_out = np.zeros_like(flux, dtype='int')
    for k in range(len(istart_i)):
        bin_out[istart_i[k]:istop_i[k]+1] = 1
    return bin_out

def wrapper(lc, gapwindow=0.1, minsep=3):
    '''
    Main wrapper to obtain and process a light curve.

    Parameters:
    -------------
    lc : light curve

    minsep : 1 or int
        minimum distance between two candidate start times in datapoints
    '''

    #find continuous observing periods
    dlr = find_gaps(lc.time)

    istart = np.array([], dtype='int')
    istop = np.array([], dtype='int')
    flux_model = np.zeros_like(lc.flux)

    for (le,ri) in dlr:
        lct = lc[le:ri]
        time, flux  = lct.time, lct.flux,
        error, flags = lct.quality, lct.quality

        flux_model_i = np.nanmedian(lct.flux) * np.ones_like(lct.flux)
        flux_diff = lct.flux - flux_model_i
        # run final flare-find on DATA - MODEL
        isflare = find_flares(flux_diff, lct.error, N1=3, N2=4, N3=3)

        # now pick out final flare candidate points from above
        candidates = np.where( isflare > 0)[0]
        #delete candidates too close to edges of time array:
        x1 = np.where( (np.abs(time[candidates] - time[-1]) < gapwindow) )
        x2 = np.where( (np.abs(time[candidates] - time[0]) < gapwindow) )
        cand1 = np.delete(candidates, x1)
        cand1 = np.delete(candidates, x2)
        if (len(candidates) < 1):#no candidates = no indices
            istart_i = np.array([])
            istop_i = np.array([])
        else:
            # find start and stop index, combine neighboring candidates
            # in to same events
            separated_candidates = np.where( ((candidates[1:] - candidates[:-1]) >= minsep) )[0]
            istart_i = cand1[ np.append([0],
                                        separated_candidates + 1) ]
            istop_i = cand1[ np.append(separated_candidates,
                                       [len(candidates) - 1]) ]

        # if start & stop times are the same, add 1 more datum on the end
        to1 = np.where((istart_i-istop_i == 0))
        if len(to1[0])>0:
            istop_i[to1] += 1

        istart = np.array(np.append(istart, istart_i + le), dtype='int')
        istop = np.array(np.append(istop, istop_i + le), dtype='int')
        flux_model[le:ri] = flux_model_i
    return istart, istop

lc = get_k2sc_lc('examples/hlsp_k2sc_k2_llc_211117077-c04_kepler_v2_lc.fits')
#lc = get_k2sc_lc('examples/hlsp_k2sc_k2_llc_210951703-c04_kepler_v2_lc.fits')
#lc = get_k2sc_lc('examples/hlsp_k2sc_k2_llc_211119999-c04_kepler_v2_lc.fits')

start, stop = wrapper(lc)
