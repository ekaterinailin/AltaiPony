from astropy.io import fits
import pandas as pd
import numpy as np
from lightkurve import KeplerLightCurve

#Ingest a K2SC light curve from file
def get_k2sc_lc(file):
    '''
    Parameters
    ----------
    file : str
        light curve file location for a Vanderburg de-trended .txt file
    Returns
    -------
    lc: light curve DataFrame with columns [time, flux]

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
    lc = hdu[1].data #'time',
 # 'cadence',
 # 'quality',
 # 'x',
 # 'y',
 # 'flux',
 # 'error',
 # 'mflags',
 # 'trtime',
 # 'trposi'

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
    outer edges of gap, left edges, right edges
    (all are indicies)
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

def find_flares(flux, error, N1=3, N2=1, N3=3,
              avg_std=False, std_window=7,
              returnbinary=False):
    '''
    The algorithm for local changes due to flares defined by
    S. W. Chang et al. (2015), Eqn. 3a-d
    http://arxiv.org/abs/1510.01005

    Note: these equations originally in magnitude units, i.e. smaller
    values are increases in brightness. The signs have been changed, but
    coefficients have not been adjusted to change from log(flux) to flux.

    Note: this algorithm originally ran over sections without "changes" as
    defined by Change Point Analysis. May have serious problems for data
    with dramatic starspot activity. If possible, remove starspot first!

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
    avg_std : bool, optional
        Should the "sigma" in this data be computed by the median of
        the rolling().std()? (Default is False)
        (Not part of original algorithm)
    std_window : float, optional
        If avg_std=True, how big of a window should it use?
        (Default is 25 data points)
        (Not part of original algorithm)
    returnbinary : bool, optional
        Should code return the start and stop indicies of flares (default,
        set to False) or a binary array where 1=flares (set to True)
        (Not part of original algorithm)

    Return:
    ------------
    '''

    med_i = np.nanmedian(flux)
    if avg_std is False:
        sig_i = np.nanstd(flux) # just the stddev of the window
    else:
        # take the average of the rolling stddev in the window.
        # better for windows w/ significant starspots being removed
        sig_i = np.nanmedian(pd.Series(flux).rolling(std_window, center=True).std())
    ca = flux - med_i
    cb = np.abs(flux - med_i) / sig_i
    cc = np.abs(flux - med_i - error) / sig_i
    # pass cuts from Eqns 3a,b,c
    ctmp = np.where((ca > 0) & (cb > N1) & (cc > N2))

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
                        (ConM[0:-1] - ConM[1:] < 0))[0] + 1

    # use the value of ConM to determine how many points away stop is
    istop_i = istart_i + (ConM[istart_i] - 1)

    istart_i = np.array(istart_i, dtype='int')
    istop_i = np.array(istop_i, dtype='int')

    if returnbinary is False:
        return istart_i, istop_i
    else:
        bin_out = np.zeros_like(flux, dtype='int')
        for k in range(len(istart_i)):
            bin_out[istart_i[k]:istop_i[k]+1] = 1
        return bin_out

def wrapper(lc, objectid='', lctype='',
          display=False, readfile=False, debug=False, dofake=True,
          dbmode='fits', gapwindow=0.1,minsep=3,
          fakefreq=.25, mode='davenport', iterations=10):
    '''
    Main wrapper to obtain and process a light curve.

    Parameters:
    -------------


    minsep : 3 or int
        minimum distance between two candidates in datapoints
    '''

    #find continuous observing periods
    dlr = find_gaps(lc.time, maxgap=0.03)

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
        isflare = find_flares(flux_diff, lct.error, N1=3, N2=4, N3=3,
                              returnbinary=True, avg_std=True)

        # now pick out final flare candidate points from above
        cand1 = np.where( isflare > 0)[0]
        #delete candidates too close to edges of time array:
        x1 = np.where( (np.abs(time[cand1] - time[-1]) < gapwindow) )
        x2 = np.where( (np.abs(time[cand1] - time[0]) < gapwindow) )
        cand1 = np.delete(cand1, x1)
        cand1 = np.delete(cand1, x2)
        if (len(cand1) < 1):#no candidates = no indices
            istart_i = np.array([])
            istop_i = np.array([])
        else:
            # find start and stop index, combine neighboring candidates in to same events
            separated_cand = np.where( (cand1[1:] - cand1[:-1] > minsep) )[0]
            istart_i = cand1[ np.append([0], separated_cand + 1) ]
            istop_i = cand1[ np.append(separated_cand, [len(cand1) - 1]) ]
        # if start & stop times are the same, add 1 more datum on the end
        to1 = np.where((istart_i-istop_i == 0))
        if len(to1[0])>0:
            istop_i[to1] += 1

        istart = np.array(np.append(istart, istart_i + le), dtype='int')
        istop = np.array(np.append(istop, istop_i + le), dtype='int')
        flux_model[le:ri] = flux_model_i

    return istart, istop

#lc = get_k2sc_lc('examples/hlsp_k2sc_k2_llc_211117077-c04_kepler_v2_lc.fits')
#lc = get_k2sc_lc('examples/hlsp_k2sc_k2_llc_210951703-c04_kepler_v2_lc.fits')
lc = get_k2sc_lc('examples/hlsp_k2sc_k2_llc_211119999-c04_kepler_v2_lc.fits')

start, stop = wrapper(lc)
print(start, stop)
