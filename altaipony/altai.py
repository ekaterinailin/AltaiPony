
import pandas as pd
import numpy as np
import logging
import copy

from scipy.stats import chi2, norm

from .utils import MAD_TO_STD, clipped_stats


LOG = logging.getLogger(__name__)

def _local_beta(z, n, window=720, min_blocks=10):
    """Local ratio of the scatter of ``n``-cadence block means of ``z`` to
    the white-noise expectation ``1 / sqrt(n)``, per cadence, at least 1.

    ``z`` is split into non-overlapping blocks of ``n`` cadences.  Around
    each block, over about ``window`` cadences, the robust scatter of the
    block means (1.4826 times their median absolute deviation from the
    running median) is compared with ``1 / sqrt(n)``.  Each cadence gets the
    value of its block.  Blocks with non-finite values are left out; where
    fewer than ``min_blocks`` are available, 1 is used.
    """
    beta = np.ones(len(z))
    n_blocks = len(z) // n
    k = max(2 * min_blocks, window // n)
    if n_blocks < k:
        return beta
    blocks = z[: n_blocks * n].reshape(n_blocks, n)
    means = pd.Series(np.where(np.isfinite(blocks).all(axis=1), blocks.mean(axis=1), np.nan))
    med = means.rolling(k, center=True, min_periods=min_blocks).median()
    mad = (means - med).abs().rolling(k, center=True, min_periods=min_blocks).median()
    beta_blocks = (MAD_TO_STD * mad * np.sqrt(n)).clip(lower=1.0).fillna(1.0).to_numpy()
    beta[: n_blocks * n] = np.repeat(beta_blocks, n)
    beta[n_blocks * n:] = beta_blocks[-1]
    return beta


def _window_flare_candidates(flux, median, sigma, N1=3, N3=3, max_window=10,
                             beta_window=720):
    """Flag cadences in windows of ``N3`` to ``max_window`` consecutive
    points that are unlikely to be noise.

    Each point's excursion ``z = (flux - median) / sigma`` is capped at
    ``N1`` (points below the median count as no excursion) and converted
    to its one-sided Gaussian tail probability ``p``.  Fisher's statistic
    ``sum(-2 ln p)`` over a window of ``n`` points follows a chi-square
    distribution with ``2n`` degrees of freedom under Gaussian noise.  A
    window is flagged if its noise probability is below that of ``N3``
    points at ``N1`` sigma, so that for ``n = N3`` the criterion is exactly
    ``N3`` consecutive points above ``N1`` sigma, and longer windows need
    a smaller but sustained excess (e.g. 10 points at 1.8 sigma for N1 = 3,
    N3 = 3).  With the cap, a single extreme cadence cannot carry a window.
    Windows with non-finite values are skipped.  Returns a boolean array.

    The test assumes white noise.  Residuals that are correlated over
    several cadences (e.g. detrending leftovers at the turning points of a
    fast, large modulation) make long windows look significant.  So for each
    window length ``n`` longer than ``N3`` the excursions are first divided
    by a local ``beta_n`` (see :func:`_local_beta`): how much more the means
    of ``n``-cadence blocks scatter within about ``beta_window`` cadences
    than white noise would.  ``beta_n`` is 1 for white noise and never below
    1.  Windows of ``N3`` points keep the plain criterion, so the detector is
    never stricter than ``N3`` consecutive points above ``N1`` sigma.
    """
    flux = np.asarray(getattr(flux, "value", flux), dtype=float)
    median = np.asarray(getattr(median, "value", median), dtype=float)
    sigma = np.asarray(getattr(sigma, "value", sigma), dtype=float)
    n_pts = len(flux)
    isflare = np.zeros(n_pts, dtype=bool)
    z = (flux - median) / sigma
    finite = np.isfinite(z)
    z0 = np.where(finite, z, 0.0)
    p_threshold = chi2.sf(-2.0 * N3 * norm.logsf(N1), 2 * N3)
    cbad = np.concatenate([[0], np.cumsum(~finite)])
    for n in range(N3, max_window + 1):
        if n > n_pts:
            break
        beta = 1.0 if n == N3 else _local_beta(z, n, window=beta_window)
        score = -2.0 * norm.logsf(np.clip(z0 / beta, 0.0, N1))
        csum = np.concatenate([[0.0], np.cumsum(score)])
        window_score = csum[n:] - csum[:-n]
        window_ok = (cbad[n:] - cbad[:-n]) == 0
        # >= with a small tolerance: N3 points capped at N1 sit exactly on
        # the threshold and must pass
        critical = chi2.isf(p_threshold, 2 * n) * (1 - 1e-9)
        hits = np.flatnonzero(window_ok & (window_score >= critical))
        for i in hits:
            isflare[i:i + n] = True
    return isflare


def find_flares_in_cont_obs_period(flux, median, error, sigma=None,
                                   N1=3, N2=2, N3=3, addtail=False,
                                   tailthreshdiff=1., max_window=None,
                                   beta_window=720):
    '''
    The algorithm for local changes due to flares defined by
    S. W. Chang et al. (2015), Eqn. 3a-d
    https://ui.adsabs.harvard.edu/abs/2015ApJ...814...35C/abstract

    Note: these equations were originally in magnitude units, i.e. smaller
    values are increases in brightness. The signs have been changed, but
    coefficients have not been adjusted to change from log(flux) to flux.

    Parameters:
    ----------
    flux : numpy array
        data to search over
    median : float or numpy array
        median value of quiescent stellar flux
    error : numpy array
        errors corresponding to data.
    sigma : numpy array
        local scatter of the flux. Array should be the same length as the
        detrended flux array. 
        If sigma=None, error is used instead.
    N1 : int or float (default is 3)
        How many times above sigma is required.
    N2 : int or float (Default is 3)
        How many times above sigma and detrended_flux_err is required
    N3 : int or float (Default is 2)
        The number of consecutive points required to flag as a flare.
    addtail : bool (Default is False)
        Optionally, add data points to the flares with a lower threshold
    tailthreshdiff: int ot float (Default is 1)
        relaxes the detection threshold for datapoints that are added to
        the decay tails of flare candidates. Tailthreshdiff is subtracted
        from N1 and N2, and should not be larger than either of the two.
    max_window : int or None (Default is None)
        If given, flag windows of ``N3`` to ``max_window`` consecutive
        points whose combined probability of being noise is below that of
        ``N3`` points at ``N1`` sigma (see :func:`_window_flare_candidates`)
        instead of requiring ``N3`` consecutive points above the thresholds.
        ``N2`` and ``addtail`` are not used then.
    beta_window : int (Default is 720)
        With ``max_window``, the number of cadences over which the local
        correction for correlated residuals is measured (see
        :func:`_local_beta`).


    Return:
    ------------
    isflare : numpy array of booleans
        datapoints are flagged with 1 if they belong to a flare candidate
    '''
    isflare = np.zeros_like(flux, dtype='bool')
  
    # If no local scatter characteristics are given, use formal error as sigma    
    if sigma is None:
        sigma = error
    if max_window is not None:
        return _window_flare_candidates(flux, median, sigma, N1=N1, N3=N3,
                                        max_window=max_window,
                                        beta_window=beta_window)
    T0 = flux - median # excursion should be positive #"N0"
    T1 = np.abs(flux - median) / sigma #N1
    T2 = np.abs(flux - median - error) / sigma #N2
    
    # apply thresholds N0-N2:
    LOG.debug(f'Factor above standard deviation: N1 = {N1},\n'
              f'Factor above standard deviation + uncertainty N2 = {N2},\n'
              f'Minimum number of consecutive data points for candidate N3 = {N3}')
              
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
    
    # OPTION: add decay phase data point with a lower detection threshold
    if addtail == True:
        
        # check for bad values of tailthreshdiff
        if ((tailthreshdiff > N1) | (tailthreshdiff > N2)):
            raise ValueError("The threshold on the decay tail should be "
                             ">0. Note N1tail = N1-tailthreshdiff and "
                             "same for N2.")
        
        # calculate new N1 and N2 for the tails
        N1tail, N2tail = N1 - tailthreshdiff, N2 - tailthreshdiff

        # add data points from tail until threshold no longer satisfied
        newstops = []
        for s in istop_i:
            while ((T0[s] > 0) & (T1[s] > N1tail) & (T2[s] > N2tail)):
                s += 1
            newstops.append(s)
        
        # overwrite old flare stop indices
        istop_i = newstops

    # Create boolean flare mask
    isflare = np.zeros_like(flux, dtype='bool')
    
    for (l,r) in list(zip(istart_i, istop_i)):
        isflare[l:r+1] = True
        
    return isflare

def _merge_elevated_gaps(istart, istop, flux, median, error, merge_sigma):
    """Merge consecutive flare candidates if the flux between them stays
    elevated, i.e. the mean of ``(flux - median) / error`` over the cadences
    between the end of one candidate and the start of the next is at least
    ``merge_sigma``.  Merges chain, so several fragments of one long flare
    become one candidate.  ``istart``/``istop`` are inclusive indices into
    ``flux``."""
    if merge_sigma is None or len(istart) < 2:
        return istart, istop
    x = (np.asarray(getattr(flux, "value", flux), dtype=float)
         - np.asarray(getattr(median, "value", median), dtype=float)) / np.asarray(
        getattr(error, "value", error), dtype=float)
    starts, stops = [istart[0]], [istop[0]]
    for a, b in zip(istart[1:], istop[1:]):
        gap = x[stops[-1] + 1:a]
        if np.isfinite(gap).any() and np.nanmean(gap) >= merge_sigma:
            stops[-1] = b
        else:
            starts.append(a)
            stops.append(b)
    return np.array(starts), np.array(stops)


def _extend_candidates(istart, istop, flux, median, error, extend_sigma):
    """Extend each candidate backwards and forwards over the neighbouring
    cadences whose excursion ``(flux - median) / error`` is at least
    ``extend_sigma``, then join candidates that touch or overlap.  The
    extension never starts a new candidate; it only moves the edges of
    existing ones, e.g. back over the peak of a faint flare whose detection
    started on the decay.  ``istart``/``istop`` are inclusive indices into
    ``flux``."""
    if extend_sigma is None or len(istart) == 0:
        return istart, istop
    x = (np.asarray(getattr(flux, "value", flux), dtype=float)
         - np.asarray(getattr(median, "value", median), dtype=float)) / np.asarray(
        getattr(error, "value", error), dtype=float)
    above = x >= extend_sigma  # NaN counts as below
    starts, stops = [], []
    for a, b in zip(istart, istop):
        a, b = int(a), int(b)
        while a > 0 and above[a - 1]:
            a -= 1
        while b < len(x) - 1 and above[b + 1]:
            b += 1
        if stops and a <= stops[-1] + 1:
            stops[-1] = max(stops[-1], b)
        else:
            starts.append(a)
            stops.append(b)
    return np.array(starts), np.array(stops)


def find_flares(flc, minsep=3, sigma=None, merge_sigma=0.25, extend_sigma=1.0,
                max_window=10, beta_window_days=1.0, **kwargs):
    '''
    Main wrapper to obtain and process a light curve.

    By default, candidates are windows of 3 to 10 consecutive cadences whose
    combined probability of being noise is below that of 3 points at 3 sigma
    (see :func:`_window_flare_candidates`), fragments are merged where the
    flux between them averages at least 0.25 sigma, and detections are
    extended over adjacent cadences above 1 sigma.  ``max_window=None``,
    ``merge_sigma=None`` and ``extend_sigma=None`` give the classic rule of
    Chang et al. (2015): 3 consecutive points above 3 sigma.

    Parameters
    -------------
    flc : light curve
        FlareLightCurve object
    minsep : 1 or int
        minimum distance between two candidate start times in datapoints
    sigma : numpy array
        local scatter of the flux. Array should be the same length as the
        detrended flux array.
    merge_sigma : float or None
        If given, merge consecutive candidates when the flux between them
        stays elevated: the mean of (detrended flux - iterative median) over
        the cadences between them is at least ``merge_sigma`` times the
        noise (``sigma`` if given, else ``detrended_flux_err``).  Long, faint
        flares are otherwise split into several short candidates wherever the
        flux dips below the detection threshold.  Default 0.25; None
        disables merging.
    extend_sigma : float or None
        If given, extend each candidate (after merging) backwards and
        forwards over all adjacent cadences whose excursion above the
        iterative median is at least ``extend_sigma`` times the noise, and
        join candidates that then touch.  Faint flares are otherwise often
        detected only on part of the decay, without their peak.  Default
        1.0; None disables the extension.
    max_window : int or None
        Longest window, in cadences, of the window detector (see
        :func:`find_flares_in_cont_obs_period`).  Default 10; None uses the
        classic rule of ``N3`` consecutive points above the thresholds.
    beta_window_days : float
        With ``max_window``, the time span over
        which the window detector measures correlated residuals, converted
        to cadences for each continuous window (see
        :func:`_window_flare_candidates`).  Default 1 day.
    kwargs : dict
        keyword arguments to pass to :func:`find_flares_in_cont_obs_period`
    Return
    ----------
    numpy arrays of start and stop cadence numbers of flare candidates
    '''
    lc = copy.deepcopy(flc)
    istart = np.array([], dtype='int')
    istop = np.array([], dtype='int')

    if lc.detrended_flux is None or np.isnan(lc.detrended_flux).all():
        raise TypeError('Flare finding only works on de-trended light curves: '
                        'call detrend() first.')
    if lc.detrended_flux_err is None:
        raise TypeError('No detrended_flux_err: estimate the noise first, e.g. '
                        'with altaipony.detrending.estimate_detrended_noise.')

    #Now work on periods of continuous observation with no gaps
    for (le,ri) in lc.cont_windows:
        kwargs["max_window"] = max_window
        if max_window is not None and ri - le > 1:
            cadence = np.nanmedian(np.diff(lc.time.value[le:ri]))
            kwargs["beta_window"] = max(1, int(round(beta_window_days / cadence)))
        error = lc.detrended_flux_err[le:ri]
        flux = lc.detrended_flux[le:ri]
        
        median = lc.it_med[le:ri]
        # run final flare-find on DATA - MODEL

        if sigma is None:
            isflare = find_flares_in_cont_obs_period(flux, median, error, **kwargs)
        else:
            isflare = find_flares_in_cont_obs_period(flux, median, error, sigma=sigma[le:ri], **kwargs)
  

        # now pick out final flare candidate indices
        candidates = np.where( isflare > 0)[0]
        if (len(candidates) < 1):#no candidates = no indices
            LOG.debug(f'INFO: No candidates were found in the ({le},{ri}) gap.')
            istart_gap = np.array([])
            istop_gap = np.array([])
        else:
            # find start and stop index, combine neighboring candidates
            # in to same events
            separated_candidates = np.where( (np.diff(candidates)) > minsep )[0]
            istart_gap = candidates[ np.append([0], separated_candidates + 1) ]
            istop_gap = candidates[ np.append(separated_candidates,
                                    [len(candidates) - 1]) ]
            istart_gap, istop_gap = _merge_elevated_gaps(
                istart_gap, istop_gap, flux, median,
                error if sigma is None else sigma[le:ri], merge_sigma,
            )
            istart_gap, istop_gap = _extend_candidates(
                istart_gap, istop_gap, flux, median,
                error if sigma is None else sigma[le:ri], extend_sigma,
            )

        #stitch indices back into the original light curve
        istart = np.array(np.append(istart, istart_gap + le), dtype='int')
        istop = np.array(np.append(istop, istop_gap + le), dtype='int')
        LOG.info('Found {} candidate(s) in the ({},{}) gap.'
                 .format(len(istart_gap), le, ri))

    if len(istart) > 0:
        l = [equivalent_duration(lc, i, j, err=True) for (i,j) in zip(istart, istop)]
        ed_rec, ed_rec_err = zip(*l)
        fl = lc.detrended_flux
        ampl_rec = [np.max(fl[i:j]) / lc.it_med[i] - 1. for (i,j) in zip(istart,istop)]
        cstart = lc.cadenceno[istart].value
        cstop = lc.cadenceno[istop].value
        tstart = lc.time[istart].value
        tstop = lc.time[istop].value

        new = pd.DataFrame({'ed_rec': ed_rec,
                           'ed_rec_err': ed_rec_err,
                           'ampl_rec': ampl_rec,
                           'istart': istart,
                           'istop': istop,
                           'cstart': cstart,
                           'cstop': cstop,
                           'tstart': tstart,
                           'tstop': tstop,
                           'total_n_valid_data_points': lc.flux.value.shape[0],
                           'dur': tstop - tstart
                          })

        lc.flares = pd.concat([lc.flares, new], ignore_index=True)

    return lc


def _find_iterative_median(detrended_flux, cont_windows, n=10, **kwargs):
    """
    Internal function to compute iterative median from arrays.
    
    Parameters
    ----------
    detrended_flux : np.ndarray
        Detrended flux array
    cont_windows : list of tuples or None
        List of (start_idx, end_idx) tuples defining continuous windows
    **kwargs : dict
        Keyword arguments to pass to sigma_clip (via clipped_stats)
    
    Returns
    -------
    it_med : np.ndarray
        Iterative median array matching the length of detrended_flux
    """
    if len(detrended_flux) == 0:
        raise ValueError("Input detrended_flux array is empty.")

    # Initialize it_med with global median
    it_med = np.full_like(detrended_flux, np.nanmedian(detrended_flux))
    
    # Median of each continuous window (the whole array if none are given)
    # that is not skewed by outliers
    if cont_windows is None or len(cont_windows) == 0:
        cont_windows = [(0, len(detrended_flux))]
    for (le, ri) in cont_windows:
        med, _, good = clipped_stats(detrended_flux[le:ri], max_iter=n, **kwargs)
        if good.any():  # else keep the global median
            it_med[le:ri] = med

    return it_med



def equivalent_duration(lc, start, stop, err=False):

    '''
    Returns the equivalend duratio of a flare event,
    found within indices [start, stop],
    calculated as the area under the residual (flux-flux_median).
    Use only on de-trended light curves!
    Returns also the uncertainty on ED following Davenport (2016)

    Parameters
    --------------
    start : int
        start time index of a flare event
    stop : int
        end time index of a flare event
    lc : FlareLightCurve

    err: False or bool
        If True will compute uncertainty on ED

    Return
    --------------
    ed : float
        equivalent duration in seconds
    ederr : float
        uncertainty in seconds
    '''

    start, stop = int(start),int(stop)+1
    lct = lc[start:stop]
    residual = lct.detrended_flux / np.nanmedian(lct.it_med)-1.
    x = lct.time.value * 60.0 * 60.0 * 24.0
    ed = np.sum(np.diff(x) * residual[:-1])

    if err == True:
        flare_chisq = chi_square(residual[:-1],
                                 lct.detrended_flux_err[:-1]/np.nanmedian(lct.it_med))
        ederr = np.sqrt(ed**2 / (stop-1-start) / flare_chisq)
        return ed, ederr
    else:
        return ed
    


def chi_square(residual, error):
    '''
    Compute the normalized chi square statistic:
    chisq =  1 / N * SUM(i) ( (data(i) - model(i))/error(i) )^2
    '''
    return np.sum( (residual / error)**2.0 ) / np.size(error)


def measure_flare(flc, sta, sto):
    """Give start and stop indices into a de-trended
    light curve, calculate flare properties assuming that
    what's inbetween is a flares, and add the result
    to FlareLightCurve.flares.

    Parameters:
    -------------
    flc : FlareLightCurve
        de-trended light curve
    sta : int
        start index of flare
    sto : int
        stop index of flare
    """
    # get ED
    ed_rec, ed_rec_err = equivalent_duration(flc, sta, sto, err=True)

    # get amplitude
    ampl_rec = np.max(flc.detrended_flux.value[sta:sto]) / flc.it_med.value[sta] - 1.0

    # get cadence numbers
    cstart = flc.cadenceno.value[sta]
    cstop = flc.cadenceno.value[sto]

    # get time stamps
    tstart = flc.time.value[sta]
    tstop = flc.time.value[sto]

    # add result to flare table
    newline = pd.Series(
        {
            "ed_rec": ed_rec,
            "ed_rec_err": ed_rec_err,
            "ampl_rec": ampl_rec,
            "istart": sta,
            "istop": sto,
            "cstart": cstart,
            "cstop": cstop,
            "tstart": tstart,
            "tstop": tstop,
            "dur": tstop - tstart,
            "total_n_valid_data_points": flc.flux.value.shape[0],
        }
    )

    flc.flares = pd.concat([flc.flares, newline.to_frame().T], ignore_index=True)
