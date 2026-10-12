
import pandas as pd
import numpy as np
import logging
import copy


from .utils import sigma_clip


LOG = logging.getLogger(__name__)

def find_flares_in_cont_obs_period(flux, median, error, sigma=None, 
                                   N1=3, N2=2, N3=3, addtail=False,
                                   tailthreshdiff=1.):
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


    Return:
    ------------
    isflare : numpy array of booleans
        datapoints are flagged with 1 if they belong to a flare candidate
    '''
    isflare = np.zeros_like(flux, dtype='bool')
  
    # If no local scatter characteristics are given, use formal error as sigma    
    if sigma is None:
        sigma = error
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


def find_flares(flc, minsep=3, sigma=None, merge_sigma=None, **kwargs):
    '''
    Main wrapper to obtain and process a light curve.

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
        flux dips below the detection threshold.  Default None (no merging).
    kwargs : dict
        keyword arguments to pass to :func:`find_flares_in_cont_obs_period`
    Return
    ----------
    numpy arrays of start and stop cadence numbers of flare candidates
    '''
    lc = copy.deepcopy(flc)
    istart = np.array([], dtype='int')
    istop = np.array([], dtype='int')

    if np.isnan(lc.detrended_flux).all():
        raise TypeError('Flare finding only works on de-trended light curves.')

    #Now work on periods of continuous observation with no gaps
    for (le,ri) in lc.cont_windows:
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
        Keyword arguments to pass to sigma_clip
    
    Returns
    -------
    it_med : np.ndarray
        Iterative median array matching the length of detrended_flux
    """
    if len(detrended_flux) == 0:
        raise ValueError("Input detrended_flux array is empty.")

    # Initialize it_med with global median
    it_med = np.full_like(detrended_flux, np.nanmedian(detrended_flux))
    
    # If no windows provided, treat entire array as one segment
    if cont_windows is None or len(cont_windows) == 0:
        good_mask = sigma_clip(detrended_flux, max_iter=n, **kwargs)
        good_flux = detrended_flux[good_mask]
        if len(good_flux) > 0:
            it_med[:] = np.nanmedian(good_flux)
    else:
        # Process each continuous window
        for (le, ri) in cont_windows:
            flux_segment = detrended_flux[le:ri]
            
            # Find median that is not skewed by outliers
            good_mask = sigma_clip(flux_segment, max_iter=n, **kwargs)
            good_flux = flux_segment[good_mask]
            if len(good_flux) > 0:
                it_med[le:ri] = np.nanmedian(good_flux)
    
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
