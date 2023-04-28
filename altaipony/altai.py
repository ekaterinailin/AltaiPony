
import pandas as pd
import numpy as np
import logging
import copy

from scipy.signal import savgol_filter

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

def find_flares(flc, minsep=3, sigma=None, **kwargs):
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
    for (le,ri) in lc.gaps:
        error = lc.detrended_flux_err[le:ri]
        flux = lc.detrended_flux[le:ri]
        
        median = lc.it_med[le:ri]
        
        time = lc.time[le:ri]
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


def detrend_savgol(lc, window_length=None, pad=3, printwl=False, **kwargs):
    '''Construct a model light curve.
    Based on original Apppaloosa (Davenport 2016)
    with Savitzky-Golay filtering from scipy,
    and iterative sigma_clipping adopted 
    from K2SC (Aigrain et al. 2016).

    
    Parameters:
    ------------
    flc : FlareLightCurve
        TESS light curve
    window_length : int
        number of datapoints for Sav.-Gol. filter,
        either one value for entire light curve
        of piecewise for gaps
    pad : int
        mask this number of data points before and
        after each outlier. Note that sigma_clip
        already expands the mask 
    printwl : bool
        if True will print final window_length
    kwargs : dict
        keyword arguments to pass to :func:`utils.sigma_clip`
    
    Return:
    -------
    FlareLightCurve
    '''
    lc = lc[np.where(~np.isnan(lc.flux.value))]
    lc.detrended_flux = np.full_like(lc.flux.value, np.nan)
    lc["flux_model"] = np.full_like(lc.flux.value, np.nan)
    lc.detrended_flux_err = lc.flux_err.value
    
    if lc.gaps is None:
        lc = lc.find_gaps()
    if (isinstance(window_length, tuple) or isinstance(window_length, list)):
        gaps = [(window_length[i], lc.gaps[i][0], lc.gaps[i][1]) for i in range(len(lc.gaps))]
    elif isinstance(window_length, int):
        gaps = [(window_length, lc.gaps[i][0], lc.gaps[i][1]) for i in range(len(lc.gaps))]
    elif window_length is None:
        gaps = [(-999, lc.gaps[i][0], lc.gaps[i][1]) for i in range(len(lc.gaps))]
    
    for (wl,le,ri) in gaps:
        
        # Do the iterative sigma clipping
        ok = np.where(sigma_clip(lc.flux.value[le:ri], **kwargs))[0] + le
        outliers = list(set(list(range(le, ri))) - set(ok)) # inverse of ok
        
        time = lc.time.value[ok]
        flux = lc.flux.value[ok]
        error = lc.flux_err.value[ok]
        
        # ----------------------------------
        # Main detrending happens here:
        
        # This is from Appaloosa: 
        if wl == -999:
            dt = np.nanmedian(time[1:] - time[0:-1])
            wl = np.floor(.1 / dt)
            if wl % 2 == 0:
                wl = wl + 1
        
        # args are flux, window_length, polyorder, mode is 
        wl = max(wl, 5) #wl must be larger than polyorder
        if printwl: 
            print(f"window length: {wl}")
        flux_model_i = savgol_filter(flux, wl, 3, mode='nearest')
        
        flux_diff = flux - flux_model_i + np.nanmean(flux_model_i)
        lc.detrended_flux[ok] = flux_diff
        lc.flux_model[ok] = flux_model_i
        
        # Find out where outlier begin and end:
        a = np.isnan(lc.detrended_flux[le:ri]).astype(int)
        
        #pad outliers-------------------
       # print(np.where(a)[0])
        x = np.where(a)[0]
        for i in range(-pad,pad+1):
            y = x+i
            y[np.where(y>len(a)-1)] = len(a)-1
            a[y] = True
        #print(list(np.where(a)[0]))
        #--------------------------------
        
        sta, fin = list(np.where(np.diff(a)==1)[0]), list(np.where(np.diff(a)==-1)[0])
        # Treat outliers at end and start of time series:
        if len(sta) > len(fin):
            fin.append(ri-le-1)
    
        elif len(sta) < len(fin):
            sta = [0] + sta

        elif ((len(sta) == len(fin)) & (len(sta)!=0)):

            if ((sta[0] > fin[0]) | (sta[-1] > fin[-1])): # outliers on both ends

                sta = [0] + sta
                fin.append(ri - le - 1)
        # Compute flux model as the mean value between 
        # start and end of flare, that is, we interpolate 
        # linearly.
        flux_model_j = np.full(np.where(a==1)[0].shape, np.nan)
        
        off = 0
        for i,j in list(zip(sta, fin)):
            d = 0
            if j+2 > ri-le-1: #treat end of time series
                k = i
            elif i == 0:
                i = 0
                d = 1
                k = j + 2
                
            else:
                k = j + 2
            k = min(len(lc.flux_model[le:ri]), k)

            upper = min(j + d - i + off, len(flux_model_j))

            # work around a bug that sometimes occurs, not sure why
            # ----------------------------------
            if k == len(lc.flux_model[le:ri]): 
                k-=1
            # ----------------------------------

            flux_model_j[off:upper] = np.nanmean(lc.flux_model[le:ri][[i,k]])
            off += j + d - i
       # print(lc.flux[le:ri][np.where(a==1)[0]], flux_model_j, np.nanmean(flux_model_i))
        lc["detrended_flux"][le:ri][np.where(a==1)[0]] = lc.flux.value[le:ri][np.where(a==1)[0]] - flux_model_j + np.nanmean(flux_model_i)

        # End of main de-trending.
        # --------------------------------------------------
   # print(lc.detrended_flux)
    lc = lc[np.isfinite(lc.time.value) &
          np.isfinite(lc.flux.value) &
          np.isfinite(lc.detrended_flux)]
    return lc


def find_iterative_median(flc, n=30, **kwargs):

    """
    Find the iterative median value for a continuous observation period using
    flare finding to identify outliers.

    Parameters
    -----------
    flc : FlareLightCurve

    n : 50 or int
        maximum number of iterations
    kwargs : dict
        keyword arguments to pass to sigma_clip

    Return
    -------
    FlareLightCurve with the it_med attribute set.
    """

    lc = copy.deepcopy(flc)

    lc.it_med = np.full_like(flc.detrended_flux, np.median(flc.detrended_flux))
    if lc.gaps is None:
        lc = lc.find_gaps()
    for (le,ri) in lc.gaps:
        flux = flc.detrended_flux[le:ri]
        #find a median that is not skewed by outliers
        good = sigma_clip(flux)
        goodflux = flux[good]
        lc.it_med[le:ri] = np.nanmedian(goodflux)

    return lc

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
