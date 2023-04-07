"""
UTF-8, Python 3

------------------
AltaiPony
------------------

Ekaterina Ilin, 2023, MIT License

This module contains custom detrending functions.
"""
import time

import copy
import numpy as np
import pandas as pd

from .altai import find_iterative_median, equivalent_duration
from .utils import sigma_clip
from .flarelc import FlareLightCurve


import astropy.units as u

from scipy.interpolate import UnivariateSpline, interp1d
from scipy import optimize




def custom_detrending(lc, spline_coarseness=30, spline_order=3,
                      savgol1=6., savgol2=3., pad=3, max_sigma=2.5, 
                      longdecay=6,):
    """Custom de-trending for TESS and Kepler 
    short cadence light curves, including TESS Cycle 3 20s
    cadence.
    
    Parameters:
    ------------
    lc : FlareLightCurve
        light curve that has at least time, flux and flux_err
    spline_coarseness : float
        time scale in hours for spline points. 
        See fit_spline for details.
    spline_order: int
        Spline order for the coarse spline fit.
        Default is cubic spline.
    savgol1 : float
        Window size for first Savitzky-Golay filter application.
        Unit is hours, defaults to 6 hours.
    savgol2 : float
        Window size for second Savitzky-Golay filter application.
        Unit is hours, defaults to 3 hours.
    pad : 3
        Outliers in Savitzky-Golay filter are padded with this
        number of data points. Defaults to 3.
    max_sigma : float
        Outlier rejection threshold in sigma. Defaults to 2.5.
    longdecay : int
        Long decay time for outlier rejection. Defaults to 6.

        
    Return:
    -------
    FlareLightCurve with detrended_flux attribute
    """
    dt = np.mean(np.diff(lc.time.value))

    # fit a spline to the general trends
    lc1, model = fit_spline(lc, spline_order=spline_order,
                            spline_coarseness=spline_coarseness)

    # replace for next step
    lc1.flux = lc1.detrended_flux.value

    # removes strong and fast variability on 5 day to 4.8 hours 
    # simple sines are probably because rotational variability is 
    # either weak and transient or strong and persistent on the timescales
    lc2 = remove_sines_iteratively(lc1)
    
    # choose a 6 hour window
    w = int((np.rint(savgol1 / 24. / dt) // 2) * 2 + 1)

    # use Savitzy-Golay to iron out the rest
    lc3 = lc2.detrend("savgol", window_length=w, pad=pad)

    # choose a three hour window
    w = int((np.rint(savgol2 / 24. / dt) // 2) * 2 + 1)

    # use Savitzy-Golay to iron out the rest
    lc4 = lc3.detrend("savgol", window_length=w, pad=pad, 
                        max_sigma=max_sigma,longdecay=longdecay)
    
    # find median value
    lc4 = find_iterative_median(lc4)

    # replace for next step
    lc4.flux = lc4.detrended_flux.value
    
    # remove exopential fringes that neither spline, 
    # nor sines, nor SavGol can remove.
    lc5 = remove_exponential_fringes(lc4)
  
    return lc5


def detrend_savgol(lc, max_sigma=2.5, longdecay=6, 
                   w=121, break_tolerance=10, **kwargs):
    """New detrending with savgol filter.
    
    Parameters:
    -----------
    
    max_sigma: float>0
        sigma clipping threshold
    longdecay: int
        adding masked datapoints to the tail if
        multiple outliers occur in a row
    w : odd int
        window length for savgol filter
    break_tolerance : int
        If there are large gaps in time, flatten will split the flux into several sub-lightcurves and apply savgol_filter to each individually. A gap is defined as a period in time larger than break_tolerance times the median gap. To disable this feature, set break_tolerance to None.
    kwargs : dict
        keyword arguments to feed LightCurve.flatten()
    """
    # fill missing cadences
    lc = interpolate_missing_cadences(lc)
    
    # normalize
    lcn = lc.normalize()
    
    # sigma clip
    m = sigma_clip(lcn.flux.value, max_sigma=2.5, longdecay=6)

    # convert bool to int
    mask = ~m * 1

    # from Appaloosa:
    # convert mask to start and stop
    reverse_counts = np.zeros_like(lcn.flux.value, dtype='int')
    for k in range(2, len(lcn.flux.value)):
        reverse_counts[-k] = (mask[-k]
                                * (reverse_counts[-(k-1)]
                                + mask[-k]))

    # find flare start where values in reverse_counts switch from 0 to >=N3 
    # SET N3=2 because we care about all longer outliers!
    istart_i = np.where((reverse_counts[1:] >= 2) &
                        (reverse_counts[:-1] - reverse_counts[1:] < 0))[0] + 1

    # use the value of reverse_counts to determine how many points away stop is
    istop_i = istart_i + (reverse_counts[istart_i])

    # get a list of masked candidates to extrapolate
    candidates = list(zip(istart_i, istop_i))

    # save the flare flux
    fluxold = lcn.flux

    # remove the flares candidates for now
    lcn.flux[mask] = np.nan

    # SAVGOL APPLIED HERE
    # https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.flatten.html?highlight=flatten#lightkurve.LightCurve.flatten
    # flatten with light curve
    # set break_tolerance to 10 by default, i.e. 20 min in a 2min cadence LC
    lcrsf  = lcn.flatten(window_length=w, break_tolerance=break_tolerance) #replace with 6h or 3h window

    # cycle over all candidates
    for i, j in candidates:

        # span the data
        mask_ij = np.arange(i,j)

        # linearinterpolate below the flare
        interpolation_ij = np.interp(lcn.time.value[mask_ij],
                                     [lcn.time.value[i],lcn.time.value[j]],
                                     [lcn.flux.value[i],lcn.flux.value[j]])
        # plt.plot(lcn.time.value[mask], fill)
        # plt.plot(lcn.time.value[mask], x)
        # plt.scatter( [lcn.time.value[mask[0]],lcn.time.value[mask[-1]]],
        #                    [lcn.flux.value[mask[0]-1],lcn.flux.value[mask[-1]+1]])

        # fill in the masked data again
        lcrsf.flux[mask_ij] = fluxold[mask_ij] / interpolation_ij
    
    # deugging helper lines:
    # %matplotlib inline
    # plt.figure(figsize=(15,4))
    # plt.plot(lcrsf.time.value, lcrsf.flux.value, color="k")
    # # plt.plot(lcrsf2.time.value, lcrsf2.flux.value, color="grey")
    # plt.plot(lcn.time.value, lcn.flux.value + 0.02, color="r")
    # plt.scatter(lcn.time[mask].value, lcn.flux[mask].value)
    # # plt.xlim(1945,1946)
    # # plt.ylim(0.98,1.03)
    
    # finally remove interpolated values
    # first, set them to NaNs
    lcrsf.flux[np.where(lcrsf.interpolated.value==1)[0]] = np.nan 
    
    # then remove
    lcrsf = lcrsf.remove_nans() 
    
    return lcrsf

def remove_sines_iteratively(flcd, niter=5, freq_unit=1/u.day, 
                             maximum_frequency=12., 
                             minimum_frequency=0.2,
                             max_sigma=3.5, longdecay=2):
    
    """Iteratively remove strong sinusoidal signal
    from light curve. Each iteration calculates a Lomb-Scargle 
    periodogram and LSQ-fits a cosine function using the dominant
    frequency as starting point. 
    
    
    Parameters:
    ------------
    flcd : FlareLightCurve
        light curve from which to remove 
    niter : int
        Maximum number of iterations. 
    freq_unit : astropy.units
        unit in which maximum_frequency and minimum_frequency
        are given
    maximum_frequency: float
        highest frequency to calculate the Lomb-Scargle periodogram
    minimum_frequency: float
        lowest frequency to calculate the Lomb-Scargle periodogram
    max_sigma : float
        Passed to altaipony.utils.sigma_clip. 
        Above this value data points
        are flagged as outliers.
    longdecay : 2
        altaipony.utils.sigma_clip expands the mask for series
        of outliers by sqrt(length of series). Longdecay doubles
        the mask expansion in the decay phase of what may be flares.
        
    Return:
    -------
    FlareLightCurve with detrended_flux attribute
            
    """
    
    # define cosine function
    def cosine(x, a, b, c, d):
        return a * np.cos(b * x + c) + d

    # make a copy of the original LC
    flcd.flux = flcd.flux.value
    flcd.flux_err = flcd.flux_err.value
    flct = copy.deepcopy(flcd)
    
    # iterate over chunks
    for le, ri in flct.find_gaps().gaps:
        
        # again make a copy of the chunk to manipulate safely
        flc = copy.deepcopy(flct[le:ri])
        
        # find median of LC
        flc = find_iterative_median(flc)
        
        # mask flares
        mask = sigma_clip(flc.flux.value, max_sigma=3.5, longdecay=2)

        # how many data points comprise the fastest period at maximum_frequency?
        full_fastest_period = 1. / maximum_frequency / np.nanmin(np.diff(flc.remove_nans().time.value))
        
        # only remove sines if LC chunk is larger than one full period of the fastest frequency
        if flc.flux.value.shape[0] > full_fastest_period:
            
            n = 0 # start counter
            snr = 3 # go into while loop at least once
            
            # iterate while there is signal, but not more than n times
            while ((snr > 1) & (n < niter)):
                t = time.process_time()
                # mask NaNs and outliers
                cond = np.invert(np.isnan(flc.time.value)) & np.invert(np.isnan(flc.flux.value)) & mask
                
                # calculate periodogram
                pg = flc[cond].to_periodogram(freq_unit=freq_unit,
                                                      maximum_frequency=maximum_frequency,
                                                      minimum_frequency=minimum_frequency)

                # fit sinusoidal
                p, p_cov = optimize.curve_fit(cosine, flc.time.value[cond], flc.flux.value[cond],
                                              p0=[np.nanstd(flc.flux.value),
                                              2*np.pi*pg.frequency_at_max_power.value,
                                              0, np.nanmean(flc.flux.value)], ftol=1e-6)
                t1 = time.process_time()
                # replace with de-trended flux but without subtracting the median
                flc.flux = flc.flux.value - cosine(flc.time.value, p[0], p[1], p[2], 0.)

                # update SNR
                snr = pg.flatten().max_power
                
                # bump iterator
                n += 1
                tf = time.process_time()
#                 print(snr, n, tf-t, tf-t1, t1-t)
      
            # replace the empty array with the fitted detrended flux
            flcd.detrended_flux[le:ri] = flc.flux.value
        
    return flcd

def remove_exponential_fringes(flcd, demask=10, max_sigma=3.5, longdecay=2):
    """Remove exponential fringes from light curve chunks.
    
    Parameters:
    -----------
    flcd : FlareLightCurve
        Mostly de-trended light curves 
        with possibly fringy fringes that need a haircut.
    demask : int
        fraction of light curve to keep in the fit even it
        deviates from the median, applied to the end and start
        of each light curve chunk.
    max_sigma : float
        Passed to altaipony.utils.sigma_clip. 
        Above this value data points
        are flagged as outliers.
    longdecay : 2
        altaipony.utils.sigma_clip expands the mask for series
        of outliers by sqrt(length of series). Longdecay doubles
        the mask expansion in the decay phase of what may be flares.
        
    """
   
    def twoexps(x, a, b, c, d, e, f, g):
        return a * np.exp(b * (c - x)) + d * np.exp(e * (f - x)) + g
    
    flct = copy.deepcopy(flcd)
    
    # initiate a detrended flux array
    flct.detrended_flux = np.full_like(flct.flux.value, np.nan)
    
    for le, ri in flct.find_gaps().gaps:
 
        f_ = copy.deepcopy(flct[le:ri])
    
        # mask outliers 
        mask = sigma_clip(f_.flux.value, max_sigma=max_sigma, longdecay=longdecay)
        ff = f_[mask]

        # get the median as a guess for the least square fit
        median = np.nanmedian(ff.it_med.value)
        
        # get noise level from the fully 
        std = np.nanstd(ff.flux.value)
        
        
        # demask the fringes because they are otherwise likely to be 
        # masked by sigma clipping
        mask[:len(f_.flux.value) // demask] = 1
        mask[-len(f_.flux.value) // demask:] = 1
        
        ff = f_[mask]
        
        # get the amplitude of the fringes
        sta, fin = ff.flux.value[0] - median, ff.flux.value[-1] - median
        
        # don't fit the fringes if they not even there
        # i.e. smaller than global noise of outlier-free LC
        noleft = np.abs(sta) < std
        noright = np.abs(fin) < std
        
        # adjust the LSQ function to number of fringes
        # also fix time offset 
        
        # only end of LC chunk fringes
        if (not noright) & (noleft):
            
            def texp(x, d, e, g):
                return twoexps(x, 0., 0., 0., d, e, ff.time.value[-1], g)
            p0 = [fin, -10., median]
            
        # only start of LC chunk fringes
        if (not noleft) & (noright):
            
            def texp(x, a, b, g):
                return twoexps(x, a, b, ff.time.value[0], 0., 0., 0., g)
            p0 = [sta, 10., median]
        
        # no fringes at all
        if (noleft) & (noright):
            
            def texp(x, g):
                return twoexps(x, 0., 0., 0., 0., 0., 0., g)
            p0 = [median]
        
        # both sides of LC chunk fringe
        if (not noleft) & (not noright):
            def texp(x, a, b, d, e, g):
                return twoexps(x, a, b, ff.time.value[0], d, e, ff.time.value[-1], g)
            p0 = [sta, 10., fin, -10., median]

        # do the LSQ fit
        
        p, p_cov = optimize.curve_fit(texp, ff.time.value, ff.flux.value,
                                      p0=p0, sigma=np.full_like(ff.flux.value, std),
                                      absolute_sigma=True, ftol=1e-6)
        # Remove the fit from the LC
        # median + full flux - model
        nflux = p[-1] + ff.flux.value - texp(ff.time.value, *p)

        # replace NaNs in detrended flux with solution
        flcd.detrended_flux[le:ri][mask] = nflux
        
        # re-introduce outliers and flare candidates
        flcd.detrended_flux[le:ri][~mask] = flcd.flux[le:ri][~mask].value
        
    return flcd

def estimate_detrended_noise(flc, mask_pos_outliers_sigma=2.5, 
                             std_window=100, ):

    flcc = copy.deepcopy(flc)
    flcc = flcc.find_gaps()

    for (le, ri) in flcc.gaps:

        flcd = copy.deepcopy(flcc[le:ri])
        mask = sigma_clip(flcd.detrended_flux.value, max_sigma=mask_pos_outliers_sigma, longdecay=2)

        flcd.detrended_flux[~mask] = np.nan
        # apply rolling window std and interpolate the masked values
        flcd.detrended_flux_err[:] = pd.Series(flcd.detrended_flux.value).rolling(std_window,
                                                                 center=True,
                                                                 min_periods=1).std().interpolate().values
        
        # and refine it:
        flcd = find_iterative_median(flcd)
        
        
        # make a copy first
        filtered = copy.deepcopy(flcd.detrended_flux.value)
        
        # get right bound of flux array
        tf = filtered.shape[0]

        # pick outliers
        mask = sigma_clip(filtered, max_sigma=mask_pos_outliers_sigma, longdecay=2)

        filtered[~mask] = np.nan    

        # apply rolling window std and interpolate the masked values
        flcc.detrended_flux_err[le:ri]= pd.Series(filtered).rolling(std_window,
                                                                 center=True,
                                                                 min_periods=1).std().interpolate().values
        
        # make it a series again so that formatting is consistent
        flcc.detrended_flux_err = pd.Series(flcc.detrended_flux_err)
    
    return flcc




def fit_spline(flc, spline_coarseness=30, spline_order=3):
    """Do a spline fit on a coarse sampling of data points.
    
    Parameters:
    ------------
    flc : FlareLightCurve
    
    spline_coarseness : int
 
    spline_order : int
        order of spline fit
        
    Return:
    --------
    FlareLightCurve with new flux attribute
    """
    flc = flc[np.where(np.isfinite(flc.flux.value))]
    flcp = copy.deepcopy(flc)

    flcp = flcp.find_gaps()
    flux_med = np.nanmedian(flcp.flux.value)
    n = int(np.rint(spline_coarseness / 
                                   24 / 
            (flcp.time.value[1] - flcp.time.value[0]))) #default 30h window
    k = spline_order
#     print(n)
    #do a first round
    model = np.full_like(flcp.flux.value, np.nan)
    for le, ri in flcp.gaps:

        rip = flcp.flux[le:ri].value.shape[0] + le
        t, f = np.zeros((rip - le)//n+2), np.zeros((rip - le)//n+2)
        if (rip - le)//n == 0:
            flcp.detrended_flux[le:ri] = flcp.flux.value[le:ri]
        elif (rip - le)//n > 0:
            news, news_mod = (rip - le)//n, (rip - le)%n 
            t[1:-1] = np.mean(flcp.time.value[le:rip - news_mod].reshape(news, n), axis=1)
            f[1:-1] =  np.median(flcp.flux.value[le:rip - news_mod].reshape(news, n), axis=1)
            t[0], t[-1] = flcp.time.value[le], flcp.time.value[rip-1]
            f[0], f[-1] = flcp.flux.value[le], flcp.flux.value[rip-1]
            
        # if the LC chunk is too short, skip spline fit
        if t.shape[0] <= k:
            flcp.detrended_flux[le:ri] = flcp.flux.value[le:ri] 
            
        # otherwise fit a spline
        else:
            p3 = UnivariateSpline(t, f, k=k, s=0)
            flcp.detrended_flux[le:ri] = flcp.flux.value[le:ri] - p3(flcp.time.value[le:ri]) + flux_med
            model[le:ri] = p3(flcp.time.value[le:ri])
    
    return flcp, model


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
    ampl_rec = np.max(flc.detrended_flux.value[sta:sto]) / flc.it_med.value[sta] - 1. 
    
    # get cadence numbers
    cstart = flc.cadenceno.value[sta]
    cstop = flc.cadenceno.value[sto]
    
    # get time stamps 
    tstart = flc.time.value[sta]
    tstop = flc.time.value[sto]
    
    # add result to flare table
    flc.flares = flc.flares.append(pd.Series(
                                 {'ed_rec': ed_rec,
                                  'ed_rec_err': ed_rec_err,
                                  'ampl_rec': ampl_rec,
                                  'istart': sta,
                                  'istop': sto,
                                  'cstart': cstart,
                                  'cstop': cstop,
                                  'tstart': tstart,
                                  'tstop': tstop,
                                  'dur': tstop - tstart,
                                  'total_n_valid_data_points': flc.flux.value.shape[0]
                                  }),ignore_index=True)
    return 


def interpolate_missing_cadences(lc, **kwargs):
    """Interpolate missing cadences in 
    light curve, skipping larger gaps in data.
    
    Parameters:
    -----------
    lc : FlareLightCurve
        the light curve
    kwargs : dict
        keyword arguments to pass to find_gaps method
    
    Return:
    -------
    interpolated FlareLightCurve
    """

    # find gaps that are too big to be interpolated with a good conscience
    gaps = lc.find_gaps().gaps

    # set up interpolated array
    time, flux, flux_err, newcadence = [], [], [], []

    # interpolate within each gap
    for i, j in gaps:

        # select gap
        gaplc = lc[i:j]

        # get old cadence
        oldx = gaplc.cadenceno.value

        # cadenceno are complete in uncorrected flux, 
        # so we fill in the removed cadences
        newx = np.arange(gaplc.cadenceno.value[0], gaplc.cadenceno.value[-1])
        newcadence.append(newx)

        # interpolate flux error
        f = interp1d(oldx, gaplc.flux_err.value)
        flux_err.append(f(newx))

        # interpolate time
        f = interp1d(oldx, gaplc.time.value)
        time.append(f(newx))

        # interpolate flux
        f = interp1d(oldx, gaplc.flux.value)
        flux.append(f(newx))

    # stitch together new light curve
    newlc = FlareLightCurve(time=np.concatenate(time),
                            flux=np.concatenate(flux),
                            flux_err=np.concatenate(flux_err),
                            targetid=lc.targetid)

    # add new cadence array
    newcadenceno = np.concatenate(newcadence)
    newlc["cadenceno"] = newcadenceno

    # flag values that have been interpolated in the new light curve
    newvals = np.sort(list(set(newcadenceno) - set(lc.cadenceno.value)))
    newvalindx = np.searchsorted(newcadenceno, newvals)
    newlc["interpolated"] = 0 # not interpolated values
    newlc.interpolated[newvalindx] = 1 # interpolated values

    return newlc
