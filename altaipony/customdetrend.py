import time

import copy
import numpy as np

from .altai import find_iterative_median
from .utils import sigma_clip


import astropy.units as u

from scipy.interpolate import UnivariateSpline
from scipy import optimize


def custom_detrending(lc, spline_coarseness=30, spline_order=3,
                      savgol1=6., savgol2=3., pad=3):
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
        
    Return:
    -------
    FlareLightCurve with detrended_flux attribute
    """
# The commented lines will help with debugging, in case the tests in test_detrend.py fail.

    dt = np.mean(np.diff(lc.time))
#     plt.figure(figsize=(16,8))
#     plt.xlim(20,21)
#     plt.plot(lc.time, lc.flux+2500, c="c", label="original light curve")
    t0 = time.clock()
    # fit a spline to the general trends
    lc1, model = fit_spline(lc, spline_order=spline_order,
                            spline_coarseness=spline_coarseness)
    
    # replace for next step
    lc1.flux = lc1.detrended_flux
    t1 = time.clock()
#     plt.plot(lc1.time, model+2500, c="r", label="rough trends")
#     plt.plot(lc1.time, lc1.detrended_flux+500, c="orange", label="rough trends removed")

    # removes strong and fast variability on 5 day to 4.8 hours 
    # simple sines are probably because rotational variability is 
    # either weak and transient or strong and persistent on the timescales
    lc2 = remove_sines_iteratively(lc1)
    t2 = time.clock()
#     plt.plot(lc2time, lc2.detrended_flux-200, label="sines removed")
    
    # choose a 6 hour window
    w = int((np.rint(savgol1 / 24. / dt) // 2) * 2 + 1)

    # use Savitzy-Golay to iron out the rest
    lc3 = lc2.detrend("savgol", window_length=w, pad=pad)
    t3 = time.clock()
    # choose a three hour window
    w = int((np.rint(savgol2 / 24. / dt) // 2) * 2 + 1)

    # use Savitzy-Golay to iron out the rest
    lc4 = lc3.detrend("savgol", window_length=w, pad=pad)
    t4 = time.clock()
#     plt.plot(lc4.time, lc4.detrended_flux-800, c="k", label="SavGol applied")
    
    # find median value
    lc4 = find_iterative_median(lc4)
    t41 = time.clock()
    # replace for next step
    lc4.flux = lc4.detrended_flux
    
    # remove exopential fringes that neither spline, 
    # nor sines, nor SavGol 
    # can remove.
    lc5 = remove_exponential_fringes(lc4)
    t5 = time.clock()
#     plt.plot(lc5.time, lc5.detrended_flux, c="magenta", label="expfunc applied")
    print(t1-t0, t2-t1, t3-t2, t4-t3, t41-t4, t5-t41, t5-t0)
#     plt.xlim(10,40)
#     plt.xlabel("time [days]")
#     plt.ylabel("flux")
#     plt.legend()    
    return lc5


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
    flct = copy.deepcopy(flcd)
    
    # iterate over chunks
    for le, ri in flct.find_gaps().gaps:
        
        # again make a copy of the chunk to manipulate safely
        flc = copy.deepcopy(flct[le:ri])
        
        # find median of LC
        flc = find_iterative_median(flc)
        
        # mask flares
        mask = sigma_clip(flc.flux, max_sigma=3.5, longdecay=2)

        # how many data points comprise the fastest period at maximum_frequency?
        full_fastest_period = 1. / maximum_frequency / np.nanmin(np.diff(flc.remove_nans().time))
        
        # only remove sines if LC chunk is larger than one full period of the fastest frequency
        if flc.flux.shape[0] > full_fastest_period:
            
            n = 0 # start counter
            snr = 3 # go into while loop at least once
            
            # iterate while there is signal, but not more than n times
            while ((snr > 1) & (n < niter)):
                t = time.clock()
                # mask NaNs and outliers
                cond = np.invert(np.isnan(flc.time)) & np.invert(np.isnan(flc.flux)) & mask
                
                # calculate periodogram
                pg = flc[cond].to_periodogram(freq_unit=freq_unit,
                                                      maximum_frequency=maximum_frequency,
                                                      minimum_frequency=minimum_frequency)

                # fit sinusoidal
                p, p_cov = optimize.curve_fit(cosine, flc.time[cond], flc.flux[cond],
                                              p0=[np.nanstd(flc.flux),
                                              2*np.pi*pg.frequency_at_max_power.value,
                                              0, np.nanmean(flc.flux)], ftol=1e-6)
                t1 = time.clock()
                # replace with de-trended flux but without subtracting the median
                flc.flux = flc.flux - cosine(flc.time, p[0], p[1], p[2], 0.)

                # update SNR
                snr = pg.flatten().max_power
                
                # bump iterator
                n += 1
                tf = time.clock()
#                 print(snr, n, tf-t, tf-t1, t1-t)
      
            # replace the empty array with the fitted detrended flux
            flcd.detrended_flux[le:ri] = flc.flux
        
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
    flct.detrended_flux = np.full_like(flct.flux, np.nan)
    
    for le, ri in flct.find_gaps().gaps:
 
        f_ = copy.deepcopy(flct[le:ri])
    
        # mask outliers 
        mask = sigma_clip(f_.flux, max_sigma=max_sigma, longdecay=longdecay)
        ff = f_[mask]

        # get the median as a guess for the least square fit
        median = np.nanmedian(ff.it_med)
        
        # get noise level from the fully 
        std = np.nanstd(ff.flux)
        
        
        # demask the fringes because they are otherwise likely to be 
        # masked by sigma clipping
        mask[:len(f_.flux) // demask] = 1
        mask[-len(f_.flux) // demask:] = 1
        
        ff = f_[mask]
        
        # get the amplitude of the fringes
        sta, fin = ff.flux[0] - median, ff.flux[-1] - median
        
        # don't fit the fringes if they not even there
        # i.e. smaller than global noise of outlier-free LC
        noleft = np.abs(sta) < std
        noright = np.abs(fin) < std
        
        # adjust the LSQ function to number of fringes
        # also fix time offset 
        
        # only end of LC chunk fringes
        if (not noright) & (noleft):
            
            def texp(x, d, e, g):
                return twoexps(x, 0., 0., 0., d, e, ff.time[-1], g)
            p0 = [fin, -10., median]
            
        # only start of LC chunk fringes
        if (not noleft) & (noright):
            
            def texp(x, a, b, g):
                return twoexps(x, a, b, ff.time[0], 0., 0., 0., g)
            p0 = [sta, 10., median]
        
        # no fringes at all
        if (noleft) & (noright):
            
            def texp(x, g):
                return twoexps(x, 0., 0., 0., 0., 0., 0., g)
            p0 = [median]
        
        # both sides of LC chunk fringe
        if (not noleft) & (not noright):
            def texp(x, a, b, d, e, g):
                return twoexps(x, a, b, ff.time[0], d, e, ff.time[-1], g)
            p0 = [sta, 10., fin, -10., median]

        # do the LSQ fit
        
        p, p_cov = optimize.curve_fit(texp, ff.time, ff.flux,
                                      p0=p0, sigma=np.full_like(ff.flux, std),
                                      absolute_sigma=True, ftol=1e-6)
        # Remove the fit from the LC
        # median + full flux - model
        nflux = p[-1] + ff.flux - texp(ff.time, *p)

        # replace NaNs in detrended flux with solution
        flcd.detrended_flux[le:ri][mask] = nflux
        
        # re-introduce outliers and flare candidates
        flcd.detrended_flux[le:ri][~mask] = flcd.flux[le:ri][~mask]
        
    return flcd



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
    flc = flc[np.where(np.isfinite(flc.flux))]
    flcp = copy.deepcopy(flc)

    flcp = flcp.find_gaps()
    flux_med = np.nanmedian(flcp.flux)
    n = int(np.rint(spline_coarseness/ 24 / (flcp.time[1] - flcp.time[0]))) #default 30h window
    k = spline_order
    #do a first round
    model = np.full_like(flcp.flux, np.nan)
    for le, ri in flcp.gaps:

        rip = flcp.flux[le:ri].shape[0] + le
        t, f = np.zeros((rip - le)//n+2), np.zeros((rip - le)//n+2)
    
        t[1:-1] = np.mean(flcp.time[le:rip - (rip - le)%n].reshape((rip - le)//n, n), axis=1)
        f[1:-1] =  np.median(flcp.flux[le:rip - (rip - le)%n].reshape((rip - le)//n, n), axis=1)
        t[0], t[-1] = flcp.time[le], flcp.time[rip-1]
        f[0], f[-1] = flcp.flux[le], flcp.flux[rip-1]
        
        # if the LC chunk is too short, skip spline fit
        if t.shape[0] <= k:
            flcp.detrended_flux[le:ri] = flcp.flux[le:ri] 
            
        # otherwise fit a spline
        else:
            p3 = UnivariateSpline(t, f, k=k)
            flcp.detrended_flux[le:ri] = flcp.flux[le:ri] - p3(flcp.time[le:ri]) + flux_med
            model[le:ri] = p3(flcp.time[le:ri])
    
    return flcp, model




