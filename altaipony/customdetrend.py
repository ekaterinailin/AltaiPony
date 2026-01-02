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

from .altai import _find_iterative_median, equivalent_duration
from .utils import sigma_clip
from .flarelc import FlareLightCurve
from lightkurve import LightCurve



import astropy.units as u

from scipy.interpolate import UnivariateSpline, interp1d
from scipy import optimize




def custom_detrending(lc, spline_coarseness=8, spline_order=3,
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
    gaps = lc.find_gaps(maxgap = 0.04).gaps

    time, flux = lc.time.value, lc.flux.value

    # fit a spline to the general trends
    m2flux, _ = fit_spline(time, flux, gaps, spline_order=spline_order,
                            spline_coarseness=spline_coarseness)
    
    # choose a 6 hour window
    w = int((np.rint(savgol1 / 24. / dt) // 2) * 2 + 1)

    # lc.flux = m2flux * u.electron / u.s
    lc["spline_detrended_flux"] = m2flux  # add for debugging

    # use Savitzy-Golay to iron out the rest
    lc3 = lc.detrend("savgol", window_length=w, pad=pad)

    # choose a three hour window
    w = int((np.rint(savgol2 / 24. / dt) // 2) * 2 + 1)

    # use Savitzy-Golay to iron out the rest
    lc4 = lc3.detrend("savgol", window_length=w, pad=pad, 
                        max_sigma=max_sigma,longdecay=longdecay)
    
    # find median value
    lc4.find_iterative_median()

    # replace for next step
    lc4.flux = lc4.detrended_flux
  
    return lc4


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
    m = sigma_clip(lcn.flux, max_sigma=2.5, longdecay=6)

    # convert bool to int
    mask = ~m * 1

    # from Appaloosa:
    # convert mask to start and stop
    reverse_counts = np.zeros_like(lcn.flux, dtype='int')
    for k in range(2, len(lcn.flux)):
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
                                     [lcn.flux[i],lcn.flux[j]])
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



def estimate_detrended_noise(flc, mask_pos_outliers_sigma=2.5, 
                             std_window=100, ):

    flcc = copy.deepcopy(flc)
    flcc = flcc.find_gaps()

    for (le, ri) in flcc.gaps:

        flcd = copy.deepcopy(flcc[le:ri])
        mask = sigma_clip(flcd.detrended_flux, max_sigma=mask_pos_outliers_sigma, longdecay=2)

        flcd.detrended_flux[~mask] = np.nan
        # apply rolling window std and interpolate the masked values
        flcd.detrended_flux_err[:] = pd.Series(flcd.detrended_flux).rolling(std_window,
                                                                 center=True,
                                                                 min_periods=1).std().interpolate().values
        
        # and refine it:
        flcd = flcd.find_iterative_median()
        
        
        # make a copy first
        filtered = copy.deepcopy(flcd.detrended_flux)
        
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




def fit_spline(time, flux, gaps, spline_coarseness=30, spline_order=3):
    """Do a spline fit on a coarse sampling of data points.
    
    Parameters:
    ------------
    flc : FlareLightCurve
    
    spline_coarseness : int
 
    spline_order : int
        order of spline fitflux
        
    Return:
    --------
    FlareLightCurve with new flux attribute
    """
    flux_med = _find_iterative_median(flux, gaps)
    n = int(np.rint(spline_coarseness / 
                                   24 / 
            (np.nanmin(np.diff(time))))) #default 30h window
    k = spline_order

    model = np.full_like(flux, np.nan)
    newflux = np.full_like(flux, np.nan)
    for le, ri in gaps:

        t, f = np.zeros((ri - le)//n+2), np.zeros((ri - le)//n+2)
        
        if (ri - le)//n == 0:
            newflux[le:ri] = flux[le:ri]
        elif (ri - le)//n > 0:
            news, news_mod = (ri - le)//n, (ri - le)%n 
            t[1:-1] = np.mean(time[le:ri - news_mod].reshape(news, n), axis=1)
            f[1:-1] =  np.median(flux[le:ri - news_mod].reshape(news, n), axis=1)
            t[0], t[-1] = time[le], time[ri-1]
            f[0], f[-1] = flux[le], flux[ri-1]
            
        # if the LC chunk is too short, fit a linear function to the full data
        if t.shape[0] <= k:
            p2 = np.polyfit(time[le:ri], flux[le:ri], 1)
            newflux[le:ri] = flux[le:ri] - np.polyval(p2, time[le:ri]) + flux_med[le:ri]
            model[le:ri] = np.polyval(p2, time[le:ri])
            
        # otherwise fit a spline
        else:
            p3 = UnivariateSpline(t, f, k=k, s=0)
            newflux[le:ri] = flux[le:ri] - p3(time[le:ri]) + flux_med[le:ri]
            model[le:ri] = p3(time[le:ri])
    
    return newflux, model


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
    newline = pd.Series(
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
                        })
    
    flc.flares = pd.concat([flc.flares, newline.to_frame().T], ignore_index=True)

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
        f = interp1d(oldx, gaplc.flux_err)
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
