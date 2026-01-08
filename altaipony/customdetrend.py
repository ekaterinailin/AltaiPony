"""
UTF-8, Python 3

------------------
AltaiPony
------------------

Ekaterina Ilin, 2023, MIT License

This module contains custom detrending functions.
"""

import numpy as np
import pandas as pd

from .altai import _find_iterative_median, equivalent_duration
from .utils import sigma_clip




import astropy.units as u

from scipy.interpolate import UnivariateSpline




def custom_detrending(lc, spline_coarseness=8, spline_order=3,
                      savgol1=6., savgol2=3., pad=3, max_sigma=2.5, 
                      longdecay=6, maxgap=10):
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
    maxgap : float
        Maximum gap size in days for spline fitting. Defaults to 10 x cadence size.

        
    Return:
    -------
    FlareLightCurve with detrended_flux attribute
    """
    dt = np.mean(np.diff(lc.time.value))
    gaps = lc.find_gaps(maxgap=maxgap * dt).gaps

    time, flux = lc.time.value, lc.flux.value
    
    # Store original flux as a column so it survives filtering operations
    lc["original_flux"] = lc.flux.value.copy()
    lc["orginal_flux_err"] = lc.flux_err.value.copy()

    # fit a spline to the general trends
    m2flux, _ = fit_spline(time, flux, gaps, spline_order=spline_order,
                           spline_coarseness=spline_coarseness)
    
    # choose a 6 hour window
    w = int((np.rint(savgol1 / 24. / dt) // 2) * 2 + 1)

    lc.flux = m2flux * u.electron / u.s
    # lc["spline_detrended_flux"] = m2flux  # add for debugging
    # use Savitzy-Golay to iron out the rest
    lc3 = lc.detrend("savgol", window_length=w, pad=pad)

    # choose a three hour window
    w = int((np.rint(savgol2 / 24. / dt) // 2) * 2 + 1)

    # use Savitzy-Golay to iron out the rest
    lc4 = lc3.detrend("savgol", window_length=w, pad=pad, 
                      max_sigma=max_sigma, longdecay=longdecay)
    
    
    # Restore original flux from the column (now properly filtered to match lc4's length)
    lc4.detrended_flux = lc4.flux.value
    lc4.detrended_flux_err = lc4.flux_err.value
    lc4.flux = lc4["original_flux"] * u.electron / u.s
    
    # Clean up the temporary column
    lc4.remove_column("original_flux")
    lc.flux = lc["original_flux"] * u.electron / u.s
    lc.flux_err = lc["orginal_flux_err"] * u.electron / u.s
    
    
    # find median value
    lc4.find_iterative_median()

    return lc4





def estimate_detrended_noise(flc, mask_pos_outliers_sigma=2.5, 
                             std_window=100):
    """
    Estimate detrended flux uncertainties using rolling standard deviation.
    
    Parameters
    ----------
    flc : FlareLightCurve
        Light curve with detrended_flux attribute
    mask_pos_outliers_sigma : float
        Sigma threshold for masking positive outliers (likely flares)
    std_window : int
        Window size for rolling standard deviation calculation
    
    Returns
    -------
    flc : FlareLightCurve
        Input light curve with detrended_flux_err attribute updated
    """
    # Find gaps if not already done
    if flc.gaps is None:
        flc = flc.find_gaps()
    
    # Extract arrays we need (avoids repeated attribute access)
    detrended_flux = flc.detrended_flux
    n_points = len(detrended_flux)
    
    # Initialize output array
    detrended_flux_err = np.full(n_points, np.nan)
    
    # Process each gap segment
    for (le, ri) in flc.gaps:
        # Extract segment
        flux_segment = detrended_flux[le:ri].copy()  # Copy just this segment
        
        # First pass: mask outliers and compute initial error estimate
        mask = sigma_clip(flux_segment, max_sigma=mask_pos_outliers_sigma, 
                         longdecay=2)
        
        # Set outliers to NaN for error calculation
        flux_segment_masked = flux_segment.copy()
        flux_segment_masked[~mask] = np.nan
        
        # Second pass: refine by finding iterative median
        it_med_segment = _find_iterative_median(
            flux_segment, 
            gaps=[(0, ri - le)]
        )
        
        # Subtract iterative median for better outlier detection
        flux_normalized = flux_segment - it_med_segment
        
        # Mask outliers again
        mask_refined = sigma_clip(flux_normalized, 
                                 max_sigma=mask_pos_outliers_sigma, 
                                 longdecay=2)
                
        # Set outliers to NaN
        flux_normalized_masked = flux_normalized.copy()
        flux_normalized_masked[~mask_refined] = np.nan
        
        # Compute final rolling std
        final_err = (pd.Series(flux_normalized_masked)
                    .rolling(std_window, center=True, min_periods=1)
                    .std()
                    .interpolate()
                    .values)

        # Store in output array
        detrended_flux_err[le:ri] = final_err
    
    # Set the result on the original lightcurve
    flc.detrended_flux_err = detrended_flux_err
    
    return flc



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


