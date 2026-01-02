import pytest


import numpy as np
import pandas as pd

from ..flarelc import FlareLightCurve

from ..customdetrend import (custom_detrending,
                       estimate_detrended_noise,
                       measure_flare)






def generate_lightcurve(errorval,  a1, a2,period1, period2, quad, cube,
                        mean=3400.):
    
    """Generate wild light curves with variability on several
    timescales.
    
    Returns:
    ---------
    FlareLightCurve with time, flux, and flux_err attributes
    """
    time = np.arange(10, 10 + 10 * np.pi,.0008)

    # define the flux
    flux = (np.random.normal(0,errorval,time.shape[0]) +
            mean + 
            a1*mean*np.sin(period1*time +1.)  +
            a2*mean*np.sin(period2*time) +
            quad*(time-25)**2 -
            cube*(time-25)**3)

    # add a gap in the data
    flux[5600:7720] = np.nan

    # add big and long flare
    l = 66
    flux[5280:5280 + l] = flux[5280:5280 + l] + np.linspace(1000,250,l)

    # add tiny flare
    l = 3
    flux[15280:15280 + l] = flux[15280:15280 + l] + np.linspace(100,60,l)

    # add intermediate flare
    l, s = 15, 25280
    flux[s:s + l] = flux[s:s + l] + np.linspace(200,60,l)

    # typically Kepler and TESS underestimate the real noise
    err = np.full_like(time,errorval/3*2)

    # define FLC
    return FlareLightCurve(time=time, flux=flux, flux_err=err)



def test_estimate_detrended_noise():
    
    # setup light curve
    time = np.linspace(10,30,200)
    
    # seed numpy to get the same error array
    np.random.seed(30)
    
    # define flux with gaussian noise and baseline flux
    flux = np.random.normal(0,40, time.shape[0]) + 200.
    
    # define light curve
    flc = FlareLightCurve(time=time)
    flc.detrended_flux=flux

    # this should work
    flces = estimate_detrended_noise(flc, mask_pos_outliers_sigma=2.5, 
                                 std_window=100)
    
    # error should be similar to input error of 40
    np.median(flces.detrended_flux_err.value) == pytest.approx(41.38048677022836)

    # re-seed and add a flare
    np.random.seed(30)
    flux = np.random.normal(0,40, time.shape[0]) + 200.
    flux[120:124] = [500,380,300,270]
    flc = FlareLightCurve(time=time)
    flc.detrended_flux = flux

    # should mask flare, error should not grow
    flces = estimate_detrended_noise(flc, mask_pos_outliers_sigma=2.5, 
                                 std_window=100)

    np.median(flces.detrended_flux_err.value) == pytest.approx(41.24232394552432)

    # re-seed and add some NaNs
    np.random.seed(30)
    flux = np.random.normal(0,40, time.shape[0]) + 200.
    flux[120:124] = [500,380,300,270]
    flux[30:40] = np.nan
    flc = FlareLightCurve(time=time) 
    flc.detrended_flux = flux

    # should work regardless
    flces = estimate_detrended_noise(flc, mask_pos_outliers_sigma=2.5, 
                                     std_window=100)

    # error should not change too much
    np.median(flces.detrended_flux_err.value) == pytest.approx(41.23144256208637)
    
    
    
def test_measure_flare():
    """Simple test: Generate light curve with flare,
    detrend, and manually measure the flare.
    """
    # generate LC
    flc = generate_lightcurve(15,.01,.03,4,.3,.1,.02)
    
    # de-trend LC
    flcc = custom_detrending(flc)

    # measure flare
    measure_flare(flcc,5280,5280+66)

    # get pandas.Series
    measured_flare = flcc.flares.iloc[0]

    # do checks
    assert measured_flare.istart == 5280
    assert measured_flare.istop == 5346
    assert measured_flare.tstart == pytest.approx(14.224)
    assert measured_flare.tstop ==  pytest.approx(14.276800)
    assert measured_flare.ed_rec == pytest.approx((250 + 750 * 0.5) / 3400 * 0.052800 * 24 * 3600, rel=.03)
    assert measured_flare.tstop -measured_flare.tstart == measured_flare.dur