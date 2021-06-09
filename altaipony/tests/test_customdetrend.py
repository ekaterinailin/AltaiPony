import pytest

import numpy as np
import pandas as pd

from ..flarelc import FlareLightCurve

from ..customdetrend import (custom_detrending,
                       estimate_detrended_noise,
                       remove_exponential_fringes,
                       remove_sines_iteratively,
                       measure_flare)

from ..altai import find_iterative_median

cases = [(.05, 0.005, 1.5, 24.4, 1.5, 0.1),
         (.1, 0.005, 1.5, 14.4, 1.5, 0.5),
         (.1, 0.05, 1.5, 8, 1.5, 0.5),
         (.01, .1, 1.5, 8, -.5, 0.25),
         (.3, .05, .5, 30, -.5, 0.25),
         ]

@pytest.mark.parametrize("a1,a2,period1,period2,quad,cube", cases)
def test_custom_detrending(a1, a2, period1, period2, quad, cube,):
    
    # fix uncertainty
    errorval = 15.
    np.random.seed(40)
    lc = generate_lightcurve(errorval, a1, a2, period1, period2, quad, cube)

#     lc.plot()
    flcc = custom_detrending(lc)

    flccc = estimate_detrended_noise(flcc, mask_pos_outliers_sigma=2.5, 
                                     std_window=100)

    flccc = find_iterative_median(flccc)


    flares = flccc.find_flares(addtail=True).flares
    print(flares)

    # check that uncertainty is 
    assert np.nanmedian(flccc.detrended_flux_err.value) == pytest.approx(errorval, abs=2)


    compare = pd.DataFrame({'istart': {0: 5280, 1: 13160, 2: 23160},
                            'istop': {0: 5346, 1: 13163, 2: 23175}})
    assert (flares[["istart","istop"]] == compare[["istart","istop"]]).all().all()
    
    assert (flares.ed_rec.values ==
            pytest.approx(np.array([802.25, 4.7907, 40.325]), rel=0.2))

    assert (flares.ampl_rec.values ==
            pytest.approx(np.array([0.28757, 0.03004, 0.064365]), rel=0.25))

    return 




cases = [(0.1, 3., 0.1, 5.),
         (0.1, 3., 0.1, 10.),
         (0.1, 8., 0.1, 10.),
         (0.05, 4., 0.1, 10.),
         (0.05, 4., 0.1, 3.),]

@pytest.mark.parametrize("a,b,c,d", cases)
def test_remove_sines_iteratively(a, b, c, d):

    # define light curve with two sinusoidal modulation
    x = np.linspace(10, 40, 1200)
    y1 = 20. + np.random.normal(0, .01, 1200) + a * np.sin(b * x) + c * np.sin(d * x)
    
    flc = FlareLightCurve(time=x, flux=y1, flux_err=np.full_like(y1, .01),)
    flc.detrended_flux = y1
    flc.detrended_flux_err = np.full_like(y1, .01)

    # find median
    flc = find_iterative_median(flc)
#     flc.plot()
    
    # apply function
    flcd = remove_sines_iteratively(flc)

#     plt.plot(flcd.time, flcd.flux)
#     plt.plot(flcd.time, flcd.detrended_flux)
    
    # do some checks
    assert flcd.detrended_flux.value.std() == pytest.approx(0.01, rel=1e-1)
    assert flcd.detrended_flux.value.max() < 20.2
    assert flcd.detrended_flux.value.min() > 19.8

cases = [(1., 40.,20.,1.,10.),
         (1., 40.,20.,-1.,10.),
         (-1., 40.,20.,-1.,10.),
         (-1., 40.,20.,1.,10.),
         (1., 40.,5.,1.,10.),
         ]

@pytest.mark.parametrize("a,b,median,c,d", cases)
def test_remove_exponential_fringes(a,b,median,c,d):
    
    # seed numpy random to exclude outliers
    np.random.seed(42)
    
    # define light curve with two positive fringes
    x = np.linspace(10,40,1200)
    y1 = (a*np.exp(-1 * (b - x) * 2) +
          median +
          c*np.exp((d - x) * 2) +
          np.random.normal(0, .0005*median, 1200))
    y1[800:810] = median + median * .05 * np.linspace(1,0,10)
    
    

    # define lightcurve
    flc = FlareLightCurve(time=x, flux=y1, 
                          flux_err=np.full_like(y1, .0005 * median))
    flc.detrended_flux = y1
    flc.detrended_flux_err = np.full_like(y1, .0005 * median)
    
    # get iterative median
    flc = find_iterative_median(flc)
    
    # run the function
    flcd = remove_exponential_fringes(flc)

#     plt.plot(flcd.time, flcd.flux)
#     plt.plot(flcd.time, flcd.detrended_flux)

    # do some checks

#     print(flcd.detrended_flux.std(), flcd.detrended_flux.min(), flcd.detrended_flux.max())
    assert flcd.detrended_flux.value[:799].std() == pytest.approx(.0005 * median, rel=1e-1)
    assert flcd.detrended_flux.value.max() == pytest.approx(median * 1.05)
    assert flcd.detrended_flux.value.min() > median * 0.995

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
    flc.detrended_flux = flux

    # this should work
    flces = estimate_detrended_noise(flc, mask_pos_outliers_sigma=2.5, 
                                 std_window=100, padleft=3, padright=10)
    
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
                                 std_window=100, padleft=3, padright=10)

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
                                     std_window=100, padleft=3, padright=10)

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
    assert measured_flare.ed_rec == pytest.approx((250 + 750 * 0.5) / 3400 * 0.052800 * 24 * 3600, rel=.01)
    assert measured_flare.ed_rec == pytest.approx(0.293085,0.1)
    assert measured_flare.tstop -measured_flare.tstart == measured_flare.dur
