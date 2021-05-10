import pytest

import numpy as np
import pandas as pd

from ..flarelc import FlareLightCurve

from ..detrend import (custom_detrending,
                       remove_exponential_fringes,
                       remove_sines_iteratively,
                       )
  
#from ..altai import find_iterative_median

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

 #   flccc = find_iterative_median(flccc)


    flares = flccc.find_flares(addtail=True).flares
    print(flares)

    # check that uncertainty is 
    assert np.nanmedian(flccc.detrended_flux_err) == pytest.approx(errorval, abs=2)


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
    
    flc = FlareLightCurve(time=x, flux=y1, 
                          flux_err=np.full_like(y1, .01),
                          detrended_flux=y1,
                          detrended_flux_err=np.full_like(y1, .01),)

 #   # find median
  #  flc = find_iterative_median(flc)
#     flc.plot()
    
    # apply function
    flcd = remove_sines_iteratively(flc)

#     plt.plot(flcd.time, flcd.flux)
#     plt.plot(flcd.time, flcd.detrended_flux)
    
    # do some checks
    assert flcd.detrended_flux.std() == pytest.approx(0.01, rel=1e-1)
    assert flcd.detrended_flux.max() < 20.2
    assert flcd.detrended_flux.min() > 19.8

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
                      flux_err=np.full_like(y1, .0005*median),
                      detrended_flux=y1,
                      detrended_flux_err=np.full_like(y1, .0005*median),)
    
    # get iterative median
    #flc = find_iterative_median(flc)
    
    # run the function
    flcd = remove_exponential_fringes(flc)

#     plt.plot(flcd.time, flcd.flux)
#     plt.plot(flcd.time, flcd.detrended_flux)

    # do some checks

#     print(flcd.detrended_flux.std(), flcd.detrended_flux.min(), flcd.detrended_flux.max())
    assert flcd.detrended_flux[:799].std() == pytest.approx(.0005*median, rel=1e-1)
    assert flcd.detrended_flux.max() == pytest.approx(median * 1.05)
    assert flcd.detrended_flux.min() > median * 0.995

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



