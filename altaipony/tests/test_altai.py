import numpy as np
import pytest

from ..altai import (find_flares,
                     find_flares_in_cont_obs_period,
                     chi_square,
                     equivalent_duration,
                     find_iterative_median,
                     detrend_savgol)
from ..flarelc import FlareLightCurve
from .test_flarelc import mock_flc

def test_detrend_savgol():
    """Test if different window_length kwargs are processed correctly."""
    
    N = int(1e4)
    time = np.linspace(2000,2050,N)
    flux = np.sin(time / 2.) * 30. + 5e4 + np.random.rand(N) * 35. + 5e-4 * ((time-2004.)**3 - 30 * (time-2004)**2)
    flux[5000:5010] = flux[5000:5010] + np.array([500,250,150,80,60,30,20,10,7,4])
    flux[4500:4809] = np.nan
    flux_err = np.random.rand(N) * 10.
    flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, flux_err=flux_err)
    
    flcds = [flc.detrend("savgol"),
             flc.detrend("savgol", window_length=201),
             flc.detrend("savgol", window_length=(101,205)),
             flc.detrend("savgol", window_length=[101,205])]
    for flcd in flcds:
            assert flcd.detrended_flux.value.shape[0] == 1e4-309
            
    N = int(1e4)
    time = np.linspace(2000,2050,N)
    np.random.seed(200)
    flux = np.sin(time / .03) * 30. + 5e4 + np.random.rand(N) * 25. + 5e-4 * ((time-2004.)**3 - 30 * (time-2004)**2)
    flux[5000:5010] = flux[5000:5010] + np.array([500,250,150,80,60,30,20,10,7,4])
    flux[4500:4809] = np.nan
    flux_err = np.random.rand(N) * 35. # this reflects the real noise
    flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, flux_err=flux_err)
    
    flcds = [flc.detrend("savgol"),
             flc.detrend("savgol", window_length=201),
             flc.detrend("savgol", window_length=(101,205)),
             flc.detrend("savgol", window_length=[25,25])]
    for flcd in flcds:
            assert flcd.detrended_flux.value.shape[0] == 1e4-309
            
    # The last de-trending iteration is the only appropriate one to give good 
    # results given the rapid variability of the light curve. So let's check
    # the outcome of this one. It should only recover the one flare we intro-
    # duced above around t=2025
    flares = flcd.find_flares().flares
    
    f = flares.iloc[0,:]
    assert f.tstart == pytest.approx(2025, abs=5e-3)
    assert f.ed_rec == pytest.approx(8.31358,abs=f.ed_rec_err)
    assert f.istart == 4691
    assert f.istop == 4695
    assert f.total_n_valid_data_points == 1e4-309
    assert f.dur == pytest.approx(f.tstop - f.tstart, rel=1e-4)

    
def test_iterative_median():

    flc = mock_flc(detrended=True)
    lc1 = find_iterative_median(flc, n=1)
    lc2 = find_iterative_median(flc, n=2)

    # test that find_iterative_median converges after one iteration for mock FLC
    assert np.median(flc.it_med.value) != np.median(lc1.it_med.value)
    assert np.median(lc1.it_med.value) == np.median(lc2.it_med.value)

def test_find_flares():
    """
    Integration test of a mock example light curve is given in test_flarelc.
    Add unit tests!
    """
    flc = mock_flc(detrended=False)
    with pytest.raises(TypeError):
        #raises error bc find_flares only works on detrended_flux
        find_flares(flc)
     
    # Check if all columns are created
    flc = mock_flc(detrended=True)
    for col in  ['istart', 'istop', 'cstart', 'cstop', 'tstart',
                 'tstop', 'ed_rec', 'ed_rec_err', 'ampl_rec', 
                 'total_n_valid_data_points', 'dur']:
        assert col in flc.flares.columns

def test_find_flares_in_cont_obs_period():
    """
    Integration test of a mock example light curve is given in test_flarelc.
    Add unit tests!
    """
    # Set up a light curve with a flare
    
    # time series (not needed but helpful for debugging)
    time = np.linspace(30,40,100)
    
    # flux array
    flux = 400 + np.random.normal(0,30,100)
    
    # add the flare
    flux[30:41] = [800, 660, 500, 490, 480, 470, 460, 455, 450, 445, 440]
    
    # give the LC characteristics
    median, error, sigma = 400., 1., 30.,

    # find the flare
    isflare = find_flares_in_cont_obs_period(flux, median, error, 
                                             sigma=sigma, N1=3, N2=2,
                                             N3=3, addtail=False)
    # check if found
    assert (np.where(isflare)[0] == np.array([30, 31, 32, 33])).all()

    # add flare decay phase to search
    isflare = find_flares_in_cont_obs_period(flux, median, error, 
                                             sigma=sigma, N1=3, N2=2,
                                             N3=3, addtail=True)


    # check if found
    assert (np.where(isflare)[0] == np.array([30, 31, 32, 33, 34, 35, 36])).all()

    # add a second flare
    flux[40:49] = [510, 500, 491, 470, 460, 455, 450, 445, 440]

    # find with tails
    isflare = find_flares_in_cont_obs_period(flux, median, error, 
                                             sigma=sigma, N1=3, N2=2, 
                                             N3=3, addtail=True)

    # check if found
    assert (np.where(isflare)[0] ==  np.array([30, 31, 32, 33, 34, 35, 
											   36, 40, 41, 42, 43, 
											   44])).all()

    # bad value for tailthreshdiff throws error
    with pytest.raises(ValueError):
        isflare = find_flares_in_cont_obs_period(flux, median, error, 
                                             sigma=sigma, N1=3, N2=2, 
                                             N3=3, addtail=True, tailthreshdiff=3)




def test_chi_square():
    """Test an abvious example"""
    residual = np.full(5,1.)
    error = np.full(5,.1)
    assert chi_square(residual, error) == 100.

def test_equivalent_duration():
    """Test a triangle shaped flare in a toy light curve."""
    detrended_flux = np.full(1000,1.)
    detrended_flux[60:70] = np.array([10,9,8,7,6,5,4,3,2,1])
    lc = FlareLightCurve(time=np.arange(1000)/86400.)
    lc.detrended_flux=detrended_flux
    lc.it_med =np.full(1000,1.)
    lc.detrended_flux_err=np.full(1000,1e-8)
   # print(lc.saturation)
    ed, ed_err = equivalent_duration(lc, 60, 70, err=True)
    assert ed == pytest.approx(45,rel=1e-8)
    assert ed_err == pytest.approx(2.665569e-08, rel=1e-4)
