import numpy as np
import pandas as pd
import pytest

import os

from inspect import currentframe, getframeinfo

from ..flarelc import FlareLightCurve
from ..altai import find_iterative_median
from ..lcio import from_path

from .. import PACKAGEDIR
from . import test_ids, test_paths, pathkepler, pathAltaiPony, pathk2TPF

def mock_flc(origin='TPF', detrended=False, ampl=1., dur=1):
    """
    Mocks a FlareLightCurve with a sinusoid variation and a single positive outlier.

    Parameter
    -----------
    origin : 'TPF' or str
        Mocks a specific origin, such as 'KLC', 'FLC' etc.
    detrended : False or bool
        If False, a sinusoid signal is added to the mock light curve.

    Return
    -------
    FlareLightCurve
    """
    n = 1000
    time = np.arange(0, n/48, 1./48.)
    pixel_time = np.outer(time,np.full((3,3), 1)).reshape((1000,3,3))
    np.random.seed(13854)

    pipeline_mask = np.array([[False, False, False],
                              [False, True,  False],
                              [False, False, False],])
    quality = np.zeros_like(time)
    np.random.seed(33)
    flux_err = np.random.rand(n)/100.
    if detrended==False:
        flux = np.sin(time/2)*7. + 500. +flux_err
        pixel_flux = np.random.rand(len(time),3,3)/100.+500.+np.sin(pixel_time/2)*7.
        pixel_flux_err = np.random.rand(len(time),3,3)/100.
    else:
        flux = 500. + flux_err
        pixel_flux = np.random.rand(len(time),3,3)/100.+500.
        pixel_flux_err = np.random.rand(len(time),3,3)/100.
    flux[15:15+dur] += 500.*ampl
    flux[15+dur:15+2*dur] += 250.*ampl
    flux[15+2*dur:15+3*dur] += 130.*ampl
    flux[15+3*dur:15+4*dur] += 80.*ampl
    quality[17] = 1024
    quality[18] = 128
    keys = {'flux' : flux, 'flux_err' : flux_err, 'time' : time,
            'pos_corr1' : np.zeros(n), 'pos_corr2' : np.zeros(n),
            'cadenceno' : np.arange(n), 'targetid' : 800000000,
            'origin' : origin, 'it_med' : np.full_like(time,500.005),
            'quality' : quality, 'pipeline_mask' : pipeline_mask,
            'pixel_flux' : pixel_flux, 'campaign' : 5, 'ra' : 22.,
            'dec' : 22., 'mission' : 'K2', 'channel' : 55, 
            'pixel_flux_err' : pixel_flux_err, 'time_format': 'bkjd'}

    if detrended == False:
        flc = FlareLightCurve(**keys)
    else:
        flc = FlareLightCurve(detrended_flux=flux,
                              detrended_flux_err=flux_err,
                              **keys)
    return flc


def test_get_saturation():
    flc = mock_flc(detrended=True)
    flc.pixel_flux[30,:] = 1e6 # add a saturated point
    
    # Find saturation overall
    r1 = flc.get_saturation()
    assert r1.saturation[30] == True
    assert r1.saturation.shape[0] == flc.flux.shape[0]
    assert (r1.saturation[:30] == False).all()
    assert (r1.saturation[31:] == False).all()
    
    
    # Find saturation with level
    r1 = flc.get_saturation(return_level=True)
    assert r1.saturation[30] == 1e6 / 10093
    assert r1.saturation.shape[0] == flc.flux.shape[0]
    assert (r1.saturation[:30] == pytest.approx(500 / 10093, rel=1e-4)) # the LC has some noise added per default, so only approximate results
    assert (r1.saturation[31:] == pytest.approx(500 / 10093, rel=1e-4)) # the LC has some noise added per default, so only approximate results
    
    # Find saturation with flares
    flc = flc.find_flares()
    
    #... without saturation attribute present
    r1 = flc.get_saturation()
    assert r1.flares.saturation_f10.iloc[0] == False
    r2 = flc.get_saturation(return_level=True)
    assert r2.flares.saturation_f10.iloc[0] == pytest.approx(0.0495,1e-2)
    r3 = flc.get_saturation(factor=1e-2)
    assert r3.flares['saturation_f0.01'].iloc[0] == True
    
    #... with saturation attribute present
    
    
    # .. that is given as a boolean array
    flc.saturation = np.full(flc.flux.shape[0], True)
    r4 = flc.get_saturation()
    assert r4.flares.saturation_f10.iloc[0] == True
    r2 = flc.get_saturation(return_level=True)
    assert r2.flares.saturation_f10.iloc[0] == True # throws a warning, too, test that later
    r3 = flc.get_saturation(factor=1e-2)
    assert r3.flares['saturation_f0.01'].iloc[0] == True # throws a warning, too, test that later
    
    # .. that is given as an array of floats
    flc.saturation = np.full(flc.flux.shape[0], 5.0)
    r4 = flc.get_saturation()
    assert r4.flares.saturation_f10.iloc[0] == False
    r2 = flc.get_saturation(return_level=True)
    assert r2.flares.saturation_f10.iloc[0] == 5 
    r3 = flc.get_saturation(factor=1e-2)
    assert r3.flares['saturation_f0.01'].iloc[0] == True 


def test_mark_flagged_flares():
    flc = mock_flc(detrended=True)
    flc = flc.find_flares()
    flc = flc.mark_flagged_flares(explain=True)
    assert flc.flares.quality.iloc[0] == 1152
    s1 = "Sudden sensitivity dropout, Cosmic ray in optimal aperture"
    s2 = "Cosmic ray in optimal aperture, Sudden sensitivity dropout"
    qs = flc.flares.explanation.iloc[0]
    assert ((qs == s1) | (qs == s2))

def test_sample_flare_recovery():
    
    # Generic case
    flc = mock_flc(detrended=True)
    
    flc, fflc = flc.sample_flare_recovery(iterations=2)
    #make sure no flares are injected overlapping true flares
    data = flc.fake_flares
    assert data[(data.istart > 14) & (data.istart < 19)].shape[0] == 0
    #test if all injected event are covered in the merged flares:
    assert data.shape[0] == 2
    assert fflc.gaps == [(0, 1000)]
    assert np.median(fflc.it_med) == pytest.approx(500.005274113832)
    
    # Custom case
    
    def func(flc):
        flc.detrended_flux =  flc.flux/2.
        flc.detrended_flux_err =  flc.flux_err/2.
        return flc
    
    flc = mock_flc(detrended=True)
    
    flcd, fflc = flc.sample_flare_recovery(iterations=10, inject_before_detrending=True,
                                          func=func, mode="custom")
    #make sure no flares are injected overlapping true flares
    data = flcd.fake_flares
    assert data[(data.istart > 14) & (data.istart < 19)].shape[0] == 0
    #test if all injected event are covered in the merged flares:
    assert data.shape[0] == 10
    assert fflc.gaps == [(0, 1000)]
    assert np.median(fflc.it_med) == pytest.approx(500.005274113832/2.)
    assert flcd.detrended_flux == pytest.approx(flc.flux/2.)


    # Custom case with detrend_kwargs
    
    def func(flc, kw=0):
        flc.detrended_flux =  flc.flux/2.
        flc.detrended_flux_err =  flc.flux_err/2.
        a = kw + 3
        assert a ==20
        return flc
    
    flc = mock_flc(detrended=True)
    
    flcd, fflc = flc.sample_flare_recovery(iterations=10, inject_before_detrending=True,
                                           func=func, mode="custom", 
                                           detrend_kwargs={"kw":17})
    #make sure no flares are injected overlapping true flares
    data = flcd.fake_flares
    assert data[(data.istart > 14) & (data.istart < 19)].shape[0] == 0
    
    #test if all injected event are covered in the merged flares:
    assert data.shape[0] == 10
    assert fflc.gaps == [(0, 1000)]
    assert np.median(fflc.it_med) == pytest.approx(500.005274113832/2.)
    assert flcd.detrended_flux == pytest.approx(flc.flux/2.)
    
    # Test that the original flare was not changed accidentally
    assert flcd.flares.loc[0,'ed_rec'] == pytest.approx(3455.8875941, rel=1e-4)
    assert flcd.flares['ed_rec_err'][0] < flcd.flares['ed_rec'][0]
    assert flcd.flares['istart'][0] == 15
    assert flcd.flares['istop'][0] == 19
    assert flcd.flares['cstop'][0] == 19
    assert flcd.flares['cstart'][0] == 15
    assert flcd.flares['tstart'][0] == pytest.approx(0.3125, rel=1e-4)
    assert flcd.flares['tstop'][0] == pytest.approx(0.395833, rel=1e-4)
    assert flcd.flares['total_n_valid_data_points'][0] == 1000
    assert flcd.flares['ampl_rec'][0] == pytest.approx(1, rel=1e-3)
    
    # Test that adding another round of injrec will append to the path
    flc = mock_flc(detrended=True)
    flcd, fflc = flc.sample_flare_recovery(iterations=10, inject_before_detrending=False,
                                           save=True)
    size = len(flcd.fake_flares)
    
    flcd, fflc = flcd.sample_flare_recovery(iterations=10, inject_before_detrending=False,
                                           save=True)
    size2 = len(flcd.fake_flares)
    assert size * 2 == size2
    
    path ='10_800000000_inj_after_5.csv'
    saved = pd.read_csv(path)
    assert saved.shape[0] == size2
    
    os.remove(path)

def test_to_fits():
    # with light curve only:
    flc = from_path(pathkepler, mode="LC", mission="Kepler")
    flc = flc.detrend("savgol")
    flc.to_fits(pathAltaiPony)
    flc = flc.find_flares()
    flc.to_fits(pathAltaiPony)
    flc = from_path(pathAltaiPony, mode="AltaiPony", mission="Kepler")
    
    # with TPF component which needs to be thrown away
    flc = from_path(pathk2TPF, mode="TPF", mission="K2")
    flc.flux_err = flc.flux_err * 100. # otherwise K2SC errors.
    flc = flc.detrend("k2sc", de_niter=3)
    flc.to_fits(pathAltaiPony)
    flc = flc.find_flares()
    flc.to_fits(pathAltaiPony)
    flc = from_path(pathAltaiPony, mode="AltaiPony", mission="K2")

def test_repr():
    pass

def test_getitem():
    pass


def test_invalid_lightcurve():
    """Invalid FlareLightCurves should not be allowed."""
    err_string = ("Input arrays have different lengths."
                  " len(time)=5, len(flux)=4")
    time = np.array([1, 2, 3, 4, 5])
    flux = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError) as err:
        FlareLightCurve(time=time, flux=flux)
    assert err_string == err.value.args[0]

def test_find_gaps():
    flux = np.random.rand(1000)
    time = np.linspace(0,30,1000)
    flux[20:200] = np.nan
    time = time[np.where(~np.isnan(flux))]
    flux = flux[np.where(~np.isnan(flux))]
    flc = FlareLightCurve(time=time, flux=flux)

    flc = flc.find_gaps()
    assert flc.gaps == [(0, 20), (20, 820)]

def test_detrend():
    
    # Test K2SC de-trending:
    flc = mock_flc()
    try:
        flc = flc.detrend("k2sc", de_niter=2,)
        shape = flc.flux.shape
        for att in ["detrended_flux", "detrended_flux_err",
                    "flux_err", "flux", "time", "quality"]:
            assert getattr(flc, att).shape == shape
        assert flc.pv[0] == pytest.approx(-3.895176160613472, rel=0.1)
    except np.linalg.linalg.LinAlgError:
        warning.warn('Detrending of mock LC failed, this happens.')
        pass
    
    # Test K2SC detrending with custom splits    
    flc = mock_flc()
    try:
        flc = flc.detrend("k2sc", de_niter=2, splits=[1.4, 2.2, 5.4])
        shape = flc.flux.shape
        for att in ["detrended_flux", "detrended_flux_err",
                    "flux_err", "flux", "time", "quality"]:
            assert getattr(flc, att).shape == shape
        assert flc.pv[0] == pytest.approx(-3.895176160613472, rel=0.1)
    except np.linalg.linalg.LinAlgError:
        warning.warn('Detrending of mock LC failed, this happens.')
        pass

    #test non TPF derived LC fails
    flc = mock_flc(origin='LC', detrended=False, ampl=1., dur=1)
    with pytest.raises(ValueError) as e:
        flc = flc.detrend("k2sc", de_niter=2,)
    #test the shapes are the same for all
    # test that the necessary attributes are kept
    
    
    # Test SAVGOL detrending
    
    ampls = [100, 10., 1, .1, .01]
    durs = [1, 2, 3]
    lcs = []
    for ampl in ampls:
        for dur in durs:
            aplc = mock_flc(ampl=ampl, dur=dur)
            daplc = aplc.detrend("savgol")
            lcs.append(daplc)

    for daplc in lcs:
        fff = find_iterative_median(daplc)
        assert fff.it_med == pytest.approx(500., rel=0.01) #median stays the same roughly
        assert aplc.flux.shape[0] == daplc.detrended_flux.shape[0] #no NaNs to throw out
        assert daplc.flux.max() > daplc.detrended_flux.max() # flare sits on a LC part above quiescent level
        assert (aplc.flux_err == daplc.detrended_flux_err).all() # uncertainties are simply kept
        # Test that shapes of arrays are kept
        for att in ["detrended_flux", "detrended_flux_err",
            "flux_err", "flux", "time", "quality"]:
            assert getattr(flc, att).shape == shape
        
    # TEST CUSTOM DETRENDING
    
    # --- create a very minimalistic light curve
    N = int(1e4)
    time = np.linspace(2000,2050,N)
    np.random.seed(200)
    flux = 5e4 + np.random.rand(N) * 35. 
    flux_err = np.random.rand(N) * 35. 
    flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, flux_err=flux_err)

    # --- test a minimum function that fails to create the desired output
    def custom_detrending(flc):
        return flc

    with pytest.raises(AttributeError) as e:
        new_flc = flc.detrend(mode="custom", func=custom_detrending)

    # -- test a minimum function that does the job    
    def custom_detrending(flc):
        flc.detrended_flux = flc.flux
        flc.detrended_flux_err = flc.flux_err
        return flc    
        
    new_flc = flc.detrend(mode="custom", func=custom_detrending)
    assert (new_flc.flux == flc.flux).all()
    assert (new_flc.flux_err == flc.flux_err).all()

    # -- test a minimum function that does the job and has kwargs
    def custom_detrending(flc, kw=0):
        flc.detrended_flux = flc.flux
        flc.detrended_flux_err = flc.flux_err
        a = kw + 3 
        assert a == 20
        return flc    
        
    new_flc = flc.detrend(mode="custom", func=custom_detrending, kw=17)
    assert (new_flc.flux == flc.flux).all()
    assert (new_flc.flux_err == flc.flux_err).all()

    # --- function should fail if no func is given

    with pytest.raises(ValueError) as e:
        new_flc = flc.detrend(mode="custom")


def test_detrend_fails():
    """If detrend fails, an error is raised with given string."""
    
    # K2SC de-trending fails because we need a TPF for it, not just a LC.
    flc =  mock_flc(origin='KLC')
    err_string = ('Only KeplerTargetPixelFile derived FlareLightCurves can be'
                          ' passed to K2SC de-trending.')
    with pytest.raises(ValueError) as err:
        flc.detrend("k2sc", de_niter=3)
    assert err_string == err.value.args[0]
    
    # De-trending fails in general when an invalid mode is passed.
    # But also a helpful message is thrown out.
    flc =  mock_flc()
    err_string = ('\nDe-trending mode blaaaah does not exist. Pass "k2sc" (K2 LCs)'
                       ' or "savgol" (Kepler, TESS).')
    with pytest.raises(ValueError) as err:
        flc.detrend("blaaaah")
    assert err_string == err.value.args[0]

def test_find_flares():
    """Test that an obvious flare is recovered sufficiently well."""
    flc = mock_flc(detrended=True)
    flc = flc.find_flares()
    assert flc.flares.loc[0,'ed_rec'] == pytest.approx(3455.8875941, rel=1e-4)
    assert flc.flares['ed_rec_err'][0] < flc.flares['ed_rec'][0]
    assert flc.flares['istart'][0] == 15
    assert flc.flares['istop'][0] == 19
    assert flc.flares['cstop'][0] == 19
    assert flc.flares['cstart'][0] == 15
    assert flc.flares['tstart'][0] == pytest.approx(0.3125, rel=1e-4)
    assert flc.flares['tstop'][0] == pytest.approx(0.395833, rel=1e-4)
    assert flc.flares['total_n_valid_data_points'][0] == 1000
    assert flc.flares['ampl_rec'][0] == pytest.approx(1, rel=1e-3)
    

def test_inject_fake_flares():
    flc = mock_flc(detrended=True)
    np.random.seed(84712)
    flc = flc.find_gaps()
    fake_flc = flc.inject_fake_flares()
    # make sure you inject only one flare per LC
    assert len(fake_flc.gaps) == fake_flc.fake_flares.shape[0]
    assert (set(fake_flc.fake_flares.columns.values.tolist()) == 
            {'amplitude', 'duration_d', 'ed_inj', 'peak_time'})
    assert fake_flc.detrended_flux_err.all() >= 1e-10
    assert fake_flc.detrended_flux.all() <= 1.
    assert fake_flc.detrended_flux.shape == flc.detrended_flux.shape
    flc = mock_flc(detrended=False)
    np.random.seed(84712)
    flc = flc.find_gaps()
    fake_flc = flc.inject_fake_flares(inject_before_detrending=True)

    # make sure you inject only one flare per LC
    assert len(fake_flc.gaps) == fake_flc.fake_flares.shape[0]
    assert (set(fake_flc.fake_flares.columns.values.tolist()) == 
            {'amplitude', 'duration_d', 'ed_inj', 'peak_time'})
    assert fake_flc.flux_err.all() >= 1e-10
    assert fake_flc.flux.all() <= 1.
    assert fake_flc.flux.shape == flc.flux.shape

def test_load_injrec_data():
    # Create a minimal empty light curve with an ID
    flcd = FlareLightCurve(targetid="GJ 1243", time=np.linspace(10,1))

    # Path to test file
    path = "altaipony/tests/testfiles/gj1243_injrec.csv"

    # Call the function for the first time
    flcd.load_injrec_data(path)

    # Check if nothing happened to the size
    assert flcd.fake_flares.shape[0] == 1010
    assert flcd.fake_flares.shape[1] == 14 

    # Loading a second time should append the new table
    flcd.load_injrec_data(path)
    
    # Twice as many rows, but same number of columns
    assert flcd.fake_flares.shape[0] == 2020
    assert flcd.fake_flares.shape[1] == 14 

    # We should get a FileNotFoundError when a bad path is passed:
    with pytest.raises(FileNotFoundError) as err:
        flcd.load_injrec_data("badpath")

def test_plot_ed_ratio_heatmap():
    # Create a minimal empty light curve with an ID
    flcd = FlareLightCurve(targetid="GJ 1243", time=np.linspace(10,1))
    
    # Path to test file
    path = "altaipony/tests/testfiles/gj1243_injrec.csv"
    flcd.load_injrec_data(path)
    
    # Test if the function is called properly with default values
    flcd.plot_ed_ratio_heatmap()


def test_plot_recovery_probability_heatmap():
    # Create a minimal empty light curve with an ID
    flcd = FlareLightCurve(targetid="GJ 1243", time=np.linspace(10,1))
    
    # Path to test file
    path = "altaipony/tests/testfiles/gj1243_injrec.csv"
    flcd.load_injrec_data(path)
    
    # Test if the function is called properly with default values
    flcd.plot_recovery_probability_heatmap()
