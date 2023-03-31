import pytest
import numpy as np
import pandas as pd
from inspect import signature

from ..fakeflares import (flare_model_mendoza2022, flare_model_davenport2014,
                          generate_fake_flare_distribution,
                          merge_fake_and_recovered_events,
                          mod_random,)


from .test_flarelc import mock_flc


def test_mod_random():
    assert mod_random(1, d=True)[0] == pytest.approx(0.48661046)
    assert mod_random(1)[0] != mod_random(1)[0]


def test_merge_fake_and_recovered_events():

    flares = {'istart' : [5, 30],
              'istop' : [15, 36],
              'cstart' : [8, 33],
              'cstop': [18, 39],
              'tstart' : [2335.5846, 2335.5846 + 12.5/24.],
              'tstop' : [2335.5846 + 5./24., 2335.5846 + 15.5/24.],
              'ed_rec' : [650, 250],
              'ed_rec_err' : [46, 15]}
    fakes = {'duration_d' : np.linspace(0.01,1,20),
             'amplitude' : np.linspace(0.01,1,20),
             'ed_inj' : np.linspace(100,50000,20),
             'peak_time' : np.linspace(2320,2340,20),}
    flares = pd.DataFrame(flares)
    fakes = pd.DataFrame(fakes)
    tab = merge_fake_and_recovered_events(fakes, flares)
    assert tab.size == 240
    assert tab.shape[0] == 20
    match = tab.dropna(how='any')
    assert match.shape[0] == 1
    row = match.iloc[0]
    assert row.peak_time >= row.tstart
    assert row.peak_time <= row.tstop
    assert row.ed_inj > row.ed_rec

def test_generate_fake_flare_distribution():

    n = 500

    dur, ampl = generate_fake_flare_distribution(n)
    assert (dur <= 20).all()
    assert (dur >= 1e-4).all()
    assert (ampl <= 1e2).all()
    assert (ampl >= 1e-4).all()
    assert len(dur) == n
    assert len(ampl) == n

    dur, ampl = generate_fake_flare_distribution(n, ampl=[1e-4, 5], dur=[0.005, 0.012],  mode='uniform')
    assert (dur <= 0.012).all()
    assert (dur >= 0.0045).all()
    assert (ampl <= 5).all()
    assert (ampl >= 1e-4).all()
    assert len(dur) == n
    assert len(ampl) == n

  

# def test_inject_fake_flares():
#     flc = mock_flc(detrended=True)
#     np.random.seed(84712)
#     flc = flc.find_gaps()
#     fake_flc = inject_fake_flares(flc)
#
#     assert fake_flc.fake_flares.size == 20
#
#     columns = set(fake_flc.fake_flares.columns.values.tolist())
#     test_columns = {'amplitude', 'duration_d', 'ed_inj', 'peak_time'}
#     assert len(columns & test_columns) == 4
#     assert fake_flc.detrended_flux_err.all() >= 1e-10
#     assert fake_flc.detrended_flux.all() <= 1.
#     assert fake_flc.detrended_flux.shape == flc.detrended_flux.shape
#     flc = mock_flc(detrended=False)
#     np.random.seed(84712)
#     flc = flc.find_gaps()
#     fake_flc = inject_fake_flares(flc, inject_before_detrending=True)
#
#     assert fake_flc.fake_flares.size == 20
#     columns = set(fake_flc.fake_flares.columns.values.tolist())
#     test_columns = {'amplitude', 'duration_d', 'ed_inj', 'peak_time'}
#     assert len(columns & test_columns) == 4
#     assert fake_flc.flux_err.all() >= 1e-10
#     assert fake_flc.flux.all() <= 1.
#     assert fake_flc.flux.shape == flc.flux.shape

def test_flare_model_davenport2014_and_equivalent_duration():

    n = 1000
    time = np.arange(0, n/48, 1./48.)
    x = time * 60.0 * 60.0 * 24.0

    # Test a large flare without upsampling
    fl_flux = flare_model_davenport2014(time, 11.400134, 1.415039, 110.981950)
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(1.22e7,rel=1e-2)
    
    # Test a flare with 0 amplitude
    fl_flux = flare_model_davenport2014(time, 11.400134, 1.415039, 0)
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == 0.

    # test a large flare with upsampling
    fl_flux = flare_model_davenport2014(time, 11.400134, 1.415039, 110.981950, upsample=True)
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(1.22e7,rel=1e-2)
    
    
    # Test a smaller undersampled flare
    fl_flux = flare_model_davenport2014(time, 11.400134, 1/48., 1.0)
    x = time * 60.0 * 60.0 * 24.0
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(1453.1179,rel=1e-2)
    
    # Test the amplitude
    fl_flux = flare_model_davenport2014(time, 1.734, 15, 1.0)
    assert np.max(fl_flux) == pytest.approx(1,rel=1e-2)

def test_flare_model_mendoza2022():

    n = 1000
    time = np.arange(-2, n/48, 1./48.)
    x = time * 60.0 * 60.0 * 24.0

    # Test a large flare without upsampling
    fl_fluxu = flare_model_mendoza2022(time, 11.400134, 1.415039, 110.981950)
    integralu = np.sum(np.diff(x) * fl_fluxu[:-1])
    assert integralu == pytest.approx(2.26e7,rel=1e-2)
    
    # Test a flare with 0 amplitude
    fl_flux0 = flare_model_mendoza2022(time, 11.400134, 1.415039, 0)
    integral0 = np.sum(np.diff(x) * fl_flux0[:-1])
    assert integral0 == 0.

    # test a large flare with upsampling
    fl_fluxup = flare_model_mendoza2022(time, 10., 1.5, 100, upsample=True,uptime=10)
    integralup = np.sum(np.diff(x[:-1]) * fl_fluxup[:-2])
    assert integralup == pytest.approx(2.2e7,rel=1e-2)
    
    
    # Test a smaller undersampled flare
    fl_fluxus = flare_model_mendoza2022(time, 10, .25, 1.0)
    integralus = np.sum(np.diff(x) * fl_fluxus[:-1])
    assert integralus == pytest.approx(44055.6396,rel=1e-2)
    
    # Test the amplitude
    fl_flux = flare_model_mendoza2022(time, 2, 15, 1)
    assert np.max(fl_flux) == pytest.approx(1,rel=10e-2)
    fl_fluxa = flare_model_mendoza2022(time, 2, 15, 1)

