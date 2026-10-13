import pytest
import numpy as np
import pandas as pd
from inspect import signature

from ..fakeflares import (flare_model_mendoza2022, flare_model_davenport2014,
                          generate_fake_flare_distribution,
                          match_flares,
                          mod_random,
                          template_equivalent_duration,)


from .test_flarelc import mock_flc


def test_mod_random():
    # deterministic with d=True, random otherwise
    assert mod_random(3, d=True) == pytest.approx(mod_random(3, d=True))
    assert mod_random(3, d=True, seed=1) != pytest.approx(mod_random(3, d=True, seed=2))
    assert mod_random(1)[0] != mod_random(1)[0]
    # numpy's global random state is left alone
    np.random.seed(5)
    expected = np.random.rand(2)
    np.random.seed(5)
    mod_random(4, d=True)
    mod_random(4)
    assert np.random.rand(2) == pytest.approx(expected)


def test_template_equivalent_duration():
    time = np.arange(0, 2, 1. / 720.)
    flux = flare_model_mendoza2022(time, 1., 0.05, 0.1)
    x = time * 60.0 * 60.0 * 24.0
    assert template_equivalent_duration(time, flux) == pytest.approx(np.sum(np.diff(x) * flux[:-1]))
    # about 2.0 ampl * fwhm, in seconds
    assert template_equivalent_duration(time, flux) == pytest.approx(1.99 * 0.1 * 0.05 * 86400, rel=0.01)


def test_match_flares():
    flares = pd.DataFrame({'tstart': [1.0, 3.0, 5.0],
                           'tstop': [1.5, 3.2, 5.5],
                           'ed_rec': [650., 250., 30.]})
    fakes = pd.DataFrame({'peak_time': [1.2, 2.9, 3.2, 1.4, 8.0],
                          'amplitude': [0.1, 0.2, 0.3, 0.4, 0.5]})
    matched, fp = match_flares(fakes, flares)
    assert matched.shape[0] == 5
    # peaks inside (1.2, 1.4), on the edge (3.2), just before a detection
    # (2.9: only its tail overlaps), and outside all detections (8.0)
    assert matched.recovered.tolist() == [True, False, True, True, False]
    assert np.allclose(matched.ed_rec.fillna(-1).to_numpy(), [650, -1, 250, 650, -1])
    assert matched.amplitude.tolist() == fakes.amplitude.tolist()
    # the detection at 5.0-5.5 contains no injected peak
    assert fp.tstart.tolist() == [5.0]
    # no detections
    matched, fp = match_flares(fakes, flares.iloc[:0])
    assert not matched.recovered.any()
    assert fp.empty


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
#     flc = flc.find_cont_windows()
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
#     flc = flc.find_cont_windows()
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

