import pytest
import numpy as np
import pandas as pd
from inspect import signature

from ..fakeflares import (inject_fake_flares,
                          aflare,
                          generate_fake_flare_distribution,
                          merge_fake_and_recovered_events)


from .test_flarelc import mock_flc

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
    assert (ampl <= 1e3).all()
    assert (ampl >= 1e-4).all()
    assert len(dur) == n
    assert len(ampl) == n

    dur, ampl = generate_fake_flare_distribution(n, mode='uniform')
    assert (dur <= 2e3/60/24).all()
    assert (dur >= 0.5/60/24).all()
    assert (ampl <= 1e3).all()
    assert (ampl >= 1e-4).all()
    assert len(dur) == n
    assert len(ampl) == n

def test_inject_fake_flares():
    flc = mock_flc(detrended=True)
    np.random.seed(84712)
    flc = flc.find_gaps()
    fake_flc = inject_fake_flares(flc)

    assert fake_flc.fake_flares.size == 20
    assert fake_flc.fake_flares.columns.values.tolist() == ['amplitude', 'duration_d',
                                                            'ed_inj', 'peak_time']
    assert fake_flc.detrended_flux_err.all() >= 1e-10
    assert fake_flc.detrended_flux.all() <= 1.
    assert fake_flc.detrended_flux.shape == flc.detrended_flux.shape
    print(fake_flc.fake_flares)
    flc = mock_flc(detrended=False)
    np.random.seed(84712)
    flc = flc.find_gaps()
    fake_flc = inject_fake_flares(flc, inject_before_detrending=True)

    assert fake_flc.fake_flares.size == 20
    assert fake_flc.fake_flares.columns.values.tolist() == ['amplitude', 'duration_d',
                                                            'ed_inj', 'peak_time']
    assert fake_flc.flux_err.all() >= 1e-10
    assert fake_flc.flux.all() <= 1.
    assert fake_flc.flux.shape == flc.flux.shape

def test_aflare_and_equivalent_duration():

    n = 1000
    time = np.arange(0, n/48, 1./48.)
    fl_flux = aflare(time, 11.400134, 1.415039, 110.981950)
    x = time * 60.0 * 60.0 * 24.0
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(1.22e7,rel=1e-2)

    fl_flux = aflare(time, 11.400134, 1.415039, 110.981950, upsample=True)
    x = time * 60.0 * 60.0 * 24.0
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(1.22e7,rel=1e-2)
