import pytest
import numpy as np
import pandas as pd
from inspect import signature

from ..fakeflares import (aflare,
                          generate_fake_flare_distribution,
                          merge_fake_and_recovered_events,
                          merge_complex_flares,
                          recovery_probability,
                          equivalent_duration_ratio,
                          mod_random,)


from .test_flarelc import mock_flc

def test_resolve_complexity():
    pass

def test_mod_random():
    assert mod_random(1, d=True)[0] == pytest.approx(0.48661046)
    assert mod_random(1)[0] != mod_random(1)[0]

def test_equivalent_duration_ratio():
    bins = 5
    minval= 1e-5
    maxval = 1200.3
    data = pd.DataFrame({'ed_rec': [minval,2e-5,2e-5,4e-4,1e-2,6e-2,0.54,1.33,4.5,12.2,44,maxval],
                         'ed_inj': [1e-4,6e-4,8e-4,4e-3,0.05,0.1,1.1,1.9,6.8,16.2,49,1500.3],
                         'cstart':1})
    ed_rat = equivalent_duration_ratio(data, bins=bins, fixed_bins=True)
    assert ed_rat.shape[0] == 5
    assert minval*0.99 == pytest.approx(ed_rat.loc[0,'min_ed_rec'])
    assert maxval*1.01 == ed_rat.loc[4,'max_ed_rec']
    assert 5. == ed_rat.loc[1,'rel_rec']


def test_recovery_probability():
    bins = 5
    minval= 3e-4
    maxval = 1200.3
    data = pd.DataFrame({'ed_rec': [0.,0.,0.,0.,0.,0.,1.33,4.5,12.2,44,901.3],
                         'ed_inj': [minval,6e-4,8e-4,4e-3,0.1,1.1,1.9,6.8,16.2,49,maxval],
                         'cstart':1})
    rec_prob = recovery_probability(data,bins=bins,fixed_bins=True)
    assert rec_prob.rec_prob.astype(float).tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert rec_prob.shape[0] == 5
    assert minval*.99 == rec_prob.loc[0,'min_ed_inj']
    assert maxval*1.01 == rec_prob.loc[4,'max_ed_inj']

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

    dur, ampl = generate_fake_flare_distribution(n, mode='uniform')
    assert (dur <= 2e4/60/24).all()
    assert (dur >= 10./60/24).all()
    assert (ampl <= 1e2).all()
    assert (ampl >= 1e-4).all()
    assert len(dur) == n
    assert len(ampl) == n

    n = 10
    dur, ampl = generate_fake_flare_distribution(n, mode='loglog', d=True)
    print(dur)
    print(ampl)
    assert dur[2] == pytest.approx(0.13889317143)
    assert ampl[2] ==  pytest.approx(0.1479375816948176)
    assert (dur <= 20).all()
    assert (dur >= 1e-4).all()
    assert (ampl <= 1e2).all()
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

def test_merge_complex_flares():
    gen = np.random.rand(13,13)
    gen[-1] = gen[3]
    cols = ['amplitude', 'cstart', 'cstop', 'duration_d', 'ed_inj', 'ed_rec',
            'ed_rec_err', 'istart', 'istop', 'peak_time', 'tstart', 'tstop',
            'ampl_rec']
    data = pd.DataFrame(data=gen, columns=cols)
    resolved_data = merge_complex_flares(data)
    assert resolved_data.shape[0] == 12
    assert len(resolved_data.columns.values) == 14
