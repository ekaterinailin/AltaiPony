import pytest
import numpy as np
from inspect import signature

from ..fakeflares import inject_fake_flares, aflare

from .test_flarelc import mock_flc

def test_aflare():
    pass

def test_merge_fake_and_recovered_events():
    pass

def test_generate_fake_flare_distribution():
    pass

def test_inject_fake_flares():
    flc = mock_flc(detrended=True)
    np.random.seed(84712)
    flc = flc.find_gaps()
    fake_flc = inject_fake_flares(flc)

    assert str(signature(inject_fake_flares)) == ('(lc, mode=\'hawley2014\','
                                                  ' gapwindow=0.1, fakefreq=0.25)')
    assert fake_flc.fake_flares.size == 20
    assert fake_flc.fake_flares.columns.values.tolist() == ['amplitude', 'duration_d',
                                                   'ed_inj', 'peak_time']
    assert fake_flc.detrended_flux_err.all() >= 1e-10
    assert fake_flc.detrended_flux.all() <= 1.
    assert fake_flc.detrended_flux.shape == flc.detrended_flux.shape

    fl_flux = aflare(fake_flc.time, 4.788063, 0.000299, 0.000027)
    x = fake_flc.time * 60.0 * 60.0 * 24.0
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(1.8e-5,rel=1e-2)
    return fake_flc
