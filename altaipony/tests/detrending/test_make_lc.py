"""Checks that ``make_lc`` builds what it promises, so that other tests can
rely on it."""

import numpy as np
import astropy.units as u


def test_flat_lc_has_requested_noise_and_scale(make_lc):
    lc, table = make_lc(duration=4.0, noise=1e-3, scale=3400.0)
    assert lc.flux.unit == u.electron / u.s
    assert len(lc.time) == 4 * 720
    assert abs(np.median(lc.flux.value) / 3400.0 - 1) < 1e-4
    assert abs(np.std(lc.flux.value) / 3400.0 / 1e-3 - 1) < 0.05
    np.testing.assert_allclose(lc.flux_err.value, 3.4)
    assert table.empty


def test_scale_multiplies_everything(make_lc):
    kw = dict(duration=4.0, modes=[dict(period=1.3, amplitude=0.01),
                                   dict(period=8 / 24, amplitude=0.02, harmonic_ratio=0.3)],
              flares=[(2.0, 0.02, 0.01)])
    lc1, t1 = make_lc(**kw)
    lc2, t2 = make_lc(scale=3400.0, **kw)
    np.testing.assert_allclose(lc2.flux.value, 3400.0 * lc1.flux.value, rtol=1e-12)
    assert t1.equals(t2)


def test_harmonic_adds_power_at_half_the_period(make_lc):
    """A mode with harmonic_ratio r has power r**2 times the fundamental's at P/2."""
    lc, _ = make_lc(duration=4.0, noise=1e-6,
                    modes=[dict(period=0.5, amplitude=0.02, harmonic_ratio=0.3)])
    t, f = lc.time.value, lc.flux.value - 1.0
    amp = [2 * abs(np.mean(f * np.exp(-2j * np.pi * t / p))) for p in (0.5, 0.25)]
    np.testing.assert_allclose(amp, [0.02, 0.006], rtol=1e-3)


def test_gaps_are_removed(make_lc):
    lc, _ = make_lc(duration=4.0, gaps=[(1.0, 0.5)])
    t = lc.time.value
    assert not ((t >= 1.0) & (t < 1.5)).any()
    assert np.isclose(np.diff(t).max(), 0.5, atol=2 / 1440)


def test_flare_peaks_at_injected_time(make_lc):
    lc, table = make_lc(duration=4.0, noise=1e-6, flares=[(2.0, 0.02, 0.01)])
    f = lc.flux.value - 1.0
    assert abs(lc.time.value[np.argmax(f)] - 2.0) <= 2 / 1440
    assert np.isclose(f.max(), 0.95 * 0.01, rtol=0.05)
    assert table.t_start[0] < 2.0 < table.t_end[0]


def test_synthetic_lc_fixture(synthetic_lc):
    lc, table = synthetic_lc
    assert len(table) == 2
    assert abs(np.median(lc.flux.value) / 3400.0 - 1) < 0.01
