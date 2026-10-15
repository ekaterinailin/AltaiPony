"""Tests of the GP step of ``custom_detrending``."""

import numpy as np
import pytest

from altaipony.detrending import custom_detrending
from altaipony.detrending.gp import fit_gp_rotation

pytestmark = pytest.mark.slow

NOISE = 3e-4


def _values(x):
    return np.asarray(getattr(x, "value", x), dtype=float)


@pytest.fixture(scope="module")
def two_period_star(make_lc):
    """25 days of 2-min data with two unrelated spot modes, at 3 d and 0.69 d,
    each with a P/2 harmonic and an amplitude that changes over 10 days. The
    periodogram picks the 3 d mode for the first rotation term.  With the
    jitter fitted to the white noise (``jitter_only``, default since
    2026-10-14) that term also follows the 0.69 d mode; with the old
    ``flux_err`` + jitter model it could not."""
    lc, _ = make_lc(duration=25.0, noise=NOISE, scale=3400.0,
                    modes=[dict(period=3.0, amplitude=0.035, harmonic_ratio=0.3, amp_mod_phase=3.5),
                           dict(period=0.69, amplitude=0.03, harmonic_ratio=0.3, amp_mod_phase=1.5)])
    return lc


@pytest.mark.parametrize("second_rotation", [False, True])
def test_two_periods_leave_white_noise(two_period_star, second_rotation):
    """With or without the second rotation term, the residual is close to the
    white noise."""
    lc = custom_detrending(two_period_star, gp_second_rotation=second_rotation)
    scatter = np.nanstd(_values(lc.detrended_flux)) / NOISE
    assert scatter < 1.5


@pytest.fixture(scope="module")
def spotted_star():
    """10 days of 2-min data: a 5 % modulation at 1.5 d, ~170x the white
    noise.  The jitter floor of the flux_err + jitter model (0.05 x the
    flux's standard deviation) would be ~6x the noise here."""
    rng = np.random.default_rng(1)
    t = np.arange(0, 10, 2 / 1440)
    f = 1 + 0.05 * np.sin(2 * np.pi * t / 1.5) + rng.normal(0, NOISE, len(t))
    return t, f


@pytest.mark.parametrize("floor", [0.1, 3.0])
def test_jitter_follows_the_noise(spotted_star, floor):
    """The jitter is fitted relative to the white noise of the data, not to
    flux_err or the modulation's amplitude, and it stays above the floor."""
    t, f = spotted_star
    _, _, params = fit_gp_rotation(t, f, 1e-3, period=1.5, flare_mask=np.zeros(len(t), bool),
                                   jitter_floor=floor)
    expected = max(floor, 1.0) * NOISE
    assert 0.7 * expected < params["jitter"] < 1.5 * expected


def test_jitter_only_ignores_flux_err(spotted_star):
    t, f = spotted_star
    mask = np.zeros(len(t), bool)
    m1 = fit_gp_rotation(t, f, 1e-3, period=1.5, flare_mask=mask)[0]
    m2 = fit_gp_rotation(t, f, 1e-5, period=1.5, flare_mask=mask)[0]
    np.testing.assert_allclose(m1, m2)


@pytest.mark.parametrize("amplitude, period, lowered", [
    (0.01, 3.0, False),   # slow, small modulation: binning by 10 is harmless
    (0.2, 0.25, True),    # fast, large modulation: 20-min bins flatten its peaks
])
def test_bin_factor_follows_curvature(amplitude, period, lowered):
    from altaipony.detrending.pipeline import _bin_factor_for_curvature
    rng = np.random.default_rng(2)
    t = np.arange(0, 5, 2 / 1440)
    model = 1 + amplitude * np.sin(2 * np.pi * t / period)
    flux = model + rng.normal(0, NOISE, len(t))
    k = _bin_factor_for_curvature(t, flux, model, np.zeros(len(t), bool), 10, 0.5)
    assert (k < 10) == lowered
