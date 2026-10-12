"""Tests of the GP step of ``custom_detrending``."""

import numpy as np
import pytest

from altaipony.detrending import custom_detrending

pytestmark = pytest.mark.slow

NOISE = 3e-4


def _values(x):
    return np.asarray(getattr(x, "value", x), dtype=float)


@pytest.fixture(scope="module")
def two_period_star(make_lc):
    """25 days of 2-min data with two unrelated spot modes, at 3 d and 0.69 d,
    each with a P/2 harmonic and an amplitude that changes over 10 days. The
    periodogram picks the 3 d mode for the first rotation term, which cannot
    follow the 0.69 d mode."""
    lc, _ = make_lc(duration=25.0, noise=NOISE, scale=3400.0,
                    modes=[dict(period=3.0, amplitude=0.035, harmonic_ratio=0.3, amp_mod_phase=3.5),
                           dict(period=0.69, amplitude=0.03, harmonic_ratio=0.3, amp_mod_phase=1.5)])
    return lc


@pytest.mark.parametrize("second_rotation, max_scatter, min_scatter", [
    (False, np.inf, 2.0),   # one rotation term leaves the 0.69 d mode in the residual
    (True, 1.5, 0.0),       # the second term fits it: residual close to the white noise
])
def test_second_rotation_term_fits_second_period(two_period_star, second_rotation,
                                                 max_scatter, min_scatter):
    lc = custom_detrending(two_period_star, gp_second_rotation=second_rotation)
    scatter = np.nanstd(_values(lc.detrended_flux)) / NOISE
    assert min_scatter < scatter < max_scatter
