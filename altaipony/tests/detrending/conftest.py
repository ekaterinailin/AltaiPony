"""Synthetic light curves for the detrending tests.

``make_lc`` builds a light curve with known ingredients (time grid with gaps,
sinusoidal variability with optional P/2 harmonics, white noise, flares) from
``altaipony.synthetic``, so that each test can check the pipeline against the
truth.
"""

import numpy as np
import pytest
import astropy.units as u

from altaipony.flarelc import FlareLightCurve
from altaipony.synthetic import make_time_grid, sinusoidal_variability, inject_flares


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: runs the full detrending pipeline (deselect with -m 'not slow')")


def make_lc(duration=8.0, cadence_min=2.0, gaps=(), modes=(), flares=(),
            noise=1e-3, scale=1.0, seed=3):
    """Synthetic ``FlareLightCurve`` with known ingredients.

    All amplitudes are relative to a quiescent level of 1; the final flux is
    multiplied by ``scale`` to mimic unnormalised flux in e-/s.

    Parameters
    ----------
    duration : float
        Length in days.
    cadence_min : float
        Cadence in minutes.
    gaps : list of (start, length)
        Data gaps in days, removed from the time grid.
    modes : list of dict
        Sinusoidal modes for ``sinusoidal_variability``: ``period`` (days),
        ``amplitude`` and optionally ``phase``, ``offset``, ``harmonic_ratio``
        and ``harmonic_phase`` (a first harmonic at P/2), ``amp_mod_phase``.
        A fast rotator is a mode with a short period, e.g.
        ``dict(period=0.4, amplitude=0.03, harmonic_ratio=0.3)``.
    flares : list of (tpeak, fwhm, ampl)
        Flares (Tovar Mendoza et al. 2022 template), in days and relative
        flux. The template peaks at ~0.95 ``ampl``.
    noise : float
        Standard deviation of the white noise, relative flux.
    scale : float
        Quiescent flux level; ``flux`` and ``flux_err`` are in e-/s.
    seed : int
        Seed of the white noise.

    Returns
    -------
    lc : FlareLightCurve
        ``time`` in days, ``flux`` and ``flux_err`` (= ``noise * scale``) in e-/s.
    flare_table : pd.DataFrame
        The injected flares: tpeak, fwhm, ampl, t_start, t_end.
    """
    rng = np.random.default_rng(seed)
    t = make_time_grid(duration, cadence_min, gaps=gaps)
    f = 1.0 + sinusoidal_variability(t, list(modes))
    f = f + rng.normal(0.0, noise, t.size)
    f, table = inject_flares(t, f, flares=list(flares))
    lc = FlareLightCurve(time=t * u.d, flux=f * scale * u.electron / u.s,
                         flux_err=np.full(t.size, noise * scale) * u.electron / u.s)
    return lc, table


@pytest.fixture(name="make_lc", scope="session")
def make_lc_fixture():
    """The ``make_lc`` factory, for tests that need custom light curves."""
    return make_lc


@pytest.fixture
def synthetic_lc():
    """A default light curve: 8 days of 2-min data in e-/s (baseline 3400),
    1 % spot modulation at P = 1.3 d, 0.1 % white noise, a gap of 0.5 d at
    day 4, and two flares. Returns ``(lc, flare_table)``."""
    return make_lc(gaps=[(4.0, 0.5)],
                   modes=[dict(period=1.3, amplitude=0.01)],
                   flares=[(2.3, 0.02, 0.01), (5.7, 0.04, 0.02)],
                   scale=3400.0)
