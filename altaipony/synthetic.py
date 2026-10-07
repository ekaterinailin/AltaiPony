"""
Synthetic light curves for tests and injection-recovery experiments.

A light curve is built from a cadence grid with data gaps, sinusoidal stellar
variability (spot modulation with a first harmonic), white noise, and flares
from the Tovar Mendoza et al. (2022) template, scaled to a quiescent baseline
in e-/s.  All amplitudes and the noise level are relative to that baseline.
Each component has a function that builds it from explicit parameters (for
tests) and, where applicable, one that draws random parameters (for
injection-recovery experiments).
"""

import numpy as np
import pandas as pd

from .fakeflares import flare_model_mendoza2022


# ---------------------------------------------------------------------------
# Cadence grid with gaps
# ---------------------------------------------------------------------------

def draw_gaps(duration_days, rng):
    """Draw random data gaps: one every 5–15 days, lasting 5 min to 1 day.

    Returns
    -------
    list of (start, length) in days
    """
    gaps = []
    t_cursor = 0.0
    while t_cursor < duration_days:
        t_cursor += rng.uniform(5.0, 15.0)
        gap_dur = rng.uniform(5.0, 1440.0) / 1440.0
        gaps.append((t_cursor, gap_dur))
        t_cursor += gap_dur
    return gaps


def make_time_grid(duration_days=25.0, cadence_min=2.0, gaps=()):
    """Uniform cadence grid starting at 0, with the cadences inside ``gaps``
    (a list of ``(start, length)`` in days) removed."""
    t_full = np.arange(0.0, duration_days, cadence_min / 1440.0)
    gap_mask = np.zeros(len(t_full), dtype=bool)
    for start, length in gaps:
        gap_mask |= (t_full >= start) & (t_full < start + length)
    return t_full[~gap_mask]


# ---------------------------------------------------------------------------
# Stellar variability
# ---------------------------------------------------------------------------

def draw_variability_modes(n_modes, rng, period_range=(0.25, 10.0),
                           amplitude_range=(0.001, 0.05), harmonic_ratio=0.3,
                           time_variable_amplitude=True):
    """Draw random sinusoidal variability modes.

    Periods are uniform in 0.25–10 d (6 h to 10 d), relative amplitudes
    uniform in 0.1–5 %, and each mode has a constant offset of up to ±0.5 %
    and a first harmonic at P/2 with ``harmonic_ratio`` times its amplitude
    and a random phase, so that the modulation is not a pure sinusoid, as for
    a spotted star.  With ``time_variable_amplitude``, each amplitude is
    modulated on a 10-day timescale (see ``sinusoidal_variability``).

    Returns
    -------
    list of dict
        Keyword arguments for ``sinusoidal_variability``.
    """
    modes = []
    for _ in range(n_modes):
        mode = dict(
            period=rng.uniform(*period_range),
            amplitude=rng.uniform(*amplitude_range),
            phase=rng.uniform(0.0, 2 * np.pi),
            offset=rng.uniform(-0.005, 0.005),
            harmonic_ratio=harmonic_ratio,
            harmonic_phase=rng.uniform(0.0, 2 * np.pi),
        )
        if time_variable_amplitude:
            mode["amp_mod_phase"] = rng.uniform(0, 2 * np.pi)
        modes.append(mode)
    return modes


def sinusoidal_variability(time, modes):
    """Sum of sinusoidal modes.

    Each mode is a dict with ``period`` (days) and ``amplitude`` (relative
    flux), and optionally ``phase`` (radians), ``offset`` (relative flux),
    ``harmonic_ratio`` and ``harmonic_phase`` (a first harmonic at P/2 with
    ``harmonic_ratio`` times the amplitude; default none), and
    ``amp_mod_phase``.  If ``amp_mod_phase`` is given, the amplitude of the
    mode and its harmonic is multiplied by
    ``1.5 * (1 + sin(2π t / 10 d + amp_mod_phase))``, i.e. it varies between
    0 and 3 times its nominal value over 10 days.
    """
    flux = np.zeros_like(time)
    for mode in modes:
        amplitude = mode["amplitude"]
        if mode.get("amp_mod_phase") is not None:
            amplitude = amplitude * (1.5 * (1 + np.sin(2 * np.pi * time / 10.0 + mode["amp_mod_phase"])))
        else:
            amplitude = amplitude * np.ones_like(time)
        phase = 2 * np.pi * time / mode["period"]
        flux += mode.get("offset", 0.0) + amplitude * (
            np.sin(phase + mode.get("phase", 0.0))
            + mode.get("harmonic_ratio", 0.0) * np.sin(2 * phase + mode.get("harmonic_phase", 0.0))
        )
    return flux


# ---------------------------------------------------------------------------
# Full synthetic light curve
# ---------------------------------------------------------------------------

def generate_synthetic_lc(
    duration_days=25.0,
    cadence_min=2.0,
    noise_ppm=None,
    n_modes=None,
    seed=None,
    baseline_range=(1000.0, 100000.0),
):
    """Generate one random synthetic light curve in e-/s.

    Parameters
    ----------
    duration_days : float
        Total length in days.  Default 25.
    cadence_min : float
        Nominal cadence in minutes.  Default 2 (TESS 2-min).
    noise_ppm : float or None
        White noise level in parts-per-million.  If None, drawn uniformly
        from [200, 2000].
    n_modes : int or None
        Number of variability modes (see ``draw_variability_modes``: periods
        6 h–10 d, amplitudes 0.1–5 %, each with a P/2 harmonic).  If None,
        drawn from {1, 2, 3} with equal probability.
    seed : int or None
        Random seed for reproducibility.
    baseline_range : tuple of float or None
        Range of the quiescent baseline in e-/s, drawn uniformly.  Default
        1000–100000.  None gives a light curve normalised to ~1.

    Returns
    -------
    time : ndarray
    flux : ndarray
    meta : dict
        ``noise_ppm``, ``n_modes``, ``seed``, ``baseline`` (e-/s, or 1),
        ``modes`` (the drawn modes) and ``min_period_hr`` (period of the
        fastest mode, in hours).
    """
    rng = np.random.default_rng(seed)

    if noise_ppm is None:
        noise_ppm = rng.uniform(200.0, 2000.0)
    if n_modes is None:
        n_modes = rng.integers(1, 4)

    time = make_time_grid(duration_days, cadence_min, draw_gaps(duration_days, rng))
    modes = draw_variability_modes(n_modes, rng)
    flux = 1.0 + sinusoidal_variability(time, modes)

    flux += rng.normal(0.0, noise_ppm * 1e-6, size=len(time))

    # Quiescent level in e-/s, from its own stream so that the shape of the
    # light curve does not depend on whether it is scaled.
    baseline = 1.0
    if baseline_range is not None:
        baseline_rng = np.random.default_rng(None if seed is None else [seed, 2])
        baseline = baseline_rng.uniform(*baseline_range)
        flux = flux * baseline

    meta = dict(noise_ppm=noise_ppm, n_modes=n_modes, seed=seed, baseline=baseline,
                modes=modes, min_period_hr=24.0 * min((m["period"] for m in modes), default=np.inf))
    return time, flux, meta


# ---------------------------------------------------------------------------
# Flares
# ---------------------------------------------------------------------------

def inject_flares(time, flux, n_flares=None, rng=None, flares=None, mean_n_flares=7.5,
                  baseline=1.0):
    """Add flares (Tovar Mendoza et al. 2022 template) to a light curve.

    Either pass ``flares`` explicitly, or let them be drawn at random:
    ``n_flares`` (default Poisson(``mean_n_flares``)) flares at random cadences, with FWHM
    log-uniform in 6–180 min and amplitude log-uniform in 2.5–50 times the
    point-to-point noise of ``flux``.

    Amplitudes are relative to the quiescent level ``baseline`` (in the units
    of ``flux``); the injected flux is ``baseline`` times the template.

    ``fwhm`` and ``ampl`` are the template's parameters: the template peaks at
    about 0.95 ``ampl``, its measured FWHM is about 1.09 ``fwhm``, and its
    equivalent duration is about 1.82 ``ampl * fwhm``.

    Parameters
    ----------
    time, flux : ndarray
    n_flares : int or None
        Number of random flares.  Ignored if ``flares`` is given.
    rng : numpy Generator or None
    flares : list of (tpeak, fwhm, ampl) or None
        Explicit flares, in days and relative flux.
    mean_n_flares : float
        Mean number of random flares per light curve when ``n_flares`` is
        None.  Default 7.5 (about 0.3 per day in a 25-day light curve).
    baseline : float
        Quiescent flux level the relative amplitudes refer to, e.g.
        ``meta["baseline"]`` from ``generate_synthetic_lc``.  Default 1.

    Returns
    -------
    flux_with_flares : ndarray
    flare_table : pd.DataFrame
        One row per flare with columns tpeak, fwhm, ampl (relative),
        t_start, t_end (the window where the flare exceeds 10 % of ``ampl``).
    """
    if flares is None:
        if rng is None:
            rng = np.random.default_rng()
        if n_flares is None:
            n_flares = rng.poisson(mean_n_flares)
        noise_est = np.nanmedian(np.abs(np.diff(flux))) * 1.4826 / np.sqrt(2) / baseline
        fwhm_min_days, fwhm_max_days = 6.0 / 1440.0, 180.0 / 1440.0
        flares = []
        for _ in range(n_flares):
            tpeak = rng.choice(time)
            fwhm = np.exp(rng.uniform(np.log(fwhm_min_days), np.log(fwhm_max_days)))
            ampl = noise_est * np.exp(rng.uniform(np.log(2.5), np.log(50.0)))
            flares.append((tpeak, fwhm, ampl))

    rows = []
    flux_out = flux.copy()
    for tpeak, fwhm, ampl in flares:
        flare_flux = flare_model_mendoza2022(time, tpeak, fwhm, ampl)
        flux_out += flare_flux * baseline
        above = flare_flux > 0.1 * ampl
        if above.any():
            t_start, t_end = time[above][0], time[above][-1]
        else:
            t_start, t_end = tpeak - 2 * fwhm, tpeak + 2 * fwhm
        rows.append(dict(tpeak=tpeak, fwhm=fwhm, ampl=ampl, t_start=t_start, t_end=t_end))

    return flux_out, pd.DataFrame(rows, columns=["tpeak", "fwhm", "ampl", "t_start", "t_end"])
