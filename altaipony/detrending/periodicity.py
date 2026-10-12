"""Lomb-Scargle periodogram helper and detection of a significant periodic signal."""

import numpy as np
from astropy.timeseries import LombScargle



def lomb_scargle(t, y, min_frequency=None, max_frequency=None, frequency=None,
                 samples_per_peak=10):
    """Compute a Lomb-Scargle periodogram of the finite points of ``y``.

    Either pass a frequency range, which is sampled with
    ``LombScargle.autopower``, or an explicit ``frequency`` grid. In the range
    case, ``min_frequency`` is raised to one cycle per data span, so no period
    longer than the data is searched. ``y`` is used as given; centre it first
    if needed.

    Parameters
    ----------
    t, y : array-like
        Times in days and values. Non-finite pairs are dropped.
    min_frequency, max_frequency : float
        Frequency range in cycles per day. Ignored if ``frequency`` is given.
    frequency : array-like or None
        Explicit frequency grid in cycles per day.
    samples_per_peak : int
        Oversampling of the autopower grid. Default 10.

    Returns
    -------
    ls : astropy.timeseries.LombScargle
        The periodogram object, e.g. for ``false_alarm_probability`` or
        ``model_parameters``.
    frequency, power : numpy.ndarray
        Frequency grid and power. Both are empty if the range is empty.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    t, y = t[finite], y[finite]
    ls = LombScargle(t, y)

    if frequency is not None:
        frequency = np.asarray(frequency, dtype=float)
        return ls, frequency, ls.power(frequency)

    span = float(t.max() - t.min()) if len(t) > 1 else 0.0
    if span > 0:
        min_frequency = max(min_frequency, 1.0 / span)
    if not min_frequency < max_frequency:
        return ls, np.array([]), np.array([])
    frequency, power = ls.autopower(
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        samples_per_peak=samples_per_peak,
    )
    return ls, frequency, power


def detect_strong_periodicity(
    time,
    flux,
    fap_threshold=1e-3,
    period_min=0.1,
    period_max=None,
):
    """Test whether the light curve contains a significant periodic signal
    using the Lomb-Scargle periodogram: the false-alarm probability of the
    highest peak must be below ``fap_threshold``.

    No amplitude criterion is applied: in injection-recovery tests, the GP
    step that follows a detection also helps stars whose variability is far
    below the per-cadence noise.

    Parameters
    ----------
    time : array_like
        Time values in days.
    flux : array_like
        Raw flux values.
    fap_threshold : float
        FAP threshold below which the signal is considered significant.
        Defaults to 1e-3.
    period_min : float
        Minimum period to search, in days.  Defaults to 0.1.
    period_max : float or None
        Maximum period to search, in days.  Defaults to half the time-span.

    Returns
    -------
    is_periodic : bool
        True when the FAP criterion is met.
    peak_period : float
        Period of the highest LS peak in days.
    fap : float
        False-alarm probability of the peak power under the
        Baluev (2008) analytic approximation.
    """
    valid = ~(np.isnan(time) | np.isnan(flux))
    t = time[valid]
    f = flux[valid]

    if len(t) < 20:
        return False, np.nan, np.nan

    # Centre flux so LS is not confused by a DC offset
    f_centred = f - np.nanmedian(f)

    if period_max is None:
        period_max = (t[-1] - t[0]) / 2.0

    ls, frequency, power = lomb_scargle(
        t, f_centred, min_frequency=1.0 / period_max, max_frequency=1.0 / period_min
    )
    if len(power) == 0:
        return False, np.nan, np.nan

    peak_idx = np.argmax(power)
    peak_period = 1.0 / frequency[peak_idx]

    # False-alarm probability (Baluev 2008 analytic approximation)
    fap = ls.false_alarm_probability(power[peak_idx], method="baluev")

    return bool(fap < fap_threshold), peak_period, fap
