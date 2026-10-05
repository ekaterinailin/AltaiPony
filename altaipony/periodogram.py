"""Lomb-Scargle periodogram helper shared by the detrending modules."""

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
