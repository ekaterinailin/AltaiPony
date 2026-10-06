"""Lomb-Scargle periodogram helper and detection of a strong periodic signal."""

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
    amplitude_threshold=0.01,
    period_min=0.1,
    period_max=None,
    flux_median=None,
):
    """Test whether the light curve contains a strong, high-amplitude periodic
    signal using the Lomb-Scargle periodogram.

    Both conditions must be satisfied simultaneously for the function to return
    ``True``:

    * The false-alarm probability of the highest peak is below
      ``fap_threshold``.
    * The semi-amplitude of a single-frequency sine fitted at the peak
      frequency exceeds ``amplitude_threshold`` × (median flux).

    Parameters
    ----------
    time : array_like
        Time values in days.
    flux : array_like
        Raw flux values.
    fap_threshold : float
        FAP threshold below which the signal is considered significant.
        Defaults to 1e-3.
    amplitude_threshold : float
        Minimum fractional semi-amplitude (relative to median flux) for the
        signal to be considered high-amplitude.  Defaults to 0.01.
    period_min : float
        Minimum period to search, in days.  Defaults to 0.1.
    period_max : float or None
        Maximum period to search, in days.  Defaults to half the time-span.

    flux_median : float or None
        Reference median used to express the semi-amplitude as a fraction.
        If None, the median of ``flux`` itself is used.  Pass the original
        raw-flux median when calling on residuals (which are centred near
        zero) to avoid division by a near-zero value.

    Returns
    -------
    is_periodic : bool
        True when both the FAP and amplitude criteria are met.
    peak_period : float
        Period of the highest LS peak in days.
    rel_amplitude : float
        Fractional semi-amplitude of the best-fit sine at the peak period
        relative to the reference median flux.
    fap : float
        False-alarm probability of the peak power under the
        Baluev (2008) analytic approximation.
    """
    valid = ~(np.isnan(time) | np.isnan(flux))
    t = time[valid]
    f = flux[valid]

    if len(t) < 20:
        return False, np.nan, np.nan, np.nan

    f_median = np.nanmedian(f)

    # Use caller-supplied reference median for amplitude normalisation if given.
    # This is essential when `flux` is a residual centred near zero, where the
    # local median would be ~0 and cause division-by-zero.
    ref_median = flux_median if flux_median is not None else f_median

    if ref_median == 0:
        return False, np.nan, np.nan, np.nan

    # Centre flux so LS is not confused by a DC offset
    f_centred = f - f_median

    if period_max is None:
        period_max = (t[-1] - t[0]) / 2.0

    ls, frequency, power = lomb_scargle(
        t, f_centred, min_frequency=1.0 / period_max, max_frequency=1.0 / period_min
    )
    if len(power) == 0:
        return False, np.nan, np.nan, np.nan

    peak_idx = np.argmax(power)
    peak_freq = frequency[peak_idx]
    peak_period = 1.0 / peak_freq
    peak_power = power[peak_idx]

    # False-alarm probability (Baluev 2008 analytic approximation)
    fap = ls.false_alarm_probability(peak_power, method="baluev")

    # Semi-amplitude from LS model coefficients at the peak frequency:
    # model = offset + a*cos(2π f t) + b*sin(2π f t)
    # semi-amplitude = sqrt(a² + b²)
    theta = ls.model_parameters(peak_freq)  # [offset, a_cos, b_sin]
    rel_amplitude = np.sqrt(theta[1] ** 2 + theta[2] ** 2) / abs(ref_median)

    is_periodic = bool((fap < fap_threshold) and (rel_amplitude > amplitude_threshold))

    return is_periodic, peak_period, rel_amplitude, fap
