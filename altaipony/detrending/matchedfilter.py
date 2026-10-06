"""Matched-filter flare detection for GP detrending.

A Gaussian Process trained on flux that still contains a flare will happily fit
that flare as if it were part of the stellar variability and subtract it away.
Per-cadence masks (a simple sigma cut on the residual) catch *narrow* flares,
whose individual points clear the threshold, but they miss *wide, moderate-SNR*
flares: each cadence sits under the per-cadence cut even though the flare's
integrated flux is large.

A matched filter is the optimal linear detector for a signal of known shape in
noise.  Cross-correlating the (baseline-subtracted, noise-whitened) residual
with a flare template coherently sums the flare's flux over its whole duration
while noise adds incoherently, converting "integrated flux" into a single
high-SNR statistic.  A flare that is invisible per cadence lights up in the
filter output.  Because the template is *asymmetric* (fast rise, slow
double-exponential decay), the filter responds to flare-shaped bumps and not to
smooth, symmetric starspot modulation — that shape mismatch is what separates
flares from rotation.

This module is self-contained: it builds the mask and returns it.  The GP
pipeline in ``detrending.pipeline`` only calls :func:`matched_filter_flare_mask` and
unions the result into the GP's initial flare mask.

The flare template is Davenport et al. (2014).  When ``altaipony`` is
importable its ``aflare`` is used verbatim so the template is *identical* to the
one used for injection-recovery; otherwise an inline implementation with the
same coefficients is used.
"""

import logging

import numpy as np
from scipy.signal import fftconvolve

from ..utils import MAD_TO_STD

logger = logging.getLogger(__name__)

# ── flare template ────────────────────────────────────────────────────────
# Prefer altaipony's aflare so the matched-filter template is bit-for-bit the
# same profile used elsewhere (e.g. injection-recovery).  Try the common
# locations/names; fall back to an inline Davenport (2014) implementation.
_aflare = None
_AFLARE_SOURCE = "inline Davenport (2014)"
for _modname, _funcname in (
    ("altaipony.fakeflares", "aflare"),
    ("altaipony.altai", "aflare"),
    ("altaipony.altai", "aflare1"),
    ("altaipony.fakeflares", "aflare1"),
):
    try:
        _mod = __import__(_modname, fromlist=[_funcname])
        _aflare = getattr(_mod, _funcname)
        _AFLARE_SOURCE = f"{_modname}.{_funcname}"
        break
    except Exception:  # pragma: no cover
        continue


def _davenport_aflare(t, tpeak, fwhm, ampl):
    """Inline Davenport et al. (2014) empirical flare profile (unit-consistent
    with altaipony's ``aflare1``).  Used only when altaipony is unavailable."""
    t = np.asarray(t, dtype=float)
    x = (t - tpeak) / fwhm
    flux = np.zeros_like(x)
    rise = (x >= -1.0) & (x < 0.0)
    flux[rise] = (
        1.0
        + 1.941 * x[rise]
        - 0.175 * x[rise] ** 2
        - 2.246 * x[rise] ** 3
        - 1.125 * x[rise] ** 4
    )
    decay = x >= 0.0
    flux[decay] = 0.6890 * np.exp(-1.600 * x[decay]) + 0.3030 * np.exp(
        -0.2783 * x[decay]
    )
    return ampl * flux


def _eval_aflare(t, tpeak, fwhm, ampl):
    """Evaluate the flare profile, preferring altaipony's ``aflare`` and
    falling back to the inline profile if it is absent or has a different
    signature."""
    if _aflare is not None:
        try:
            return np.asarray(_aflare(t, tpeak, fwhm, ampl), dtype=float)
        except Exception:  # pragma: no cover - signature mismatch
            pass
    return _davenport_aflare(t, tpeak, fwhm, ampl)


def _flare_template(cadence, fwhm, rise_widths=1.0, decay_widths=6.0):
    """Build a normalised flare template sampled at the data cadence.

    The template spans ``rise_widths`` FWHM before the peak through
    ``decay_widths`` FWHM after it, and is zero-meaned and L2-normalised so
    that (a) the matched-filter output has unit variance under white noise
    (hence reads directly in sigma) and (b) it is insensitive to a constant
    offset in the residual.

    Returns
    -------
    template : ndarray
        Zero-mean, unit-norm template.
    peak_idx : int
        Index of the flare peak within ``template`` (needed to align the
        filter response to the flare peak rather than the template centroid).
    """
    t = np.arange(-rise_widths * fwhm, decay_widths * fwhm + cadence, cadence)
    s = _eval_aflare(t, 0.0, fwhm, 1.0)
    peak_idx = int(np.argmax(s))
    s = s - s.mean()
    norm = np.sqrt(np.sum(s**2))
    if norm == 0 or not np.isfinite(norm):
        return None, 0
    return s / norm, peak_idx


# ── helpers ─────────────────────────────────────────────────────────────────
def _segment_slices(time, segment_gap_factor=10):
    """Yield (start, stop) slices splitting ``time`` at real gaps."""
    n = len(time)
    if n == 0:
        return []
    if n == 1:
        return [(0, 1)]
    cad = np.nanmedian(np.diff(time))
    breaks = np.where(np.diff(time) > segment_gap_factor * cad)[0] + 1
    bounds = np.concatenate([[0], breaks, [n]]).astype(int)
    return list(zip(bounds[:-1], bounds[1:]))


def _robust_noise(residual, n_iter=2):
    """Robust per-array scatter (1.4826·MAD) with light outlier rejection so
    flares themselves do not inflate the noise estimate."""
    r = residual[np.isfinite(residual)]
    if r.size < 3:
        return np.nan
    keep = np.ones(r.size, dtype=bool)
    sigma = np.nan
    for _ in range(n_iter):
        med = np.median(r[keep])
        mad = np.median(np.abs(r[keep] - med))
        sigma = MAD_TO_STD * mad
        if sigma == 0 or not np.isfinite(sigma):
            break
        keep = np.abs(r - med) < 3.0 * sigma
        if keep.sum() < 3:
            break
    return sigma


def _cross_correlate(w, s, peak_idx):
    """Cross-correlate whitened residual ``w`` with template ``s`` so the
    output ``c[i]`` is the template amplitude for a flare *peaking* at cadence
    ``i`` (in sigma units when ``w`` is unit-variance and ``||s|| = 1``)."""
    n, L = len(w), len(s)
    if n < L:
        return np.zeros(n)
    full = fftconvolve(w, s[::-1], mode="full")  # length n + L - 1
    start = (L - 1) - peak_idx
    return full[start : start + n]


# ── public API ────────────────────────────────────────────────────────────
def matched_filter_statistic(
    time,
    residual,
    noise=None,
    fwhm_grid=(0.02, 0.05, 0.1, 0.15, 0.25),
    segment_gap_factor=10,
):
    """Return the matched-filter detection statistic over a bank of flare
    widths.

    The residual is whitened (per segment) by ``noise`` — a robust MAD scatter
    when not supplied — and cross-correlated with an ``aflare`` template at each
    FWHM in ``fwhm_grid``.  The per-cadence maximum over the bank is the
    statistic (a flare of any width in the bank produces a peak); the winning
    FWHM index is returned too so the flare's extent can be masked.

    Parameters
    ----------
    time, residual : ndarray
        Baseline-subtracted residual (stellar modulation already removed).
    noise : ndarray or float or None
        Per-cadence noise for whitening.  If None, estimated robustly per
        segment from the residual scatter.
    fwhm_grid : sequence of float
        Flare FWHMs to match, in days.
    segment_gap_factor : int
        Time-gap multiple (of the median cadence) that starts a new segment;
        correlation never spans a gap.

    Returns
    -------
    mf : ndarray
        Detection statistic (sigma units), same shape as ``time``.
    best_fwhm : ndarray
        FWHM (days) of the best-matching template at each cadence.
    """
    time = np.asarray(time, dtype=float)
    residual = np.asarray(residual, dtype=float)
    n = len(time)
    mf = np.zeros(n)
    best_fwhm = np.full(n, np.nan)
    fwhm_grid = np.atleast_1d(fwhm_grid).astype(float)

    if noise is not None and np.isscalar(noise):
        noise = np.full(n, float(noise))

    for a, b in _segment_slices(time, segment_gap_factor):
        seg_t = time[a:b]
        seg_r = residual[a:b]
        m = b - a
        if m < 5:
            continue
        cad = np.nanmedian(np.diff(seg_t))
        if not np.isfinite(cad) or cad <= 0:
            continue

        if noise is None:
            sig = _robust_noise(seg_r)
        else:
            seg_noise = np.asarray(noise)[a:b]
            sig = np.nanmedian(seg_noise[np.isfinite(seg_noise)])
        if not np.isfinite(sig) or sig <= 0:
            continue

        # Whiten: fill non-finite residual with 0 so it contributes nothing.
        w = np.where(np.isfinite(seg_r), (seg_r - np.nanmedian(seg_r)) / sig, 0.0)

        seg_mf = mf[a:b]
        seg_best = best_fwhm[a:b]
        for fwhm in fwhm_grid:
            s, peak_idx = _flare_template(cad, fwhm)
            if s is None or len(s) >= m:
                continue
            c = _cross_correlate(w, s, peak_idx)
            better = c > seg_mf
            seg_mf[better] = c[better]
            seg_best[better] = fwhm
        mf[a:b] = seg_mf
        best_fwhm[a:b] = seg_best

    return mf, best_fwhm


def matched_filter_flare_mask(
    time,
    residual,
    noise=None,
    snr_threshold=5.0,
    fwhm_grid=(0.02, 0.05, 0.1, 0.15),
    grow_sigma=1.5,
    expand_cadences=2,
    max_flare_days=0.6,
    segment_gap_factor=10,
    return_statistic=False,
):
    """Boolean flare mask from the matched-filter statistic.

    Contiguous runs of cadences above ``snr_threshold`` are grouped into flare
    *events*; each event is grown outward from its peak — asymmetrically, and
    for free — until the whitened residual falls back to the noise floor
    (``grow_sigma``), covering the fast rise and slow decay without a
    width-dependent guess.  ``expand_cadences`` pads each side and
    ``max_flare_days`` caps the growth so a mis-fit can't mask a whole segment.

    Parameters
    ----------
    time, residual : ndarray
        Baseline-subtracted residual (stellar modulation already removed).
    noise : ndarray or float or None
        Whitening noise (robust MAD per segment if None).
    snr_threshold : float
        Detection threshold in matched-filter sigma.  The statistic has unit
        variance under white noise, so ~5 keeps the per-cadence false-alarm
        rate negligible while still catching wide flares, whose *integrated*
        statistic sits far above their per-cadence SNR.
    fwhm_grid : sequence of float
        Flare FWHMs (days) to match.
    grow_sigma : float
        Grow each event outward while the whitened residual exceeds this many
        sigma, so the mask tracks the actual flare shape.
    expand_cadences : int
        Extra cadences padded on each side of every event.
    max_flare_days : float
        Hard cap on the half-width an event may grow to (days).
    segment_gap_factor : int
        Gap multiple that starts a new segment.
    return_statistic : bool
        If True, also return ``(mf, best_fwhm)``.

    Returns
    -------
    mask : ndarray of bool
        True where a flare is detected (aligned to ``time``).
    (mf, best_fwhm) : tuple, optional
        Returned only if ``return_statistic``.
    """
    time = np.asarray(time, dtype=float)
    residual = np.asarray(residual, dtype=float)
    n = len(time)
    mf, best_fwhm = matched_filter_statistic(
        time, residual, noise, fwhm_grid, segment_gap_factor
    )

    mask = np.zeros(n, dtype=bool)
    if noise is not None and np.isscalar(noise):
        noise_arr = np.full(n, float(noise))
    elif noise is not None:
        noise_arr = np.asarray(noise, dtype=float)
    else:
        noise_arr = None

    for a, b in _segment_slices(time, segment_gap_factor):
        seg_mf = mf[a:b]
        seg_r = residual[a:b]
        m = b - a
        if m < 5:
            continue
        cad = np.nanmedian(np.diff(time[a:b]))
        if not np.isfinite(cad) or cad <= 0:
            continue
        max_grow = int(np.ceil(max_flare_days / cad))

        # Whitened residual for the grow step.
        if noise_arr is None:
            sig = _robust_noise(seg_r)
        else:
            seg_noise = noise_arr[a:b]
            sig = np.nanmedian(seg_noise[np.isfinite(seg_noise)])
        if not np.isfinite(sig) or sig <= 0:
            continue
        w = np.where(np.isfinite(seg_r), (seg_r - np.nanmedian(seg_r)) / sig, 0.0)

        above = seg_mf >= snr_threshold
        if not above.any():
            continue

        # Group contiguous super-threshold cadences into events.
        edges = np.diff(above.astype(int))
        starts = list(np.where(edges == 1)[0] + 1)
        stops = list(np.where(edges == -1)[0] + 1)
        if above[0]:
            starts = [0] + starts
        if above[-1]:
            stops = stops + [m]

        seg_mask = mask[a:b]
        for s0, s1 in zip(starts, stops):
            peak = s0 + int(np.argmax(seg_mf[s0:s1]))
            # Grow left/right from the peak while the residual stays elevated,
            # allowing a couple of dips below the floor before stopping.
            lo = peak
            misses = 0
            while lo > 0 and (peak - lo) < max_grow:
                if w[lo - 1] > grow_sigma:
                    misses = 0
                else:
                    misses += 1
                    if misses > 2:
                        break
                lo -= 1
            hi = peak
            misses = 0
            while hi < m - 1 and (hi - peak) < max_grow:
                if w[hi + 1] > grow_sigma:
                    misses = 0
                else:
                    misses += 1
                    if misses > 2:
                        break
                hi += 1
            seg_mask[
                max(0, lo - expand_cadences) : min(m, hi + expand_cadences + 1)
            ] = True
        mask[a:b] = seg_mask

    if return_statistic:
        return mask, (mf, best_fwhm)
    return mask
