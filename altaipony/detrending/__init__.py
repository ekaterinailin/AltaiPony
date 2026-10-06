"""Light-curve detrending.

``custom_detrending`` (in ``pipeline``) removes stellar variability in three
stages: a baseline fit (``baselines``: segmented polynomials, multi-sine, or
spline), two Savitzky-Golay passes, and, for periodic stars, a Gaussian-process
model of the rotational modulation (``gp``). ``estimate_detrended_noise``
(``noise``) estimates the noise of the result.
"""

from .pipeline import custom_detrending
from .noise import estimate_detrended_noise

__all__ = ["custom_detrending", "estimate_detrended_noise"]
