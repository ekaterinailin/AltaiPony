"""Light-curve detrending.

``custom_detrending`` (in ``pipeline``) removes stellar variability with a
Gaussian-process model (``gp``), trained on the flux without the flares.  A
segmented-polynomial baseline (``baselines.polynomial``) provides the initial
flare mask, and a matched filter (``matchedfilter``) extends it.  ``estimate_detrended_noise`` (``noise``)
estimates the noise of the result.
"""

from .pipeline import custom_detrending
from .noise import estimate_detrended_noise

__all__ = ["custom_detrending", "estimate_detrended_noise"]
