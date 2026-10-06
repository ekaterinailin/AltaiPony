"""Noise estimate of a detrended light curve."""

import numpy as np
import pandas as pd

from ..altai import _find_iterative_median
from ..utils import upper_outlier_threshold


def estimate_detrended_noise(
    flc, mask_pos_outliers_sigma=2.5, std_window=100, longdecay=6
):
    """
    Estimate detrended flux uncertainties using rolling standard deviation.

    Flare masking is intentionally **one-sided**: only data points whose flux
    exceeds the segment median by more than ``mask_pos_outliers_sigma`` × MAD
    are excluded.  Clipping negative deviations (genuine troughs from spot
    modulation or imperfect detrending) would artificially inflate the noise
    estimate at those phases, so they are left in.  MAD is used instead of
    std so that the threshold itself is not inflated by any surviving flare
    signal.

    Parameters
    ----------
    flc : FlareLightCurve
        Light curve with detrended_flux attribute
    mask_pos_outliers_sigma : float
        One-sided sigma threshold: points with residual > median +
        ``mask_pos_outliers_sigma`` × 1.4826 × MAD are treated as flare
        contamination and set to NaN before the rolling std is computed.
        Defaults to 2.5.
    std_window : int
        Window size for the rolling standard deviation (in cadences).
        Defaults to 100.
    longdecay : int
        Long-decay parameter forwarded to ``_find_iterative_median``.
        Defaults to 6.

    Returns
    -------
    flc : FlareLightCurve
        Input light curve with detrended_flux_err attribute updated.
    """
    # Find gaps if not already done
    if flc.gaps is None:
        flc = flc.find_gaps()

    # Extract arrays we need (avoids repeated attribute access)
    detrended_flux = np.asarray(
        getattr(flc.detrended_flux, "value", flc.detrended_flux), dtype=float
    )
    n_points = len(detrended_flux)

    # Initialize output array
    detrended_flux_err = np.full(n_points, np.nan)

    # Process each gap segment
    for le, ri in flc.gaps:
        # Extract segment
        flux_segment = detrended_flux[le:ri].copy()

        # --- first pass: one-sided positive clip on the raw segment ----------
        # Only mask upward outliers (flares).  Clipping the lower tail would
        # bias the noise estimate high at spot minima.
        upper_threshold1 = upper_outlier_threshold(flux_segment, mask_pos_outliers_sigma)
        mask = flux_segment <= upper_threshold1  # True = keep

        flux_segment_masked = flux_segment.copy()
        flux_segment_masked[~mask] = np.nan

        # --- second pass: refine on iterative-median-subtracted residuals ----
        # The iterative median tracks slow baseline drifts; subtracting it
        # centres the segment near zero so the MAD threshold is not biased
        # by any residual low-frequency trend.
        it_med_segment = _find_iterative_median(
            flux_segment,
            gaps=[(0, ri - le)],
            longdecay=longdecay,
        )
        flux_normalized = flux_segment - it_med_segment

        # One-sided clip on the centred residuals.  Same logic: positive-only.
        upper_threshold2 = upper_outlier_threshold(flux_normalized, mask_pos_outliers_sigma)
        mask_refined = flux_normalized <= upper_threshold2

        flux_normalized_masked = flux_normalized.copy()
        flux_normalized_masked[~mask_refined] = np.nan

        # Compute rolling std on the cleaned, centred residuals
        final_err = (
            pd.Series(flux_normalized_masked)
            .rolling(std_window, center=True, min_periods=1)
            .std()
            .interpolate()
            .values
        )

        # Store in output array
        detrended_flux_err[le:ri] = final_err

    # Set the result on the original lightcurve
    flc.detrended_flux_err = detrended_flux_err

    return flc
