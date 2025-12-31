import logging
import warnings
import astropy.units as u

import numpy as np

from altaipony.flarelc import FlareLightCurve
from lightkurve import LightCurve

LOG = logging.getLogger(__name__)


def to_flare_lightcurve(lc):
    """
    Convert a Lightkurve LightCurve to an AltaiPony FlareLightCurve.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        Lightkurve LightCurve object

    Returns
    -------
    FlareLightCurve
        AltaiPony FlareLightCurve object

    Raises
    ------
    TypeError
        If lc is not a LightCurve instance
    ValueError
        If lightcurve has no valid data points
    """
    
    # Validate input
    if not isinstance(lc, LightCurve):
        raise TypeError(f"Expected LightCurve, got {type(lc).__name__}")
    
    # Change class
    lc.__class__ = FlareLightCurve
    
    # Remove nans and infs
    valid_mask = (np.isfinite(lc.time.value) & 
                  np.isfinite(lc.flux.value) & 
                  np.isfinite(lc.flux_err.value))
    
    lc = lc[valid_mask]
    
    # Check if any data remains
    if len(lc) == 0:
        raise ValueError("No valid data points after removing NaNs and infs")
    
    # Add columns for detrended flux and error
    lc["detrended_flux"] = np.full_like(lc.flux.value, np.nan)
    lc["detrended_flux_err"] = np.full_like(lc.flux_err.value, np.nan)
    
    # Setup flare table
    lc._init_flare_table()
    
    # Set qcs to either quarter, campaign, or sector
    # Fixed typo: CAMPIGN -> CAMPAIGN
    lc.meta["qcs"] = lc.meta.get("QUARTER", 
                                 lc.meta.get("CAMPAIGN", 
                                            lc.meta.get("SECTOR", None)))
    
    # Set cadence in seconds from TIMEDEL in days
    timedel = lc.meta.get("TIMEDEL", None)
    
    try:
        lc.meta["cadence"] = float(timedel) * u.day.to(u.second)
    except (TypeError, ValueError):
        warnings.warn(f"Could not convert TIMEDEL '{timedel}' to cadence", 
                        UserWarning)
      # Try to calculate from time array
        if len(lc.time) > 1:
            dt = np.median(np.diff(lc.time.value))
            lc.meta["cadence"] = float(dt) * u.day.to(u.second)
        else:
            lc.meta["cadence"] = None
    
    # Make all meta keys lowercase
    lc.meta = {k.lower(): v for k, v in lc.meta.items()}
    
    return lc