import numpy as np
from .utils import sigma_clip


def detrend_savgol(lc, og_flux, og_flux_err, max_sigma=2.5, longdecay=6, 
                   w=121, break_tolerance=10, **kwargs):
    """New detrending with savgol filter.
    
    Parameters:
    -----------
    
    max_sigma: float>0
        sigma clipping threshold
    longdecay: int
        adding masked datapoints to the tail if
        multiple outliers occur in a row
    w : odd int
        window length for savgol filter
    break_tolerance : int
        If there are large gaps in time, flatten will split the flux into 
        several sub-lightcurves and apply savgol_filter to each individually. 
        A gap is defined as a period in time larger than break_tolerance times 
        the median gap. To disable this feature, set break_tolerance to None.
    kwargs : dict
        keyword arguments to feed LightCurve.flatten()
    """
    
    # normalize
    lcn = lc.normalize()
    
    # sigma clip
    m = sigma_clip(lcn.flux, max_sigma=max_sigma, longdecay=longdecay)

    # convert bool to int
    mask = ~m * 1

    # from Appaloosa:
    # convert mask to start and stop
    reverse_counts = np.zeros_like(lcn.flux, dtype='int')
    for k in range(1, len(lcn.flux)):
        reverse_counts[-k] = (mask[-k]
                                * (reverse_counts[-(k-1)]
                                + mask[-k]))

    # find flare start where values in reverse_counts switch from 0 to >=N3 
    # SET N3=1 because we care about all outliers!
    istart_i = np.where((reverse_counts[1:] >= 1) &
                        (reverse_counts[:-1] - reverse_counts[1:] < 0))[0] + 1

    # use the value of reverse_counts to determine how many points away stop is
    istop_i = istart_i + (reverse_counts[istart_i]) - 1

    # get a list of masked candidates to extrapolate
    candidates = list(zip(istart_i, istop_i))

    fluxold = lcn.flux.copy()

    # remove the flares candidates for now
    lcn.flux[mask] = np.nan

    # SAVGOL APPLIED HERE
    # https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.flatten.html
    # flatten with light curve
    # set break_tolerance to 10 by default, i.e. 20 min in a 2min cadence LC
    lcrsf = lcn.flatten(window_length=w, break_tolerance=break_tolerance)
   
    # fill lcrsf nans with median value
    lcrsf.flux[np.isnan(lcrsf.flux)] = np.nanmedian(lcrsf.flux)

    # cycle over all candidates
    for i, j in candidates:

        # span the data
        mask_ij = np.arange(i, j)
        
        # Get interpolation anchor points - use nearest non-NaN values
        # Handle edge cases where i or j might be at boundaries
        left_idx = max(0, i - 1)
        right_idx = min(len(lcn.flux) - 1, j)
        
        # Find valid anchor points for interpolation
        left_val = fluxold[left_idx] if not np.isnan(fluxold[left_idx]) else np.nanmedian(fluxold)
        right_val = fluxold[right_idx] if not np.isnan(fluxold[right_idx]) else np.nanmedian(fluxold)
        
        # linear interpolate below the flare
        interpolation_ij = np.interp(lcn.time.value[mask_ij],
                                     [lcn.time.value[left_idx], lcn.time.value[right_idx]],
                                     [left_val, right_val])
   
        # fill in the masked data again
        # Avoid division by zero
        interpolation_ij = np.where(interpolation_ij == 0, 1e-10, interpolation_ij)
        lcrsf.flux[mask_ij] = fluxold[mask_ij] / interpolation_ij
    
    # Track which indices to keep (non-interpolated cadences)
    if hasattr(lcrsf, 'interpolated') and 'interpolated' in lcrsf.colnames:
        keep_mask = lcrsf.interpolated.value == 0
    else:
        keep_mask = np.ones(len(lcrsf), dtype=bool)
    
    # Filter the light curve
    lcrsf = lcrsf[keep_mask]
    
    # Filter og_flux and og_flux_err to match
    og_flux_filtered = og_flux.value[keep_mask] if hasattr(og_flux, 'value') else og_flux[keep_mask]
    og_flux_err_filtered = og_flux_err.value[keep_mask] if hasattr(og_flux_err, 'value') else og_flux_err[keep_mask]

    # store detrended flux and restore original flux
    lcrsf.detrended_flux = lcrsf.flux.value * np.nanmedian(og_flux_filtered)
    lcrsf.detrended_flux_err = og_flux_err_filtered
    
    # Restore original flux (filtered to match)
    if hasattr(og_flux, 'unit'):
        lcrsf.flux = og_flux_filtered * og_flux.unit
        lcrsf.flux_err = og_flux_err_filtered * og_flux_err.unit
    else:
        lcrsf.flux = og_flux_filtered
        lcrsf.flux_err = og_flux_err_filtered

    return lcrsf