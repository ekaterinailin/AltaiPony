import numpy as np
from .utils import sigma_clip


def detrend_savgol(lc, og_flux, og_flux_err, max_sigma=2.5, longdecay=6, 
                   w=121, break_tolerance=10, **kwargs):
    """New detrending with savgol filter.
    
    Parameters:
    -----------
    lc : FlareLightCurve
        Light curve (already interpolated if needed)
    og_flux : array
        Original flux before interpolation
    og_flux_err : array
        Original flux error before interpolation
    max_sigma: float>0
        sigma clipping threshold
    longdecay: int
        adding masked datapoints to the tail if multiple outliers occur in a row
    w : odd int
        window length for savgol filter
    break_tolerance : int
        Gap threshold for splitting light curve
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
        mask_ij = np.arange(i, j)
        
        left_idx = max(0, i - 1)
        right_idx = min(len(lcn.flux) - 1, j)
        
        left_val = fluxold[left_idx] if not np.isnan(fluxold[left_idx]) else np.nanmedian(fluxold)
        right_val = fluxold[right_idx] if not np.isnan(fluxold[right_idx]) else np.nanmedian(fluxold)
        
        interpolation_ij = np.interp(lcn.time.value[mask_ij],
                                     [lcn.time.value[left_idx], lcn.time.value[right_idx]],
                                     [left_val, right_val])
   
        interpolation_ij = np.where(interpolation_ij == 0, 1e-10, interpolation_ij)
        lcrsf.flux[mask_ij] = fluxold[mask_ij] / interpolation_ij
    
    # Filter based on interpolated column
    if hasattr(lcrsf, 'interpolated') and 'interpolated' in lcrsf.colnames:
        print("Filtering out interpolated data points based on 'interpolated' column.")
        keep_mask = lcrsf.interpolated.value == 0
        
        # Filter the light curve to remove interpolated points
        lcrsf = lcrsf[keep_mask]
        
        print(f"After filtering: lcrsf length = {len(lcrsf)}, og_flux length = {len(og_flux)}")
        
        # After removing interpolated points, lcrsf should have the same length as og_flux
        # They should align 1-to-1 (no indexing needed)
        if len(lcrsf) != len(og_flux):
            raise ValueError(
                f"Length mismatch after filtering: lcrsf has {len(lcrsf)} points "
                f"but og_flux has {len(og_flux)} points. "
                f"This suggests the interpolation didn't preserve cadence numbers correctly."
            )
        
        # Use og_flux directly - they're already aligned
        og_flux_filtered = og_flux.value if hasattr(og_flux, 'value') else og_flux
        og_flux_err_filtered = og_flux_err.value if hasattr(og_flux_err, 'value') else og_flux_err
    else:
        print("No 'interpolated' column found; keeping all data points.")
        og_flux_filtered = og_flux.value if hasattr(og_flux, 'value') else og_flux
        og_flux_err_filtered = og_flux_err.value if hasattr(og_flux_err, 'value') else og_flux_err

    # store detrended flux and restore original flux
    lcrsf.detrended_flux = lcrsf.flux.value * np.nanmedian(og_flux_filtered)
    lcrsf.detrended_flux_err = og_flux_err_filtered
    
    # Restore original flux
    if hasattr(og_flux, 'unit'):
        lcrsf.flux = og_flux_filtered * og_flux.unit
        lcrsf.flux_err = og_flux_err_filtered * og_flux_err.unit
    else:
        lcrsf.flux = og_flux_filtered
        lcrsf.flux_err = og_flux_err_filtered

    return lcrsf