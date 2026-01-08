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
        If there are large gaps in time, flatten will split the flux into several sub-lightcurves and apply savgol_filter to each individually. A gap is defined as a period in time larger than break_tolerance times the median gap. To disable this feature, set break_tolerance to None.
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
                        (reverse_counts[:-1] - reverse_counts[1:] < 0))[0]  +1

    # use the value of reverse_counts to determine how many points away stop is
    istop_i = istart_i + (reverse_counts[istart_i]) -1

    # get a list of masked candidates to extrapolate
    candidates = list(zip(istart_i, istop_i))

    fluxold = lcn.flux.copy()

    # remove the flares candidates for now
    lcn.flux[mask] = np.nan

    

    # SAVGOL APPLIED HERE
    # https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.flatten.html?highlight=flatten#lightkurve.LightCurve.flatten
    # flatten with light curve
    # set break_tolerance to 10 by default, i.e. 20 min in a 2min cadence LC
    lcrsf  = lcn.flatten(window_length=w, break_tolerance=break_tolerance) #replace with 6h or 3h window

    # cycle over all candidates
    for i, j in candidates:

        # span the data
        mask_ij = np.arange(i,j)
        # linearinterpolate below the flare

        interpolation_ij = np.interp(lcn.time.value[mask_ij],
                                     [lcn.time.value[i],lcn.time.value[j]],
                                     [lcn.flux[i],lcn.flux[j]])
   
        # fill in the masked data again
        lcrsf.flux[mask_ij] = fluxold[mask_ij] / interpolation_ij
    
    # then ignore the interpolated points again
    lcrsf = lcrsf[lcrsf.interpolated.value==0]

    # store detrended flux and restore original flux

    lcrsf.detrended_flux = lcrsf.flux.value * np.nanmedian(og_flux.value)
    lcrsf.detrended_flux_err = og_flux_err.value
    lcrsf.flux = og_flux
    lcrsf.flux_err = og_flux_err


    
    return lcrsf

