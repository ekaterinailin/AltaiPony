'''
Use this file to keep various detrending methods
'''

import numpy as np
import pandas as pd

def MultiBoxcar(time, flux, error, numpass=3, kernel=2.0,
                sigclip=3, pcentclip=5, returnindx=False,
                debug=False):
    '''
    Boxcar smoothing with multi-pass outlier rejection. Uses both errors
    and local scatter for rejection Uses Pandas rolling median filter.

    Parameters
    time : 1-d numpy array
    flux : 1-d numpy array
    error : 1-d numpy array
    numpass : int, optional
        the number of passes to make over the data. (Default = 3)
    kernel : float, optional
        the boxcar size in hours. (Default is 2.0)
        Note: using whole numbers is probably wise here.
    sigclip : int, optional
        Number of times the standard deviation to clip points at
        (Default is 5)
    pcentclip : int, optional
        % of data to clip for outliers, i.e. 5= keep 5th-95th percentile
        (Default is 5)
    debug : bool, optional
        used to print out troubleshooting things (default=False)

    Returns
    -------
    The smoothed light curve model
    '''
    flux_i = pd.DataFrame({'flux':flux,'error_i':error,'time_i':time})
    time_i = np.array(time)
    error_i = error
    indx_i = np.arange(len(time)) # for tracking final indx used
    exptime = np.nanmedian(time_i[1:]-time_i[:-1])

    nptsmooth = int(kernel/24.0 / exptime)

    if (nptsmooth < 4):
        nptsmooth = 4

    if debug is True:
        print('# of smoothing points: '+str(nptsmooth))

    # now take N passes of rejection on it
    for k in range(0, numpass):
        # rolling median in this data span with the kernel size
        flux_i['flux_i_sm'] = flux_i.flux.rolling(nptsmooth, center=True).median()
        flux_i = flux_i.dropna(how='any')

        if (flux_i.shape[0] > 1):
            flux_i['diff_k'] = flux_i.flux-flux_i.flux_i_sm
            lims = np.nanpercentile(flux_i.diff_k, (pcentclip, 100-pcentclip))

            # iteratively reject points
            # keep points within sigclip (for phot errors), or
            # within percentile clip (for scatter)
            ok = np.logical_or((np.abs(flux_i.diff_k / flux_i.error_i) < sigclip),
                               (lims[0] < flux_i.diff_k) * (flux_i.diff_k < lims[1]))
            if debug is True:
                print('k = '+str(k))
                print('number of accepted points: '+str(len(ok[0])))

            flux_i = flux_i[ok]


    flux_sm = np.interp(time, flux_i.time_i, flux_i.flux)

    indx_out = flux_i.index.values

    if returnindx is False:
        return flux_sm
    else:
        return np.array(indx_out, dtype='int')
