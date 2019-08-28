import logging
import numpy as np

LOG = logging.getLogger(__name__)

def k2sc_quality_cuts(data):
    """
    Apply all the quality checks that k2sc uses internally.

    Parameters
    ------------
    data : KeplerLightCurve or TargetPixelFile

    Return
    --------
    KeplerLightCurve or TargetPixelFile where ``time``, ``centroid_col``, and
    ``centroid_row`` all have finite values.
    """

    data2 = data[np.isfinite(data.time) &
                 np.isfinite(data.pos_corr1) &
                 np.isfinite(data.pos_corr2)]

    return data2

# From the K2SC fork

from numpy import isfinite, nan, median, abs, ones_like, where, rint, sqrt



def medsig(a):
    """Return median and outlier-robust estimate of standard deviation
       (1.48 x median of absolute deviations).
    """
    l = isfinite(a)
    nfinite = l.sum()
    if nfinite == 0:
        return nan, nan
    if nfinite == 1:
        return a[l], nan
    med = median(a[l])
    sig = 1.48 * median(abs(a[l] - med))
    return med, sig


def sigma_clip(a, max_iter=10, max_sigma=5, 
               separate_masks=False, mexc=None,
               debug=False):
    """Iterative sigma-clipping routine that 
    separates not finite points, and down-
    and upwards outliers.
    """
    
    # perform sigma-clipping on finite points only, or custom indices given by mexc
    mexc  = isfinite(a) if mexc is None else isfinite(a) & mexc
    
    #init different masks for up- and downward outliers
    mhigh = ones_like(mexc)
    mlow  = ones_like(mexc)
    mask  = ones_like(mexc)
    
    # iteratively (with i) clip outliers above(below) (-)max_sigma *sig
    i, nm = 0, None
    while (nm != mask.sum()) and (i < max_iter):
        mask = mexc & mhigh & mlow
        nm = mask.sum()
        med, sig = medsig(a[mask])
        mhigh[mexc] = a[mexc] - med <  max_sigma*sig #indices of okay values above median
        mlow[mexc]  = a[mexc] - med > -max_sigma*sig #indices of okay values below median
        i += 1
        mask = mexc & mhigh & mlow
        LOG.debug("iteration {} at normalized median flux {:.5f} \pm {:.5f}".format(i, med, sig))
        LOG.debug("upper mask size before expansion = ", mhigh.shape[0])
        mhigh = expand_mask(mhigh)
        LOG.debug("upper mask size after expansion = ", mhigh.shape[0], "\n")
    if separate_masks:
        return mlow, mhigh
    else:
        return mlow & mhigh

def expand_mask(a, divval=3):
    """Expand the mask if multiple outliers occur in a row."""
    i,j,k = 0, 0, 0
    while i<len(a):
        v=a[i]
        
        if (v==0) & (j==0):
            k += 1
            j = 1
            i += 1

        elif (v==0) & (j==1):
            k += 1
            i += 1

        elif (v==1) & (j==0):
            i += 1
        
        elif (v==1) & (j==1):
            if k >= 3:
                addto = int(rint(sqrt(k/divval)))
                a[i - k - addto : i - k] = 0
                a[i : i + addto] = 0
                i += addto
            else:
                i += 1
            j = 0
            k = 0
                  
    return a
