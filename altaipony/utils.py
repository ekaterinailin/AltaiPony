import logging
from numpy import isfinite, nan, median, abs, ones_like, where, rint, sqrt, random

LOG = logging.getLogger(__name__)

def k2sc_quality_cuts(data):
    """
    Apply all the quality checks that K2SC (Aigrain et al. 2016) 
    uses internally.

    Parameters
    ------------
    data : KeplerLightCurve or TargetPixelFile

    Return
    --------
    KeplerLightCurve or TargetPixelFile where ``time``, ``centroid_col``, and
    ``centroid_row`` all have finite values.
    """

    data2 = data[isfinite(data.time) &
                 isfinite(data.pos_corr1) &
                 isfinite(data.pos_corr2)]

    return data2


def medsig(a):
    """Return median and outlier-robust estimate
    of standard deviation
       (1.48 x median of absolute deviations).
    Adapted from K2SC (Aigrain et al. 2016).
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


def sigma_clip(a, max_iter=10, max_sigma=3., 
               separate_masks=False, mexc=None):
    """Iterative sigma-clipping routine that 
    separates not finite points, and down-
    and upwards outliers.
    
    Adapted from (Aigrain et al. 2016)
    
    1: good data point
    0: masked outlier
    
    Parameters:
    ------------
    max_iter : int
        how often do we want to recalculate sigma to get
        ever smaller outliers?
    max_sigma : float
        where do we clip the outliers?
    separate_masks : bools
        if True will give to boolean arrays for
        positive and negative outliers.
    mexc : boolean array
        custom mask to additionally account for
    
    Return:
    -------
    boolean array (all) or two boolean arrays (positive/negative)
    with the final outliers as zeros.
    """
    
    # perform sigma-clipping on finite points only, or custom indices given by mexc
    mexc  = isfinite(a) if mexc is None else isfinite(a) & mexc
    #init different masks for up- and downward outliers
    mhigh = ones_like(mexc)
    mlow  = ones_like(mexc)
    mask  = ones_like(mexc)
    
    # iteratively (with i) clip outliers above(below) (-)max_sigma *sig
    i, nm = 0, None
    
    while (nm != mask.sum()) & (i < max_iter):
    
        # Okay values are finite and not outliers
        mask = mexc & mhigh & mlow
        
        # Safety check if the mask looks fine
        nm = mask.sum()
        
        # Calculate median and MAD adjusted standard deviation
        med, sig = medsig(a[mask])
    
        mhigh[mexc] = a[mexc] - med <  max_sigma * sig #indices of okay values above median
        mlow[mexc]  = a[mexc] - med > -max_sigma * sig #indices of okay values below median
    
        # Okay values are finite and not outliers
        mask = mexc & mhigh & mlow
    
        LOG.debug(f"iteration {i} at normalized median flux {med:.5f} \pm {sig:.5f}")
        LOG.debug(f"upper mask size before expansion = {mhigh.shape[0]}")
    
        # Expand the mask left and right
        mhigh = expand_mask(mhigh)
    
        LOG.debug("upper mask size after expansion = {mhigh.shape[0]}\n Should be the same as before.")
        
        i += 1
    
    if separate_masks:
        return mlow, mhigh
    
    else:
        return mlow & mhigh


def expand_mask(a, divval=1):
    """Expand the mask if multiple outliers occur in a row.
    Add 3 x sqrt(#outliers in a row / divval) masked points
    before and after the outlier sequence.
    
    Parameters:
    -----------
    a : bool array
        mask
    divval : float
        optional parameter to set the length of the
        expanded mask
     
    Return:
    -------
    array - expanded mask
    """
    i, j, k = 0, 0, 0
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
            if k >= 2:
                addto = int(rint(3 * sqrt(k/divval)))
                a[i - k - addto : i - k] = 0
                a[i : i + addto] = 0
                i += addto
            else:
                i += 1
            j = 0
            k = 0
                 
    return a

def generate_random_power_law_distribution(a, b, g, size=1, seed=None):
    """Power-law generator for pdf(x)\propto x^{g-1}
    for a<=x<=b
    """
    if seed is not None:
        random.seed(seed)
    r = random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag) * r)**(1. / g)
