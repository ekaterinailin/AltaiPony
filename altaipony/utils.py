import copy

from scipy.interpolate import interp1d
import os
import pandas as pd
import numpy as np
			
import logging
LOG = logging.getLogger(__name__)


def split_gaps(gaps, splits):
    """Helper function that splices up a list
    of tuples into more tuples at values defined by
    splits, like:
    ```
    gaps = [(0., 20.), (21., 34.), (37., 41.)]
    splits = [1.5, 14., 39.]
    result = split_gaps(gaps, splits)
    >>> result = [(0., 1.5), (1.5, 14.), (14., 20.), 
    >>>           (21.0, 34.), (37., 39.), (39., 41.)]
    
    Parameters:
    ------------
    gaps : list of tuples of floats or ints
        
    splits: list of floats or ints
    
    Return:
    -------
    list of tuples of floats or ints - extended gaps
    
    """
    try:
        # transform gaps to an array
        npgaps = np.array(gaps).T

        # find where the existing gaps must be split up
        splitloc = [np.where((s > npgaps[0]) & (s < npgaps[1]))[0][0] for s in splits]
        
    except IndexError:
        raise IndexError(f"The splits you passed are wrong or NaN. "
                         f"They should be values between {gaps[0][0]} and {gaps[-1][1]}.")

    # sort the user's inputs
    df = pd.DataFrame({"splits":splits,
                       "splitlocs":splitloc})

    # create an independent duplicate
    gaps2 = copy.deepcopy(gaps)


    # group splitting locations
    for loc, g in df.groupby(splitloc):

        # remove gaps that will be replaced by new ones
        gaps2.remove(gaps[loc])

        # take left boundary from old gap, 
        # then append new splits that go inbetween, 
        # and then add the right boundary
        l = [gaps[loc][0]] + list(g.splits.values) + [gaps[loc][1]]

        # reformat the list into a set of gaps
        newgaps = [(i,j) for i, j in zip(l[:-1],l[1:])]

        # insert new gaps into the new list of gaps
        gaps2[loc:loc] = newgaps 

    # sort in ascending order
    gaps2.sort(key=lambda x: x[0])

    return gaps2 


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

    data2 = data[np.isfinite(data.time.value) &
                 np.isfinite(data.pos_corr1.value) &
                 np.isfinite(data.pos_corr2.value)]

    return data2


def medsig(a):
    """Return median and outlier-robust estimate
    of standard deviation
       (1.48 x median of absolute deviations).
    Adapted from K2SC (Aigrain et al. 2016).
    """
    l = np.isfinite(a)
    nfinite = l.sum()
    if nfinite == 0:
        return np.nan, np.nan
    if nfinite == 1:
        return a[l], np.nan
    med = np.median(a[l])
    sig = 1.48 * np.median(np.abs(a[l] - med))
    return med, sig


def sigma_clip(a, max_iter=10, max_sigma=3., 
               separate_masks=False, mexc=None, **kwargs):
    """Iterative sigma-clipping routine that 
    separates not finite points, and down-
    and upwards outliers.
    
    Adapted from (Aigrain et al. 2016)
    
    1: good data point
    0: masked outlier
    
    Parameters:
    ------------
    a : np.array
        flux array
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
    kwargs : dict
        keyword arguments to pass to expand_mask
    
    Return:
    -------
    boolean array (all) or two boolean arrays (positive/negative)
    with the final outliers as zeros.
    """
    
    # perform sigma-clipping on finite points only, or custom indices given by mexc
    mexc  = np.isfinite(a) if mexc is None else np.isfinite(a) & mexc
    #init different masks for up- and downward outliers
    mhigh = np.ones_like(mexc)
    mlow  = np.ones_like(mexc)
    mask  = np.ones_like(mexc)
    
    # iteratively (with i) clip outliers above(below) (-)max_sigma *sig
    i, nm = 0, None
    
    while (nm != mask.sum()) & (i < max_iter):
    
        # Okay values are finite and not outliers
        mask = mexc & mhigh & mlow
        
        # Safety check if the mask looks fine
        nm = mask.sum()
        if nm > 1:
            # Calculate median and MAD adjusted standard deviation
            
            med, sig = medsig(a[mask])
            mhigh[mexc] = a[mexc] - med <  max_sigma * sig #indices of okay values above median
            mlow[mexc]  = a[mexc] - med > -max_sigma * sig #indices of okay values below median
           
            # Okay values are finite and not outliers
            mask = mexc & mhigh & mlow
            
            LOG.debug(f"iteration {i} at normalized median flux {med:.5f} \pm {sig:.5f}")
            LOG.debug(f"upper mask size before expansion = {mhigh.shape[0]}")
       
            # Expand the mask left and right
            mhigh = expand_mask(mhigh, **kwargs)
      
            LOG.debug("upper mask size after expansion = {mhigh.shape[0]}\n Should be the same as before.")
            
            i += 1
    
    if separate_masks:
        return mlow, mhigh
    
    else:
        return mlow & mhigh


def expand_mask(a, longdecay=1):
    """Expand the mask if multiple outliers occur in a row.
    Add sqrt(#outliers in a row) masked points
    before and after the outlier sequence.
    
    Yes the code looks is ugly, but it's faster than 
    
    Parameters:
    -----------
    a : bool array
        mask
    longdecay : int
        optional parameter to expand the mask more by 
        this factor after the series of outliers 
     
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
                
                addto = int(np.rint(np.sqrt(k)))
                
                a[i - k - addto : i - k] = 0
                a[i : i + longdecay * addto] = 0
                i += longdecay * addto
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
        np.random.seed(seed)
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag) * r)**(1. / g)





def get_response_curve(mission=None, custom_path=None, base_dir="static"):
    """
    Load and interpolate a response curve either from a built-in mission or a user-specified file.

    Returns
    -------
    wav : array
        Wavelength grid (Angstroms)
    resp : array
        Instrumental response at each wavelength
    """
    if custom_path:
        path = custom_path
    else:
        name_map = {
            "Kepler": "kepler_resp.csv",
            "K2": "kepler_resp.csv",
            "TESS": "tess-response-function.csv"
        }
        if mission is None or mission.lower() not in name_map:
            raise ValueError("Unknown mission or no mission provided.")

        base_path = os.path.join(os.path.dirname(__file__), base_dir)
        path = os.path.join(base_path, name_map[mission.lower()])

    df = pd.read_csv(path)
    
    
    required = {"lambda", "resp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Invalid response file format at {path}. "
            f"Missing column(s): {', '.join(missing)}. "
            f"Expected columns: ['lambda', 'resp']"
            )

    
    wav_raw = df["lambda"].values
    resp_raw = df["resp"].values
    wav = np.linspace(wav_raw.min(), wav_raw.max(), 1000)
    resp = interp1d(wav_raw, resp_raw, kind="cubic", fill_value=0, bounds_error=False)(wav)

    return wav, resp
