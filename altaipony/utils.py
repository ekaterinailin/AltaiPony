import numpy as np

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
