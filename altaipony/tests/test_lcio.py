from ..lcio import (from_TargetPixel_source, from_KeplerLightCurve_source,
                   from_K2SC_source, from_K2SC_file, from_KeplerLightCurve)
from test_flarelc import test_FlareLightCurve

ID1 = '211119999'
campaign = 4
size = 3294

def FlareLightCurve_testhelper(flc):
    """
    Test that reading in a FlareLightCurve does not kill or change any
    KeplerLightCurve attributes.

    Parameters
    -----------
    flc : FlareLightCurve
        lightcurve of EPIC 211119999
    """

    assert flc.campaign == campaign
    assert flc.flux.shape[0] == size
    assert flc.flux_err.shape[0] == size
    assert flc.time.shape[0] == size
    assert flc.remove_nans().flux.shape[0] == size
    assert flc.quality_bitmask == None
    assert lc.centroid_col.shape[0] == size
    assert lc.centroid_row.shape[0] == size
    assert flc.time_format == 'bkjd'
    assert flc.time_scale == 'tbd'
    assert flc.quarter == None
    assert flc.ra == 56.90868
    assert flc.dec == 24.891865
    assert flc.remove_outliers().flux.shape[0] == size-2
    assert flc.targetid == 211119999
    assert flc.channel == 52
    assert flc.correct().flux.shape[0] == size
    assert flc.flatten().flux.shape[0] == size

def test_from_TargetPixel_source():
    flc = from_TargetPixel_source(ID1)
    FlareLightCurve_testhelper(flc)
    pass

def test_from_KeplerLightCurve_source():
    pass

def test_from_K2SC_source():
    pass

def test_from_K2SC_file():
    pass

def test_from_KeplerLightCurve():
    pass
