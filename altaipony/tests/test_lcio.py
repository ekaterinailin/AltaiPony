import pytest
from testfixtures import LogCapture
from ..lcio import (from_TargetPixel_source, from_KeplerLightCurve_source,
                   from_K2SC_source, from_K2SC_file, from_KeplerLightCurve)

ID1 = '211119999'
path = 'examples/hlsp_k2sc_k2_llc_211119999-c04_kepler_v2_lc.fits'
campaign = 4
size = 3423

def FlareLightCurve_testhelper(flc, size=size, campaign=campaign, from_tpf = False):
    """
    Test that reading in a FlareLightCurve does not kill or change any
    KeplerLightCurve attributes.

    Parameters
    -----------
    flc : FlareLightCurve
        lightcurve of EPIC 211119999
    from_k2sc : False or bool
        if light curve is created from a processed ``K2SC`` file one cadence is
        thrown out from the resulting ``FlareLightCurve``.
    """
    if from_tpf == True:
        s = size + 1
    else:
        s = size

    assert flc.campaign == campaign
    assert flc.flux.shape[0] == s
    assert flc.flux_err.shape[0] == s
    assert flc.time.shape[0] == s
    assert ((flc.quality_bitmask == 'none') or (flc.quality_bitmask == None))
    assert flc.centroid_col.shape[0] == s
    assert flc.centroid_row.shape[0] == s
    assert flc.time_format == 'bkjd'
    assert flc.time_scale == 'tdb'
    assert flc.quarter == None
    assert flc.ra == 56.90868
    assert flc.dec == 24.891865
    assert flc.targetid == 211119999
    assert flc.channel == 52
    assert flc.remove_nans().flux.shape[0] == s
    assert flc.remove_outliers().flux.shape[0] == s-2
    assert flc.correct().flux.shape[0] == s
    assert flc.flatten().flux.shape[0] == s


def test_from_TargetPixel_source():
    flc = from_TargetPixel_source(ID1)
    FlareLightCurve_testhelper(flc, from_tpf=True)
    pass

def test_from_KeplerLightCurve_source():
    flc = from_KeplerLightCurve_source(ID1)
    FlareLightCurve_testhelper(flc)
    pass

@pytest.fixture(autouse=True)
def capture():
    with LogCapture() as capture:
        yield capture

def test_from_K2SC_source(size=size):
    flc = from_K2SC_source(ID1)
    FlareLightCurve_testhelper(flc)
    assert flc.detrended_flux_err.shape[0] == size
    assert flc.detrended_flux.shape[0] == size
    assert flc.detrended_flux.shape[0] == size
    assert flc.flares == None
    assert flc.gaps == None
    #also test if a local path throws warning
    #test if a list of IDs is correctly resolved - must return a list of FlareLightCurves
    #also test if a list of paths is correctly resolved - must return a list of FlareLightCurves
    pass

def test_from_K2SC_file():
    flc = from_K2SC_file(path)
    FlareLightCurve_testhelper(flc)
    pass

def test_from_KeplerLightCurve():
    #is currently implicitly tested by test_from_K2SC_source and test_from_TargetPixel_source
    pass
