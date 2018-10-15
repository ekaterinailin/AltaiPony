import os
import pytest
from testfixtures import LogCapture
from ..lcio import (from_TargetPixel_source, from_KeplerLightCurve_source,
                   from_K2SC_source, from_K2SC_file, from_KeplerLightCurve)
from ..__init__ import PACKAGEDIR

ids = ['211119999', '210951703', 211117077]
paths = [os.path.join(PACKAGEDIR, 'examples',
         'hlsp_k2sc_k2_llc_{}-c04_kepler_v2_lc.fits'.format(id_)) for id_ in ids]
campaign = 4
sizes = [3423, 3423, 3423]
iterator = zip(ids, paths, sizes)

@pytest.fixture(autouse=True)
def capture():
    with LogCapture() as capture:
        yield capture

def FlareLightCurve_testhelper(flc, iterator=iterator, campaign=campaign, from_tpf = False):
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
    for (ID, path, size) in iterator:
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
    for ID in ids:
        flc = from_TargetPixel_source(ID)
        FlareLightCurve_testhelper(flc, from_tpf=True)


def test_from_KeplerLightCurve_source():
    for ID in ids:
        flc = from_KeplerLightCurve_source(ID)
        FlareLightCurve_testhelper(flc)

def test_from_K2SC_source(iterator=iterator):

    for (ID, path, size) in iterator:
        for target in [ID, path]:
            flc = from_K2SC_source(target)
            FlareLightCurve_testhelper(flc)
            assert flc.detrended_flux_err.shape[0] == size
            assert flc.detrended_flux.shape[0] == size
            assert flc.detrended_flux.shape[0] == size
            assert flc.flares == None
            assert flc.gaps == None
        #also test if a local path throws warning
        #test if a list of IDs is correctly resolved - must return a list of FlareLightCurves
        #also test if a list of paths is correctly resolved - must return a list of FlareLightCurves

def test_from_K2SC_file():

    for path in paths:
        flc = from_K2SC_file(path)
        FlareLightCurve_testhelper(flc)

def test_from_KeplerLightCurve():
    #is currently implicitly tested by test_from_K2SC_source and test_from_TargetPixel_source
    pass
