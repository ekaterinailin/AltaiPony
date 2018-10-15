import os
import pytest
from ..lcio import (from_TargetPixel_source, from_KeplerLightCurve_source,
                   from_K2SC_source, from_K2SC_file, from_KeplerLightCurve)
from ..__init__ import PACKAGEDIR

ids = ['211119999', '210951703', 211117077]
paths = [os.path.join(PACKAGEDIR, 'examples',
         'hlsp_k2sc_k2_llc_{}-c04_kepler_v2_lc.fits'.format(id_)) for id_ in ids]
campaign = 4
sizes = [3423, 3422, 3423]
ra = [56.90868, 57.10626, 55.967295]
dec = [24.891865, 22.211572, 24.841616]
channel = [52, 47, 52]
iterator = list(zip(ids, paths, sizes, ra, dec, channel))


def FlareLightCurve_testhelper(flc, ID, s, ra, dec, channel, from_tpf = False):
    """
    Test that reading in a FlareLightCurve does not kill or change any
    KeplerLightCurve attributes.

    Parameters
    -----------
    flc : FlareLightCurve
        lightcurve of target with given ID
    ID :

    size :

    ra :

    dec :

    channel :

    from_tpf : False or bool
        if light curve is created from a processed ``K2SC`` file one cadence is
        thrown out from the resulting ``FlareLightCurve``.
    """
    def greq(x, s, from_tpf):
        if from_tpf==True:
            assert x>=s
        else:
            assert x == s

    greq(flc.flux.shape[0], s, from_tpf)
    greq(flc.flux_err.shape[0], s, from_tpf)
    greq(flc.time.shape[0], s, from_tpf)
    greq(flc.centroid_col.shape[0], s, from_tpf)
    greq(flc.centroid_row.shape[0], s, from_tpf)
    greq(flc.remove_nans().flux.shape[0], s, from_tpf)
    greq(flc.correct().flux.shape[0], s, from_tpf)
    greq(flc.flatten().flux.shape[0], s, from_tpf)

    assert flc.campaign == campaign
    assert ((flc.quality_bitmask == 'none') or (flc.quality_bitmask == None))
    assert flc.time_format == 'bkjd'
    assert flc.time_scale == 'tdb'
    assert flc.quarter == None
    assert flc.ra == ra
    assert flc.dec == dec
    assert flc.targetid == int(ID)
    assert flc.channel == channel



def test_from_TargetPixel_source():
    for (ID, path, size, ra, dec, channel) in iterator:
        flc = from_TargetPixel_source(ID)
        FlareLightCurve_testhelper(flc, ID, size, ra, dec, channel, from_tpf=True)


def test_from_KeplerLightCurve_source():
    for (ID, path, size, ra, dec, channel) in iterator:
        flc = from_KeplerLightCurve_source(ID)
        FlareLightCurve_testhelper(flc, ID, size, ra, dec, channel)

def test_from_K2SC_source():

    for (ID, path, size, ra, dec, channel) in iterator:
        for target in [path, ID]:
            flc = from_K2SC_source(target)
            FlareLightCurve_testhelper(flc, ID, size, ra, dec, channel)
            assert flc.detrended_flux_err.shape[0] == size
            assert flc.detrended_flux.shape[0] == size
            assert flc.detrended_flux.shape[0] == size
            assert flc.flares == None
            assert flc.gaps == None
        #also test if a local path throws warning
        #test if a list of IDs is correctly resolved - must return a list of FlareLightCurves
        #also test if a list of paths is correctly resolved - must return a list of FlareLightCurves

def test_from_K2SC_file():

    for (ID, size, path, ra, dec, channel) in iterator:
        flc = from_K2SC_file(path)
        FlareLightCurve_testhelper(flc, ID, size, ra, dec, channel)

def test_from_KeplerLightCurve():
    #is currently implicitly tested by test_from_K2SC_source and test_from_TargetPixel_source
    pass
