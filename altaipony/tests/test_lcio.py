import os
import pytest
import pandas as pd
from ..lcio import (from_TargetPixel_source, from_KeplerLightCurve_source,
                   from_K2SC_source, from_K2SC_file, from_KeplerLightCurve)
from .__init__ import test_ids, test_paths


campaign = 4
ra = [56.90868, 57.10626, 55.967295]
dec = [24.891865, 22.211572, 24.841616]
channel = [52, 47, 52]
iterator = list(zip(test_ids, test_paths, ra, dec, channel))


def FlareLightCurve_testhelper(flc, ID, ra, dec, channel, from_tpf = False):
    """
    Test that reading in a FlareLightCurve does not kill or change any
    KeplerLightCurve attributes.

    Parameters
    -----------
    flc : FlareLightCurve
        lightcurve of target with given ID
    ID :
        EPIC ID
    ra : float
        RA
    dec : float
        declination
    channel : int
        channel on the CCD detector
    from_tpf : False or bool
        if light curve is created from a processed ``K2SC`` file one cadence is
        thrown out from the resulting ``FlareLightCurve``.
    """
    assert flc.time.shape == flc.flux.shape
    assert flc.time.shape == flc.flux_err.shape

    assert flc.campaign == campaign
    #Wait until Nick's bugfix #325 is an official version feature
    #assert flc.quality_bitmask == 'default'
    assert flc.time_format == 'bkjd'
    assert flc.time_scale == 'tdb'
    assert flc.quarter == None
    assert flc.ra == ra
    assert flc.dec == dec
    assert flc.targetid == int(ID)
    assert flc.channel == channel


def test_from_TargetPixel_source():
    '''
    Test if a ``FlareLightCurve`` is created from a ``TargetPixelFile`` properly
    when calling an EPIC ID.
    '''
    #Can we load a path, too? ->later
    for (ID, path, ra, dec, channel) in iterator:
        flc = from_TargetPixel_source(ID)
        FlareLightCurve_testhelper(flc, ID, ra, dec, channel, from_tpf=True)

def test_from_KeplerLightCurve_source():
    '''
    Test if a ``FlareLightCurve`` is created from a ``KeplerLightCurve`` properly
    when calling an EPIC ID.
    '''
    #Can we load a path, too? -> later
    for (ID, path, ra, dec, channel) in iterator:
        flc = from_KeplerLightCurve_source(ID)
        FlareLightCurve_testhelper(flc, ID, ra, dec, channel)

def test_from_K2SC_source():
    '''
    Test if a ``FlareLightCurve`` is created from a ``K2SC`` file properly
    when calling an EPIC ID or local path.
    '''
    for (ID, path, ra, dec, channel) in iterator:
        for target in [path, ID]:
            flc = from_K2SC_source(target)
            FlareLightCurve_testhelper(flc, ID, ra, dec, channel)
            assert flc.detrended_flux_err.shape[0] == flc.detrended_flux.shape[0]
            assert flc.flares.empty
            assert flc.gaps == None
        #also test if a local path throws warning
        #test if a list of IDs is correctly resolved - must return a list of FlareLightCurves
        #also test if a list of paths is correctly resolved - must return a list of FlareLightCurves

def test_from_K2SC_file():
    '''
    Test if a ``FlareLightCurve`` is created from a ``K2SC`` file properly
    when calling a local path.
    '''
    for (ID, path, ra, dec, channel) in iterator:
        flc = from_K2SC_file(path)
        FlareLightCurve_testhelper(flc, ID, ra, dec, channel)
    for (ID, path, ra, dec, channel) in iterator:
        flc = from_K2SC_file(path, add_TPF=False)
        assert flc.time.shape == flc.flux.shape
        assert flc.time.shape == flc.flux_err.shape
        assert flc.time_format == None
        assert flc.time_scale == None
        assert flc.quarter == None
        assert flc.ra == None
        assert flc.dec == None
        assert flc.targetid == int(ID)
        assert flc.channel == None


def test_from_KeplerLightCurve():
    #is implicitly tested by test_from_K2SC_source and test_from_TargetPixel_source
    pass
