import pytest

import numpy as np

from ..lcio import from_path, from_mast, _from_path_AltaiPony
from . import pathkepler, pathk2LC, pathk2TPF, pathtess, pathAltaiPony



@pytest.mark.parametrize("path,mode,ID,mission,campaign,quarter,sector",
                         [(pathkepler,"LC", 10002792, "Kepler", None, 2, None ),
                          (pathk2LC,"LC", 211117077, "K2", 4, None, None ),
                          (pathk2TPF,"TPF", 210994964, "K2", 4, None, None ),
                          (pathtess,"LC", 358108509, "TESS", None, None, 1)
                          ])

def test_from_path(path, mode, ID, mission, campaign, quarter, sector):
    
    flc = from_path(path, mode=mode, mission=mission)
    assert flc.targetid == ID
    assert flc.mission == mission
    assert flc.flux.shape[0] == flc.time.shape[0]
    assert flc.campaign == campaign
    assert flc.quarter == quarter
    assert flc.sector == sector
    assert (np.isnan(flc.flux) == False).all()
    assert (np.isnan(flc.time) == False).all()
    assert np.isnan(flc.detrended_flux).all()
    assert (~np.isnan(flc.flux_err)).all()
    assert flc.flares.shape[0] == 0
    flct = flc[10:20]
    assert flct.flux.shape[0] == flct.time.shape[0]
    assert flct.flux.shape[0] == flct.flux_err.shape[0]
    assert flct.flux.shape[0] == flct.detrended_flux_err.shape[0]
    assert flct.flux.shape[0] == 10
    
    
def test__from_path_AltaiPony():
    path = "altaipony/examples/pony010002792-2009259160929_llc_test_from_path_AltaiPony.fits"
    flc = from_path(pathkepler, mode="LC", mission="Kepler")
    flc.to_fits(path)
    rflc = _from_path_AltaiPony(path)
    assert rflc.channel == flc.channel
    assert rflc.quarter == flc.quarter
    assert rflc.ra == flc.ra
    assert rflc.dec == flc.dec
    assert rflc.time_scale == flc.time_scale
    assert rflc.time_format == flc.time_format
    assert rflc.mission == "Kepler"
    kws = ['time', 'flux', 'flux_err', 'centroid_col',
           'centroid_row', 'quality', 'cadenceno',
           'detrended_flux', 'detrended_flux_err',]
    assert (len(rflc.flux) == np.array([len(getattr(rflc,x)) for x in kws])).all()
    assert rflc.targetid == 10002792
    assert rflc.origin == "KLC"
    assert np.isnan(rflc.detrended_flux).all()
    assert np.isnan(rflc.detrended_flux_err).all()
    
    

@pytest.mark.parametrize("ID,mission,c,mode,cadence,sector,campaign,quarter",
                         [(395130640,"TESS", 11,"LC", "short", 11, None, None ),
                          (211119999, "K2", 4, "LC", "long", None, 4, None),
                          (211119999, "K2", 4, "TPF", "long", None, 4, None),
                          (9726699, "Kepler", 6, "LC", "long", None, None, 6)
                          ])
def test_from_mast(ID, mission, c, mode, cadence, sector, campaign, quarter):
    flc = from_mast(ID, mission, c, mode=mode, cadence=cadence)
    assert flc.targetid == ID
    assert flc.mission == mission
    assert flc.flux.shape[0] == flc.time.shape[0]
    assert flc.campaign == campaign
    assert flc.quarter == quarter
    assert flc.sector == sector
    assert (np.isnan(flc.flux) == False).all()
    assert (np.isnan(flc.time) == False).all()
    assert np.isnan(flc.detrended_flux).all()
    assert (~np.isnan(flc.flux_err)).all()
    assert flc.flares.shape[0] == 0
    flct = flc[10:20]
    assert flct.flux.shape[0] == flct.time.shape[0]
    assert flct.flux.shape[0] == flct.flux_err.shape[0]
    assert flct.flux.shape[0] == flct.detrended_flux_err.shape[0]
    assert flct.flux.shape[0] == 10
    
