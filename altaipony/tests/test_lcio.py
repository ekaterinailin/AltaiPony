import pytest

import numpy as np

from ..lcio import from_path, from_mast, _from_path_AltaiPony
from . import pathkepler, pathtess



@pytest.mark.parametrize("path,mode,ID,mission,campaign,quarter,sector",
                         [(pathkepler,"LC", 10002792, "kepler", None, 2, None ),
                          (pathtess,"LC", 358108509, "tess", None, None, 1)
                          ])

def test_from_path(path, mode, ID, mission, campaign, quarter, sector):
    
    flc = from_path(path, mode=mode, mission=mission)
    assert flc.targetid == ID
    assert flc.mission == mission
    assert flc.flux.value.shape[0] == flc.time.value.shape[0]
    if campaign is not None:
        assert flc.campaign == campaign
    if quarter is not None:
        assert flc.quarter == quarter
    if sector is not None:    
        assert flc.sector == sector    
    assert (np.isnan(flc.flux.value) == False).all()
    assert (np.isnan(flc.time.value) == False).all()
    
    assert np.isnan(flc.detrended_flux.value).all()
    assert (~np.isnan(flc.flux_err.value)).all()
    assert flc.flares.shape[0] == 0
    flct = flc[10:20]
    assert flct.flux.value.shape[0] == flct.time.value.shape[0]
    assert flct.flux.value.shape[0] == flct.flux_err.value.shape[0]
    assert flct.flux.value.shape[0] == flct.detrended_flux_err.value.shape[0]
    assert flct.flux.value.shape[0] == 10
    
    
def test__from_path_AltaiPony():
    flc = from_path("altaipony/examples/kplr010002792-2010174085026_llc.fits","LC", "kepler")
    path = "altaipony/examples/pony010002792-2010174085026_llc_test_from_path_AltaiPony.fits"
    #flc = from_path(pathkepler, mode="LC", mission="Kepler")
    flc.to_fits(path, overwrite=True)
    rflc = _from_path_AltaiPony(path)
    assert rflc.channel == flc.channel
    assert rflc.quarter == flc.quarter
    assert rflc.ra == flc.ra
    assert rflc.dec == flc.dec
    assert rflc.mission == "kepler"
    kws = ['time', 'flux', 'flux_err', 'centroid_col',
           'centroid_row', 'quality', 'cadenceno',
           'detrended_flux', 'detrended_flux_err',]
    assert (len(rflc.flux.value) == np.array([len(getattr(rflc,x)) for x in kws])).all()
    assert rflc.targetid == 10002792
    assert rflc.origin == "FLC"
    assert np.isnan(rflc.detrended_flux.value).all()
    assert np.isnan(rflc.detrended_flux_err.value).all()
    
    

@pytest.mark.parametrize("ID,mission,c,mode,cadence,sector,campaign,quarter,lflc",
                         [("TIC 395130640","tess", 11,"LC", "short", 11, None, None,1 ),
                          ("KIC 9726699", "kepler", 6, "LC", "long", None, None, 6,1),
                          ("KIC 100004076", "kepler", 14, "LC", "short", None, None, 14,3),
                          ("TIC 395130640","tess", None,"LC", "short", 11, None, None,2),
                          ("KIC 9726699", "kepler", None, "LC", "long", None, None, 0,15),
                          ("KIC 100004076", "kepler", None, "LC", "short", None, None, 14,3)
                          ])
def test_from_mast(ID, mission, c, mode, cadence, sector, campaign, quarter, lflc):
    flc = from_mast(ID, mission, c, mode=mode, cadence=cadence)
    # Only for the KIC 100004706 target:
    if isinstance(flc, list):
        assert len(flc) >= lflc 
        flc = flc[0]
    # -----------------------------------    
    assert flc.targetid == int(ID.split(" ")[1])
    assert flc.mission == mission
 
    assert flc.flux.value.shape[0] == flc.time.value.shape[0]
    if campaign is not None:
        assert flc.campaign == campaign
    if quarter is not None:
        assert flc.quarter == quarter
    if sector is not None:    
        assert flc.sector == sector
    assert (np.isnan(flc.flux.value) == False).all()
    assert (np.isnan(flc.time.value) == False).all()
    print(flc.columns)
    assert np.isnan(flc.detrended_flux.value).all()
    assert (~np.isnan(flc.flux_err.value)).all()
    assert flc.flares.shape[0] == 0
    flct = flc[10:20]
    assert flct.flux.value.shape[0] == flct.time.value.shape[0]
    assert flct.flux.value.shape[0] == flct.flux_err.value.shape[0]
    assert flct.flux.value.shape[0] == flct.detrended_flux_err.value.shape[0]
    assert flct.flux.value.shape[0] == 10
    
