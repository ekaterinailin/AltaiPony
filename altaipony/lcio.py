import warnings
import logging

import numpy as np
import astropy.units as u

from astropy.io import fits

from altaipony.flarelc import FlareLightCurve

from lightkurve import (search_lightcurvefile,
                        search_targetpixelfile,
                        KeplerLightCurveFile,
                        TessLightCurveFile,
                        KeplerTargetPixelFile,
                        TessTargetPixelFile)
                        
LOG = logging.getLogger(__name__)

# ----------------------------------------------------------
# Read in data from MAST, either as LC (TESS, Kepler, K2) or TPF (K2)
# No reading in of Kepler or TESS TPFs as de-trending is not implemented for them.

def from_mast(targetid, mission, c, mode="LC", **kwargs):
    """Download light curve derived from
    TPF or LC directly from MAST using the
    great search functionality in lightkurve,
    and construct a FlareLightCurve.
    
    Parameters:
    ------------
    targetid : str or int
        TIC, EPIC or KIC ID
    mission : str
        "Kepler", "K2", or TESS"
    c : int
        quarter, campaign, or sector
    mode : str
        "TPF" or "LC", default is "LC"
    kwargs : dict
        Keyword arguments to pass to _from_mast_<mission>()
        functions, like cadence ("short" or "long").
        
    Return:
    --------
    FlareLightCurve
    """
    
    if mission=="K2":
        if mode == "LC":
            warnings.warn("\nYou cannot do K2SC de-trending on a light curve only." 
                 "Pass mode='TPF' to be able to run FLC.detrend('k2sc') later.")
        return _from_mast_K2(targetid, mode, c, **kwargs)
    
    elif mission == "Kepler":
        return _from_mast_Kepler(targetid, c, **kwargs)
    
    elif mission == "TESS":
        return _from_mast_TESS(targetid, c, **kwargs)
    
    return


def _from_mast_K2(targetid, mode, c, flux_type="PDCSAP_FLUX",
                  cadence="long", aperture_mask="default",
                  download_dir=None):
    
    if mode == "TPF":
        
        tpffilelist = search_targetpixelfile(targetid, mission="K2",
                                             campaign=c, cadence=cadence)
        tpf = tpffilelist.download(download_dir=download_dir)
        
        if aperture_mask == "default":
            aperture_mask = tpf.pipeline_mask
            
        lc = tpf.to_lightcurve(aperture_mask=aperture_mask)
        
        flc = _convert_TPF_to_FLC(tpf, lc)
        
        return flc
    
    elif mode == "LC":
        
        flcfilelist = search_lightcurvefile(targetid, mission="K2",
                                            campaign=c, cadence=cadence)
        flcfile = flcfilelist.download(download_dir=download_dir)
        lc = flcfile.get_lightcurve(flux_type)
        flc = _convert_LC_to_FLC(lc, origin="KLC")
        return flc


def _from_mast_Kepler(targetid, c, flux_type="PDCSAP_FLUX", cadence="long",
                      download_dir=None):
    flcfilelist = search_lightcurvefile(targetid, mission="Kepler",
                                        quarter=c, cadence=cadence)
    if len(flcfilelist)==1:
        flcfile = flcfilelist.download(download_dir=download_dir)
        lc = flcfile.get_lightcurve(flux_type)

        flc = _convert_LC_to_FLC(lc, origin="KLC")
        return flc

    elif len(flcfilelist)>1:
        warnings.warn(f"Multiple Kepler light curves for {targetid}"
                      f" in quarter {c}. Downloading all to list.")
        lclist = []
        flcfiles = flcfilelist.download_all(download_dir=download_dir)
        for flcfile in flcfiles:
            lc = flcfile.get_lightcurve(flux_type)
            flc = _convert_LC_to_FLC(lc, origin="KLC")
            lclist.append(flc)
        return lclist


def _from_mast_TESS(targetid, c, flux_type="PDCSAP_FLUX", cadence="long",
                    download_dir=None):
    flcfilelist = search_lightcurvefile(targetid, mission="TESS",
                                        sector=c, cadence=cadence)
    flcfile = flcfilelist.download(download_dir=download_dir)
    print(flcfile)
    lc = flcfile.get_lightcurve(flux_type)
    flc = _convert_LC_to_FLC(lc, origin="TLC", sector=c)    
    return flc


# ----------------------------------------------------------


# ----------------------------------------------------------
# Read in local TPF (Kepler, K2, TESS) or LC (Kepler, K2, TESS, AltaiPony)

def from_path(path, mode, mission, **kwargs):
    """Construct a FlareLightCurve from
    a local LC or TPF file. Also loads
    AltaiPony-detrended light curves.
    
    Parameters:
    -------------
    path : str
        Path to local file.
    mode : str
        "LC", "TPF", "AltaiPony"
    mission : str
        "Kepler", "K2", "TESS"
    kwargs : dict
        Keyword arguments to pass to _from_path_<XXX>
        functions 
    """
    if mode == "LC":
        return _from_path_LC(path, mission, **kwargs)
    elif mode == "TPF":
        return _from_path_TPF(path, mission, **kwargs)
    elif mode == "AltaiPony":
        return _from_path_AltaiPony(path, **kwargs)
    else:
        raise KeyError("Invalid mode. Pass 'LC', 'TPF', or 'AltaiPony'.")
    return

def _from_path_LC(path, mission, flux_type="PDCSAP_FLUX"):
    
    origins = {"Kepler":"KLC", "K2":"KLC","TESS":"TLC"}
    
    if ((mission == "Kepler") | (mission == "K2")):
        lcf = KeplerLightCurveFile(path)
    elif mission == "TESS":
        lcf = TessLightCurveFile(path)
        
    else:
        raise KeyError("Invalid mission. Pass 'Kepler', 'K2', or 'TESS'.")
        
    lc = lcf.get_lightcurve(flux_type)
    flc = _convert_LC_to_FLC(lc, origin=origins[mission])
    return flc


def _from_path_TPF(path, mission, aperture_mask="default"):
    
    origins = {"Kepler":"KLC", "K2":"KLC","TESS":"TLC"}
    
    if ((mission == "Kepler") | (mission == "K2")):
        tpf = KeplerTargetPixelFile(path)
        
    elif mission == "TESS":
        tpf = TessTargetPixelFile(path)
        
    if aperture_mask == "default":
            aperture_mask = tpf.pipeline_mask

    lc = tpf.to_lightcurve(aperture_mask=aperture_mask)
    
    flc = _convert_TPF_to_FLC(tpf, lc)
    
    return flc
    
    
def _from_path_AltaiPony(path):
    
    rhdul =  fits.open(path)
    attrs = dict()
    for k, v in rhdul[0].header.items():
        if str.lower(k) not in ['simple', 'bitpix', 'naxis', 'extend']:
            if str.lower(k) == "keplerid": #rename keplerid if it appears
                k = "targetid"
            attrs[str.lower(k)] = v
    
    for k in ['time', 'flux', 'flux_err', 'centroid_col',
              'centroid_row', 'quality', 'cadenceno',
              'detrended_flux', 'detrended_flux_err',
              'quality_bitmask', 'saturation']:
        try:
            attrs[k] = rhdul[1].data[k].byteswap().newbyteorder()
        except KeyError:
            LOG.info("Warning: Keyword {} not in file.".format(k))
            continue   
            
    return FlareLightCurve(**attrs)

# ----------------------------------------------------------

# ----------------------------------------------------------
# Internal type conversion functions

def _convert_TPF_to_FLC(tpf, lc):
    keys = {'primary_header' : tpf.hdu[0].header,
            'data_header' : tpf.hdu[1].header,
            'pos_corr1' : tpf.pos_corr1,
            'pos_corr2' : tpf.pos_corr2,
            'pixel_flux' : tpf.flux,
            'pixel_flux_err' : tpf.flux_err,
            'pipeline_mask' : tpf.pipeline_mask}

    attributes = lc.__dict__
    z = attributes.copy()
    z.update(keys)
    if "_flux_unit" in z.keys():
        del z["_flux_unit"]
    if '_required_columns_relax' in z.keys():
        del z['_required_columns_relax']

    flc = FlareLightCurve(time_unit=u.day, origin="TPF",
                          flux_unit = u.electron/u.s, **z)
    if flc.pos_corr1 is None:
        flc.pos_corr1 = flc.centroid_col
    if flc.pos_corr2 is None:
        flc.pos_corr2 = flc.centroid_row
    flc = flc[np.isfinite(flc.time) &
              np.isfinite(flc.flux) &
              np.isfinite(flc.pos_corr1) &
              np.isfinite(flc.pos_corr2) &
              np.isfinite(flc.cadenceno) ]
    return flc


def _convert_LC_to_FLC(lc, origin=None, **kwargs):
    attributes = lc.__dict__
    attributes.update(kwargs)
    
    if "_flux_unit" in attributes.keys():
        del attributes["_flux_unit"]
    flc = FlareLightCurve(time_unit=u.day, origin=origin,
                          flux_unit = u.electron/u.s,
                           **attributes)
    flc = flc[np.isfinite(flc.time) &
              np.isfinite(flc.flux) &
              np.isfinite(flc.cadenceno)]
    return flc

# ----------------------------------------------------------

