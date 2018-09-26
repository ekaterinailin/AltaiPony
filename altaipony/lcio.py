from lightkurve import KeplerLightCurveFile, KeplerTargetPixelFile
from mast import download_kepler_products
from astropy.io import fits
import os

from flarelc import FlareLightCurve

# Naming convention:
# from_* : IO method for some data type (TPF, KLC, K2SC)
# *_source : accept both EPIC IDs and paths
# *_file : accept only local paths
# *_archive : accept only EPIC IDs (not used yet)


def from_TargetPixel_source(target, **kwargs):
    tpf = KeplerTargetPixelFile.from_archive(target, **kwargs)
    lc = tpf.to_lightcurve()
    return from_KeplerLightCurve(lc)


def from_KeplerLightCurve_source(target):

    lcf = KeplerLightCurveFile.from_archive(target)
    lc = lcf.get_lightcurve('SAP_FLUX')
    return from_KeplerLightCurve(lc)


def from_KeplerLightCurve(lc):
    #populate to reconcile KLC with FLC
    print(dir(lc))
    return FlareLightCurve(time=lc.time, flux=lc.flux, flux_err=lc.flux_err)


def from_K2SC_file(path):

    hdu = fits.open(path)
    data_rec = hdu[1].data
    print(data_rec.names)
    flc = FlareLightCurve(time=data_rec.time, flux=data_rec.flux)
    hdu.close()
    del data_rec
    del hdu
    return flc


def from_K2SC_source(target, filetype='Lightcurve', cadence='long', quarter=None,
              campaign=None, month=None, radius=None,
              targetlimit=1):


    if os.path.exists(str(target)) or str(target).startswith('http'):
        log.warning('Warning: from_archive() is not intended to accept a '
                    'direct path, use from_K2SC_File(path) instead.')
        path = [target]
    else:
        path = download_kepler_products(target=target, filetype=filetype,
                                        cadence=cadence, campaign=campaign,
                                        month=month, radius=radius,
                                        targetlimit=targetlimit)
    if len(path) == 1:

        return from_K2SC_file(path[0])
    return [from_K2SC_file(p) for p in path]
