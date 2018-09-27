from lightkurve import KeplerLightCurveFile, KeplerTargetPixelFile
from .flarelc import FlareLightCurve
from .mast import download_kepler_products
from astropy.io import fits
import os



# Naming convention:
# from_* : IO method for some data type (TPF, KLC, K2SC)
# *_source : accept both EPIC IDs and paths
# *_file : accept only local paths
# *_archive : accept only EPIC IDs (not used yet)


def from_TargetPixel_source(target, **kwargs):
    """
    Accepts paths and EPIC IDs as targets. Either fetches a ``KeplerTargetPixelFile``
    from MAST via ID or directly from a path, then creates a lightcurve with
    default Kepler/K2 pixel mask.

    Parameters:
    ------------
    target : str or int
        EPIC ID (e.g., 211119999) or path to zipped ``KeplerTargetPixelFile``
    **kwargs : dict
        Keyword arguments to pass to ``KeplerTargetPixelFile.from_archive``
        <https://lightkurve.keplerscience.org/api/lightkurve.targetpixelfile.
        KeplerTargetPixelFile.html#lightkurve.targetpixelfile.
        KeplerTargetPixelFile.from_archive>
    """
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
    dr = hdu[1].data
    print(dr.names)
    flc = FlareLightCurve(time=dr.time, flux=dr.flux, cadenceno=dr.cadence)
    hdu.close()
    del dr
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
