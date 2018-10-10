import os
import inspect
import logging

from lightkurve import KeplerLightCurveFile, KeplerTargetPixelFile, KeplerLightCurve
from .flarelc import FlareLightCurve
from .mast import download_kepler_products
from astropy.io import fits

LOG = logging.getLogger(__name__)

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
    kwargs : dict
        Keyword arguments to pass to ``KeplerTargetPixelFile.from_archive``
        <https://lightkurve.keplerscience.org/api/lightkurve.targetpixelfile.
        KeplerTargetPixelFile.html#lightkurve.targetpixelfile.
        KeplerTargetPixelFile.from_archive>
    """
    tpf = KeplerTargetPixelFile.from_archive(target, **kwargs)
    lc = tpf.to_lightcurve()
    return from_KeplerLightCurve(lc)


def from_KeplerLightCurve_source(target, lctype='SAP_FLUX',**kwargs):
    """
    Accepts paths and EPIC IDs as targets. Either fetches a ``KeplerLightCurveFile``
    from MAST via ID or directly from a path, then creates a ``FlareLightCurve``
    preserving all data from ``KeplerLightCurve``.

    Parameters:
    ------------
    target : str or int
        EPIC ID (e.g., 211119999) or path to zipped ``KeplerLightCurveFile``
    lctype: 'SAP_FLUX' or 'PDCSAP_FLUX'
        takes in either raw or PDC flux, default is 'SAP_FLUX' because it seems
        to work best with the K2SC detrending pipeline
    kwargs : dict
        Keyword arguments to pass to ``KeplerLightCurveFile.from_archive``_
        .. _``KeplerLightCurveFile.from_archive``: https://lightkurve.keplerscience.org/
        api/lightkurve.lightcurvefile.KeplerLightCurveFile.html#lightkurve.
        lightcurvefile.KeplerLightCurveFile.from_archive

    Return:
    --------
    ``FlareLightCurve``
    """

    lcf = KeplerLightCurveFile.from_archive(target, **kwargs)
    lc = lcf.get_lightcurve(lctype)

    return from_KeplerLightCurve(lc)


def from_KeplerLightCurve(lc):
    #populate to reconcile KLC with FLC
    print(dir(lc))
    #get all KeplerLightCurve attributes and pass them to the FLC
    kwnames = inspect.getargspec(KeplerLightCurve)[0][1:]
    data = [getattr(lc, names) for names in kwnames]
    kwargs = dict(zip(kwnames, data))
    return FlareLightCurve(**kwargs)


def from_K2SC_file(path):

    hdu = fits.open(path)
    dr = hdu[1].data
    print(dr.names)
    targetid = path.split('-')[0][-9:]
    flc = FlareLightCurve(time=dr.time, detrended_flux=dr.flux, flux_err=dr.error,
                          cadenceno=dr.cadence, flux_trends = dr.trtime,
                          targetid=targetid)
    hdu.close()
    del dr
    return flc


def from_K2SC_source(target, filetype='Lightcurve', cadence='long', quarter=None,
                     campaign=None, month=None, radius=None, targetlimit=1):


    if os.path.exists(str(target)) or str(target).startswith('http'):
        LOG.warning('Warning: from_archive() is not intended to accept a '
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
