import os
import inspect
import logging
import numpy as np

import k2sc.core as k2sc_flag_values

from astropy.io import fits
from lightkurve import KeplerLightCurveFile, KeplerTargetPixelFile, KeplerLightCurve

from .flarelc import FlareLightCurve
from .mast import download_kepler_products

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

    Parameters
    ------------
    target : str or int
        EPIC ID (e.g., 211119999) or path to zipped ``KeplerTargetPixelFile``
    kwargs : dict
        Keyword arguments to pass to `KeplerTargetPixelFile.from_archive()
        <https://lightkurve.keplerscience.org/api/lightkurve.targetpixelfile.KeplerTargetPixelFile.html#lightkurve.targetpixelfile.KeplerTargetPixelFile.from_archive>`_
    """
    tpf = KeplerTargetPixelFile.from_archive(target, quality_bitmask='none',
                                             **kwargs)
    lc = tpf.to_lightcurve()
    lc = k2sc_quality_cuts(lc)

    return from_KeplerLightCurve(lc)


def from_KeplerLightCurve_source(target, lctype='SAP_FLUX',**kwargs):
    """
    Accepts paths and EPIC IDs as targets. Either fetches a ``KeplerLightCurveFile``
    from MAST via ID or directly from a path, then creates a ``FlareLightCurve``
    preserving all data from ``KeplerLightCurve``.

    Parameters
    ------------
    target : str or int
        EPIC ID (e.g., 211119999) or path to zipped ``KeplerLightCurveFile``
    lctype: 'SAP_FLUX' or 'PDCSAP_FLUX'
        takes in either raw or PDC flux, default is 'SAP_FLUX' because it seems
        to work best with the K2SC detrending pipeline
    kwargs : dict
        Keyword arguments to pass to `KeplerLightCurveFile.from_archive
        <https://lightkurve.keplerscience.org/api/lightkurve.lightcurvefile.KeplerLightCurveFile.html#lightkurve.lightcurvefile.KeplerLightCurveFile.from_archive>`_

    Returns
    --------
    FlareLightCurve
    """

    lcf = KeplerLightCurveFile.from_archive(target, quality_bitmask='none',
                                            **kwargs)
    lc = lcf.get_lightcurve(lctype)
    lc = k2sc_quality_cuts(lc)

    return from_KeplerLightCurve(lc)


def from_KeplerLightCurve(lc):
    """
    Convert a ``KeplerLightCurve`` to a ``FlareLightCurve``. Just get all
    ``KeplerLightCurve`` attributes and pass them to the ``FlareLightCurve``.

    Parameters
    -------------
    lc: KeplerLightCurve
        light curve as used in lightkurve

    Returns
    -----------
    FlareLightCurve
    """
    #populate to reconcile KLC with FLC
    print(dir(lc))

    return FlareLightCurve(**vars(lc))


def from_K2SC_file(path, campaign=None, lctype='SAP_FLUX', **kwargs):
    """
    Read in a K2SC de-trended light curve and convert it to a ``FlareLightCurve``.

    Parameters
    ------------
    path: str
        path to light curve
    campaign: int or None
        K2 observing campaign
    kwargs: dict
        Keyword arguments to pass to `KeplerLightCurveFile.from_archive
        <https://lightkurve.keplerscience.org/api/lightkurve.lightcurvefile.KeplerLightCurveFile.html#lightkurve.lightcurvefile.KeplerLightCurveFile.from_archive>`_

    Returns
    --------
    FlareLightCurve

    """

    hdu = fits.open(path)
    dr = hdu[1].data


    targetid = int(path.split('-')[0][-9:])
    klcf = KeplerLightCurveFile.from_archive(targetid, quality_bitmask='none',
                                             campaign=campaign, **kwargs)
    klc = klcf.get_lightcurve(lctype)

    klc = k2sc_quality_cuts(klc)

    #Only use those cadences that are present in both files:
    values, counts = np.unique(np.append(klc.cadenceno, dr.cadence), return_counts=True)
    cadences = values[ np.where( counts == 2 ) ] #you could check if counts can be 3 or more and throw an exception in that case
    dr = dr[ np.isin( dr.cadence, cadences) ]
    klc = klc[ np.isin( klc.cadenceno, cadences) ]

    flc = FlareLightCurve(time=dr.time, flux=klc.flux, detrended_flux=dr.flux,
                          detrended_flux_err=dr.error, cadenceno=dr.cadence,
                          flux_trends = dr.trtime, targetid=targetid,
                          campaign=klc.campaign, centroid_col=klc.centroid_col,
                          centroid_row=klc.centroid_row,time_format=klc.time_format,
                          time_scale=klc.time_scale, ra=klc.ra, dec=klc.dec,
                          channel=klc.channel)
    hdu.close()
    del dr
    return flc


def from_K2SC_source(target, filetype='Lightcurve', cadence='long', quarter=None,
                     campaign=None, month=None, radius=None, targetlimit=1):
    """
    Read in a K2SC de-trended light curve and convert it to a ``FlareLightCurve``.

    Parameters
    ------------
    path : str
        path to light curve

    Returns
    --------
    FlareLightCurve

    """


    if os.path.exists(str(target)) or str(target).startswith('http'):
        LOG.warning('Warning: from_archive() is not intended to accept a '
                    'direct path, use from_K2SC_File(path) instead.')
        path = [target]
    else:
        path, campaign = download_kepler_products(target=target, filetype=filetype,
                                        cadence=cadence, campaign=campaign,
                                        month=month, radius=radius,
                                        targetlimit=targetlimit)
    if len(path) == 1:
        return from_K2SC_file(path[0], campaign=campaign[0])
    return [from_K2SC_file(p, campaign=c) for p,c in zip(path, campaign)]

def k2sc_quality_cuts(data):
    """
    Apply all the quality checks that k2sc uses internally.
    """
    data = data[np.isfinite(data.time)]
    data = data[np.isfinite(data.centroid_col)]
    data = data[np.isfinite(data.centroid_row)]

    return data
