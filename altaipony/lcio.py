import os
import inspect
import logging
import numpy as np

import k2sc.core as k2sc_flag_values

from astropy import units as u
from astropy.io.fits.hdu.hdulist import fitsopen

from lightkurve import KeplerLightCurveFile, KeplerTargetPixelFile, KeplerLightCurve

from .flarelc import FlareLightCurve
from .mast import download_kepler_products
from .utils import k2sc_quality_cuts

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
    k2sc_keys = {'primary_header' : tpf.hdu[0].header,
                 'data_header' : tpf.hdu[1].header,
                 'pos_corr1' : tpf.pos_corr1,
                 'pos_corr2' : tpf.pos_corr2,}

    lc = tpf.to_lightcurve()
    lc = from_KeplerLightCurve(lc, origin = 'TPF', **k2sc_keys)

    return lc


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
        takes in either raw or *PDC* flux, default is 'SAP_FLUX' because it seems
        to work best with the K2SC detrending pipeline
    kwargs : dict
        Keyword arguments to pass to `KeplerLightCurveFile.from_archive
        <https://lightkurve.keplerscience.org/api/lightkurve.lightcurvefile.KeplerLightCurveFile.html#lightkurve.lightcurvefile.KeplerLightCurveFile.from_archive>`_

    Returns
    --------
    FlareLightCurve
    """

    lcf = KeplerLightCurveFile.from_archive(target, quality_bitmask=None, **kwargs)
    lc = lcf.get_lightcurve(lctype)
    flc = from_KeplerLightCurve(lc, origin='KLC')
    return flc


def from_KeplerLightCurve(lc, origin='KLC', **kwargs):
    """
    Convert a ``KeplerLightCurve`` to a ``FlareLightCurve``. Just get all
    ``KeplerLightCurve`` attributes and pass them to the ``FlareLightCurve``.

    Parameters
    -------------
    lc: KeplerLightCurve

    origin : 'KLC' or str
        Indicates the origin of the FlareLightCurve, can take 'FLC, 'KLC', 'TPF'
        and 'K2SC'.
    kwargs: dict
        Keyword arguments to pass to FlareLightCurve.

    Returns
    -----------
    FlareLightCurve
    """
    flc = FlareLightCurve(**vars(lc), time_unit=u.day, origin=origin,
                           flux_unit = u.electron/u.s, **kwargs)
    flc[np.isfinite(flc.time)]

    return flc


def from_K2SC_file(path, campaign=None, lctype='SAP_FLUX', **kwargs):
    """
    Read in a K2SC de-trended light curve and convert it to a ``FlareLightCurve``.

    Parameters
    ------------
    path : str
        path to light curve
    campaign : None or int
        K2 observing campaign
    lctype : 'SAP_FLUX' or 'PDCSAP_FLUX'
        Takes in either raw or _PDC_ flux, default is 'SAP_FLUX' because it seems
        to work best with the K2SC detrending pipeline.
    kwargs : dict
        Keyword arguments to pass to `KeplerLightCurveFile.from_archive
        <https://lightkurve.keplerscience.org/api/lightkurve.lightcurvefile.KeplerLightCurveFile.html#lightkurve.lightcurvefile.KeplerLightCurveFile.from_archive>`_

    Returns
    --------
    FlareLightCurve

    """

    hdu = fitsopen(path)
    dr = hdu[1].data
    targetid = int(path.split('-')[0][-9:])
    klcf = KeplerLightCurveFile.from_archive(targetid, quality_bitmask='none',
                                             campaign=campaign, **kwargs)
    klc = klcf.get_lightcurve(lctype)
    #Only use those cadences that are present in both files:
    values, counts = np.unique(np.append(klc.cadenceno, dr.cadence), return_counts=True)
    cadences = values[ np.where( counts == 2 ) ] #you could check if counts can be 3 or more and throw an exception in that case
    #note that order of cadences is irrelevant for the following to be right
    dr = dr[ np.isin( dr.cadence, cadences) ]
    klc = klc[ np.isin( klc.cadenceno, cadences) ]

    flc = FlareLightCurve(time=dr.time, flux=klc.flux, detrended_flux=dr.flux,
                          detrended_flux_err=dr.error, cadenceno=dr.cadence,
                          flux_trends = dr.trtime, targetid=targetid,
                          campaign=klc.campaign, centroid_col=klc.centroid_col,
                          centroid_row=klc.centroid_row,time_format=klc.time_format,
                          time_scale=klc.time_scale, ra=klc.ra, dec=klc.dec,
                          channel=klc.channel, time_unit=u.day,
                          flux_unit = u.electron/u.s, origin='K2SC',
                          pos_corr1=dr.x, pos_corr2=dr.y)
    hdu.close()
    del dr

    flc = flc[(np.isfinite(flc.detrended_flux)) &
              (np.isfinite(flc.detrended_flux_err))]
    return flc


def from_K2SC_source(target, campaign=None):
    """
    Read in a K2SC de-trended light curve and convert it to a ``FlareLightCurve``.

    Parameters
    ------------
    target : str
        ID or path to K2SC light curve
    campaign : None or int
        K2 Campaign number
    Returns
    --------
    FlareLightCurve

    """

    if os.path.exists(str(target)) or str(target).startswith('http'):

        LOG.warning('Warning: from_archive() is not intended to accept a '
                    'direct path, use from_K2SC_File(path) instead.'
                    'Now using from_K2SC_File({})'.format(target))
        path = [target]
        campaign = [campaign]

    else:
        keys = {'filetype' : 'Lightcurve',
                'cadence' : 'long',
                'campaign' : None,
                'month' : None,
                'radius' : None,
                'targetlimit' : 1}
        path, campaign = download_kepler_products(target=target, **keys)
    if len(path) == 1:
        return from_K2SC_file(path[0], campaign=campaign[0])
    return [from_K2SC_file(p, campaign=c) for p,c in zip(path, campaign)]
