import os
import inspect
import logging
import time
import numpy as np

import k2sc.core as k2sc_flag_values

from astropy import units as u
from astropy.io.fits.hdu.hdulist import fitsopen

from lightkurve import KeplerLightCurveFile
from lightkurve import search_targetpixelfile, search_lightcurvefile

from .flarelc import FlareLightCurve
from .mast import download_kepler_products
from .utils import k2sc_quality_cuts

LOG = logging.getLogger(__name__)

# Naming convention:
# from_* : IO method for some data type (TPF, KLC, K2SC)
# *_source : accept both EPIC IDs and paths
# *_file : accept only local paths
# *_archive : accept only EPIC IDs (not used yet)


def from_TargetPixel_source(target, download_dir=None, **kwargs):
    """
    Accepts paths and EPIC IDs as targets. Either fetches a ``KeplerTargetPixelFile``
    from MAST via ID or directly from a path, then creates a lightcurve with
    default Kepler/K2 pixel mask.

    Parameters
    ------------
    target : str or int
        EPIC ID (e.g., 211119999) or path to zipped ``KeplerTargetPixelFile``
    download_dir : str 
        directory to store the file in
    kwargs : dict
        Keyword arguments to pass to `lightkurve.search_targetpixelfile()
        <http://docs.lightkurve.org/api/lightkurve.search.search_targetpixelfile.html#lightkurve.search.search_targetpixelfile>`_
    """
    tpf_list = search_targetpixelfile(target, **kwargs)
    
    if len(tpf_list) > 1:
        LOG.error('Target data identifier must be unique. Provide campaign or cadence.')
        return
    else:
        tpf = tpf_list.download(download_dir=download_dir)
        keys = {'primary_header' : tpf.hdu[0].header,
                'data_header' : tpf.hdu[1].header,
                'pos_corr1' : tpf.pos_corr1,
                'pos_corr2' : tpf.pos_corr2,
                'pixel_flux' : tpf.flux,
                'pixel_flux_err' : tpf.flux_err,}

        lc = tpf.to_lightcurve()
        lc = from_KeplerLightCurve(lc, origin = 'TPF', **keys)
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
        Keyword arguments to pass to `lightkurve.search_lightcurvefile()
        <http://docs.lightkurve.org/api/lightkurve.search.search_lightcurvefile.html>`_

    Returns
    --------
    FlareLightCurve
    """

    lcf_list = search_lightcurvefile(target, **kwargs)
    if len(lcf_list) > 1:
        LOG.error('Target data identifier must be unique. Provide campaign or cadence.')
        return
    else:
        lcf = lcf_list.download()
        lc = lcf.get_lightcurve(lctype)
        flc = from_KeplerLightCurve(lc, origin='KLC')
        LOG.warning('Using from_KeplerLightCurve_source limits AltaiPony\'s functionality'
                    ' to lightkurve\'s K2SFF de-trending, and flare finding. better '
                    'use from_TargetPixel_source or from_K2SC_source.')
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
    attributes = lc.__dict__
    z = attributes.copy()
    z.update(kwargs)
    flc = FlareLightCurve(time_unit=u.day, origin=origin,
                           flux_unit = u.electron/u.s, **z)
    if flc.pos_corr1 is None:
        flc.pos_corr1 = flc.centroid_col
    if flc.pos_corr2 is None:
        flc.pos_corr2 = flc.centroid_row
    flc = flc[np.isfinite(flc.time) &
              np.isfinite(flc.flux) &
              np.isfinite(flc.pos_corr1) &
              np.isfinite(flc.pos_corr2) &
              np.isfinite(flc.cadenceno)]

    return flc


def from_K2SC_file(path, add_TPF=True, **kwargs):
    """
    Read in a K2SC de-trended light curve and convert it to a ``FlareLightCurve``.

    Parameters
    ------------
    path : str
        path to light curve
    add_TPF : True or bool
        Default fetches TPF for additional info. Required for
        K2SC de-trending.
    kwargs : dict
        Keyword arguments to pass to :func:lightkurve.search_targetpixelfile

    Returns
    --------
    FlareLightCurve

    """

    hdu = fitsopen(path)
    dr = hdu[1].data

    targetid = int(path.split('-')[0][-9:])

    if add_TPF == True:

        tpf_list = search_targetpixelfile(targetid, **kwargs)
        #raw flux,campaign
        if len(tpf_list) > 1:
            LOG.error('Target data identifier must be unique. Provide campaign or cadence.')
            return
        else:
            ktpf = tpf_list.download()
            klc = ktpf.to_lightcurve()
            #Only use those cadences that are present in all TPF, KLC, and K2SC LC:
            values, counts = np.unique(np.concatenate((klc.cadenceno, dr['CADENCE'], ktpf.cadenceno)),
                                       return_counts=True)
            cadences = values[ np.where( counts == 3 ) ] #you could check if counts can be 4 or more and throw an exception in that case
            #note that order of cadences is irrelevant for the following to be right
            dr = dr[ np.isin( dr['CADENCE '], cadences) ]
            klc = klc[ np.isin( klc.cadenceno, cadences) ]
            ktpf = ktpf[ np.isin( ktpf.cadenceno, cadences)]

            flc = FlareLightCurve(time=dr['TIME'], flux=klc.flux, detrended_flux=dr['FLUX'],
                                  detrended_flux_err=dr['ERROR'], cadenceno=dr['CADENCE'],
                                  flux_trends = dr['TRTIME'], targetid=targetid,
                                  campaign=klc.campaign, centroid_col=klc.centroid_col,
                                  centroid_row=klc.centroid_row,time_format=klc.time_format,
                                  time_scale=klc.time_scale, ra=klc.ra, dec=klc.dec,
                                  channel=klc.channel, time_unit=u.day,
                                  flux_unit = u.electron/u.s, origin='K2SC',
                                  pos_corr1=dr['X'], pos_corr2=dr['Y'], quality=klc.quality,
                                  pixel_flux=ktpf.flux, pixel_flux_err=ktpf.flux_err,
                                  quality_bitmask=ktpf.quality_bitmask,
                                  pipeline_mask=ktpf.pipeline_mask )
    elif add_TPF == False:
        flc = FlareLightCurve(time=dr['TIME'], flux=None, detrended_flux=dr['FLUX'],
                              detrended_flux_err=dr['ERROR'], cadenceno=dr['CADENCE'],
                              flux_trends=dr['TRTIME'], targetid=targetid,
                              campaign=None, centroid_col=dr["X"],
                              centroid_row=dr['Y'], time_format=None,
                              time_scale=None, ra=hdu[1].header['RA_OBJ'], dec=hdu[1].header['DEC_OBJ'],
                              channel=None, time_unit=u.day,
                              flux_unit=u.electron / u.s, origin='K2SC',
                              pos_corr1=dr['X'], pos_corr2=dr['Y'], quality=dr.quality,
                              pixel_flux=None, pixel_flux_err=None,
                              quality_bitmask=None,
                              pipeline_mask=None)

    hdu.close()
    del dr

    flc = flc[(np.isfinite(flc.detrended_flux)) &
              (np.isfinite(flc.detrended_flux_err))]
    return flc


def from_KeplerLightCurve_file(path, flux_type, **kwargs):
    '''Read in a Kepler light curve file as
    FlareLightCurve.

    Parameters:
    -----------
    path : str
        path to file
    flux_type: str
        PDCSAP_FLUX or SAP_FLUX
    kwargs : dict
        keyword arguments to pass to func:from_KeplerLightCurve
    Return:
    -------
    FlareLightCurve
    '''
    klcf = KeplerLightCurveFile(path)
    klc = klcf.get_lightcurve(flux_type)
    return from_KeplerLightCurve(klc, **kwargs)


# def from_fits_file(path, flux_type='PDCSAP_FLUX', **kwargs):
#     '''Loads a FlareLightCurve from some fits file.
#     If targetid is not given in the header, function
#     takes the last 9 digits before the first "-"
#     in the filename. In case of a K2 light curve
#     file this would be the EPIC ID.
#
#     Parameters:
#     -----------
#     path : str
#         path to light curve fits file
#     flux_type : str
#         either SAP_FLUX or PDCSAP_FLUX. Important
#         if you pass a KeplerLightCurve.
#     kwargs :dict
#         keyword arguments that will be passed to
#         func:from_KeplerLightCurve
#
#     Return:
#     -----------
#     FlareLightCurve
#     '''
#     hdu = fitsopen(path)
#     dr = hdu[1].data
#
#     if 'TARGETID' in hdu[1].header:
#         targetid = hdu[1].header['TARGETID']
#         klcf = KeplerLightCurveFile(path)
#         klc = klcf.get_lightcurve(flux_type)
#         flc = from_KeplerLightCurve(klc, **kwargs)
#         return flc
#     elif 'KEPLERID' in hdu[1].header:
#         return from_KeplerLightCurve_file(path, flux_type, **kwargs)
#     else:
#         targetid = int(path.split('-')[0][-9:])
#         keys = {    'flux':'RAW_FLUX',
#                     'time':'TIME',
#                     'detrended_flux':'FLUX',
#                     'detrended_flux_err':'ERROR',
#                     'cadenceno':'CADENCE',
#                     'flux_trends' : 'TRTIME',
#                     'campaign':'CAMPAIGN',
#                     'centroid_col':'CENTROID_COL',
#                     'centroid_row':'CENTROID_ROW',
#                     'time_format':'TIME_FORMAT',
#                     'time_scale':'TIME_SCALE',
#                     'ra':'RA',
#                     'dec':'DEC',
#                     'channel':'CHANNEL',
#                     'time_unit':'TIME_UNIT',
#                     'flux_unit' : 'FLUX_UNIT',
#                     'origin':'ORIGIN',
#                     'pos_corr1':'X',
#                     'pos_corr2':'Y',
#                     'quality':'QUALITY',
#                     'pixel_flux':'PIXEL_FLUX',
#                     'pixel_flux_err':'PIXEL_FLUX_ERR',
#                     'quality_bitmask':'QUALITY_BITMASK',
#                     'pipeline_mask':'PIPELINE_MASK' }
#
#         for k, v in keys.items():# populate the kwargs
#             if v in hdu[1].header:
#                 keys[k] = hdu[1].header[v]
#             elif ((v.lower() in list(dr.names)) | (v in list(dr.names))) :
#                 keys[k] = dr[v]
#             else:
#                 keys[k] = None
#         flc = FlareLightCurve(targetid=targetid, **keys)
#         hdu.close()
#         del dr
#         flc = flc[(np.isfinite(flc.detrended_flux)) &
#                   (np.isfinite(flc.detrended_flux_err))]
#         return flc

def from_K2SC_source(target, campaign=None):
    """
    Read in a K2SC de-trended light curve and convert it to a ``FlareLightCurve``.

    Parameters
    ------------
    target : str
        ID or path to K2SC light curve-
    campaign : None or int
        K2 Campaign number.

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
                'campaign' : campaign,
                'month' : None,
                'radius' : 1,
                'targetlimit' : 1}
        path, campaign = download_kepler_products(target=target, **keys)
    if len(path) == 1:
        return from_K2SC_file(path[0], campaign=campaign[0])
    else:
        return [from_K2SC_file(p, campaign=c) for p,c in zip(path, campaign)]
