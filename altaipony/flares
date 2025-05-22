from astropy import units as u
from astropy.modeling import models
from astropy.constants import sigma_sb
import numpy as np


def flare_factor(teff, radius, wav, resp,  tflare=10000):
    """Calculate the flare energy factor in ergs, following Shibayama et al. (2013).
    	Equivalent duration [s] x flare_factor [erg/s] = flare energy [erg]

    Parameters
    ----------
    teff : float
        Stellar effective temperature in Kelvin.
    radius : float
        Stellar radius in solar radii.
    wav : array
        Array of wavelengths in nanometers.
    resp : array
        Array of bandpass responses.
     tflare : float
        Flare temperature in Kelvin.
    
    Returns
    -------
    factor : float
        Flare energy factor in ergs/s.
    """

    # blackbody
    bb = models.BlackBody(temperature=teff * u.K, scale = 1 * u.erg / (u.cm ** 2 * u.s * u.AA * u.sr))

    # blackbody flux in TESS band
    bbwavs = bb(wav * u.nm)  * resp

    fluxs = np.trapz(bbwavs.value, wav)

    # blackbody
    bb = models.BlackBody(temperature=tflare * u.K, scale = 1 * u.erg / (u.cm ** 2 * u.s * u.AA * u.sr))

    # blackbody flux in TESS band
    bbwavf = bb(wav * u.nm)  * resp

    fluxf = np.trapz(bbwavf.value, wav)

    ratio = fluxs / fluxf

    factor = ratio * np.pi * (radius * u.R_sun) ** 2 * sigma_sb * (tflare * u.K)**4

    # print(factor.to("erg/s"))   

    return factor.to("erg/s")
