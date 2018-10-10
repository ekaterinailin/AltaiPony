import os
import numpy as np
import pytest
from inspect import currentframe, getframeinfo

from ..flarelc import FlareLightCurve
from ..lcio import from_K2SC_file

from .. import PACKAGEDIR


#example paths:

target1 = 'examples/hlsp_k2sc_k2_llc_210951703-c04_kepler_v2_lc.fits'
target2 = 'examples/hlsp_k2sc_k2_llc_211119999-c04_kepler_v2_lc.fits'
target3 = 'examples/hlsp_k2sc_k2_llc_211117077-c04_kepler_v2_lc.fits'

#From lightkurve
def test_invalid_lightcurve():
    """Invalid FlareLightCurves should not be allowed."""
    err_string = ("Input arrays have different lengths."
                  " len(time)=5, len(flux)=4")
    time = np.array([1, 2, 3, 4, 5])
    flux = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError) as err:
        FlareLightCurve(time=time, flux=flux)
    assert err_string == err.value.args[0]

def test_find_gaps():
    filename = os.path.join(PACKAGEDIR, 'examples', 'hlsp_k2sc_k2_llc_210951703-c04_kepler_v2_lc.fits')
    lc = from_K2SC_file(filename)
    lc.find_gaps()
    assert lc.gaps == [(0, 2582), (2582, 3424)]

    
