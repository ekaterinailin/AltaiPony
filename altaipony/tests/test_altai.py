
from ..altai import get_k2sc_lc, wrapper

def test_wrapper():
    lc = get_k2sc_lc('examples/hlsp_k2sc_k2_llc_211119999-c04_kepler_v2_lc.fits')
    assert wrapper(lc) == ([104974],[104978])
