from ..mast import search_kepler_products, download_products
from . import test_ids, test_paths
import pytest

#use tests from lightkurve

@pytest.mark.remote_data
def test_search_kepler_lightcurve_products():
    """Tests `lightkurve.mast.search_kepler_lightcurve_products`."""
    assert(len(search_kepler_products(211119999, filetype='Lightcurve')) == 1)
    #212292519 This one has SC data
    assert(len(search_kepler_products(212292519, filetype='Lightcurve')) == 1)

def test_download_products():
    for ID in test_ids:
        tab = search_kepler_products(ID)
        pat = download_products(tab)
        assert pat == ('./mastDownload/K2/k2sc{0}-c04_lc/hlsp_k2sc_k2'
                       '_llc_{0}-c04_kepler_v2_lc.fits'.format(ID))
    pass

def test_download_kepler_products():
    pass

def test_query_kepler_products():
    pass

def test_search_kepler_products():
    for ID in test_ids:
        tab = search_kepler_products(ID)
        assert tab['qoc'] == 4
        assert len(tab) == 1
        assert tab['project'] == 'hlsp_k2sc'
