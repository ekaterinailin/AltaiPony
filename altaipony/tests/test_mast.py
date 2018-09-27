from ..mast import search_kepler_products
import pytest

@pytest.mark.remote_data
def test_search_kepler_lightcurve_products():
    """Tests `lightkurve.mast.search_kepler_lightcurve_products`."""
    assert(len(search_kepler_products(211119999, filetype='Lightcurve')) == 1)
    #212292519 This one has SC data
    assert(len(search_kepler_products(212292519, filetype='Lightcurve')) == 1)
