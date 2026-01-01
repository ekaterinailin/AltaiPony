import numpy as np
import pandas as pd
import pytest

import os
from pathlib import Path

from ..utils import get_response_curve
from ..flarelc import FlareLightCurve
from ..lcio import to_flare_lightcurve
from ..altai import find_iterative_median


from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from lightkurve import LightCurve


def mock_flc(origin='TPF', detrended=False, ampl=1., dur=1):
    """
    Mocks a FlareLightCurve with a sinusoid variation and a single positive outlier.

    Parameter
    -----------
    origin : 'TPF' or str
        Mocks a specific origin, such as 'KLC', 'FLC' etc.
    detrended : False or bool
        If False, a sinusoid signal is added to the mock light curve.

    Return
    -------
    FlareLightCurve
    """
    n = 1000
    time = np.arange(0, n/48, 1./48.)
    pixel_time = np.outer(time,np.full((3,3), 1)).reshape((1000,3,3))
    np.random.seed(13854)

    pipeline_mask = np.array([[False, False, False],
                              [False, True,  False],
                              [False, False, False],])
    quality = np.zeros_like(time)
    np.random.seed(33)
    flux_err = np.random.rand(n)/100.
    if detrended==False:
        flux = np.sin(time/2)*7. + 500. +flux_err
        pixel_flux = np.random.rand(len(time),3,3)/100.+500.+np.sin(pixel_time/2)*7.
        pixel_flux_err = np.random.rand(len(time),3,3)/100.
    else:
        flux = 500. + flux_err
        pixel_flux = np.random.rand(len(time),3,3)/100.+500.
        pixel_flux_err = np.random.rand(len(time),3,3)/100.
    flux[15:15+dur] += 500.*ampl
    flux[15+dur:15+2*dur] += 250.*ampl
    flux[15+2*dur:15+3*dur] += 130.*ampl
    flux[15+3*dur:15+4*dur] += 80.*ampl
    quality[17] = 1024
    quality[18] = 128
    keys = {'flux' : flux, 'flux_err' : flux_err, 'time' : time,
            'pos_corr1' : np.zeros(n), 'pos_corr2' : np.zeros(n),
            'cadenceno' : np.arange(n), 'it_med' : np.full_like(time,500.005),
            'quality' : quality,}
    meta = {'targetid' : 800000000,
            'origin' : origin,  'pipeline_mask' : pipeline_mask,
            'pixel_flux' : pixel_flux, 'campaign' : 5, 'ra' : 22.,
            'dec' : 22., 'mission' : 'K2', 'channel' : 55, 
            'pixel_flux_err' : pixel_flux_err, 'time_format': 'bkjd',
            'primary_header':3, 'data_header':2}

    if detrended == False:
        flc = FlareLightCurve(keys, meta=meta)
    else:
        keys["detrended_flux"]=flux
        keys["detrended_flux_err"]=flux_err
        flc = FlareLightCurve(data=keys, meta=meta)
    return flc



def test_mark_flagged_flares():
    flc = mock_flc(detrended=True)
    flc = flc.find_flares()
    flc = flc.mark_flagged_flares(explain=True)
    assert flc.flares.quality.iloc[0] == 1152
    s1 = "Sudden sensitivity dropout, Cosmic ray in optimal aperture"
    s2 = "Cosmic ray in optimal aperture, Sudden sensitivity dropout"
    qs = flc.flares.explanation.iloc[0]
    assert ((qs == s1) | (qs == s2))

def test_sample_flare_recovery():
    
    # Generic case
    flc = mock_flc(detrended=True)
    
    flc, fflc = flc.sample_flare_recovery(iterations=2, **{"model":"davenport2014"},)
    #make sure no flares are injected overlapping true flares
    data = flc.fake_flares
    assert data[(data.istart > 14) & (data.istart < 19)].shape[0] == 0
    #test if all injected event are covered in the merged flares:
    assert data.shape[0] == 2
    assert fflc.gaps == [(0, 1000)]
    assert np.median(fflc.it_med.value) == pytest.approx(500.005274113832)
    
    # Custom case
    
    def func(flc):
        flc["detrended_flux"] =  flc.flux/2.
        flc["detrended_flux_err"] =  flc.flux_err/2.
        return flc
    
    flc = mock_flc(detrended=True)
    
    flcd, fflc = flc.sample_flare_recovery(iterations=10, inject_before_detrending=True,
                                          func=func, **{"model":"davenport2014"}, mode="custom")
    #make sure no flares are injected overlapping true flares
    data = flcd.fake_flares
    assert data[(data.istart > 14) & (data.istart < 19)].shape[0] == 0
    #test if all injected event are covered in the merged flares:
    assert data.shape[0] == 10
    assert fflc.gaps == [(0, 1000)]
    assert np.median(fflc.it_med.value) == pytest.approx(500.005274113832/2.)
    assert flcd.detrended_flux.value == pytest.approx(flc.flux/2.)


    # Custom case with detrend_kwargs
    
    def func(flc, kw=0):
        flc["detrended_flux"] =  flc.flux/2.
        flc["detrended_flux_err"] =  flc.flux_err/2.
        a = kw + 3
        assert a ==20
        return flc
    
    flc = mock_flc(detrended=True)
    
    flcd, fflc = flc.sample_flare_recovery(iterations=10, inject_before_detrending=True,
                                           func=func, mode="custom", **{"model":"davenport2014"},
                                           detrend_kwargs={"kw":17})
    #make sure no flares are injected overlapping true flares
    data = flcd.fake_flares
    assert data[(data.istart > 14) & (data.istart < 19)].shape[0] == 0
    
    #test if all injected event are covered in the merged flares:
    assert data.shape[0] == 10
    assert fflc.gaps == [(0, 1000)]
    assert float(np.median(fflc.it_med)) == pytest.approx(500.005274113832/2.)
    assert flcd.detrended_flux == pytest.approx(flc.flux/2.)
       
    flcd.flares = flcd.flares.astype(float)
    # Test that the original flare was not changed accidentally
    assert flcd.flares.loc[0,'ed_rec'] == pytest.approx(3455.8875941, rel=1e-4)
    assert flcd.flares['ed_rec_err'][0] < flcd.flares['ed_rec'][0]
    assert flcd.flares['istart'][0] == 15
    assert flcd.flares['istop'][0] == 19
    assert flcd.flares['cstop'][0] == 19
    assert flcd.flares['cstart'][0] == 15
    assert flcd.flares['tstart'][0] == pytest.approx(0.3125, rel=1e-4)
    assert flcd.flares['tstop'][0] == pytest.approx(0.395833, rel=1e-4)
    assert flcd.flares['total_n_valid_data_points'][0] == 1000
    assert flcd.flares['ampl_rec'][0] == pytest.approx(1, rel=1e-3)
    
    # Test that adding another round of injrec will append to the path
    flc = mock_flc(detrended=True)
    flcd, fflc = flc.sample_flare_recovery(iterations=10, inject_before_detrending=False,
                                           save=True, **{"model":"davenport2014"},)
    size = len(flcd.fake_flares)
    
    flcd, fflc = flcd.sample_flare_recovery(iterations=10, inject_before_detrending=False,
                                           save=True, **{"model":"davenport2014"},)
    size2 = len(flcd.fake_flares)
    assert size < size2
    
    path ='10_800000000_inj_after_5.csv'
    saved = pd.read_csv(path)
    assert saved.shape[0] == size2
    
    os.remove(path)


def test_repr():
    pass

def test_getitem():
    pass


def test_invalid_lightcurve():
    """Invalid FlareLightCurves should not be allowed."""
    err_string = ("Input arrays have different lengths."
                  " len(time)=5, len(flux)=4")
    time = np.array([1, 2, 3, 4, 5])
    flux = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError) as err:
        FlareLightCurve(time=time, flux=flux)


def test_find_gaps():
    flux = np.random.rand(1000)
    time = np.linspace(0,30,1000)
    flux[20:200] = np.nan
    time = time[np.where(~np.isnan(flux))]
    flux = flux[np.where(~np.isnan(flux))]
    flc = FlareLightCurve(time=time, flux=flux)

    flc = flc.find_gaps()
    assert flc.gaps == [(0, 20), (20, 820)]

def test_detrend():
    # Test SAVGOL detrending
    
    ampls = [100, 10., 1, .1, .01]
    durs = [1, 2, 3]
    lcs = []
    for ampl in ampls:
        for dur in durs:
            aplc = mock_flc(ampl=ampl, dur=dur)
            daplc = aplc.detrend("savgol")
            lcs.append(daplc)

    for daplc in lcs:
        print("FFFDF", daplc.detrended_flux)
        fff = find_iterative_median(daplc)
        print("DF", fff.detrended_flux)
        shape = fff.flux.value.shape
        assert fff.it_med.value == pytest.approx(500., rel=0.01) #median stays the same roughly
        assert aplc.flux.value.shape[0] == daplc.detrended_flux.value.shape[0] #no NaNs to throw out
        assert daplc.flux.value.max() > daplc.detrended_flux.value.max() # flare sits on a LC part above quiescent level
        assert (aplc.flux_err.value == daplc.detrended_flux_err.value).all() # uncertainties are simply kept
        # Test that shapes of arrays are kept
        for att in ["detrended_flux", "detrended_flux_err",
            "flux_err", "flux", "time", "quality"]:
            assert getattr(fff, att).value.shape == shape
        
    # TEST CUSTOM DETRENDING
    
    # --- create a very minimalistic light curve
    N = int(1e4)
    time = np.linspace(2000,2050,N)
    np.random.seed(200)
    flux = 5e4 + np.random.rand(N) * 35. 
    flux_err = np.random.rand(N) * 35. 
    flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, flux_err=flux_err)

    # --- test a minimum function that fails to create the desired output
    def custom_detrending(flc):
        return flc

    with pytest.raises(AttributeError) as e:
        new_flc = flc.detrend(mode="custom", func=custom_detrending)

    # -- test a minimum function that does the job    
    def custom_detrending(flc):
        flc["detrended_flux"] = flc.flux
        flc["detrended_flux_err"] = flc.flux_err
        return flc    
        
    new_flc = flc.detrend(mode="custom", func=custom_detrending)
    assert (new_flc.flux.value == flc.flux.value).all()
    assert (new_flc.flux_err.value == flc.flux_err.value).all()

    # -- test a minimum function that does the job and has kwargs
    def custom_detrending(flc, kw=0):
        flc["detrended_flux"] = flc.flux
        flc["detrended_flux_err"] = flc.flux_err
        a = kw + 3 
        assert a == 20
        return flc    
        
    new_flc = flc.detrend(mode="custom", func=custom_detrending, kw=17)
    assert (new_flc.flux.value == flc.flux.value).all()
    assert (new_flc.flux_err.value == flc.flux_err.value).all()

    # --- function should fail if no func is given

    with pytest.raises(ValueError) as e:
        new_flc = flc.detrend(mode="custom")


def test_detrend_fails():
    """If detrend fails, an error is raised with given string."""
    
    # De-trending fails in general when an invalid mode is passed.
    # But also a helpful message is thrown out.
    flc =  mock_flc()
    err_string = ('\nDe-trending mode blaaaah does not exist. Pass "savgol" (for a Savitzky-Golay filter based detrending)'
                       ' or "custom" (to pass a custom detrending function to func=).')
    with pytest.raises(ValueError) as err:
        flc.detrend("blaaaah")
    assert err_string == err.value.args[0]

def test_find_flares():
    """Test that an obvious flare is recovered sufficiently well."""
    flc = mock_flc(detrended=True)
    flc = flc.find_flares()
    #print(flc.flares)
    assert flc.flares.loc[0,'ed_rec'] == pytest.approx(3455.8875941, rel=1e-4)
    assert flc.flares['ed_rec_err'][0] < flc.flares['ed_rec'][0]
    assert flc.flares['istart'][0] == 15
    assert flc.flares['istop'][0] == 19
    assert flc.flares['cstop'][0] == 19
    assert flc.flares['cstart'][0] == 15
    assert flc.flares['tstart'][0] == pytest.approx(0.3125, rel=1e-4)
    assert flc.flares['tstop'][0] == pytest.approx(0.395833, rel=1e-4)
    assert flc.flares['total_n_valid_data_points'][0] == 1000
    assert flc.flares['ampl_rec'][0] == pytest.approx(1, rel=1e-3)

    
def test_get_response_curve(tmp_path):
    """Test all expected errors from get_response_curve()."""

    #Unknown mission should raise ValueError
    with pytest.raises(ValueError, match="Unknown mission"):
        get_response_curve(mission="invalidmission")

    #bad custom path
    with pytest.raises(FileNotFoundError):
        get_response_curve(custom_path=tmp_path / "does_not_exist.csv")

    #invalid format in CSV
    # Create a CSV file with bad headers
    bad_csv = tmp_path / "bad_response.csv"
    df = pd.DataFrame({"bad": [1, 2, 3], "wrong": [0.1, 0.2, 0.3]})
    df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="Invalid response file format"):
        get_response_curve(custom_path=bad_csv)
    
    

def test_get_energies():
    """Test that get_energies works and raises correct errors/warnings."""

    import warnings

    #no flares should raise ValueError
    flc = mock_flc(detrended=True)
    with pytest.raises(ValueError, match="No flares found"):
        flc.get_energies(teff=5000, radius=0.9)

    #flare light curve
    flc = mock_flc(detrended=True)
    flc = flc.find_flares()
    ncols_before = flc.flares.shape[1]

    #mismatched wav/resp lengths should raise ValueError
    wav = np.array([1, 2, 3])
    resp = np.array([1, 2])  # mismatch
    with pytest.raises(ValueError, match="must have the same length"):
        flc.get_energies(teff=5000, radius=0.9, wav=wav, resp=resp)

    #too few wav/resp points should raise ValueError
    wav = np.array([1, 2, 3])
    resp = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="contain at least"):
        flc.get_energies(teff=5000, radius=0.9, wav=wav, resp=resp)

    #valid custom input should work
    wav = np.linspace(300, 900, 100)
    resp = np.ones(100)
    flc = flc.get_energies(teff=5000, radius=0.9, wav=wav, resp=resp)
    assert "bolometric_energy_erg" in flc.flares.columns
    assert flc.flares.shape[1] == ncols_before + 1  #column count increase
    assert flc.flares["bolometric_energy_erg"].notnull().all()
    assert (flc.flares["bolometric_energy_erg"] > 0).all()
    
    # Invalid teff: string instead of float
    with pytest.raises(ValueError, match="hot is not a valid unit"):
        flc.get_energies(teff="hot", radius=0.9, wav=wav, resp=resp)
    with pytest.raises(ValueError, match="big is not a valid unit"):
        flc.get_energies(teff=5000, radius="big", wav=wav, resp=resp)

    # Invalid radius: None instead of float
    with pytest.raises(TypeError):
        flc.get_energies(teff=5000, radius=None, wav=wav, resp=resp)
    with pytest.raises(TypeError):
        flc.get_energies(teff=None, radius=0.9, wav=wav, resp=resp)

    #warn if both direct arrays and mission/path provided
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        flc.get_energies(teff=5000, radius=0.9, wav=wav, resp=resp, mission="kepler")
        assert any("Ignoring mission/path" in str(warn.message) for warn in w)

    #empty wav/resp arrays should raise ValueError
    wav = np.array([])
    resp = np.array([])
    with pytest.raises(ValueError, match="contain at least"):
        flc.get_energies(teff=5000, radius=0.9, wav=wav, resp=resp)


def test_inject_fake_flares():
    flc = mock_flc(detrended=True)
    np.random.seed(84712)
    flc = flc.find_gaps()
    fake_flc = flc.inject_fake_flares()
    # make sure you inject only one flare per LC
    assert len(fake_flc.gaps) == fake_flc.fake_flares.shape[0]
    assert (set(fake_flc.fake_flares.columns.values.tolist()) == 
            {'amplitude', 'duration_d', 'ed_inj', 'peak_time'})
    assert fake_flc.detrended_flux_err.value.all() >= 1e-10
    assert fake_flc.detrended_flux.value.all() <= 1.
    assert fake_flc.detrended_flux.value.shape == flc.detrended_flux.value.shape
    flc = mock_flc(detrended=False)
    np.random.seed(84712)
    flc = flc.find_gaps()
    fake_flc = flc.inject_fake_flares(inject_before_detrending=True)

    # make sure you inject only one flare per LC
    assert len(fake_flc.gaps) == fake_flc.fake_flares.shape[0]
    assert (set(fake_flc.fake_flares.columns.values.tolist()) == 
            {'amplitude', 'duration_d', 'ed_inj', 'peak_time'})
    assert fake_flc.flux_err.value.all() >= 1e-10
    assert fake_flc.flux.value.all() <= 1.
    assert fake_flc.flux.value.shape == flc.flux.value.shape

def test_load_injrec_data():
    # Create a minimal empty light curve with an ID
    flcd = FlareLightCurve(targetid="GJ 1243", time=np.linspace(10,1))

    # Path to test file
    path = "altaipony/tests/testfiles/gj1243_injrec.csv"

    # Call the function for the first time
    flcd.load_injrec_data(path)

    # Check if nothing happened to the size
    assert flcd.fake_flares.shape[0] == 1010
    assert flcd.fake_flares.shape[1] == 14 

    # Loading a second time should append the new table
    flcd.load_injrec_data(path)
    
    # Twice as many rows, but same number of columns
    assert flcd.fake_flares.shape[0] == 2020
    assert flcd.fake_flares.shape[1] == 14 

    # We should get a FileNotFoundError when a bad path is passed:
    with pytest.raises(FileNotFoundError) as err:
        flcd.load_injrec_data("badpath")

def test_plot_ed_ratio_heatmap():
    # Create a minimal empty light curve with an ID
    flcd = FlareLightCurve(targetid="GJ 1243", time=np.linspace(10,1))
    
    # Path to test file
    path = "altaipony/tests/testfiles/gj1243_injrec.csv"
    flcd.load_injrec_data(path)
    
    # Test if the function is called properly with default values
    flcd.plot_ed_ratio_heatmap()


def test_plot_recovery_probability_heatmap():
    # Create a minimal empty light curve with an ID
    flcd = FlareLightCurve(targetid="GJ 1243", time=np.linspace(10,1))
    
    # Path to test file
    path = "altaipony/tests/testfiles/gj1243_injrec.csv"
    flcd.load_injrec_data(path)
    
    # Test if the function is called properly with default values
    flcd.plot_recovery_probability_heatmap()


# ------------------------------------------------------------------------------
# TESTING save_to_fits()
# ------------------------------------------------------------------------------

#



class TestSaveToFits:
    """Test suite for FlareLightCurve.save_to_fits()"""
    
    @pytest.fixture
    def mock_flc(self):
        """Create a minimal FlareLightCurve for testing"""
        time = Time(np.linspace(2450000, 2450010, 100), format='jd')
        flux = np.random.normal(1.0, 0.01, 100) * (u.electron / u.s)
        flux_err = np.full(100, 0.01) * (u.electron / u.s)
        
        meta = {
            'TARGETID': 123456789,
            'MISSION': 'Kepler',
            'QUARTER': 5,
            'TIMEDEL': 1765./24./3600.,  # Kepler long cadence in days
        }
        
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err, meta=meta)
        flc = to_flare_lightcurve(lc)
        flc["detrended_flux"] = np.random.normal(0, 0.001, 100)
        flc["detrended_flux_err"] = flux_err.value
        
        return flc
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Provide a temporary directory for file operations"""
        return tmp_path
    
    # ========== Test Group A: File Naming ==========
    
    def test_default_filename_format(self, mock_flc, temp_dir):
        """Test that default filename follows expected pattern without bad characters"""
        path = mock_flc.save_to_fits(loc=str(temp_dir))
        filename = Path(path).name
        
        # Check pattern: flc_{targetid}_{mission}_{qcs}_{cadence}.fits
        assert filename.startswith('flc_123456789_')
        assert 'Kepler' in filename
        assert filename.endswith('.fits')
        
        # No spaces or problematic characters
        assert ' ' not in filename
        assert '\t' not in filename
        assert '\n' not in filename
        for bad_char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            assert bad_char not in filename
    
    def test_custom_filename(self, mock_flc, temp_dir):
        """Test that custom filename is respected"""
        custom_name = "my_custom_file.fits"
        path = mock_flc.save_to_fits(loc=str(temp_dir), name=custom_name)
        
        assert Path(path).name == custom_name
    
    def test_filename_with_missing_metadata(self, mock_flc, temp_dir):
        """Test filename generation when metadata fields are missing"""
        del mock_flc.meta['mission']
        del mock_flc.meta['qcs']
        
        path = mock_flc.save_to_fits(loc=str(temp_dir))
        filename = Path(path).name
        
        # Should use 'Unknown' for missing fields
        assert 'Unknown' in filename
        assert filename.endswith('.fits')
    
    # ========== Test Group B: File Location ==========
    
    def test_saves_to_specified_location(self, mock_flc, temp_dir):
        """Test that file is saved to specified directory"""
        path = mock_flc.save_to_fits(loc=str(temp_dir))
        
        assert Path(path).exists()
        assert Path(path).parent == temp_dir
    
    def test_saves_to_current_directory_by_default(self, mock_flc, monkeypatch):
        """Test that file saves to current directory when loc=None"""
        test_dir = Path.cwd() / "test_fits_output"
        test_dir.mkdir(exist_ok=True)
        
        try:
            monkeypatch.chdir(test_dir)
            path = mock_flc.save_to_fits()
            
            assert Path(path).exists()
            assert Path(path).parent == test_dir
        finally:
            # Cleanup
            if Path(path).exists():
                Path(path).unlink()
            test_dir.rmdir()
    
    def test_returns_full_path(self, mock_flc, temp_dir):
        """Test that function returns the full path to saved file"""
        path = mock_flc.save_to_fits(loc=str(temp_dir))
        
        assert isinstance(path, str)
        assert Path(path).is_absolute()
        assert Path(path).exists()
    
    # ========== Test Group C: HDU Contents ==========
    
    def test_hdu_structure(self, mock_flc, temp_dir):
        """Test that FITS file has correct HDU structure"""
        path = mock_flc.save_to_fits(loc=str(temp_dir))
        
        with fits.open(path) as hdul:
            # Should have at least 2 HDUs: Primary + Table
            assert len(hdul) >= 2
            assert isinstance(hdul[0], fits.PrimaryHDU)
            assert isinstance(hdul[1], fits.BinTableHDU)
    
    def test_metadata_in_primary_header(self, mock_flc, temp_dir):
        """Test that metadata is written to primary HDU header"""
        path = mock_flc.save_to_fits(loc=str(temp_dir))
        
        with fits.open(path) as hdul:
            header = hdul[0].header
            
            # Check that metadata keys are present (uppercase, max 8 chars)
            assert 'TARGETID' in header
            assert header['TARGETID'] == 123456789
            assert 'MISSION' in header
            assert header['MISSION'] == 'Kepler'
            assert 'QCS' in header
            assert header['QCS'] == 5
            assert 'CADENCE' in header
            assert np.isclose(header['CADENCE'], 1765.0)
    
    def test_required_columns_in_table(self, mock_flc, temp_dir):
        """Test that all required columns are in the table HDU"""
        path = mock_flc.save_to_fits(loc=str(temp_dir))
        
        with fits.open(path) as hdul:
            table = hdul[1].data
            colnames = [col.lower() for col in hdul[1].columns.names]
            
            # Required columns
            assert 'time' in colnames
            assert 'flux' in colnames
            assert 'flux_err' in colnames
            assert 'cadenceno' in colnames
    
    def test_optional_columns_preserved(self, mock_flc, temp_dir):
        """Test that optional columns are preserved when present"""
        path = mock_flc.save_to_fits(loc=str(temp_dir))
        
        with fits.open(path) as hdul:
            colnames = [col.lower() for col in hdul[1].columns.names]
            
            # Optional columns that were set
            assert 'detrended_flux' in colnames
            assert 'detrended_flux_err' in colnames
    
    def test_data_preservation(self, mock_flc, temp_dir):
        """Test that data values are correctly preserved"""
        original_flux = mock_flc.flux.value.copy()
        original_time = mock_flc.time.value.copy()
        
        path = mock_flc.save_to_fits(loc=str(temp_dir))
        
        with fits.open(path) as hdul:
            table = hdul[1].data
            
            # Data should match original (within numerical precision)
            saved_flux = table['flux']
            saved_time = table['time']
            
            assert np.allclose(saved_flux, original_flux, rtol=1e-6)
            assert np.allclose(saved_time, original_time, rtol=1e-6)
    
    # ========== Test Group D: Error Handling ==========
    
    def test_missing_targetid_raises_error(self, temp_dir):
        """Test that missing targetid raises appropriate error"""
        time = Time(np.linspace(2450000, 2450010, 100), format='jd')
        flux = np.random.normal(1.0, 0.01, 100) * (u.electron / u.s)
        flux_err = np.full(100, 0.01) * (u.electron / u.s)
        
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        lc.__class__ = FlareLightCurve
        
        # Should raise AttributeError or similar when accessing targetid
        with pytest.raises((AttributeError, KeyError, TypeError)):
            lc.save_to_fits(loc=str(temp_dir))
    
    def test_invalid_path_raises_error(self, mock_flc):
        """Test that invalid path raises appropriate error"""
        invalid_path = "/nonexistent/directory/that/does/not/exist"
        
        with pytest.raises((OSError, FileNotFoundError)):
            mock_flc.save_to_fits(loc=invalid_path)
    
    def test_overwrite_behavior(self, mock_flc, temp_dir):
        """Test that overwrite parameter works correctly"""
        # Save once
        path = mock_flc.save_to_fits(loc=str(temp_dir), name="test_overwrite.fits")
        assert Path(path).exists()
        
        # Save again with overwrite=True (default)
        path2 = mock_flc.save_to_fits(loc=str(temp_dir), name="test_overwrite.fits", overwrite=True)
        assert path2 == path
        assert Path(path2).exists()
        
        # Save again with overwrite=False should raise error
        with pytest.raises(OSError):
            mock_flc.save_to_fits(loc=str(temp_dir), name="test_overwrite.fits", overwrite=False)
    
    def test_nan_handling_in_optional_columns(self, mock_flc, temp_dir):
        """Test that columns with all NaN are handled correctly"""
        # it_med defaults to all NaN
        assert np.all(np.isnan(mock_flc.it_med))
        
        # Should save without error (it_med should be excluded by _add_column_if_valid)
        path = mock_flc.save_to_fits(loc=str(temp_dir))
        
        with fits.open(path) as hdul:
            colnames = [col.lower() for col in hdul[1].columns.names]
            # it_med should not be in columns since it's all NaN
            assert 'it_med' not in colnames


