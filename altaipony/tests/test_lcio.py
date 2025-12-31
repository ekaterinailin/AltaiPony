import pytest
import numpy as np
from astropy.time import Time
import astropy.units as u
from lightkurve import LightCurve
from altaipony.flarelc import FlareLightCurve


from altaipony.lcio import to_flare_lightcurve


class TestToFlareLightCurve:
    """Test suite for to_flare_lightcurve function"""
    
    @pytest.fixture
    def mock_lightcurve(self):
        """Create a minimal mock LightCurve object"""
        time = Time(np.linspace(2450000, 2450010, 100), format='jd')
        flux = np.random.normal(1.0, 0.01, 100)
        flux_err = np.full(100, 0.01)
        
        meta = {
            'TARGETID': 123456789,
            'MISSION': 'Kepler',
            'QUARTER': 5,
            'TIMEDEL': 0.0206833,  # ~30 min in days
        }
        
        return LightCurve(time=time, flux=flux, flux_err=flux_err, meta=meta)
    
    @pytest.fixture
    def mock_lightcurve_campaign(self):
        """Mock K2 lightcurve with CAMPAIGN"""
        time = Time(np.linspace(2450000, 2450010, 100), format='jd')
        flux = np.random.normal(1.0, 0.01, 100)
        flux_err = np.full(100, 0.01)
        
        meta = {
            'TARGETID': 987654321,
            'MISSION': 'K2',
            'CAMPAIGN': 12,
            'TIMEDEL': 0.0206833,
        }
        
        return LightCurve(time=time, flux=flux, flux_err=flux_err, meta=meta)
    
    @pytest.fixture
    def mock_lightcurve_sector(self):
        """Mock TESS lightcurve with SECTOR"""
        time = Time(np.linspace(2450000, 2450010, 100), format='jd')
        flux = np.random.normal(1.0, 0.01, 100)
        flux_err = np.full(100, 0.01)
        
        meta = {
            'TARGETID': 111222333,
            'MISSION': 'TESS',
            'SECTOR': 42,
            'TIMEDEL': 0.00138889,  # 2 min in days
        }
        
        return LightCurve(time=time, flux=flux, flux_err=flux_err, meta=meta)
    
    def test_class_conversion(self, mock_lightcurve):
        """Test that LightCurve is converted to FlareLightCurve"""
        flc = to_flare_lightcurve(mock_lightcurve)
        assert isinstance(flc, FlareLightCurve)
        assert flc.__class__ == FlareLightCurve
    

    def test_removes_nan_time(self):
        """Test that NaN times are removed"""
        time = Time(np.linspace(2450000, 2450010, 100),format='jd')
        # Modify the underlying array to include NaN
        time_copy = time.copy()
        time_copy._time.jd1[10] = np.nan  # Modify internal representation
       
        flux = np.random.normal(1.0, 0.01, 100)
        flux_err = np.full(100, 0.01)
        
        lc = LightCurve(time=time_copy, flux=flux, flux_err=flux_err)
        flc = to_flare_lightcurve(lc)
        
        assert len(flc) == 99
        assert np.all(np.isfinite(flc.time.value))
    
    def test_removes_nan_flux(self):
        """Test that NaN flux values are removed"""
        time = Time(np.linspace(2450000, 2450010, 100), format='jd')
        flux = np.random.normal(1.0, 0.01, 100)
        flux[5:10] = np.nan
        flux_err = np.full(100, 0.01)
        
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        flc = to_flare_lightcurve(lc)
        
        assert len(flc) == 95
        assert np.all(np.isfinite(flc.flux.value))
    
    def test_removes_inf_values(self):
        """Test that inf values are removed"""
        time = Time(np.linspace(2450000, 2450010, 100), format='jd')
        flux = np.random.normal(1.0, 0.01, 100)
        flux[20] = np.inf
        flux[21] = -np.inf
        flux_err = np.full(100, 0.01)
        
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        flc = to_flare_lightcurve(lc)
        
        assert len(flc) == 98
        assert np.all(np.isfinite(flc.flux.value))
    
    def test_empty_lightcurve_raises_error(self):
        """Test that lightcurve with all NaNs raises ValueError"""
        time = Time(np.full(100, 0), format='jd')
        # Modify the underlying array to include NaN
        time_copy = time.copy()
        time_copy._time.jd1[::] = np.nan  # adding NaNs no straightforward in Time
        flux = np.random.normal(1.0, 0.01, 100)
        flux_err = np.full(100, 0.01)
        
        lc = LightCurve(time=time_copy, flux=flux, flux_err=flux_err)
        
        with pytest.raises(ValueError, match="No valid data points"):
            to_flare_lightcurve(lc)
    
    def test_detrended_flux_columns_added(self, mock_lightcurve):
        """Test that detrended_flux columns are added with NaN values"""
        flc = to_flare_lightcurve(mock_lightcurve)
        
        assert 'detrended_flux' in flc.colnames
        assert 'detrended_flux_err' in flc.colnames
        assert np.all(np.isnan(flc['detrended_flux']))
        assert np.all(np.isnan(flc['detrended_flux_err']))
        assert len(flc['detrended_flux']) == len(flc.time)
    
    def test_flare_tables_initialized(self, mock_lightcurve):
        """Test that flare tables are initialized"""
        flc = to_flare_lightcurve(mock_lightcurve)
        
        assert hasattr(flc, 'flares')
        assert hasattr(flc, 'fake_flares')
        assert len(flc.flares) == 0
        assert len(flc.fake_flares) == 0
    
    def test_qcs_from_quarter(self, mock_lightcurve):
        """Test that qcs is set from QUARTER"""
        flc = to_flare_lightcurve(mock_lightcurve)
        assert flc.meta['qcs'] == 5
    
    def test_qcs_from_campaign(self, mock_lightcurve_campaign):
        """Test that qcs is set from CAMPAIGN"""
        flc = to_flare_lightcurve(mock_lightcurve_campaign)
        assert flc.meta['qcs'] == 12
    
    def test_qcs_from_sector(self, mock_lightcurve_sector):
        """Test that qcs is set from SECTOR"""
        flc = to_flare_lightcurve(mock_lightcurve_sector)
        assert flc.meta['qcs'] == 42
    
    def test_qcs_none_when_missing(self):
        """Test that qcs is None when all keys are missing"""
        time = Time(np.linspace(2450000, 2450010, 100), format='jd')
        flux = np.random.normal(1.0, 0.01, 100)
        flux_err = np.full(100, 0.01)
        
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err, meta={'TIMEDEL': 0.0206833})
        
        flc = to_flare_lightcurve(lc)
        
        assert flc.meta['qcs'] is None
    
    def test_cadence_from_timedel(self, mock_lightcurve):
        """Test that cadence is calculated from TIMEDEL"""
        flc = to_flare_lightcurve(mock_lightcurve)
        expected_cadence = 0.0206833 * u.day.to(u.second)
        
        assert 'cadence' in flc.meta
        assert np.isclose(flc.meta['cadence'], expected_cadence, rtol=1e-5)
    
    def test_cadence_calculated_when_timedel_missing(self):
        """Test that cadence is calculated from time array when TIMEDEL missing"""
        time = Time(np.linspace(2450000, 2450010, 100), format='jd')
        flux = np.random.normal(1.0, 0.01, 100)
        flux_err = np.full(100, 0.01)
        
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err, meta={})
        flc = to_flare_lightcurve(lc)
        
        expected_dt = np.median(np.diff(time.value))
        expected_cadence = expected_dt * u.day.to(u.second)
        
        assert flc.meta['cadence'] is not None
        assert np.isclose(flc.meta['cadence'], expected_cadence, rtol=1e-3)
    
    def test_cadence_none_for_single_point(self):
        """Test that cadence is None for single data point"""
        time = Time([2450000], format='jd')
        flux = np.array([1.0])
        flux_err = np.array([0.01])
        
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err, meta={})
        flc = to_flare_lightcurve(lc)
        
        assert flc.meta['cadence'] is None
    
    def test_invalid_timedel_warns(self):
        """Test that invalid TIMEDEL values issue warning"""
        time = Time(np.linspace(2450000, 2450010, 100), format='jd')
        flux = np.random.normal(1.0, 0.01, 100)
        flux_err = np.full(100, 0.01)
        
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err, 
                       meta={'TIMEDEL': 'invalid'})
        
        with pytest.warns(UserWarning, match="Could not convert TIMEDEL"):
            flc = to_flare_lightcurve(lc)

        # Should fall back to calculation from time array
        assert flc.meta['cadence'] is not None
    
    def test_meta_keys_lowercase(self, mock_lightcurve):
        """Test that all meta keys are converted to lowercase"""
        flc = to_flare_lightcurve(mock_lightcurve)
        
        for key in flc.meta.keys():
            assert key.islower(), f"Key {key} is not lowercase"
        
        # Check specific keys
        assert 'targetid' in flc.meta
        assert 'mission' in flc.meta
        assert 'quarter' in flc.meta
        assert 'TARGETID' not in flc.meta
        assert 'MISSION' not in flc.meta
    
    def test_preserves_targetid(self, mock_lightcurve):
        """Test that targetid is preserved"""
        flc = to_flare_lightcurve(mock_lightcurve)
        assert flc.meta['targetid'] == 123456789
    
    def test_preserves_mission(self, mock_lightcurve):
        """Test that mission is preserved"""
        flc = to_flare_lightcurve(mock_lightcurve)
        assert flc.meta['mission'] == 'Kepler'
    
    def test_integration_full_workflow(self, mock_lightcurve):
        """Integration test: full conversion workflow"""
        flc = to_flare_lightcurve(mock_lightcurve)
        
        # Check class
        assert isinstance(flc, FlareLightCurve)
        
        # Check data integrity
        assert len(flc) == 100
        assert np.all(np.isfinite(flc.time.value))
        assert np.all(np.isfinite(flc.flux.value))
        
        # Check columns
        assert 'detrended_flux' in flc.colnames
        assert 'detrended_flux_err' in flc.colnames
        
        # Check metadata
        assert flc.meta['qcs'] == 5
        assert flc.meta['cadence'] is not None
        assert all(k.islower() for k in flc.meta.keys())
        
        # Check flare tables
        assert len(flc.flares) == 0
        assert len(flc.fake_flares) == 0

    def test_cadenceno_retained(self):
        """Test that cadenceno is retained from LightCurve"""
        time = Time(np.linspace(2450000, 2450010, 100), format='jd')
        flux = np.random.normal(1.0, 0.01, 100)
        flux_err = np.full(100, 0.01)
        
        # Create LightCurve with specific cadenceno values
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err, 
                    meta={'TARGETID': 123456, 'TIMEDEL': 0.0206833})
        lc['cadenceno'] = np.arange(5000, 5100)  # Custom cadence numbers
        
        # Convert to FlareLightCurve
        flc = to_flare_lightcurve(lc)
        
        # Verify cadenceno is retained
        assert 'cadenceno' in flc.colnames
        assert len(flc.cadenceno) == 100
        assert np.all(flc.cadenceno == np.arange(5000, 5100))
        
        # Verify it's the correct data type
        assert isinstance(flc.cadenceno, np.ndarray)
