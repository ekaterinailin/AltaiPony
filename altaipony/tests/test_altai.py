import numpy as np
import pytest

from ..altai import (find_flares,
                     find_flares_in_cont_obs_period,
                     chi_square,
                     equivalent_duration,
                     _find_iterative_median)

from ..flarelc import FlareLightCurve
from .test_flarelc import mock_flc

import matplotlib.pyplot as plt



def test_find_flares():
    """
    Integration test of a mock example light curve is given in test_flarelc.
    Add unit tests!
    """
    flc = mock_flc(detrended=False)
    with pytest.raises(TypeError):
        #raises error bc find_flares only works on detrended_flux
        find_flares(flc)
     
    # Check if all columns are created
    flc = mock_flc(detrended=True)
    for col in  ['istart', 'istop', 'cstart', 'cstop', 'tstart',
                 'tstop', 'ed_rec', 'ed_rec_err', 'ampl_rec', 
                 'total_n_valid_data_points', 'dur']:
        assert col in flc.flares.columns

def test_find_flares_in_cont_obs_period():
    """
    Integration test of a mock example light curve is given in test_flarelc.
    Add unit tests!
    """
    # Set up a light curve with a flare
    
    # time series (not needed but helpful for debugging)
    time = np.linspace(30,40,100)
    
    # flux array
    flux = 400 + np.random.normal(0,30,100)
    
    # add the flare
    flux[30:41] = [800, 660, 500, 490, 480, 470, 460, 455, 450, 445, 440]
    
    # give the LC characteristics
    median, error, sigma = 400., 1., 30.,

    # find the flare
    isflare = find_flares_in_cont_obs_period(flux, median, error, 
                                             sigma=sigma, N1=3, N2=2,
                                             N3=3, addtail=False)
    # check if found
    assert (np.where(isflare)[0] == np.array([30, 31, 32, 33])).all()

    # add flare decay phase to search
    isflare = find_flares_in_cont_obs_period(flux, median, error, 
                                             sigma=sigma, N1=3, N2=2,
                                             N3=3, addtail=True)


    # check if found
    assert (np.where(isflare)[0] == np.array([30, 31, 32, 33, 34, 35, 36])).all()

    # add a second flare
    flux[40:49] = [510, 500, 491, 470, 460, 455, 450, 445, 440]

    # find with tails
    isflare = find_flares_in_cont_obs_period(flux, median, error, 
                                             sigma=sigma, N1=3, N2=2, 
                                             N3=3, addtail=True)

    # check if found
    assert (np.where(isflare)[0] ==  np.array([30, 31, 32, 33, 34, 35, 
											   36, 40, 41, 42, 43, 
											   44])).all()

    # bad value for tailthreshdiff throws error
    with pytest.raises(ValueError):
        isflare = find_flares_in_cont_obs_period(flux, median, error, 
                                             sigma=sigma, N1=3, N2=2, 
                                             N3=3, addtail=True, tailthreshdiff=3)




def test_chi_square():
    """Test an abvious example"""
    residual = np.full(5,1.)
    error = np.full(5,.1)
    assert chi_square(residual, error) == 100.

def test_equivalent_duration():
    """Test a triangle shaped flare in a toy light curve."""
    detrended_flux = np.full(1000,1.)
    detrended_flux[60:70] = np.array([10,9,8,7,6,5,4,3,2,1])
    lc = FlareLightCurve(time=np.arange(1000)/86400.)
    lc.detrended_flux=detrended_flux
    lc.it_med =np.full(1000,1.)
    lc.detrended_flux_err=np.full(1000,1e-8)
   # print(lc.saturation)
    ed, ed_err = equivalent_duration(lc, 60, 70, err=True)
    assert ed == pytest.approx(45,rel=1e-8)
    assert ed_err == pytest.approx(2.665569e-08, rel=1e-4)



class TestFindIterativeMedian:
    """Test suite for _find_iterative_median function"""
    
    # ========== Test Normal Operation ==========
    
    def test_returns_correct_shape(self):
        """Test that output has same shape as input"""
        flux = np.random.normal(1.0, 0.1, 100)
        gaps = [(0, 100)]
        
        result = _find_iterative_median(flux, gaps, n=3)
        
        assert result.shape == flux.shape
        assert len(result) == len(flux)
    
    def test_single_segment_no_gaps(self):
        """Test with no gaps (entire array as one segment)"""
        flux = np.random.normal(1.0, 0.1, 100)
        
        result = _find_iterative_median(flux, gaps=None, n=3)
        
        assert result.shape == flux.shape
        # All values should be the same (global median)
        assert np.all(result == result[0])
        # Should be close to true median
        assert result[0] == pytest.approx(np.nanmedian(flux), abs=0.1)
    
    def test_single_segment_empty_gaps_list(self):
        """Test with empty gaps list"""
        flux = np.random.normal(1.0, 0.1, 100)
        
        result = _find_iterative_median(flux, gaps=[], n=3)
        
        assert result.shape == flux.shape
        assert np.all(result == result[0])
    
    def test_multiple_segments(self):
        """Test with multiple segments (gaps)"""
        # Create three segments with different means
        seg1 = np.random.normal(1.0, 0.05, 50)
        seg2 = np.random.normal(2.0, 0.05, 50)
        seg3 = np.random.normal(1.5, 0.05, 50)
        flux = np.concatenate([seg1, seg2, seg3])
        
        gaps = [(0, 50), (50, 100), (100, 150)]
        
        result = _find_iterative_median(flux, gaps, n=3)
        
        # Each segment should have a different median
        median1 = result[25]
        median2 = result[75]
        median3 = result[125]
        
        assert median1 == pytest.approx(1.0, abs=0.1)
        assert median2 == pytest.approx(2.0, abs=0.1)
        assert median3 == pytest.approx(1.5, abs=0.1)
        
        # Within each segment, all values should be the same
        assert np.all(result[0:50] == median1)
        assert np.all(result[50:100] == median2)
        assert np.all(result[100:150] == median3)
    
    def test_outliers_excluded(self):
        """Test that outliers are properly excluded from median calculation"""
        # Create data with outliers
        flux = np.ones(100) * 1.0
        flux[10] = 10.0  # Strong outlier
        flux[20] = 0.1   # Weak outlier
        
        gaps = [(0, 100)]
        
        result = _find_iterative_median(flux, gaps, n=3, max_sigma=3.0)
        
        # Median should be close to 1.0, not affected by outliers
        assert result[0] == pytest.approx(1.0, abs=0.05)
    
    def test_kwargs_passed_to_sigma_clip(self):
        """Test that kwargs are properly passed to sigma_clip"""
        flux = np.random.normal(1,0.5,100) * 1.0
        flux[10:15] = 1.9  # Add some outliers
        
        gaps = [(0, 100)]
        
        # With strict clipping
        result_strict = _find_iterative_median(flux, gaps, n=5, max_sigma=2.0)
        
        # With loose clipping
        result_loose = _find_iterative_median(flux, gaps, n=5, max_sigma=5.0)
        
        # Results should be different (loose allows more outliers)
        # Strict should be closer to 1.0
        assert result_strict[0] == pytest.approx(1.0, abs=0.05)
        # Loose may include some outlier influence
        assert result_loose[0] > result_strict[0]
    
    # ========== Test Edge Cases ==========
    
    def test_empty_array(self):
        """Test with empty array. Should throw ValueError."""
        flux = np.array([])
        gaps = []
        
        with pytest.raises(ValueError, match="Input detrended_flux array is empty."):
            _find_iterative_median(flux, gaps, n=3)
            
    def test_single_value(self):
        """Test with single value"""
        flux = np.array([1.5])
        gaps = [(0, 1)]
        
        result = _find_iterative_median(flux, gaps, n=3)
        
        assert len(result) == 1
        assert result[0] == 1.5
    
    def test_all_nan_values(self):
        """Test with all NaN values"""
        flux = np.full(100, np.nan)
        gaps = [(0, 100)]
        
        result = _find_iterative_median(flux, gaps, n=3)
        
        # Result should be all NaN
        assert np.all(np.isnan(result))
        assert result.shape == flux.shape
    
    def test_some_nan_values(self):
        """Test with some NaN values mixed in"""
        flux = np.random.normal(1.0, 0.1, 100)
        flux[10:20] = np.nan  # Add NaN gap
        
        gaps = [(0, 100)]
        
        result = _find_iterative_median(flux, gaps, n=3)
        
        # Should compute median ignoring NaNs
        assert not np.all(np.isnan(result))
        expected_median = np.nanmedian(flux)
        assert result[0] == pytest.approx(expected_median, abs=0.1)
    
    def test_segment_with_all_nans(self):
        """Test when one segment is all NaNs"""
        seg1 = np.random.normal(1.0, 0.1, 50)
        seg2 = np.full(50, np.nan)
        seg3 = np.random.normal(2.0, 0.1, 50)
        flux = np.concatenate([seg1, seg2, seg3])
        
        gaps = [(0, 50), (50, 100), (100, 150)]
        
        result = _find_iterative_median(flux, gaps, n=3)
        
        # First and third segments should have finite values
        assert np.isfinite(result[0])
        assert np.isfinite(result[100])
        
        # Second segment should be all NaNs
        assert np.all(np.isnan(result[50:100]))
    
    def test_segment_all_clipped(self):
        """Test when all values in a segment are clipped"""
        # Create a segment where all values are outliers
        flux = np.random.normal(1.0, 0.01, 100)
        flux[40:60] = 1.0 + np.random.normal(0, 5.0, 20)  # Very noisy segment
        
        gaps = [(0, 40), (40, 60), (60, 100)]
        
        result = _find_iterative_median(flux, gaps, n=3, max_sigma=1.0)
        
        # All segments should have finite medians
        assert np.all(np.isfinite(result))
    
    # ========== Test Data Integrity ==========
    
    def test_does_not_modify_input(self):
        """Test that input array is not modified"""
        flux = np.random.normal(1.0, 0.1, 100)
        flux_copy = flux.copy()
        gaps = [(0, 100)]
        
        _find_iterative_median(flux, gaps, n=3)
        
        # Original array should be unchanged
        assert np.allclose(flux, flux_copy)
    
    def test_returns_copy_not_view(self):
        """Test that result is a copy, not a view"""
        flux = np.random.normal(1.0, 0.1, 100)
        gaps = [(0, 100)]
        
        result = _find_iterative_median(flux, gaps, n=3)
        
        # Modifying result should not affect input
        result[0] = 999.0
        assert flux[0] != 999.0
    
    # ========== Test Gap Handling ==========
    
    def test_single_point_gap(self):
        """Test with a gap containing only one point"""
        flux = np.random.normal(1.0, 0.1, 100)
        gaps = [(0, 50), (50, 51), (51, 100)]  # Single point gap
        
        result = _find_iterative_median(flux, gaps, n=3)
        
        assert result.shape == flux.shape
        assert np.all(np.isfinite(result))
    
    # ========== Test expected behavior ==========
    
    def test_median_calculation_accuracy(self):
        """Test that median is calculated accurately"""
        # Create data with known median
        flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gaps = [(0, 5)]
        
        # high max_sigma to avoid clipping
        result = _find_iterative_median(flux, gaps, n=1, max_sigma=10.0)
        
        # Median should be 3.0
        assert result[0] == pytest.approx(3.0)
        assert np.all(result == 3.0)
    
    def test_iterative_sigma_clipping_converges(self):
        """Test that iterative clipping converges properly"""
        # Data with progressive outliers
        flux = np.ones(100) * 1.0
        flux[90:95] = [2.0, 3.0, 4.0, 5.0, 6.0]  # Progressive outliers
        
        gaps = [(0, 100)]
        
        # Multiple iterations should converge
        result = _find_iterative_median(flux, gaps, n=10, max_sigma=2.0)
        
        # Should be close to 1.0 (outliers excluded)
        assert result[0] == pytest.approx(1.0, abs=0.05)
    
    # ========== Test max_iter effect ==========
    
    def test_n_parameter_effects(self):
        """Test that n (max_iter) parameter has expected effect"""
        np.random.seed(983)
        flux = np.random.normal(1, 0.01, 150)
        flux[10] = 5.0  # Outliers
        flux[11] = 18.
        flux[44] = 34
        flux[[5,6,7,8,9,20,21,22,23,25,49]] = 1.06
        flux[[15,16,17,18,19,30,31,32,33,24,48]] = 1.05
        
        gaps = [(0, 150)]
        
        # With n=1 (less iteration)
        result_n1 = _find_iterative_median(flux, gaps, n=1, max_sigma=5.0)
        
        # With n=10 (more iteration)
        result_n10 = _find_iterative_median(flux, gaps, n=10, max_sigma=5.0)
        print(result_n1[0], result_n10[0])
        # More iterations should converge better
        assert np.isfinite(result_n1[0])
        assert np.isfinite(result_n10[0])
        # Both should exclude outliers
        assert result_n1[0] > result_n10[0] 

