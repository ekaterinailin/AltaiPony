
import pytest

import numpy as np

from ..flarelc import FlareLightCurve
from ..utils import split_gaps, expand_mask

import copy

def test_expand_mask():
    # set up a test array with a single outlier and a sequence
    a = np.ones(25).astype(int)
    a[3] = 0
    a[[15,16]] = 0

    # check result
    assert (expand_mask(copy.deepcopy(a), longdecay=2) == 
            np.array([1, 1, 1, 0, 1,
                      1, 1, 1, 1, 1,
                      1, 1, 1, 1, 0,
                      0, 0, 0, 0, 1,
                      1, 1, 1, 1, 1])).all()





def test_split_gaps_basic():
    """Test basic functionality with the example from docstring"""
    gaps = [(0., 20.), (21., 34.), (37., 41.)]
    splits = [1.5, 14., 39.]
    result = split_gaps(gaps, splits)
    
    expected = [(0., 1.5), (1.5, 14.), (14., 20.), 
                (21.0, 34.), (37., 39.), (39., 41.)]
    
    assert result == expected, f"Expected {expected}, but got {result}"


def test_split_gaps_multiple_splits_in_one_gap():
    """Test multiple splits within a single gap"""
    gaps = [(0., 100.), (150., 200.)]
    splits = [25., 50., 75.]
    result = split_gaps(gaps, splits)
    
    expected = [(0., 25.), (25., 50.), (50., 75.), (75., 100.), 
                (150., 200.)]
    
    assert result == expected
    assert len(result) == 5, "Should have 5 gaps total"


def test_split_gaps_single_split():
    """Test with a single split value"""
    gaps = [(10., 30.), (40., 60.)]
    splits = [20.]
    result = split_gaps(gaps, splits)
    
    expected = [(10., 20.), (20., 30.), (40., 60.)]
    
    assert result == expected
    assert len(result) == 3


def test_split_gaps_invalid_splits_raises_error():
    """Test that invalid split values raise an IndexError"""
    gaps = [(0., 20.), (30., 50.)]
    
    # Split value outside any gap range
    invalid_splits = [25.]  # This is between gaps, not inside any gap
    
    with pytest.raises(IndexError) as exc_info:
        split_gaps(gaps, invalid_splits)
    
    assert "splits you passed are wrong" in str(exc_info.value)


def test_split_gaps_preserves_gap_order():
    """Test that gaps remain sorted after splitting"""
    gaps = [(100., 200.), (10., 50.), (250., 300.)]
    gaps.sort(key=lambda x: x[0])  # Pre-sort
    splits = [25., 150.]
    
    result = split_gaps(gaps, splits)
    
    # Check that result is sorted
    for i in range(len(result) - 1):
        assert result[i][0] < result[i+1][0], "Gaps should be in ascending order"
    
    # Check specific values
    assert (10., 25.) in result
    assert (25., 50.) in result
    assert (100., 150.) in result
    assert (150., 200.) in result




class TestExpandMaskBasic:
    """Basic functionality tests."""
    
    def test_all_ones_unchanged(self):
        """Array of all 1s should remain unchanged."""
        a = np.ones(10, dtype=int)
        result = expand_mask(a.copy())
        np.testing.assert_array_equal(result, a)
    
    def test_all_zeros_unchanged(self):
        """Array of all 0s should remain unchanged (already masked)."""
        a = np.zeros(10, dtype=int)
        result = expand_mask(a.copy())
        np.testing.assert_array_equal(result, a)
    
    def test_empty_array(self):
        """Empty array should return empty array."""
        a = np.array([], dtype=int)
        result = expand_mask(a.copy())
        assert len(result) == 0
    
    def test_single_element_zero(self):
        """Single zero element."""
        a = np.array([0])
        result = expand_mask(a.copy())
        np.testing.assert_array_equal(result, [0])
    
    def test_single_element_one(self):
        """Single one element."""
        a = np.array([1])
        result = expand_mask(a.copy())
        np.testing.assert_array_equal(result, [1])


class TestSingleOutlierNoExpansion:
    """Single outliers (run length 1) should NOT trigger expansion."""
    
    def test_single_zero_middle(self):
        """Single zero in middle should not expand."""
        a = np.array([1, 1, 1, 0, 1, 1, 1])
        result = expand_mask(a.copy())
        np.testing.assert_array_equal(result, a)
    
    def test_single_zero_start(self):
        """Single zero at start should not expand."""
        a = np.array([0, 1, 1, 1, 1])
        result = expand_mask(a.copy())
        np.testing.assert_array_equal(result, a)
    
    def test_single_zero_end(self):
        """Single zero at end should not expand."""
        a = np.array([1, 1, 1, 1, 0])
        result = expand_mask(a.copy())
        np.testing.assert_array_equal(result, a)
    
    def test_multiple_isolated_zeros(self):
        """Multiple isolated single zeros should not expand."""
        a = np.array([1, 0, 1, 1, 0, 1, 0, 1])
        result = expand_mask(a.copy())
        np.testing.assert_array_equal(result, a)


class TestRunExpansion:
    """Test expansion logic for runs of length >= 2."""
    
    def test_run_of_two_expands_by_one(self):
        """Run of 2 zeros: sqrt(2) ≈ 1.4 -> rounds to 1."""
        # Add 1 before and 1 after
        a = np.array([1, 1, 1, 0, 0, 1, 1, 1])
        result = expand_mask(a.copy())
        expected = np.array([1, 1, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_run_of_three_expands_by_two(self):
        """Run of 3 zeros: sqrt(3) ≈ 1.7 -> rounds to 2."""
        a = np.array([1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        result = expand_mask(a.copy())
        expected = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_run_of_four_expands_by_two(self):
        """Run of 4 zeros: sqrt(4) = 2 -> exactly 2."""
        a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
        result = expand_mask(a.copy())
        expected = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_run_of_nine_expands_by_three(self):
        """Run of 9 zeros: sqrt(9) = 3 -> exactly 3."""
        a = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = expand_mask(a.copy())
        # 3 before, 9 zeros, 3 after
        expected = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(result, expected)


class TestBoundaryConditions:
    """Test expansion at array boundaries."""
    
    def test_run_at_start_no_room_before(self):
        """Run at start - can't expand before, but should expand after."""
        a = np.array([0, 0, 1, 1, 1, 1, 1])
        result = expand_mask(a.copy())
        # sqrt(2) rounds to 1, expand 1 after
        expected = np.array([0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_run_at_end_no_room_after(self):
        """Run at end - can expand before, but can't expand after."""
        a = np.array([1, 1, 1, 1, 1, 0, 0])
        result = expand_mask(a.copy())
        # sqrt(2) rounds to 1, expand 1 before
        expected = np.array([1, 1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(result, expected)
    
    def test_run_at_start_partial_expand_before(self):
        """Run near start - partial expansion before."""
        a = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1])
        result = expand_mask(a.copy())
        # sqrt(4) = 2, but only 1 space before
        expected = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_run_at_end_partial_expand_after(self):
        """Run near end - partial expansion after."""
        a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1])
        result = expand_mask(a.copy())
        # sqrt(4) = 2, but only 1 space after
        expected = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(result, expected)


class TestLongdecayParameter:
    """Test the longdecay multiplier for post-run expansion."""
    
    def test_longdecay_2(self):
        """Longdecay=2 doubles the expansion after the run."""
        a = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1])
        result = expand_mask(a.copy(), longdecay=2)
        # sqrt(2) rounds to 1, expand 1 before, 2 after
        expected = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_longdecay_3(self):
        """Longdecay=3 triples the expansion after the run."""
        a = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        result = expand_mask(a.copy(), longdecay=3)
        # sqrt(2) rounds to 1, expand 1 before, 3 after
        expected = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_longdecay_0(self):
        """Longdecay=0 means no expansion after the run."""
        a = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        result = expand_mask(a.copy(), longdecay=0)
        # sqrt(2) rounds to 1, expand 1 before, 0 after
        expected = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)


class TestMultipleRuns:
    """Test handling of multiple runs in the same array."""
    
    def test_two_separate_runs(self):
        """Two separate runs should both be expanded."""
        a = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
        result = expand_mask(a.copy())
        # Each run of 2: sqrt(2) rounds to 1
        expected = np.array([1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_runs_of_different_lengths(self):
        """Runs of different lengths get different expansions."""
        a = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
        result = expand_mask(a.copy())
        # First run of 2: expand by 1
        # Second run of 4: expand by 2
        expected = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_adjacent_expansions_overlap(self):
        """When expansions would overlap, both apply (0 wins)."""
        a = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1])
        result = expand_mask(a.copy())
        # Both runs expand by 1, middle overlaps
        expected = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(result, expected)


class TestInputNotMutated:
    """Ensure the original input array is not modified."""
    
    def test_original_unchanged(self):
        """Original array should not be mutated."""
        original = np.array([1, 1, 0, 0, 1, 1])
        original_copy = original.copy()
        _ = expand_mask(original)
        np.testing.assert_array_equal(original, original_copy)


class TestDataTypes:
    """Test different input data types."""
    
    def test_float_input(self):
        """Float input should work (0.0 and 1.0)."""
        a = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        result = expand_mask(a.copy())
        expected = np.array([1, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(result, expected)
    
    def test_bool_input(self):
        """Boolean input should work."""
        a = np.array([True, True, False, False, True, True])
        result = expand_mask(a.copy())
        expected = np.array([True, False, False, False, False, True])
        np.testing.assert_array_equal(result, expected)


class TestSqrtRounding:
    """Verify sqrt rounding behavior for various run lengths."""
    
    @pytest.mark.parametrize("run_length,expected_expansion", [
        (2, 1),   # sqrt(2) = 1.41 -> 1
        (3, 2),   # sqrt(3) = 1.73 -> 2
        (4, 2),   # sqrt(4) = 2.0  -> 2
        (5, 2),   # sqrt(5) = 2.24 -> 2
        (6, 2),   # sqrt(6) = 2.45 -> 2
        (7, 3),   # sqrt(7) = 2.65 -> 3
        (8, 3),   # sqrt(8) = 2.83 -> 3
        (9, 3),   # sqrt(9) = 3.0  -> 3
        (10, 3),  # sqrt(10) = 3.16 -> 3
        (16, 4),  # sqrt(16) = 4.0 -> 4
        (25, 5),  # sqrt(25) = 5.0 -> 5
    ])
    def test_expansion_amount(self, run_length, expected_expansion):
        """Verify expansion amount for various run lengths."""
        # Create array with enough padding
        padding = expected_expansion + 5
        a = np.ones(run_length + 2 * padding, dtype=int)
        a[padding:padding + run_length] = 0
        
        result = expand_mask(a.copy())
        
        # Count zeros in result
        zero_count = np.sum(result == 0)
        # Should be: original run + expansion before + expansion after
        expected_zeros = run_length + expected_expansion + expected_expansion
        
        assert zero_count == expected_zeros, (
            f"Run of {run_length}: expected {expected_zeros} zeros, got {zero_count}"
        )

