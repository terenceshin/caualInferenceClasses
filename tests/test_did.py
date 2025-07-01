"""
Tests for Difference-in-Differences implementation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from causal_inference.methods import DifferenceInDifferences
from causal_inference.datasets import generate_did_data
from causal_inference.core.exceptions import DataValidationError


class TestDifferenceInDifferences:
    """Test suite for DifferenceInDifferences class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        return generate_did_data(
            n_units=50,
            n_periods=8,
            treatment_period=5,
            treatment_effect=2.0,
            random_seed=42
        )
    
    def test_initialization(self, sample_data):
        """Test DiD initialization."""
        did = DifferenceInDifferences(
            data=sample_data,
            outcome_var='outcome',
            unit_var='unit_id',
            time_var='period',
            treatment_var='treated',
            post_var='post',
            treatment_start=5
        )
        
        assert did.outcome_var == 'outcome'
        assert did.unit_var == 'unit_id'
        assert did.time_var == 'period'
        assert did.treatment_var == 'treated'
        assert did.post_var == 'post'
        assert did.treatment_start == 5
        assert 'did_term' in did.data.columns
    
    def test_missing_columns(self):
        """Test error handling for missing columns."""
        bad_data = pd.DataFrame({
            'outcome': [1, 2, 3],
            'unit_id': [1, 1, 2],
            'treated': [1, 1, 0]
            # Missing required columns: period, post
        })

        with pytest.raises(DataValidationError):
            DifferenceInDifferences(
                data=bad_data,
                outcome_var='outcome',
                unit_var='unit_id',
                time_var='period',  # This column doesn't exist
                treatment_var='treated',
                post_var='post',  # This column doesn't exist
                treatment_start=5
            )
    
    def test_basic_did_estimation(self, sample_data):
        """Test basic DiD estimation."""
        did = DifferenceInDifferences(
            data=sample_data,
            outcome_var='outcome',
            unit_var='unit_id',
            time_var='period',
            treatment_var='treated',
            post_var='post',
            treatment_start=5
        )
        
        results = did.estimate_basic_did()
        
        # Check that results contain expected keys
        expected_keys = [
            'control_pre', 'control_post', 'treatment_pre', 'treatment_post',
            'control_diff', 'treatment_diff', 'did_estimate'
        ]
        for key in expected_keys:
            assert key in results
        
        # Check that DiD estimate is reasonable (should be close to true effect of 2.0)
        assert isinstance(results['did_estimate'], (int, float))
        assert abs(results['did_estimate'] - 2.0) < 1.0  # Allow some noise
    
    def test_regression_did_estimation(self, sample_data):
        """Test regression-based DiD estimation."""
        did = DifferenceInDifferences(
            data=sample_data,
            outcome_var='outcome',
            unit_var='unit_id',
            time_var='period',
            treatment_var='treated',
            post_var='post',
            treatment_start=5
        )
        
        results = did.estimate_regression_did()
        
        # Check that results contain expected keys
        expected_keys = [
            'model', 'did_coefficient', 'did_pvalue', 
            'did_confidence_interval', 'r_squared'
        ]
        for key in expected_keys:
            assert key in results
        
        # Check that coefficient is reasonable
        assert isinstance(results['did_coefficient'], (int, float))
        assert abs(results['did_coefficient'] - 2.0) < 1.0
    
    def test_parallel_trends_check(self, sample_data):
        """Test parallel trends assumption check."""
        did = DifferenceInDifferences(
            data=sample_data,
            outcome_var='outcome',
            unit_var='unit_id',
            time_var='period',
            treatment_var='treated',
            post_var='post',
            treatment_start=5
        )
        
        results = did.check_parallel_trends(plot=False)
        
        # Check that results contain expected keys
        expected_keys = [
            'treatment_slope', 'control_slope', 'slope_difference',
            'trend_test_pvalue', 'assumption_satisfied'
        ]
        for key in expected_keys:
            assert key in results
        
        assert isinstance(results['assumption_satisfied'], bool)
    
    def test_placebo_test(self, sample_data):
        """Test placebo test."""
        did = DifferenceInDifferences(
            data=sample_data,
            outcome_var='outcome',
            unit_var='unit_id',
            time_var='period',
            treatment_var='treated',
            post_var='post',
            treatment_start=5
        )
        
        results = did.placebo_test()
        
        # Check that results contain expected keys
        expected_keys = [
            'fake_treatment_time', 'placebo_coefficient', 
            'placebo_pvalue', 'test_passed'
        ]
        for key in expected_keys:
            assert key in results
        
        assert isinstance(results['test_passed'], bool)
    
    def test_summary_method(self, sample_data):
        """Test summary method."""
        did = DifferenceInDifferences(
            data=sample_data,
            outcome_var='outcome',
            unit_var='unit_id',
            time_var='period',
            treatment_var='treated',
            post_var='post',
            treatment_start=5
        )
        
        # Before estimation
        summary_before = did.summary()
        assert "No estimation results" in summary_before
        
        # After estimation
        did.estimate_basic_did()
        summary_after = did.summary()
        assert "DiD Estimate" in summary_after
        assert isinstance(summary_after, str)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])
