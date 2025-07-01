"""
Tests for Regression Discontinuity Design implementation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from causal_inference.methods import RegressionDiscontinuityDesign
from causal_inference.datasets import generate_rdd_data
from causal_inference.core.exceptions import DataValidationError


class TestRegressionDiscontinuityDesign:
    """Test suite for RegressionDiscontinuityDesign class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        return generate_rdd_data(
            n_obs=200,
            cutoff=0.0,
            treatment_effect=3.0,
            noise_std=1.0,
            random_seed=42
        )
    
    def test_initialization(self, sample_data):
        """Test RDD initialization."""
        rdd = RegressionDiscontinuityDesign(
            data=sample_data,
            outcome_var='outcome',
            running_var='running_var',
            cutoff=0.0
        )
        
        assert rdd.outcome_var == 'outcome'
        assert rdd.running_var == 'running_var'
        assert rdd.cutoff == 0.0
        assert rdd.treatment_var == 'treatment'
        assert 'running_var_centered' in rdd.data.columns
        assert 'treatment' in rdd.data.columns
    
    def test_initialization_with_existing_treatment(self, sample_data):
        """Test RDD initialization with existing treatment variable."""
        # Add a treatment variable to the data
        sample_data['my_treatment'] = (sample_data['running_var'] >= 0.0).astype(int)
        
        rdd = RegressionDiscontinuityDesign(
            data=sample_data,
            outcome_var='outcome',
            running_var='running_var',
            cutoff=0.0,
            treatment_var='my_treatment'
        )
        
        assert rdd.treatment_var == 'my_treatment'
    
    def test_missing_running_variable(self):
        """Test error handling for missing running variable."""
        bad_data = pd.DataFrame({
            'outcome': [1, 2, 3],
            'other_var': [1, 2, 3]
            # Missing running_var
        })
        
        with pytest.raises(DataValidationError):
            RegressionDiscontinuityDesign(
                data=bad_data,
                outcome_var='outcome',
                running_var='running_var',  # Doesn't exist
                cutoff=0.0
            )
    
    def test_cutoff_outside_range(self, sample_data):
        """Test error handling for cutoff outside data range."""
        with pytest.raises(ValueError):
            RegressionDiscontinuityDesign(
                data=sample_data,
                outcome_var='outcome',
                running_var='running_var',
                cutoff=100.0  # Way outside data range
            )
    
    def test_continuity_check(self, sample_data):
        """Test continuity assumption check."""
        rdd = RegressionDiscontinuityDesign(
            data=sample_data,
            outcome_var='outcome',
            running_var='running_var',
            cutoff=0.0
        )
        
        continuity_results = rdd.check_continuity_assumption(plot=False)
        
        assert 'assumption_satisfied' in continuity_results
        assert isinstance(continuity_results['assumption_satisfied'], bool)
        
        # If we have sufficient data, should have density measures
        if not continuity_results.get('insufficient_data', False):
            assert 'density_left' in continuity_results
            assert 'density_right' in continuity_results
            assert 'density_ratio' in continuity_results
    
    def test_bandwidth_selection(self, sample_data):
        """Test bandwidth selection."""
        rdd = RegressionDiscontinuityDesign(
            data=sample_data,
            outcome_var='outcome',
            running_var='running_var',
            cutoff=0.0
        )
        
        # Test Imbens-Kalyanaraman bandwidth
        bandwidth = rdd.optimal_bandwidth_selection(method='imbens_kalyanaraman')
        
        assert isinstance(bandwidth, (int, float))
        assert bandwidth > 0
        assert 'optimal_bandwidth' in rdd.results_
    
    def test_rdd_estimation(self, sample_data):
        """Test RDD effect estimation."""
        rdd = RegressionDiscontinuityDesign(
            data=sample_data,
            outcome_var='outcome',
            running_var='running_var',
            cutoff=0.0
        )
        
        # Test with specified bandwidth
        effect_results = rdd.estimate_rdd_effect(bandwidth=1.0, polynomial_order=1)
        
        expected_keys = [
            'treatment_effect', 'standard_error', 't_statistic', 'p_value',
            'ci_lower', 'ci_upper', 'n_obs', 'r_squared'
        ]
        for key in expected_keys:
            assert key in effect_results
        
        # Check that effect is reasonable (should be close to true effect of 3.0)
        assert isinstance(effect_results['treatment_effect'], (int, float))
        assert effect_results['n_obs'] > 0
    
    def test_polynomial_orders(self, sample_data):
        """Test different polynomial orders."""
        rdd = RegressionDiscontinuityDesign(
            data=sample_data,
            outcome_var='outcome',
            running_var='running_var',
            cutoff=0.0
        )
        
        # Test linear
        linear_results = rdd.estimate_rdd_effect(bandwidth=1.0, polynomial_order=1)
        assert 'treatment_effect' in linear_results
        
        # Test quadratic
        quadratic_results = rdd.estimate_rdd_effect(bandwidth=1.0, polynomial_order=2)
        assert 'treatment_effect' in quadratic_results
        
        # Test invalid polynomial order
        with pytest.raises(ValueError):
            rdd.estimate_rdd_effect(bandwidth=1.0, polynomial_order=3)
    
    def test_kernels(self, sample_data):
        """Test different kernel functions."""
        rdd = RegressionDiscontinuityDesign(
            data=sample_data,
            outcome_var='outcome',
            running_var='running_var',
            cutoff=0.0
        )
        
        kernels = ['triangular', 'uniform', 'epanechnikov']
        
        for kernel in kernels:
            results = rdd.estimate_rdd_effect(bandwidth=1.0, kernel=kernel)
            assert 'treatment_effect' in results
    
    def test_insufficient_bandwidth(self, sample_data):
        """Test error handling for insufficient bandwidth."""
        rdd = RegressionDiscontinuityDesign(
            data=sample_data,
            outcome_var='outcome',
            running_var='running_var',
            cutoff=0.0
        )
        
        # Very small bandwidth should cause error
        with pytest.raises(ValueError):
            rdd.estimate_rdd_effect(bandwidth=0.001)
    
    def test_summary_method(self, sample_data):
        """Test summary method."""
        rdd = RegressionDiscontinuityDesign(
            data=sample_data,
            outcome_var='outcome',
            running_var='running_var',
            cutoff=0.0
        )
        
        # Before estimation
        summary_before = rdd.summary()
        assert "No estimation results" in summary_before
        
        # After estimation
        rdd.estimate_rdd_effect(bandwidth=1.0)
        summary_after = rdd.summary()
        assert "Treatment Effect" in summary_after
        assert isinstance(summary_after, str)
    
    def test_estimate_method(self, sample_data):
        """Test the main estimate method."""
        rdd = RegressionDiscontinuityDesign(
            data=sample_data,
            outcome_var='outcome',
            running_var='running_var',
            cutoff=0.0
        )
        
        # The estimate method should run RDD estimation
        results = rdd.estimate()
        
        assert 'treatment_effect' in results
        assert 'standard_error' in results
        assert 'p_value' in results
    
    def test_run_full_analysis(self, sample_data):
        """Test the full analysis pipeline."""
        rdd = RegressionDiscontinuityDesign(
            data=sample_data,
            outcome_var='outcome',
            running_var='running_var',
            cutoff=0.0
        )
        
        # Run full analysis
        results = rdd.run_full_analysis(plot=False)
        
        assert isinstance(results, dict)
        assert 'rdd_effect' in results
        assert 'optimal_bandwidth' in results


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])
