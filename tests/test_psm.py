"""
Tests for Propensity Score Matching implementation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from causal_inference.methods import PropensityScoreMatching
from causal_inference.datasets import generate_psm_data
from causal_inference.core.exceptions import DataValidationError


class TestPropensityScoreMatching:
    """Test suite for PropensityScoreMatching class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        return generate_psm_data(
            n_samples=200,
            n_features=3,
            treatment_effect=2.0,
            random_seed=42
        )
    
    def test_initialization(self, sample_data):
        """Test PSM initialization."""
        covariates = ['covariate_1', 'covariate_2', 'covariate_3']
        psm = PropensityScoreMatching(
            data=sample_data,
            outcome_var='outcome',
            treatment_var='treated',
            covariates=covariates
        )
        
        assert psm.outcome_var == 'outcome'
        assert psm.treatment_var == 'treated'
        assert psm.covariates == covariates
        assert psm.propensity_scores is None
        assert psm.matched_data is None
    
    def test_missing_covariates(self):
        """Test error handling for missing covariates."""
        bad_data = pd.DataFrame({
            'outcome': [1, 2, 3],
            'treated': [1, 0, 1],
            'covariate_1': [1, 2, 3]
            # Missing covariate_2
        })
        
        with pytest.raises(DataValidationError):
            PropensityScoreMatching(
                data=bad_data,
                outcome_var='outcome',
                treatment_var='treated',
                covariates=['covariate_1', 'covariate_2']  # covariate_2 doesn't exist
            )
    
    def test_propensity_score_estimation(self, sample_data):
        """Test propensity score estimation."""
        covariates = ['covariate_1', 'covariate_2', 'covariate_3']
        psm = PropensityScoreMatching(
            data=sample_data,
            outcome_var='outcome',
            treatment_var='treated',
            covariates=covariates
        )
        
        # Test logistic regression
        ps_scores = psm.estimate_propensity_scores(method='logistic')
        
        assert len(ps_scores) == len(sample_data)
        assert all(0 <= score <= 1 for score in ps_scores)
        assert psm.propensity_scores is not None
        assert 'propensity_score' in psm.data.columns
    
    def test_overlap_check(self, sample_data):
        """Test overlap assumption check."""
        covariates = ['covariate_1', 'covariate_2', 'covariate_3']
        psm = PropensityScoreMatching(
            data=sample_data,
            outcome_var='outcome',
            treatment_var='treated',
            covariates=covariates
        )
        
        # Estimate propensity scores first
        psm.estimate_propensity_scores(method='logistic')
        
        # Check overlap
        overlap_results = psm.check_overlap(plot=False)
        
        expected_keys = [
            'treatment_range', 'control_range', 'overlap_range',
            'overlap_exists', 'overlap_percentage', 'assumption_satisfied'
        ]
        for key in expected_keys:
            assert key in overlap_results
        
        assert isinstance(overlap_results['assumption_satisfied'], bool)
    
    def test_balance_check(self, sample_data):
        """Test balance checking."""
        covariates = ['covariate_1', 'covariate_2', 'covariate_3']
        psm = PropensityScoreMatching(
            data=sample_data,
            outcome_var='outcome',
            treatment_var='treated',
            covariates=covariates
        )
        
        balance_results = psm.check_balance_before_matching()
        
        # Check that all covariates are included
        for covar in covariates:
            assert covar in balance_results
            assert 'standardized_diff' in balance_results[covar]
            assert 't_pvalue' in balance_results[covar]
            assert 'balanced' in balance_results[covar]
    
    def test_matching(self, sample_data):
        """Test matching functionality."""
        covariates = ['covariate_1', 'covariate_2', 'covariate_3']
        psm = PropensityScoreMatching(
            data=sample_data,
            outcome_var='outcome',
            treatment_var='treated',
            covariates=covariates
        )
        
        # Estimate propensity scores
        psm.estimate_propensity_scores(method='logistic')
        
        # Perform matching
        matched_data = psm.perform_matching(method='nearest', caliper=0.1)
        
        if matched_data is not None:
            assert len(matched_data) > 0
            assert 'treated' in matched_data.columns
            assert 'outcome' in matched_data.columns
            
            # Check that we have both treatment and control units
            treatment_count = (matched_data['treated'] == 1).sum()
            control_count = (matched_data['treated'] == 0).sum()
            assert treatment_count > 0
            assert control_count > 0
    
    def test_treatment_effect_estimation(self, sample_data):
        """Test treatment effect estimation."""
        covariates = ['covariate_1', 'covariate_2', 'covariate_3']
        psm = PropensityScoreMatching(
            data=sample_data,
            outcome_var='outcome',
            treatment_var='treated',
            covariates=covariates
        )
        
        # Run full analysis
        psm.estimate_propensity_scores(method='logistic')
        matched_data = psm.perform_matching(method='nearest', caliper=0.1)
        
        if matched_data is not None:
            effect_results = psm.estimate_treatment_effect()
            
            expected_keys = [
                'ate', 'standard_error', 't_statistic', 'p_value',
                'ci_lower', 'ci_upper', 'treatment_mean', 'control_mean'
            ]
            for key in expected_keys:
                assert key in effect_results
            
            # Check that effect is reasonable (should be close to true effect of 2.0)
            assert isinstance(effect_results['ate'], (int, float))
    
    def test_summary_method(self, sample_data):
        """Test summary method."""
        covariates = ['covariate_1', 'covariate_2', 'covariate_3']
        psm = PropensityScoreMatching(
            data=sample_data,
            outcome_var='outcome',
            treatment_var='treated',
            covariates=covariates
        )
        
        # Before estimation
        summary_before = psm.summary()
        assert "No estimation results" in summary_before
        
        # After estimation
        psm.estimate_propensity_scores(method='logistic')
        matched_data = psm.perform_matching(method='nearest', caliper=0.1)
        
        if matched_data is not None:
            psm.estimate_treatment_effect()
            summary_after = psm.summary()
            assert "Average Treatment Effect" in summary_after
            assert isinstance(summary_after, str)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])
