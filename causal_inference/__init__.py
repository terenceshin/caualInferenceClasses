"""
Causal Inference Library
========================

A comprehensive Python library for causal inference methods including:
- Difference-in-Differences (DiD)
- Regression Discontinuity Design (RDD)
- Instrumental Variables (IV)
- Propensity Score Matching (PSM)

Example usage:
    >>> from causal_inference.methods import DifferenceInDifferences
    >>> did = DifferenceInDifferences(data, outcome_var='y', ...)
    >>> results = did.estimate_basic_did()
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .methods.difference_in_differences import DifferenceInDifferences
from .methods.propensity_score_matching import PropensityScoreMatching
from .methods.regression_discontinuity_design import RegressionDiscontinuityDesign
from .datasets.synthetic import generate_did_data, generate_rdd_data, generate_psm_data

# Define what gets imported with "from causal_inference import *"
__all__ = [
    'DifferenceInDifferences',
    'PropensityScoreMatching',
    'RegressionDiscontinuityDesign',
    'generate_did_data',
    'generate_rdd_data',
    'generate_psm_data',
]
