"""
Causal inference methods module.
"""

from .difference_in_differences import DifferenceInDifferences
from .propensity_score_matching import PropensityScoreMatching
from .regression_discontinuity_design import RegressionDiscontinuityDesign

__all__ = [
    'DifferenceInDifferences',
    'PropensityScoreMatching',
    'RegressionDiscontinuityDesign'
]
