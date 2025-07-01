"""
Custom exceptions for the causal inference library.
"""


class CausalInferenceError(Exception):
    """Base exception class for causal inference errors."""
    pass


class DataValidationError(CausalInferenceError):
    """Raised when data validation fails."""
    pass


class EstimationError(CausalInferenceError):
    """Raised when estimation fails."""
    pass


class ModelSpecificationError(CausalInferenceError):
    """Raised when model specification is invalid."""
    pass
