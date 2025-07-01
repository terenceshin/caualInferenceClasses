"""
Base classes and common functionality for causal inference methods.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List


class CausalInferenceBase(ABC):
    """
    Abstract base class for all causal inference methods.
    
    This class provides common functionality and enforces a consistent
    interface across different causal inference techniques.
    """
    
    def __init__(self, data: pd.DataFrame, outcome_var: str, 
                 treatment_var: str, **kwargs):
        """
        Initialize the causal inference method.
        
        Parameters
        ----------
        data : pd.DataFrame
            The dataset containing all variables
        outcome_var : str
            Name of the outcome variable
        treatment_var : str
            Name of the treatment variable
        """
        self.data = data.copy()
        self.outcome_var = outcome_var
        self.treatment_var = treatment_var
        self.results_ = None
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate input data and variables."""
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        if self.outcome_var not in self.data.columns:
            raise ValueError(f"Outcome variable '{self.outcome_var}' not found in data")
        
        if self.treatment_var not in self.data.columns:
            raise ValueError(f"Treatment variable '{self.treatment_var}' not found in data")
    
    @abstractmethod
    def estimate(self) -> Dict[str, Any]:
        """
        Estimate the causal effect.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing estimation results
        """
        pass
    
    @abstractmethod
    def summary(self) -> str:
        """
        Return a summary of the estimation results.
        
        Returns
        -------
        str
            Formatted summary of results
        """
        pass
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the estimation results.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Results dictionary if estimation has been run, None otherwise
        """
        return self.results_
