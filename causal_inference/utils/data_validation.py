"""
Data validation utilities for causal inference.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from ..core.exceptions import DataValidationError


def validate_panel_data(data: pd.DataFrame, unit_var: str, time_var: str, 
                       outcome_var: str, treatment_var: str) -> Dict[str, Any]:
    """
    Validate panel data structure for causal inference analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset to validate
    unit_var : str
        Name of the unit identifier column
    time_var : str
        Name of the time variable column
    outcome_var : str
        Name of the outcome variable column
    treatment_var : str
        Name of the treatment variable column
        
    Returns
    -------
    Dict[str, Any]
        Validation results and summary statistics
        
    Raises
    ------
    DataValidationError
        If validation fails
    """
    validation_results = {}
    
    # Check required columns exist
    required_cols = [unit_var, time_var, outcome_var, treatment_var]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")
    
    # Check for missing values
    missing_counts = data[required_cols].isnull().sum()
    if missing_counts.any():
        validation_results['missing_values'] = missing_counts.to_dict()
        print("⚠ Warning: Missing values detected:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count} missing values")
    
    # Check treatment variable is binary
    treatment_values = sorted(data[treatment_var].dropna().unique())
    if treatment_values != [0, 1]:
        raise DataValidationError(
            f"Treatment variable must be binary (0, 1), found: {treatment_values}"
        )
    
    # Panel structure validation
    n_units = data[unit_var].nunique()
    n_periods = data[time_var].nunique()
    expected_obs = n_units * n_periods
    actual_obs = len(data)
    
    validation_results.update({
        'n_units': n_units,
        'n_periods': n_periods,
        'expected_observations': expected_obs,
        'actual_observations': actual_obs,
        'is_balanced': actual_obs == expected_obs,
        'completeness_rate': actual_obs / expected_obs
    })
    
    # Treatment group composition
    treatment_units = data[data[treatment_var] == 1][unit_var].nunique()
    control_units = data[data[treatment_var] == 0][unit_var].nunique()
    
    validation_results.update({
        'treatment_units': treatment_units,
        'control_units': control_units,
        'treatment_share': treatment_units / n_units
    })
    
    print("✓ Data validation completed:")
    print(f"  Units: {n_units} (Treatment: {treatment_units}, Control: {control_units})")
    print(f"  Time periods: {n_periods}")
    print(f"  Panel balance: {'Balanced' if validation_results['is_balanced'] else 'Unbalanced'}")
    print(f"  Completeness: {validation_results['completeness_rate']:.1%}")
    
    return validation_results


def check_balance(data: pd.DataFrame, treatment_var: str, 
                 covariates: List[str]) -> Dict[str, Any]:
    """
    Check balance of covariates between treatment and control groups.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset
    treatment_var : str
        Name of the treatment variable
    covariates : List[str]
        List of covariate column names to check
        
    Returns
    -------
    Dict[str, Any]
        Balance test results
    """
    from scipy import stats
    
    balance_results = {}
    
    print("Covariate Balance Check:")
    print("-" * 40)
    
    for covar in covariates:
        if covar not in data.columns:
            print(f"⚠ Warning: {covar} not found in data")
            continue
            
        treatment_group = data[data[treatment_var] == 1][covar].dropna()
        control_group = data[data[treatment_var] == 0][covar].dropna()
        
        # Calculate means and standard deviations
        treat_mean = treatment_group.mean()
        control_mean = control_group.mean()
        treat_std = treatment_group.std()
        control_std = control_group.std()
        
        # Standardized difference
        pooled_std = np.sqrt((treat_std**2 + control_std**2) / 2)
        std_diff = (treat_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # T-test
        t_stat, p_value = stats.ttest_ind(treatment_group, control_group)
        
        balance_results[covar] = {
            'treatment_mean': treat_mean,
            'control_mean': control_mean,
            'difference': treat_mean - control_mean,
            'standardized_difference': std_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'balanced': abs(std_diff) < 0.1 and p_value > 0.05
        }
        
        status = "✓" if balance_results[covar]['balanced'] else "⚠"
        print(f"{status} {covar}:")
        print(f"    Treatment: {treat_mean:.3f}, Control: {control_mean:.3f}")
        print(f"    Std. Diff: {std_diff:.3f}, p-value: {p_value:.3f}")
    
    return balance_results
