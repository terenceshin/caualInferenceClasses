"""
Plotting utilities for causal inference visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any


def plot_parallel_trends(data: pd.DataFrame, time_var: str, outcome_var: str, 
                        treatment_var: str, treatment_start: Optional[float] = None,
                        figsize: tuple = (12, 6)) -> None:
    """
    Plot parallel trends for difference-in-differences analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset
    time_var : str
        Name of the time variable
    outcome_var : str
        Name of the outcome variable
    treatment_var : str
        Name of the treatment variable
    treatment_start : float, optional
        Time when treatment begins (for vertical line)
    figsize : tuple
        Figure size (width, height)
    """
    # Calculate group means over time
    trend_data = data.groupby([time_var, treatment_var])[outcome_var].mean().reset_index()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot treatment and control groups
    treatment_data = trend_data[trend_data[treatment_var] == 1]
    control_data = trend_data[trend_data[treatment_var] == 0]
    
    ax.plot(treatment_data[time_var], treatment_data[outcome_var], 
            'b-o', label='Treatment Group', linewidth=2, markersize=6)
    ax.plot(control_data[time_var], control_data[outcome_var], 
            'r-s', label='Control Group', linewidth=2, markersize=6)
    
    # Add treatment start line if provided
    if treatment_start is not None:
        ax.axvline(x=treatment_start, color='gray', linestyle='--',
                  alpha=0.7, label='Treatment Start')
    
    ax.set_xlabel(time_var.replace('_', ' ').title())
    ax.set_ylabel(outcome_var.replace('_', ' ').title())
    ax.set_title('Parallel Trends Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_treatment_effects(results: Dict[str, Any], method_names: Optional[List[str]] = None,
                          figsize: tuple = (10, 6)) -> None:
    """
    Plot treatment effect estimates with confidence intervals.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing estimation results
    method_names : List[str], optional
        Names of methods to include in plot
    figsize : tuple
        Figure size (width, height)
    """
    if method_names is None:
        method_names = list(results.keys())
    
    estimates = []
    ci_lower = []
    ci_upper = []
    methods = []
    
    for method in method_names:
        if method in results:
            result = results[method]
            
            # Extract estimate
            if 'did_estimate' in result:
                estimate = result['did_estimate']
                # For basic DiD, we don't have confidence intervals
                ci_low = estimate
                ci_high = estimate
            elif 'did_coefficient' in result:
                estimate = result['did_coefficient']
                if 'did_confidence_interval' in result:
                    ci_low, ci_high = result['did_confidence_interval']
                else:
                    ci_low = estimate
                    ci_high = estimate
            else:
                continue
                
            estimates.append(estimate)
            ci_lower.append(ci_low)
            ci_upper.append(ci_high)
            methods.append(method.replace('_', ' ').title())
    
    if not estimates:
        print("No valid estimates found for plotting")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(methods))
    
    # Plot estimates with error bars
    ax.errorbar(estimates, y_pos, 
                xerr=[np.array(estimates) - np.array(ci_lower),
                      np.array(ci_upper) - np.array(estimates)],
                fmt='o', capsize=5, capthick=2, markersize=8)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel('Treatment Effect Estimate')
    ax.set_title('Treatment Effect Estimates with Confidence Intervals')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_covariate_balance(balance_results: Dict[str, Dict[str, float]], 
                          figsize: tuple = (10, 8)) -> None:
    """
    Plot covariate balance between treatment and control groups.
    
    Parameters
    ----------
    balance_results : Dict[str, Dict[str, float]]
        Results from balance check
    figsize : tuple
        Figure size (width, height)
    """
    covariates = list(balance_results.keys())
    std_diffs = [balance_results[cov]['standardized_difference'] for cov in covariates]
    p_values = [balance_results[cov]['p_value'] for cov in covariates]
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot standardized differences
    colors = ['red' if abs(diff) > 0.1 else 'green' for diff in std_diffs]
    ax1.barh(covariates, std_diffs, color=colors, alpha=0.7)
    ax1.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold (0.1)')
    ax1.axvline(x=-0.1, color='red', linestyle='--', alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_xlabel('Standardized Difference')
    ax1.set_title('Covariate Balance: Standardized Differences')
    ax1.legend()
    
    # Plot p-values
    colors = ['red' if p < 0.05 else 'green' for p in p_values]
    ax2.barh(covariates, p_values, color=colors, alpha=0.7)
    ax2.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='Significance (0.05)')
    ax2.set_xlabel('P-value')
    ax2.set_title('Covariate Balance: P-values')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def plot_residuals_diagnostics(model, figsize: tuple = (12, 8)) -> None:
    """
    Plot regression diagnostics for model residuals.
    
    Parameters
    ----------
    model : statsmodels regression result
        Fitted regression model
    figsize : tuple
        Figure size (width, height)
    """
    residuals = model.resid
    fitted = model.fittedvalues
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Residuals vs Fitted
    ax1.scatter(fitted, residuals, alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    
    # Histogram of residuals
    ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    
    # Scale-Location plot
    standardized_residuals = np.sqrt(np.abs(residuals / residuals.std()))
    ax4.scatter(fitted, standardized_residuals, alpha=0.6)
    ax4.set_xlabel('Fitted Values')
    ax4.set_ylabel('√|Standardized Residuals|')
    ax4.set_title('Scale-Location Plot')
    
    plt.tight_layout()
    plt.show()
