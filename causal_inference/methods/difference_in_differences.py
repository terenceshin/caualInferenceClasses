"""
Difference-in-Differences implementation for causal inference.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from datetime import datetime, timedelta
import warnings
from typing import Dict, Any, Optional, List, Union

from ..core.base import CausalInferenceBase
from ..core.exceptions import DataValidationError, EstimationError

warnings.filterwarnings('ignore')


class DifferenceInDifferences(CausalInferenceBase):
    """
    A comprehensive class for Difference-in-Differences analysis with assumption checking
    and robustness tests.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing the variables for analysis
    outcome_var : str
        Name of the outcome variable column
    unit_var : str
        Name of the unit identifier column (e.g., 'market', 'store_id', 'user_id')
    time_var : str
        Name of the time variable column (e.g., 'week', 'month', 'period')
    treatment_var : str
        Name of the treatment group indicator column (1 = treatment, 0 = control)
    post_var : str
        Name of the post-treatment period indicator column (1 = post, 0 = pre)
    treatment_start : int, float, or str
        The time period when treatment begins. Can be:
        - int/float: for numeric time periods (e.g., week 13)
        - str: for date strings in 'YYYY-MM-DD' format (e.g., '2024-03-25')
    """
    
    def __init__(self, data: pd.DataFrame, outcome_var: str, unit_var: str, 
                 time_var: str, treatment_var: str, post_var: str, 
                 treatment_start: Union[int, float, str]):
        # Initialize base class
        super().__init__(data, outcome_var, treatment_var)
        
        # DiD-specific attributes
        self.unit_var = unit_var
        self.time_var = time_var
        self.post_var = post_var

        # Handle different treatment_start formats
        self.treatment_start = self._parse_treatment_start(treatment_start)
        self.treatment_start_original = treatment_start  # Keep original for reference

        # Determine if we're working with dates
        self.is_date_based = isinstance(treatment_start, str)

        # Validate DiD-specific data first
        self._validate_did_data()

        # Convert time variable to datetime if working with dates
        if self.is_date_based:
            self.data[self.time_var] = pd.to_datetime(self.data[self.time_var])

        # Create DiD interaction term
        self.data['did_term'] = self.data[treatment_var] * self.data[post_var]

        # Store results
        self.assumption_checks = {}

    def _parse_treatment_start(self, treatment_start: Union[int, float, str]) -> Union[int, float, pd.Timestamp]:
        """Parse treatment_start into appropriate format"""
        if isinstance(treatment_start, str):
            # Assume it's a date string in YYYY-MM-DD format
            try:
                return pd.to_datetime(treatment_start)
            except:
                raise ValueError(f"Invalid date format for treatment_start: {treatment_start}. Use 'YYYY-MM-DD' format.")
        else:
            # Numeric value (int or float)
            return treatment_start

    def _validate_did_data(self) -> None:
        """Validate DiD-specific input data and variables"""
        required_cols = [self.unit_var, self.time_var, self.post_var]
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")
            
        # Check binary variables
        for var in [self.treatment_var, self.post_var]:
            unique_vals = sorted(self.data[var].unique())
            if unique_vals != [0, 1]:
                raise DataValidationError(f"{var} must be binary (0, 1), found: {unique_vals}")
                
        print(f"✓ Data validation passed")
        print(f"  - Units: {self.data[self.unit_var].nunique()}")
        print(f"  - Time periods: {self.data[self.time_var].nunique()}")
        print(f"  - Treatment units: {self.data[self.data[self.treatment_var]==1][self.unit_var].nunique()}")
        print(f"  - Control units: {self.data[self.data[self.treatment_var]==0][self.unit_var].nunique()}")

    def estimate(self) -> Dict[str, Any]:
        """
        Main estimation method - runs basic DiD estimation.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing estimation results
        """
        return self.estimate_basic_did()
    
    def summary(self) -> str:
        """
        Return a summary of the estimation results.
        
        Returns
        -------
        str
            Formatted summary of results
        """
        if self.results_ is None:
            return "No estimation results available. Run estimate() first."
        
        summary_lines = []
        summary_lines.append("Difference-in-Differences Estimation Results")
        summary_lines.append("=" * 50)
        
        if 'basic_did' in self.results_:
            basic_results = self.results_['basic_did']
            summary_lines.append(f"DiD Estimate: {basic_results['did_estimate']:.3f}")
            summary_lines.append(f"Treatment Group Change: {basic_results['treatment_diff']:.3f}")
            summary_lines.append(f"Control Group Change: {basic_results['control_diff']:.3f}")
        
        return "\n".join(summary_lines)

    def check_parallel_trends(self, pre_period_end: Optional[Union[int, float, str]] = None,
                            plot: bool = True) -> Dict[str, Any]:
        """
        Check parallel trends assumption - critical for DiD validity

        Parameters:
        -----------
        pre_period_end : int, float, or str, optional
            Last period to include in pre-treatment analysis. If None, uses treatment_start - 1
        plot : bool, default True
            Whether to create visualization

        Returns:
        --------
        dict : Results of parallel trends test
        """
        if pre_period_end is None:
            if self.is_date_based:
                # For dates, subtract 1 day
                pre_period_end = self.treatment_start - timedelta(days=1)
            else:
                # For numeric, subtract 1
                pre_period_end = self.treatment_start - 1

        print("\n" + "="*60)
        print("ASSUMPTION CHECK: PARALLEL TRENDS")
        print("="*60)

        # Filter to pre-treatment period
        pre_df = self.data[self.data[self.time_var] <= pre_period_end].copy()

        # Calculate trends for each group
        trend_data = pre_df.groupby([self.time_var, self.treatment_var])[self.outcome_var].mean().reset_index()

        treatment_data = trend_data[trend_data[self.treatment_var] == 1]
        control_data = trend_data[trend_data[self.treatment_var] == 0]

        # Convert time to numeric for regression if it's date-based
        if self.is_date_based:
            # Convert dates to numeric (days since first date)
            min_date = treatment_data[self.time_var].min()
            treatment_time_numeric = (treatment_data[self.time_var] - min_date).dt.days
            control_time_numeric = (control_data[self.time_var] - min_date).dt.days
            pre_df_time_numeric = (pre_df[self.time_var] - min_date).dt.days
        else:
            treatment_time_numeric = treatment_data[self.time_var]
            control_time_numeric = control_data[self.time_var]
            pre_df_time_numeric = pre_df[self.time_var]

        # Fit linear trends
        treatment_slope, treatment_intercept, treatment_r, treatment_p, _ = stats.linregress(
            treatment_time_numeric, treatment_data[self.outcome_var])
        control_slope, control_intercept, control_r, control_p, _ = stats.linregress(
            control_time_numeric, control_data[self.outcome_var])

        # Statistical test for difference in trends
        # Regression-based test
        pre_df['time_treat_interact'] = pre_df_time_numeric * pre_df[self.treatment_var]
        X = sm.add_constant(pd.DataFrame({
            self.time_var: pre_df_time_numeric,
            self.treatment_var: pre_df[self.treatment_var],
            'time_treat_interact': pre_df['time_treat_interact']
        }))
        y = pre_df[self.outcome_var]
        trend_model = sm.OLS(y, X).fit()

        slope_diff = abs(treatment_slope - control_slope)
        trend_test_pvalue = trend_model.pvalues['time_treat_interact']

        results = {
            'treatment_slope': treatment_slope,
            'control_slope': control_slope,
            'slope_difference': slope_diff,
            'trend_test_pvalue': trend_test_pvalue,
            'treatment_data': treatment_data,
            'control_data': control_data
        }

        print(f"Pre-treatment trends:")
        print(f"  Treatment group slope: {treatment_slope:.3f} (p-value: {treatment_p:.3f})")
        print(f"  Control group slope: {control_slope:.3f} (p-value: {control_p:.3f})")
        print(f"  Difference in slopes: {slope_diff:.3f}")
        print(f"  Statistical test p-value: {trend_test_pvalue:.3f}")

        if trend_test_pvalue > 0.05:
            print("✓ PARALLEL TRENDS: ASSUMPTION SATISFIED (p > 0.05)")
            results['assumption_satisfied'] = True
        else:
            print("⚠ PARALLEL TRENDS: ASSUMPTION VIOLATED (p < 0.05)")
            results['assumption_satisfied'] = False

        if plot:
            self._plot_trends(treatment_data, control_data)

        self.assumption_checks['parallel_trends'] = results
        return results

    def check_common_shocks(self) -> Dict[str, Any]:
        """Check for differential shocks between treatment and control groups"""
        print("\n" + "="*60)
        print("ASSUMPTION CHECK: NO DIFFERENTIAL SHOCKS")
        print("="*60)

        # Calculate period-over-period changes
        df_sorted = self.data.sort_values([self.unit_var, self.time_var])
        df_sorted['outcome_change'] = df_sorted.groupby(self.unit_var)[self.outcome_var].pct_change()

        # Compare volatility between groups
        treatment_volatility = df_sorted[df_sorted[self.treatment_var] == 1]['outcome_change'].std()
        control_volatility = df_sorted[df_sorted[self.treatment_var] == 0]['outcome_change'].std()

        # Statistical test for equal variances
        treatment_changes = df_sorted[df_sorted[self.treatment_var] == 1]['outcome_change'].dropna()
        control_changes = df_sorted[df_sorted[self.treatment_var] == 0]['outcome_change'].dropna()

        # Levene's test for equal variances
        levene_stat, levene_p = stats.levene(treatment_changes, control_changes)

        results = {
            'treatment_volatility': treatment_volatility,
            'control_volatility': control_volatility,
            'volatility_ratio': treatment_volatility / control_volatility if control_volatility != 0 else np.inf,
            'levene_statistic': levene_stat,
            'levene_pvalue': levene_p
        }

        print(f"Volatility comparison:")
        print(f"  Treatment group volatility: {treatment_volatility:.4f}")
        print(f"  Control group volatility: {control_volatility:.4f}")
        print(f"  Ratio: {results['volatility_ratio']:.2f}")
        print(f"  Levene's test p-value: {levene_p:.3f}")

        if levene_p > 0.05:
            print("✓ COMMON SHOCKS: ASSUMPTION SATISFIED")
            results['assumption_satisfied'] = True
        else:
            print("⚠ COMMON SHOCKS: POTENTIAL DIFFERENTIAL SHOCKS DETECTED")
            results['assumption_satisfied'] = False

        self.assumption_checks['common_shocks'] = results
        return results

    def check_composition_stability(self) -> Dict[str, Any]:
        """Check that unit composition is stable over time"""
        print("\n" + "="*60)
        print("ASSUMPTION CHECK: STABLE COMPOSITION")
        print("="*60)

        # Check units per period
        units_per_period = self.data.groupby(self.time_var)[self.unit_var].nunique()
        total_units = self.data[self.unit_var].nunique()

        # Check for balanced panel
        expected_obs = total_units * self.data[self.time_var].nunique()
        actual_obs = len(self.data)

        results = {
            'total_units': total_units,
            'units_per_period_min': units_per_period.min(),
            'units_per_period_max': units_per_period.max(),
            'is_balanced': units_per_period.nunique() == 1 and units_per_period.iloc[0] == total_units,
            'expected_observations': expected_obs,
            'actual_observations': actual_obs,
            'completeness_rate': actual_obs / expected_obs
        }

        print(f"Composition analysis:")
        print(f"  Total units: {total_units}")
        print(f"  Units per period range: {units_per_period.min()} - {units_per_period.max()}")
        print(f"  Completeness rate: {results['completeness_rate']:.2%}")

        if results['is_balanced']:
            print("✓ COMPOSITION STABILITY: BALANCED PANEL")
            results['assumption_satisfied'] = True
        else:
            print("⚠ COMPOSITION STABILITY: UNBALANCED PANEL")
            results['assumption_satisfied'] = False

        self.assumption_checks['composition_stability'] = results
        return results

    def estimate_basic_did(self) -> Dict[str, Any]:
        """Basic 2x2 DiD estimation using group means"""
        print("\n" + "="*60)
        print("BASIC DiD ESTIMATION (2x2 Design)")
        print("="*60)

        # Calculate group means
        means = self.data.groupby([self.treatment_var, self.post_var])[self.outcome_var].mean()

        # Extract the four means
        control_pre = means[(0, 0)]
        control_post = means[(0, 1)]
        treatment_pre = means[(1, 0)]
        treatment_post = means[(1, 1)]

        # Calculate differences
        control_diff = control_post - control_pre
        treatment_diff = treatment_post - treatment_pre
        did_estimate = treatment_diff - control_diff

        results = {
            'control_pre': control_pre,
            'control_post': control_post,
            'treatment_pre': treatment_pre,
            'treatment_post': treatment_post,
            'control_diff': control_diff,
            'treatment_diff': treatment_diff,
            'did_estimate': did_estimate
        }

        print(f"Group means:")
        print(f"  Control - Pre: {control_pre:.2f}")
        print(f"  Control - Post: {control_post:.2f}")
        print(f"  Treatment - Pre: {treatment_pre:.2f}")
        print(f"  Treatment - Post: {treatment_post:.2f}")
        print(f"\nDifferences:")
        print(f"  Control group change: {control_diff:.2f}")
        print(f"  Treatment group change: {treatment_diff:.2f}")
        print(f"  DiD estimate: {did_estimate:.2f}")

        self.results_ = {'basic_did': results}
        return results

    def estimate_regression_did(self, robust_se: bool = True) -> Dict[str, Any]:
        """DiD estimation using regression framework"""
        print("\n" + "="*60)
        print("REGRESSION-BASED DiD ESTIMATION")
        print("="*60)

        # Prepare regression variables
        X = self.data[[self.treatment_var, self.post_var, 'did_term']].copy()
        X = sm.add_constant(X)
        y = self.data[self.outcome_var]

        # Fit model
        cov_type = 'HC3' if robust_se else 'nonrobust'
        model = sm.OLS(y, X).fit(cov_type=cov_type)

        # Extract key results
        did_coef = model.params['did_term']
        did_pvalue = model.pvalues['did_term']
        did_ci = model.conf_int().loc['did_term']

        results = {
            'model': model,
            'did_coefficient': did_coef,
            'did_pvalue': did_pvalue,
            'did_confidence_interval': did_ci,
            'r_squared': model.rsquared,
            'robust_se': robust_se
        }

        print(f"Regression results:")
        print(f"  Intercept (Control, Pre): {model.params['const']:.2f}")
        print(f"  Treatment Group Effect: {model.params[self.treatment_var]:.2f}")
        print(f"  Post Period Effect: {model.params[self.post_var]:.2f}")
        print(f"  DiD Effect: {did_coef:.2f}")
        print(f"  p-value: {did_pvalue:.4f}")
        print(f"  95% CI: [{did_ci[0]:.2f}, {did_ci[1]:.2f}]")
        print(f"  R²: {model.rsquared:.3f}")

        if did_pvalue < 0.05:
            print("✓ STATISTICALLY SIGNIFICANT at 5% level")
        else:
            print("⚠ NOT STATISTICALLY SIGNIFICANT at 5% level")

        if self.results_ is None:
            self.results_ = {}
        self.results_['regression_did'] = results
        return results

    def placebo_test(self, fake_treatment_time: Optional[Union[int, float, str]] = None) -> Dict[str, Any]:
        """Placebo test using fake treatment timing"""
        print("\n" + "="*60)
        print("ROBUSTNESS CHECK: PLACEBO TEST")
        print("="*60)

        if fake_treatment_time is None:
            # Default to halfway through pre-treatment period
            pre_periods = self.data[self.data[self.time_var] < self.treatment_start][self.time_var].unique()
            if self.is_date_based:
                # For dates, find the median date
                pre_periods_sorted = sorted(pre_periods)
                mid_idx = len(pre_periods_sorted) // 2
                fake_treatment_time = pre_periods_sorted[mid_idx]
            else:
                fake_treatment_time = np.median(pre_periods)

        # Use only pre-treatment data
        placebo_df = self.data[self.data[self.time_var] < self.treatment_start].copy()

        # Create fake treatment variables
        placebo_df['fake_post'] = (placebo_df[self.time_var] >= fake_treatment_time).astype(int)
        placebo_df['fake_did'] = placebo_df[self.treatment_var] * placebo_df['fake_post']

        # Run regression
        X = sm.add_constant(placebo_df[[self.treatment_var, 'fake_post', 'fake_did']])
        y = placebo_df[self.outcome_var]
        model = sm.OLS(y, X).fit()

        placebo_coef = model.params['fake_did']
        placebo_pvalue = model.pvalues['fake_did']

        results = {
            'fake_treatment_time': fake_treatment_time,
            'placebo_coefficient': placebo_coef,
            'placebo_pvalue': placebo_pvalue,
            'placebo_model': model
        }

        print(f"Placebo test (fake treatment at {fake_treatment_time}):")
        print(f"  Placebo DiD coefficient: {placebo_coef:.2f}")
        print(f"  p-value: {placebo_pvalue:.4f}")

        if placebo_pvalue > 0.05:
            print("✓ PLACEBO TEST PASSED (no false positive)")
            results['test_passed'] = True
        else:
            print("⚠ PLACEBO TEST FAILED (possible pre-existing differences)")
            results['test_passed'] = False

        if self.results_ is None:
            self.results_ = {}
        self.results_['placebo_test'] = results
        return results

    def run_all_checks(self, control_vars: Optional[List[str]] = None, plot: bool = True) -> Dict[str, Any]:
        """Run all assumption checks and estimations"""
        print("RUNNING COMPREHENSIVE DiD ANALYSIS")
        print("="*60)

        # Assumption checks
        self.check_parallel_trends(plot=plot)
        self.check_common_shocks()
        self.check_composition_stability()

        # Estimations
        self.estimate_basic_did()
        self.estimate_regression_did()

        # Robustness checks
        self.placebo_test()

        # Summary
        self.print_summary()

        return self.results_

    def print_summary(self) -> None:
        """Print a summary of all results"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)

        # Assumption checks summary
        print("Assumption Checks:")
        for check_name, check_results in self.assumption_checks.items():
            status = "✓ PASSED" if check_results.get('assumption_satisfied', False) else "⚠ FAILED"
            print(f"  {check_name.replace('_', ' ').title()}: {status}")

        # DiD estimates summary
        print("\nDiD Estimates:")
        if self.results_ and 'basic_did' in self.results_:
            print(f"  Basic DiD: {self.results_['basic_did']['did_estimate']:.2f}")
        if self.results_ and 'regression_did' in self.results_:
            coef = self.results_['regression_did']['did_coefficient']
            pval = self.results_['regression_did']['did_pvalue']
            print(f"  Regression DiD: {coef:.2f} (p={pval:.3f})")

        # Robustness checks summary
        print("\nRobustness Checks:")
        if self.results_ and 'placebo_test' in self.results_:
            status = "✓ PASSED" if self.results_['placebo_test']['test_passed'] else "⚠ FAILED"
            print(f"  Placebo Test: {status}")

    def _plot_trends(self, treatment_data: pd.DataFrame, control_data: pd.DataFrame) -> None:
        """Internal method to plot trends"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(treatment_data[self.time_var], treatment_data[self.outcome_var],
                'b-o', label='Treatment Group', linewidth=2, markersize=6)
        ax.plot(control_data[self.time_var], control_data[self.outcome_var],
                'r-s', label='Control Group', linewidth=2, markersize=6)

        if self.is_date_based:
            ax.axvline(x=self.treatment_start, color='gray', linestyle='--',
                      alpha=0.7, label='Treatment Start')
        else:
            ax.axvline(x=self.treatment_start - 0.5, color='gray', linestyle='--',
                      alpha=0.7, label='Treatment Start')

        ax.set_xlabel(self.time_var.title())
        ax.set_ylabel(self.outcome_var.title())
        ax.set_title('Parallel Trends Check: Pre-treatment Period')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
