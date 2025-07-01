import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DifferenceInDifferences:
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
    
    def __init__(self, data, outcome_var, unit_var, time_var, treatment_var, post_var, treatment_start):
        self.data = data.copy()
        self.outcome_var = outcome_var
        self.unit_var = unit_var
        self.time_var = time_var
        self.treatment_var = treatment_var
        self.post_var = post_var

        # Handle different treatment_start formats
        self.treatment_start = self._parse_treatment_start(treatment_start)
        self.treatment_start_original = treatment_start  # Keep original for reference

        # Determine if we're working with dates
        self.is_date_based = isinstance(treatment_start, str)

        # Convert time variable to datetime if working with dates
        if self.is_date_based:
            self.data[self.time_var] = pd.to_datetime(self.data[self.time_var])

        # Create DiD interaction term
        self.data['did_term'] = self.data[treatment_var] * self.data[post_var]

        # Store results
        self.results = {}
        self.assumption_checks = {}

        # Validate data
        self._validate_data()

    def _parse_treatment_start(self, treatment_start):
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

    def _validate_data(self):
        """Validate input data and variables"""
        required_cols = [self.outcome_var, self.unit_var, self.time_var, 
                        self.treatment_var, self.post_var]
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check binary variables
        for var in [self.treatment_var, self.post_var]:
            unique_vals = sorted(self.data[var].unique())
            if unique_vals != [0, 1]:
                raise ValueError(f"{var} must be binary (0, 1), found: {unique_vals}")
                
        print(f"✓ Data validation passed")
        print(f"  - Units: {self.data[self.unit_var].nunique()}")
        print(f"  - Time periods: {self.data[self.time_var].nunique()}")
        print(f"  - Treatment units: {self.data[self.data[self.treatment_var]==1][self.unit_var].nunique()}")
        print(f"  - Control units: {self.data[self.data[self.treatment_var]==0][self.unit_var].nunique()}")
    
    def check_parallel_trends(self, pre_period_end=None, plot=True):
        """
        Check parallel trends assumption - critical for DiD validity
        
        Parameters:
        -----------
        pre_period_end : int or float, optional
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
    
    def check_common_shocks(self):
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
    
    def check_composition_stability(self):
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
    
    def estimate_basic_did(self):
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
        
        self.results['basic_did'] = results
        return results
    
    def estimate_regression_did(self, robust_se=True):
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
            
        self.results['regression_did'] = results
        return results
    
    def estimate_did_with_controls(self, control_vars=None, unit_fe=True, time_fe=False):
        """DiD with additional control variables and fixed effects"""
        print("\n" + "="*60)
        print("DiD WITH CONTROLS AND FIXED EFFECTS")
        print("="*60)

        # Start with basic DiD variables
        X_vars = [self.treatment_var, self.post_var, 'did_term']

        # Add control variables
        if control_vars:
            available_controls = [var for var in control_vars if var in self.data.columns]
            X_vars.extend(available_controls)
            print(f"Added control variables: {available_controls}")

        # Start with the base data and ensure numeric types
        X_data = self.data[X_vars].copy()

        # Convert all columns to numeric, ensuring consistent data types
        for col in X_data.columns:
            X_data[col] = pd.to_numeric(X_data[col], errors='coerce')

        # Add unit fixed effects
        if unit_fe:
            unit_dummies = pd.get_dummies(self.data[self.unit_var], prefix='unit', drop_first=True)
            # Ensure dummy variables are integers (0/1)
            unit_dummies = unit_dummies.astype(int)
            X_data = pd.concat([X_data, unit_dummies], axis=1)
            print(f"Added unit fixed effects ({unit_dummies.shape[1]} dummies)")

        # Add time fixed effects
        if time_fe:
            time_dummies = pd.get_dummies(self.data[self.time_var], prefix='time', drop_first=True)
            # Ensure dummy variables are integers (0/1)
            time_dummies = time_dummies.astype(int)
            X_data = pd.concat([X_data, time_dummies], axis=1)
            print(f"Added time fixed effects ({time_dummies.shape[1]} dummies)")

        # Add constant and fit model
        X_data = sm.add_constant(X_data)

        # Ensure all data is numeric before passing to statsmodels
        X_data = X_data.astype(float)

        y = self.data[self.outcome_var]
        # Ensure y is also numeric
        y = pd.to_numeric(y, errors='coerce')

        model = sm.OLS(y, X_data).fit(cov_type='HC3')
        
        # Extract results
        did_coef = model.params['did_term']
        did_pvalue = model.pvalues['did_term']
        did_ci = model.conf_int().loc['did_term']
        
        results = {
            'model': model,
            'did_coefficient': did_coef,
            'did_pvalue': did_pvalue,
            'did_confidence_interval': did_ci,
            'r_squared': model.rsquared,
            'control_vars': control_vars,
            'unit_fe': unit_fe,
            'time_fe': time_fe
        }
        
        print(f"Results with controls:")
        print(f"  DiD Effect: {did_coef:.2f}")
        print(f"  p-value: {did_pvalue:.4f}")
        print(f"  95% CI: [{did_ci[0]:.2f}, {did_ci[1]:.2f}]")
        print(f"  R²: {model.rsquared:.3f}")
        
        self.results['controlled_did'] = results
        return results
    
    def placebo_test(self, fake_treatment_time=None):
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
            
        self.results['placebo_test'] = results
        return results
    
    def exclusion_test(self):
        """Test robustness by excluding one unit at a time"""
        print("\n" + "="*60)
        print("ROBUSTNESS CHECK: UNIT EXCLUSION TEST")
        print("="*60)
        
        units = self.data[self.unit_var].unique()
        did_estimates = []
        exclusion_results = {}
        
        for unit in units:
            df_subset = self.data[self.data[self.unit_var] != unit].copy()
            
            # Re-run basic regression
            X = sm.add_constant(df_subset[[self.treatment_var, self.post_var, 'did_term']])
            y = df_subset[self.outcome_var]
            model = sm.OLS(y, X).fit()
            
            did_coef = model.params['did_term']
            did_estimates.append(did_coef)
            exclusion_results[unit] = did_coef
            
            print(f"  Excluding {unit}: DiD = {did_coef:.2f}")
        
        results = {
            'exclusion_estimates': exclusion_results,
            'estimate_range': (min(did_estimates), max(did_estimates)),
            'estimate_std': np.std(did_estimates),
            'estimate_mean': np.mean(did_estimates)
        }
        
        print(f"\nExclusion test summary:")
        print(f"  Range of estimates: {min(did_estimates):.2f} to {max(did_estimates):.2f}")
        print(f"  Standard deviation: {np.std(did_estimates):.2f}")
        print(f"  Mean estimate: {np.mean(did_estimates):.2f}")
        
        self.results['exclusion_test'] = results
        return results
    
    def run_all_checks(self, control_vars=None, plot=True):
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
        self.estimate_did_with_controls(control_vars=control_vars)
        
        # Robustness checks
        self.placebo_test()
        self.exclusion_test()
        
        # Summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
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
        if 'basic_did' in self.results:
            print(f"  Basic DiD: {self.results['basic_did']['did_estimate']:.2f}")
        if 'regression_did' in self.results:
            coef = self.results['regression_did']['did_coefficient']
            pval = self.results['regression_did']['did_pvalue']
            print(f"  Regression DiD: {coef:.2f} (p={pval:.3f})")
        if 'controlled_did' in self.results:
            coef = self.results['controlled_did']['did_coefficient']
            pval = self.results['controlled_did']['did_pvalue']
            print(f"  Controlled DiD: {coef:.2f} (p={pval:.3f})")
        
        # Robustness checks summary
        print("\nRobustness Checks:")
        if 'placebo_test' in self.results:
            status = "✓ PASSED" if self.results['placebo_test']['test_passed'] else "⚠ FAILED"
            print(f"  Placebo Test: {status}")
        if 'exclusion_test' in self.results:
            range_est = self.results['exclusion_test']['estimate_range']
            print(f"  Exclusion Test Range: {range_est[0]:.2f} to {range_est[1]:.2f}")
    
    def _plot_trends(self, treatment_data, control_data):
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


class PropensityScoreMatching:
    """
    A comprehensive class for Propensity Score Matching analysis with multiple matching methods,
    assumption checking, and robustness tests.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing the variables for analysis
    outcome_var : str
        Name of the outcome variable column
    treatment_var : str
        Name of the treatment indicator column (1 = treatment, 0 = control)
    covariates : list
        List of covariate column names to use for propensity score estimation
    unit_id : str, optional
        Name of unit identifier column (for tracking matched pairs)
    """
    
    def __init__(self, data, outcome_var, treatment_var, covariates, unit_id=None):
        self.data = data.copy()
        self.outcome_var = outcome_var
        self.treatment_var = treatment_var
        self.covariates = covariates
        self.unit_id = unit_id if unit_id else 'unit_id'
        
        # Create unit_id if not provided
        if self.unit_id not in self.data.columns:
            self.data[self.unit_id] = range(len(self.data))
        
        # Storage for results
        self.propensity_scores = None
        self.matched_data = None
        self.results = {}
        self.assumption_checks = {}
        
        # Validate data
        self._validate_data()
        
    def _validate_data(self):
        """Validate input data and variables"""
        required_cols = [self.outcome_var, self.treatment_var] + self.covariates
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check treatment variable is binary
        unique_vals = sorted(self.data[self.treatment_var].unique())
        if unique_vals != [0, 1]:
            raise ValueError(f"Treatment variable must be binary (0, 1), found: {unique_vals}")
            
        # Check for missing values in key variables
        missing_data = self.data[required_cols].isnull().sum()
        if missing_data.any():
            print("⚠️  Missing values detected:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"  {col}: {count} missing values")
                
        print(f"✅ Data validation passed")
        print(f"  - Total observations: {len(self.data):,}")
        print(f"  - Treatment units: {(self.data[self.treatment_var] == 1).sum():,}")
        print(f"  - Control units: {(self.data[self.treatment_var] == 0).sum():,}")
        print(f"  - Covariates: {len(self.covariates)}")
    
    def estimate_propensity_scores(self, method='logistic', include_interactions=False, 
                                  polynomial_degree=1):
        """
        Estimate propensity scores using various methods
        
        Parameters:
        -----------
        method : str, default 'logistic'
            Method for propensity score estimation ('logistic', 'random_forest')
        include_interactions : bool, default False
            Whether to include interaction terms between covariates
        polynomial_degree : int, default 1
            Degree of polynomial terms to include
        """
        print("\n" + "="*60)
        print("PROPENSITY SCORE ESTIMATION")
        print("="*60)
        
        # Prepare feature matrix
        X = self.data[self.covariates].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Add polynomial terms
        if polynomial_degree > 1:
            for degree in range(2, polynomial_degree + 1):
                for col in self.covariates:
                    X[f"{col}_poly_{degree}"] = X[col] ** degree
        
        # Add interaction terms
        if include_interactions:
            for i, col1 in enumerate(self.covariates):
                for col2 in self.covariates[i+1:]:
                    X[f"{col1}_x_{col2}"] = X[col1] * X[col2]
        
        y = self.data[self.treatment_var]
        
        # Estimate propensity scores
        if method == 'logistic':
            # Standardize features for logistic regression
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_scaled, y)
            propensity_scores = model.predict_proba(X_scaled)[:, 1]
            
            self.propensity_model = model
            self.scaler = scaler
            
        elif method == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X, y)
            propensity_scores = model.predict_proba(X)[:, 1]
            
            self.propensity_model = model
            
        else:
            raise ValueError("Method must be 'logistic' or 'random_forest'")
        
        # Store results
        self.propensity_scores = propensity_scores
        self.data['propensity_score'] = propensity_scores
        
        # Model diagnostics
        auc_score = roc_auc_score(y, propensity_scores)
        
        print(f"Propensity score estimation complete:")
        print(f"  Method: {method}")
        print(f"  Features used: {X.shape[1]}")
        print(f"  AUC: {auc_score:.3f}")
        print(f"  Propensity score range: [{propensity_scores.min():.3f}, {propensity_scores.max():.3f}]")
        
        # Check for extreme propensity scores
        extreme_low = (propensity_scores < 0.1).sum()
        extreme_high = (propensity_scores > 0.9).sum()
        
        if extreme_low > 0 or extreme_high > 0:
            print(f"⚠️  Extreme propensity scores detected:")
            print(f"  < 0.1: {extreme_low} observations")
            print(f"  > 0.9: {extreme_high} observations")
        
        self.results['propensity_estimation'] = {
            'method': method,
            'auc': auc_score,
            'extreme_scores': extreme_low + extreme_high,
            'feature_count': X.shape[1]
        }
        
        return propensity_scores
    
    def check_overlap(self, plot=True):
        """Check common support/overlap assumption"""
        print("\n" + "="*60)
        print("ASSUMPTION CHECK: COMMON SUPPORT/OVERLAP")
        print("="*60)
        
        if self.propensity_scores is None:
            raise ValueError("Must estimate propensity scores first")
        
        treatment_ps = self.propensity_scores[self.data[self.treatment_var] == 1]
        control_ps = self.propensity_scores[self.data[self.treatment_var] == 0]
        
        # Calculate overlap statistics
        treatment_range = (treatment_ps.min(), treatment_ps.max())
        control_range = (control_ps.min(), control_ps.max())
        
        overlap_min = max(treatment_range[0], control_range[0])
        overlap_max = min(treatment_range[1], control_range[1])
        overlap_range = (overlap_min, overlap_max)
        
        # Check if there's meaningful overlap
        overlap_exists = overlap_min < overlap_max
        
        # Calculate percentage of observations in overlap region
        in_overlap = ((self.propensity_scores >= overlap_min) & 
                     (self.propensity_scores <= overlap_max))
        overlap_pct = in_overlap.mean() * 100
        
        # Kolmogorov-Smirnov test for distribution differences
        ks_stat, ks_pvalue = stats.ks_2samp(treatment_ps, control_ps)
        
        results = {
            'treatment_range': treatment_range,
            'control_range': control_range,
            'overlap_range': overlap_range,
            'overlap_exists': overlap_exists,
            'overlap_percentage': overlap_pct,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue
        }
        
        print(f"Overlap analysis:")
        print(f"  Treatment range: [{treatment_range[0]:.3f}, {treatment_range[1]:.3f}]")
        print(f"  Control range: [{control_range[0]:.3f}, {control_range[1]:.3f}]")
        print(f"  Overlap range: [{overlap_range[0]:.3f}, {overlap_range[1]:.3f}]")
        print(f"  Observations in overlap: {overlap_pct:.1f}%")
        print(f"  KS test p-value: {ks_pvalue:.3f}")
        
        if overlap_exists and overlap_pct > 80:
            print("✅ COMMON SUPPORT: GOOD OVERLAP")
            results['assumption_satisfied'] = True
        elif overlap_exists and overlap_pct > 50:
            print("⚠️ COMMON SUPPORT: MODERATE OVERLAP")
            results['assumption_satisfied'] = False
        else:
            print("❌ COMMON SUPPORT: POOR OVERLAP")
            results['assumption_satisfied'] = False
        
        if plot:
            self._plot_overlap(treatment_ps, control_ps)
        
        self.assumption_checks['overlap'] = results
        return results
    
    def check_balance_before_matching(self):
        """Check covariate balance before matching"""
        print("\n" + "="*60)
        print("COVARIATE BALANCE: BEFORE MATCHING")
        print("="*60)
        
        balance_results = {}
        
        for covar in self.covariates:
            treatment_vals = self.data[self.data[self.treatment_var] == 1][covar]
            control_vals = self.data[self.data[self.treatment_var] == 0][covar]
            
            # Calculate standardized difference
            pooled_std = np.sqrt(((treatment_vals.var() + control_vals.var()) / 2))
            std_diff = (treatment_vals.mean() - control_vals.mean()) / pooled_std
            
            # T-test
            t_stat, t_pvalue = stats.ttest_ind(treatment_vals, control_vals)
            
            balance_results[covar] = {
                'treatment_mean': treatment_vals.mean(),
                'control_mean': control_vals.mean(),
                'standardized_diff': std_diff,
                't_statistic': t_stat,
                't_pvalue': t_pvalue,
                'balanced': abs(std_diff) < 0.1
            }
            
            print(f"{covar:20s}: Std diff = {std_diff:6.3f}, p-value = {t_pvalue:.3f}")
        
        # Overall balance assessment
        large_imbalances = sum(1 for v in balance_results.values() if abs(v['standardized_diff']) > 0.25)
        
        print(f"\nBalance summary:")
        print(f"  Variables with |std diff| > 0.25: {large_imbalances}/{len(self.covariates)}")
        
        if large_imbalances == 0:
            print("✅ GOOD BALANCE before matching")
        elif large_imbalances <= len(self.covariates) * 0.3:
            print("⚠️ MODERATE IMBALANCE before matching")
        else:
            print("❌ POOR BALANCE before matching")
        
        self.assumption_checks['balance_before'] = balance_results
        return balance_results
    
    def perform_matching(self, method='nearest', caliper=None, replacement=False, 
                        ratio=1, random_state=42):
        """
        Perform propensity score matching
        
        Parameters:
        -----------
        method : str, default 'nearest'
            Matching method ('nearest', 'caliper', 'stratification')
        caliper : float, optional
            Maximum distance for caliper matching (e.g., 0.01)
        replacement : bool, default False
            Whether to allow replacement in matching
        ratio : int, default 1
            Number of control units to match to each treatment unit
        """
        print("\n" + "="*60)
        print(f"PROPENSITY SCORE MATCHING: {method.upper()}")
        print("="*60)
        
        if self.propensity_scores is None:
            raise ValueError("Must estimate propensity scores first")
        
        np.random.seed(random_state)
        
        treatment_data = self.data[self.data[self.treatment_var] == 1].copy()
        control_data = self.data[self.data[self.treatment_var] == 0].copy()
        
        matched_pairs = []
        
        if method == 'nearest':
            matched_pairs = self._nearest_neighbor_matching(
                treatment_data, control_data, caliper, replacement, ratio)
                
        elif method == 'caliper':
            if caliper is None:
                caliper = 0.01  # Default caliper
            matched_pairs = self._caliper_matching(
                treatment_data, control_data, caliper, replacement, ratio)
                
        elif method == 'stratification':
            matched_pairs = self._stratification_matching(
                treatment_data, control_data, ratio)
        else:
            raise ValueError("Method must be 'nearest', 'caliper', or 'stratification'")
        
        # Create matched dataset
        if matched_pairs:
            self.matched_data = pd.concat(matched_pairs, ignore_index=True)
            
            print(f"Matching complete:")
            print(f"  Treatment units matched: {(self.matched_data[self.treatment_var] == 1).sum()}")
            print(f"  Control units matched: {(self.matched_data[self.treatment_var] == 0).sum()}")
            print(f"  Total matched observations: {len(self.matched_data)}")
            
            # Calculate match quality
            self._assess_match_quality()
            
        else:
            print("❌ No successful matches found")
            self.matched_data = None
            
        return self.matched_data
    
    def _nearest_neighbor_matching(self, treatment_data, control_data, caliper, replacement, ratio):
        """Perform nearest neighbor matching"""
        matched_pairs = []
        used_controls = set()
        
        for _, treatment_unit in treatment_data.iterrows():
            treatment_ps = treatment_unit['propensity_score']
            
            # Calculate distances to all control units
            available_controls = control_data
            if not replacement:
                available_controls = control_data[~control_data[self.unit_id].isin(used_controls)]
            
            if len(available_controls) == 0:
                continue
                
            distances = np.abs(available_controls['propensity_score'] - treatment_ps)
            
            # Apply caliper if specified
            if caliper:
                valid_matches = distances <= caliper
                if not valid_matches.any():
                    continue
                available_controls = available_controls[valid_matches]
                distances = distances[valid_matches]
            
            # Find closest matches
            n_matches = min(ratio, len(available_controls))
            closest_indices = distances.nsmallest(n_matches).index
            
            # Add treatment unit
            matched_pairs.append(treatment_unit.to_frame().T)
            
            # Add matched control units
            for idx in closest_indices:
                control_unit = available_controls.loc[idx]
                matched_pairs.append(control_unit.to_frame().T)
                
                if not replacement:
                    used_controls.add(control_unit[self.unit_id])
        
        return matched_pairs
    
    def _caliper_matching(self, treatment_data, control_data, caliper, replacement, ratio):
        """Perform caliper matching"""
        return self._nearest_neighbor_matching(treatment_data, control_data, caliper, replacement, ratio)
    
    def _stratification_matching(self, treatment_data, control_data, ratio):
        """Perform stratification matching"""
        matched_pairs = []
        
        # Create propensity score strata
        n_strata = 5
        ps_min = self.propensity_scores.min()
        ps_max = self.propensity_scores.max()
        strata_bounds = np.linspace(ps_min, ps_max, n_strata + 1)
        
        for i in range(n_strata):
            lower_bound = strata_bounds[i]
            upper_bound = strata_bounds[i + 1]
            
            # Get units in this stratum
            in_stratum = ((self.data['propensity_score'] >= lower_bound) & 
                         (self.data['propensity_score'] <= upper_bound))
            
            stratum_treatment = treatment_data[in_stratum.loc[treatment_data.index]]
            stratum_control = control_data[in_stratum.loc[control_data.index]]
            
            if len(stratum_treatment) == 0 or len(stratum_control) == 0:
                continue
            
            # Sample controls within stratum
            n_controls_needed = len(stratum_treatment) * ratio
            n_controls_available = len(stratum_control)
            
            if n_controls_available >= n_controls_needed:
                sampled_controls = stratum_control.sample(n_controls_needed, random_state=42)
            else:
                sampled_controls = stratum_control
            
            matched_pairs.extend([stratum_treatment, sampled_controls])
        
        return matched_pairs
    
    def _assess_match_quality(self):
        """Assess quality of matching"""
        if self.matched_data is None:
            return
        
        print(f"\nMatch quality assessment:")
        
        # Propensity score balance
        matched_treatment_ps = self.matched_data[self.matched_data[self.treatment_var] == 1]['propensity_score']
        matched_control_ps = self.matched_data[self.matched_data[self.treatment_var] == 0]['propensity_score']
        
        ps_diff = abs(matched_treatment_ps.mean() - matched_control_ps.mean())
        print(f"  Propensity score difference: {ps_diff:.4f}")
        
        # Covariate balance after matching
        self.check_balance_after_matching()
    
    def check_balance_after_matching(self):
        """Check covariate balance after matching"""
        print("\n" + "="*60)
        print("COVARIATE BALANCE: AFTER MATCHING")
        print("="*60)
        
        if self.matched_data is None:
            print("No matched data available")
            return
        
        balance_results = {}
        
        for covar in self.covariates:
            treatment_vals = self.matched_data[self.matched_data[self.treatment_var] == 1][covar]
            control_vals = self.matched_data[self.matched_data[self.treatment_var] == 0][covar]
            
            # Calculate standardized difference
            pooled_std = np.sqrt(((treatment_vals.var() + control_vals.var()) / 2))
            std_diff = (treatment_vals.mean() - control_vals.mean()) / pooled_std
            
            # T-test
            t_stat, t_pvalue = stats.ttest_ind(treatment_vals, control_vals)
            
            balance_results[covar] = {
                'treatment_mean': treatment_vals.mean(),
                'control_mean': control_vals.mean(),
                'standardized_diff': std_diff,
                't_statistic': t_stat,
                't_pvalue': t_pvalue,
                'balanced': abs(std_diff) < 0.1
            }
            
            print(f"{covar:20s}: Std diff = {std_diff:6.3f}, p-value = {t_pvalue:.3f}")
        
        # Overall balance assessment
        well_balanced = sum(1 for v in balance_results.values() if abs(v['standardized_diff']) < 0.1)
        
        print(f"\nBalance summary:")
        print(f"  Well-balanced variables (|std diff| < 0.1): {well_balanced}/{len(self.covariates)}")
        
        if well_balanced == len(self.covariates):
            print("✅ EXCELLENT BALANCE after matching")
        elif well_balanced >= len(self.covariates) * 0.8:
            print("✅ GOOD BALANCE after matching")
        elif well_balanced >= len(self.covariates) * 0.6:
            print("⚠️ MODERATE BALANCE after matching")
        else:
            print("❌ POOR BALANCE after matching")
        
        self.assumption_checks['balance_after'] = balance_results
        return balance_results
    
    def estimate_treatment_effect(self, method='simple_difference'):
        """
        Estimate treatment effect on matched sample
        
        Parameters:
        -----------
        method : str, default 'simple_difference'
            Method for effect estimation ('simple_difference', 'regression_adjustment')
        """
        print("\n" + "="*60)
        print("TREATMENT EFFECT ESTIMATION")
        print("="*60)
        
        if self.matched_data is None:
            raise ValueError("Must perform matching first")
        
        treatment_outcomes = self.matched_data[self.matched_data[self.treatment_var] == 1][self.outcome_var]
        control_outcomes = self.matched_data[self.matched_data[self.treatment_var] == 0][self.outcome_var]
        
        if method == 'simple_difference':
            # Simple difference in means
            ate = treatment_outcomes.mean() - control_outcomes.mean()
            
            # Standard error for difference in means
            se = np.sqrt(treatment_outcomes.var()/len(treatment_outcomes) + 
                        control_outcomes.var()/len(control_outcomes))
            
            # T-test
            t_stat, p_value = stats.ttest_ind(treatment_outcomes, control_outcomes)
            
        elif method == 'regression_adjustment':
            # Regression on matched sample with covariates
            X = sm.add_constant(self.matched_data[[self.treatment_var] + self.covariates])
            y = self.matched_data[self.outcome_var]
            
            model = sm.OLS(y, X).fit(cov_type='HC3')
            ate = model.params[self.treatment_var]
            se = model.bse[self.treatment_var]
            t_stat = model.tvalues[self.treatment_var]
            p_value = model.pvalues[self.treatment_var]
            
        # Confidence interval
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        # Effect size (as percentage of control mean)
        control_mean = control_outcomes.mean()
        effect_pct = (ate / control_mean) * 100 if control_mean != 0 else 0
        
        results = {
            'ate': ate,
            'standard_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'effect_percentage': effect_pct,
            'treatment_mean': treatment_outcomes.mean(),
            'control_mean': control_mean,
            'method': method
        }
        
        print(f"Treatment effect estimation:")
        print(f"  Method: {method}")
        print(f"  ATE: {ate:.3f}")
        print(f"  Standard Error: {se:.3f}")
        print(f"  p-value: {p_value:.3f}")
        print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"  Effect size: {effect_pct:.1f}% of control mean")
        
        if p_value < 0.05:
            print("✅ STATISTICALLY SIGNIFICANT at 5% level")
        else:
            print("⚠️ NOT STATISTICALLY SIGNIFICANT at 5% level")
        
        self.results['treatment_effect'] = results
        return results
    
    def sensitivity_analysis(self, hidden_bias_range=(1.0, 2.0, 0.1)):
        """
        Rosenbaum sensitivity analysis for hidden bias
        
        Parameters:
        -----------
        hidden_bias_range : tuple
            (min_gamma, max_gamma, step) for sensitivity analysis
        """
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSIS: HIDDEN BIAS")
        print("="*60)
        
        if self.matched_data is None:
            raise ValueError("Must perform matching first")
        
        print("Note: This is a simplified sensitivity analysis.")
        print("Full Rosenbaum bounds require specialized implementation.")
        
        # Simple approach: test different levels of unobserved confounding
        gammas = np.arange(*hidden_bias_range)
        sensitivity_results = []
        
        baseline_effect = self.results.get('treatment_effect', {}).get('ate', 0)
        
        for gamma in gammas:
            # Simulate effect of unobserved confounder
            # This is a simplified version - full implementation would be more complex
            adjusted_effect = baseline_effect / gamma
            
            sensitivity_results.append({
                'gamma': gamma,
                'adjusted_effect': adjusted_effect,
                'effect_reduction': (1 - adjusted_effect/baseline_effect) * 100 if baseline_effect != 0 else 0
            })
        
        print(f"Sensitivity to hidden bias (Γ = degree of hidden bias):")
        for result in sensitivity_results[:6]:  # Show first few
            print(f"  Γ = {result['gamma']:.1f}: Effect = {result['adjusted_effect']:.3f} "
                  f"({result['effect_reduction']:+.1f}%)")
        
        self.results['sensitivity_analysis'] = sensitivity_results
        return sensitivity_results
    
    def placebo_test(self, placebo_outcome=None):
        """
        Placebo test using alternative outcome that shouldn't be affected by treatment
        
        Parameters:
        -----------
        placebo_outcome : str, optional
            Name of placebo outcome variable. If None, uses a permuted version of treatment
        """
        print("\n" + "="*60)
        print("ROBUSTNESS CHECK: PLACEBO TEST")
        print("="*60)
        
        if self.matched_data is None:
            raise ValueError("Must perform matching first")
        
        if placebo_outcome and placebo_outcome in self.matched_data.columns:
            # Use alternative outcome
            original_outcome = self.outcome_var
            self.outcome_var = placebo_outcome
            
            placebo_results = self.estimate_treatment_effect()
            
            # Restore original outcome
            self.outcome_var = original_outcome
            
            print(f"Placebo test with {placebo_outcome}:")
            print(f"  Should find no effect if matching is valid")
            
        else:
            # Permutation test - randomly assign treatment
            np.random.seed(42)
            placebo_data = self.matched_data.copy()
            placebo_data[self.treatment_var] = np.random.permutation(placebo_data[self.treatment_var])
            
            treatment_outcomes = placebo_data[placebo_data[self.treatment_var] == 1][self.outcome_var]
            control_outcomes = placebo_data[placebo_data[self.treatment_var] == 0][self.outcome_var]
            
            placebo_effect = treatment_outcomes.mean() - control_outcomes.mean()
            _, placebo_pvalue = stats.ttest_ind(treatment_outcomes, control_outcomes)
            
            placebo_results = {
                'placebo_effect': placebo_effect,
                'placebo_pvalue': placebo_pvalue,
                'test_type': 'permutation'
            }
            
            print(f"Placebo test (permuted treatment):")
            print(f"  Placebo effect: {placebo_effect:.3f}")
            print(f"  p-value: {placebo_pvalue:.3f}")
        
        if placebo_results.get('placebo_pvalue', 1) > 0.05:
            print("✅ PLACEBO TEST PASSED")
            placebo_results['test_passed'] = True
        else:
            print("⚠️ PLACEBO TEST FAILED - possible unobserved confounding")
            placebo_results['test_passed'] = False
        
        self.results['placebo_test'] = placebo_results
        return placebo_results
    
    def run_full_analysis(self, matching_method='nearest', caliper=None, plot=True):
        """Run complete PSM analysis pipeline"""
        print("RUNNING COMPREHENSIVE PSM ANALYSIS")
        print("="*60)
        
        # Step 1: Estimate propensity scores
        self.estimate_propensity_scores()
        
        # Step 2: Check assumptions
        self.check_overlap(plot=plot)
        self.check_balance_before_matching()
        
        # Step 3: Perform matching
        self.perform_matching(method=matching_method, caliper=caliper)
        
        if self.matched_data is not None:
            # Step 4: Check balance after matching
            self.check_balance_after_matching()
            
            # Step 5: Estimate treatment effect
            self.estimate_treatment_effect()
            
            # Step 6: Robustness checks
            self.sensitivity_analysis()
            self.placebo_test()
            
            # Step 7: Summary
            self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*60)
        print("PSM ANALYSIS SUMMARY")
        print("="*60)
        
        # Propensity score model
        if 'propensity_estimation' in self.results:
            ps_results = self.results['propensity_estimation']
            print(f"Propensity Score Model:")
            print(f"  Method: {ps_results['method']}")
            print(f"  AUC: {ps_results['auc']:.3f}")
            print(f"  Extreme scores: {ps_results['extreme_scores']}")
        
        # Assumption checks
        print(f"\nAssumption Checks:")
        if 'overlap' in self.assumption_checks:
            overlap_ok = self.assumption_checks['overlap']['assumption_satisfied']
            print(f"  Common Support: {'✅ SATISFIED' if overlap_ok else '⚠️ VIOLATED'}")
        
        # Balance improvement
        if 'balance_before' in self.assumption_checks and 'balance_after' in self.assumption_checks:
            before_imbalanced = sum(1 for v in self.assumption_checks['balance_before'].values() 
                                  if abs(v['standardized_diff']) > 0.1)
            after_imbalanced = sum(1 for v in self.assumption_checks['balance_after'].values() 
                                 if abs(v['standardized_diff']) > 0.1)
            print(f"  Balance Improvement: {before_imbalanced} → {after_imbalanced} imbalanced variables")
        
        # Treatment effect
        if 'treatment_effect' in self.results:
            effect = self.results['treatment_effect']
            print(f"\nTreatment Effect:")
            print(f"  ATE: {effect['ate']:.3f} (p={effect['p_value']:.3f})")
            print(f"  95% CI: [{effect['ci_lower']:.3f}, {effect['ci_upper']:.3f}]")
            print(f"  Effect size: {effect['effect_percentage']:.1f}% of control mean")
        
        # Robustness
        if 'placebo_test' in self.results:
            placebo_passed = self.results['placebo_test']['test_passed']
            print(f"\nRobustness Checks:")
            print(f"  Placebo Test: {'✅ PASSED' if placebo_passed else '⚠️ FAILED'}")
    
    def _plot_overlap(self, treatment_ps, control_ps):
        """Plot propensity score distributions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(control_ps, bins=30, alpha=0.7, label='Control', color='red', density=True)
        ax1.hist(treatment_ps, bins=30, alpha=0.7, label='Treatment', color='blue', density=True)
        ax1.set_xlabel('Propensity Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Propensity Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot([control_ps, treatment_ps], labels=['Control', 'Treatment'])
        ax2.set_ylabel('Propensity Score')
        ax2.set_title('Propensity Score Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_balance(self):
        """Plot covariate balance before and after matching"""
        if ('balance_before' not in self.assumption_checks or 
            'balance_after' not in self.assumption_checks):
            print("Balance checks must be run first")
            return
        
        before = self.assumption_checks['balance_before']
        after = self.assumption_checks['balance_after']
        
        covariates = list(before.keys())
        before_diffs = [before[var]['standardized_diff'] for var in covariates]
        after_diffs = [after[var]['standardized_diff'] for var in covariates]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(covariates))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, before_diffs, width, label='Before Matching', alpha=0.7)
        bars2 = ax.bar(x + width/2, after_diffs, width, label='After Matching', alpha=0.7)
        
        ax.set_xlabel('Covariates')
        ax.set_ylabel('Standardized Difference')
        ax.set_title('Covariate Balance: Before vs After Matching')
        ax.set_xticks(x)
        ax.set_xticklabels(covariates, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add reference lines for good balance
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Balance threshold')
        ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

class RegressionDiscontinuityDesign:
    """
    A comprehensive class for Regression Discontinuity Design analysis with multiple
    estimation methods, assumption checking, and robustness tests.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing the variables for analysis
    outcome_var : str
        Name of the outcome variable column
    running_var : str
        Name of the running variable (assignment variable) column
    treatment_var : str, optional
        Name of the treatment indicator column. If None, will be created based on cutoff
    cutoff : float
        The cutoff value for treatment assignment
    """
    
    def __init__(self, data, outcome_var, running_var, cutoff, treatment_var=None):
        self.data = data.copy()
        self.outcome_var = outcome_var
        self.running_var = running_var
        self.cutoff = cutoff
        self.treatment_var = treatment_var
        
        # Create treatment variable if not provided
        if self.treatment_var is None:
            self.treatment_var = 'treatment'
            self.data[self.treatment_var] = (self.data[self.running_var] >= self.cutoff).astype(int)
        
        # Center running variable around cutoff
        self.data['running_var_centered'] = self.data[self.running_var] - self.cutoff
        
        # Storage for results
        self.results = {}
        self.assumption_checks = {}
        
        # Validate data
        self._validate_data()
        
    def _validate_data(self):
        """Validate input data and variables"""
        required_cols = [self.outcome_var, self.running_var]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check for missing values
        missing_data = self.data[required_cols].isnull().sum()
        if missing_data.any():
            print("⚠️ Missing values detected:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"  {col}: {count} missing values")
                
        # Check cutoff is within data range
        running_min = self.data[self.running_var].min()
        running_max = self.data[self.running_var].max()
        
        if not (running_min <= self.cutoff <= running_max):
            raise ValueError(f"Cutoff {self.cutoff} outside data range [{running_min}, {running_max}]")
            
        print(f"✅ RDD data validation passed")
        print(f"  - Total observations: {len(self.data):,}")
        print(f"  - Running variable range: [{running_min:.2f}, {running_max:.2f}]")
        print(f"  - Cutoff: {self.cutoff}")
        print(f"  - Treatment units: {self.data[self.treatment_var].sum():,}")
        print(f"  - Control units: {(self.data[self.treatment_var] == 0).sum():,}")
    
    def check_continuity_assumption(self, plot=True):
        """
        Check continuity of baseline characteristics at cutoff
        Key assumption: No manipulation of running variable
        """
        print("\n" + "="*60)
        print("ASSUMPTION CHECK: CONTINUITY (NO MANIPULATION)")
        print("="*60)
        
        # Test density continuity using McCrary test (simplified)
        bin_width = (self.data[self.running_var].max() - self.data[self.running_var].min()) / 50
        
        # Create bins around cutoff
        bins_left = np.arange(self.cutoff - 5*bin_width, self.cutoff, bin_width)
        bins_right = np.arange(self.cutoff, self.cutoff + 5*bin_width, bin_width)
        
        # Count observations in each bin
        counts_left = []
        counts_right = []
        
        for i in range(len(bins_left)-1):
            count = ((self.data[self.running_var] >= bins_left[i]) & 
                    (self.data[self.running_var] < bins_left[i+1])).sum()
            counts_left.append(count)
            
        for i in range(len(bins_right)-1):
            count = ((self.data[self.running_var] >= bins_right[i]) & 
                    (self.data[self.running_var] < bins_right[i+1])).sum()
            counts_right.append(count)
        
        # Test for discontinuity in density
        if len(counts_left) > 0 and len(counts_right) > 0:
            density_left = np.mean(counts_left[-2:])  # Average of last 2 bins before cutoff
            density_right = np.mean(counts_right[:2])  # Average of first 2 bins after cutoff
            
            density_ratio = density_right / density_left if density_left > 0 else float('inf')
            
            # Simple test - more sophisticated would use McCrary test
            manipulation_suspected = abs(np.log(density_ratio)) > 0.5  # 50% jump
            
            results = {
                'density_left': density_left,
                'density_right': density_right,
                'density_ratio': density_ratio,
                'manipulation_suspected': manipulation_suspected,
                'bin_width': bin_width
            }
            
            print(f"Density continuity test:")
            print(f"  Density left of cutoff: {density_left:.2f}")
            print(f"  Density right of cutoff: {density_right:.2f}")
            print(f"  Ratio (right/left): {density_ratio:.3f}")
            
            if manipulation_suspected:
                print("⚠️ POTENTIAL MANIPULATION detected - large density jump")
                results['assumption_satisfied'] = False
            else:
                print("✅ NO OBVIOUS MANIPULATION - density appears continuous")
                results['assumption_satisfied'] = True
                
        else:
            print("❌ Insufficient data near cutoff for density test")
            results = {'assumption_satisfied': False, 'insufficient_data': True}
        
        if plot:
            self._plot_density_continuity()
            
        self.assumption_checks['continuity'] = results
        return results
    
    def check_covariate_continuity(self, covariates=None):
        """
        Check if baseline covariates are continuous at cutoff
        If covariates jump at cutoff, suggests manipulation
        """
        print("\n" + "="*60)
        print("ASSUMPTION CHECK: COVARIATE CONTINUITY")
        print("="*60)
        
        if covariates is None:
            # Try to identify potential covariates automatically
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            covariates = [col for col in numeric_cols 
                         if col not in [self.outcome_var, self.running_var, 
                                       self.treatment_var, 'running_var_centered']]
            
        if not covariates:
            print("No covariates specified or found")
            return {}
        
        covariate_results = {}
        
        print(f"Testing continuity for {len(covariates)} covariates:")
        
        for covar in covariates:
            if covar not in self.data.columns:
                continue
                
            # Run RDD on covariate (should find no effect if no manipulation)
            covar_effect = self._estimate_rdd_effect(
                outcome=self.data[covar],
                running_var=self.data['running_var_centered'],
                bandwidth=None,  # Use optimal bandwidth
                polynomial_order=1
            )
            
            effect_size = covar_effect['treatment_effect']
            p_value = covar_effect['p_value']
            
            covariate_results[covar] = {
                'effect': effect_size,
                'p_value': p_value,
                'continuous': p_value > 0.05
            }
            
            status = "✅" if p_value > 0.05 else "⚠️"
            print(f"  {covar:20s}: Effect = {effect_size:8.3f}, p = {p_value:.3f} {status}")
        
        # Overall assessment
        discontinuous_vars = sum(1 for result in covariate_results.values() 
                               if not result['continuous'])
        
        print(f"\nCovariate continuity summary:")
        print(f"  Variables with discontinuities: {discontinuous_vars}/{len(covariate_results)}")
        
        if discontinuous_vars == 0:
            print("✅ ALL COVARIATES CONTINUOUS - strong evidence against manipulation")
            overall_satisfied = True
        elif discontinuous_vars <= len(covariate_results) * 0.2:
            print("⚠️ FEW DISCONTINUITIES - likely acceptable")
            overall_satisfied = True
        else:
            print("❌ MANY DISCONTINUITIES - manipulation concerns")
            overall_satisfied = False
            
        covariate_results['overall_satisfied'] = overall_satisfied
        self.assumption_checks['covariate_continuity'] = covariate_results
        return covariate_results
    
    def optimal_bandwidth_selection(self, method='imbens_kalyanaraman'):
        """
        Select optimal bandwidth for RDD estimation
        
        Parameters:
        -----------
        method : str, default 'imbens_kalyanaraman'
            Method for bandwidth selection ('imbens_kalyanaraman', 'cross_validation')
        """
        print("\n" + "="*60)
        print("OPTIMAL BANDWIDTH SELECTION")
        print("="*60)
        
        if method == 'imbens_kalyanaraman':
            bandwidth = self._imbens_kalyanaraman_bandwidth()
        elif method == 'cross_validation':
            bandwidth = self._cross_validation_bandwidth()
        else:
            raise ValueError("Method must be 'imbens_kalyanaraman' or 'cross_validation'")
        
        print(f"Optimal bandwidth ({method}): {bandwidth:.3f}")
        
        # Check if bandwidth provides sufficient observations
        within_bandwidth = (np.abs(self.data['running_var_centered']) <= bandwidth).sum()
        print(f"Observations within bandwidth: {within_bandwidth:,}")
        
        if within_bandwidth < 50:
            print("⚠️ WARNING: Very few observations within optimal bandwidth")
            print("   Consider using wider bandwidth or checking data quality")
        
        self.results['optimal_bandwidth'] = {
            'bandwidth': bandwidth,
            'method': method,
            'observations_within': within_bandwidth
        }
        
        return bandwidth
    
    def _imbens_kalyanaraman_bandwidth(self):
        """Imbens-Kalyanaraman optimal bandwidth selection"""
        # Simplified implementation of IK bandwidth
        
        # Get data near cutoff for pilot estimation
        pilot_bandwidth = np.std(self.data['running_var_centered']) / 2
        
        pilot_data = self.data[np.abs(self.data['running_var_centered']) <= pilot_bandwidth].copy()
        
        if len(pilot_data) < 20:
            print("⚠️ Insufficient data for bandwidth selection, using rule of thumb")
            return np.std(self.data['running_var_centered']) / 4
        
        # Estimate second derivatives (simplified)
        X_left = pilot_data[pilot_data['running_var_centered'] < 0]['running_var_centered']
        y_left = pilot_data[pilot_data['running_var_centered'] < 0][self.outcome_var]
        
        X_right = pilot_data[pilot_data['running_var_centered'] >= 0]['running_var_centered']
        y_right = pilot_data[pilot_data['running_var_centered'] >= 0][self.outcome_var]
        
        # Simple rule of thumb based on data characteristics
        n = len(self.data)
        range_running = self.data['running_var_centered'].max() - self.data['running_var_centered'].min()
        
        # IK-style bandwidth (simplified formula)
        bandwidth = 1.84 * (range_running / 4) * (n ** (-1/5))
        
        return max(bandwidth, range_running / 20)  # Minimum bandwidth
    
    def _cross_validation_bandwidth(self):
        """Cross-validation bandwidth selection"""
        # Test different bandwidths
        bandwidths = np.linspace(0.1, 2.0, 20) * np.std(self.data['running_var_centered'])
        cv_scores = []
        
        for bw in bandwidths:
            # Split data for cross-validation
            subset = self.data[np.abs(self.data['running_var_centered']) <= bw].copy()
            
            if len(subset) < 20:
                cv_scores.append(float('inf'))
                continue
                
            # Simple leave-one-out CV (simplified)
            mse_scores = []
            for i in range(min(50, len(subset))):  # Limit for computational efficiency
                train_data = subset.drop(subset.index[i])
                test_point = subset.iloc[i]
                
                # Fit model on training data
                effect_result = self._estimate_rdd_effect(
                    outcome=train_data[self.outcome_var],
                    running_var=train_data['running_var_centered'],
                    bandwidth=bw,
                    polynomial_order=1
                )
                
                # Predict for test point
                predicted = self._predict_rdd(test_point['running_var_centered'], effect_result, bw)
                actual = test_point[self.outcome_var]
                
                mse_scores.append((predicted - actual) ** 2)
            
            cv_scores.append(np.mean(mse_scores))
        
        # Select bandwidth with minimum CV error
        optimal_idx = np.argmin(cv_scores)
        return bandwidths[optimal_idx]
    
    def _predict_rdd(self, running_var_value, effect_result, bandwidth):
        """Helper function to predict outcome for RDD model"""
        # Simplified prediction - in practice would use full model
        if running_var_value >= 0:
            return effect_result.get('intercept_right', 0) + effect_result.get('treatment_effect', 0)
        else:
            return effect_result.get('intercept_left', 0)
    
    def estimate_rdd_effect(self, bandwidth=None, polynomial_order=1, kernel='triangular'):
        """
        Estimate RDD treatment effect
        
        Parameters:
        -----------
        bandwidth : float, optional
            Bandwidth for local estimation. If None, uses optimal bandwidth
        polynomial_order : int, default 1
            Order of polynomial for local regression (1=linear, 2=quadratic)
        kernel : str, default 'triangular'
            Kernel function ('triangular', 'uniform', 'epanechnikov')
        """
        print("\n" + "="*60)
        print("RDD TREATMENT EFFECT ESTIMATION")
        print("="*60)
        
        if bandwidth is None:
            bandwidth = self.optimal_bandwidth_selection()
        else:
            print(f"Using specified bandwidth: {bandwidth:.3f}")
        
        # Estimate effect
        effect_result = self._estimate_rdd_effect(
            outcome=self.data[self.outcome_var],
            running_var=self.data['running_var_centered'],
            bandwidth=bandwidth,
            polynomial_order=polynomial_order,
            kernel=kernel
        )
        
        # Store results
        effect_result.update({
            'bandwidth': bandwidth,
            'polynomial_order': polynomial_order,
            'kernel': kernel
        })
        
        # Print results
        print(f"RDD estimation results:")
        print(f"  Treatment effect: {effect_result['treatment_effect']:.3f}")
        print(f"  Standard error: {effect_result['standard_error']:.3f}")
        print(f"  t-statistic: {effect_result['t_statistic']:.3f}")
        print(f"  p-value: {effect_result['p_value']:.3f}")
        print(f"  95% CI: [{effect_result['ci_lower']:.3f}, {effect_result['ci_upper']:.3f}]")
        print(f"  Observations used: {effect_result['n_obs']}")
        
        if effect_result['p_value'] < 0.05:
            print("✅ STATISTICALLY SIGNIFICANT at 5% level")
        else:
            print("⚠️ NOT statistically significant at 5% level")
        
        self.results['rdd_effect'] = effect_result
        return effect_result
    
    def _estimate_rdd_effect(self, outcome, running_var, bandwidth, polynomial_order=1, kernel='triangular'):
        """Internal method for RDD estimation"""
        
        # Select observations within bandwidth
        if bandwidth is not None:
            within_bw = np.abs(running_var) <= bandwidth
            outcome_bw = outcome[within_bw]
            running_var_bw = running_var[within_bw]
        else:
            outcome_bw = outcome
            running_var_bw = running_var
        
        if len(outcome_bw) < 10:
            raise ValueError("Insufficient observations within bandwidth")
        
        # Create treatment indicator
        treatment_bw = (running_var_bw >= 0).astype(int)
        
        # Create polynomial features
        if polynomial_order == 1:
            # Linear specification
            X = pd.DataFrame({
                'constant': 1,
                'running_var': running_var_bw,
                'treatment': treatment_bw,
                'running_var_treatment': running_var_bw * treatment_bw
            })
        elif polynomial_order == 2:
            # Quadratic specification
            X = pd.DataFrame({
                'constant': 1,
                'running_var': running_var_bw,
                'running_var_sq': running_var_bw ** 2,
                'treatment': treatment_bw,
                'running_var_treatment': running_var_bw * treatment_bw,
                'running_var_sq_treatment': (running_var_bw ** 2) * treatment_bw
            })
        else:
            raise ValueError("Only polynomial orders 1 and 2 are supported")
        
        # Apply kernel weights
        if kernel == 'triangular' and bandwidth is not None:
            weights = np.maximum(0, 1 - np.abs(running_var_bw) / bandwidth)
        elif kernel == 'uniform':
            weights = np.ones(len(running_var_bw))
        elif kernel == 'epanechnikov' and bandwidth is not None:
            weights = np.maximum(0, 0.75 * (1 - (running_var_bw / bandwidth) ** 2))
        else:
            weights = np.ones(len(running_var_bw))
        
        # Weighted least squares
        X_weighted = X.multiply(np.sqrt(weights), axis=0)
        y_weighted = outcome_bw * np.sqrt(weights)
        
        # Fit model
        model = sm.OLS(y_weighted, X_weighted).fit()
        
        # Extract treatment effect (coefficient on treatment dummy)
        treatment_effect = model.params['treatment']
        standard_error = model.bse['treatment']
        t_statistic = model.tvalues['treatment']
        p_value = model.pvalues['treatment']
        
        # Confidence interval
        ci_lower = treatment_effect - 1.96 * standard_error
        ci_upper = treatment_effect + 1.96 * standard_error
        
        return {
            'treatment_effect': treatment_effect,
            'standard_error': standard_error,
            't_statistic': t_statistic,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': len(outcome_bw),
            'model': model,
            'r_squared': model.rsquared
        }
    
    def sensitivity_analysis(self, bandwidth_range=None, polynomial_orders=[1, 2]):
        """
        Test sensitivity to bandwidth and polynomial order choices
        
        Parameters:
        -----------
        bandwidth_range : tuple, optional
            (min_bw, max_bw) for testing. If None, uses range around optimal
        polynomial_orders : list, default [1, 2]
            Polynomial orders to test
        """
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSIS")
        print("="*60)
        
        if bandwidth_range is None:
            optimal_bw = self.results.get('optimal_bandwidth', {}).get('bandwidth')
            if optimal_bw is None:
                optimal_bw = self.optimal_bandwidth_selection()
            bandwidth_range = (optimal_bw * 0.5, optimal_bw * 2.0)
        
        # Test different bandwidths
        bandwidths = np.linspace(bandwidth_range[0], bandwidth_range[1], 10)
        
        sensitivity_results = []
        
        print("Testing sensitivity to bandwidth choice:")
        print("Bandwidth    Poly Order    Effect    Std Err    P-value")
        print("-" * 55)
        
        for bw in bandwidths:
            for poly_order in polynomial_orders:
                try:
                    result = self._estimate_rdd_effect(
                        outcome=self.data[self.outcome_var],
                        running_var=self.data['running_var_centered'],
                        bandwidth=bw,
                        polynomial_order=poly_order
                    )
                    
                    sensitivity_results.append({
                        'bandwidth': bw,
                        'polynomial_order': poly_order,
                        'treatment_effect': result['treatment_effect'],
                        'standard_error': result['standard_error'],
                        'p_value': result['p_value'],
                        'n_obs': result['n_obs']
                    })
                    
                    print(f"{bw:8.3f}        {poly_order}       {result['treatment_effect']:6.3f}    {result['standard_error']:6.3f}    {result['p_value']:6.3f}")
                    
                except Exception as e:
                    print(f"{bw:8.3f}        {poly_order}       ERROR: {str(e)[:20]}")
        
        # Analyze sensitivity
        if sensitivity_results:
            effects = [r['treatment_effect'] for r in sensitivity_results]
            effect_range = max(effects) - min(effects)
            effect_std = np.std(effects)
            
            print(f"\nSensitivity summary:")
            print(f"  Effect range: {min(effects):.3f} to {max(effects):.3f}")
            print(f"  Effect standard deviation: {effect_std:.3f}")
            print(f"  Range as % of mean effect: {(effect_range / np.mean(effects)) * 100:.1f}%")
            
            if effect_range / abs(np.mean(effects)) < 0.2:
                print("✅ ROBUST - effect stable across specifications")
            else:
                print("⚠️ SENSITIVE - effect varies significantly across specifications")
        
        self.results['sensitivity_analysis'] = sensitivity_results
        return sensitivity_results
    
    def placebo_test(self, placebo_cutoffs=None):
        """
        Test for effects at false cutoffs where no effect should exist
        
        Parameters:
        -----------
        placebo_cutoffs : list, optional
            List of false cutoff values to test. If None, uses quantiles
        """
        print("\n" + "="*60)
        print("ROBUSTNESS CHECK: PLACEBO TEST")
        print("="*60)
        
        if placebo_cutoffs is None:
            # Use quantiles of running variable as placebo cutoffs
            quantiles = [0.2, 0.3, 0.7, 0.8]
            placebo_cutoffs = [self.data[self.running_var].quantile(q) for q in quantiles]
            placebo_cutoffs = [c for c in placebo_cutoffs if c != self.cutoff]
        
        placebo_results = []
        
        print("Testing for effects at false cutoffs:")
        print("Placebo Cutoff    Effect    Std Err    P-value")
        print("-" * 45)
        
        for placebo_cutoff in placebo_cutoffs:
            # Create placebo treatment variable
            placebo_data = self.data.copy()
            placebo_data['placebo_treatment'] = (placebo_data[self.running_var] >= placebo_cutoff).astype(int)
            placebo_data['placebo_running_centered'] = placebo_data[self.running_var] - placebo_cutoff
            
            try:
                # Estimate effect at placebo cutoff
                placebo_result = self._estimate_rdd_effect(
                    outcome=placebo_data[self.outcome_var],
                    running_var=placebo_data['placebo_running_centered'],
                    bandwidth=None,  # Use automatic bandwidth selection
                    polynomial_order=1
                )
                
                placebo_results.append({
                    'placebo_cutoff': placebo_cutoff,
                    'effect': placebo_result['treatment_effect'],
                    'standard_error': placebo_result['standard_error'],
                    'p_value': placebo_result['p_value'],
                    'significant': placebo_result['p_value'] < 0.05
                })
                
                sig_marker = "*" if placebo_result['p_value'] < 0.05 else ""
                print(f"{placebo_cutoff:12.2f}    {placebo_result['treatment_effect']:6.3f}    {placebo_result['standard_error']:6.3f}    {placebo_result['p_value']:6.3f}{sig_marker}")
                
            except Exception as e:
                print(f"{placebo_cutoff:12.2f}    ERROR: {str(e)[:30]}")
        
        # Assess placebo test results
        if placebo_results:
            significant_placebos = sum(1 for r in placebo_results if r['significant'])
            
            print(f"\nPlacebo test summary:")
            print(f"  False cutoffs tested: {len(placebo_results)}")
            print(f"  Significant effects found: {significant_placebos}")
            
            if significant_placebos == 0:
                print("✅ PLACEBO TEST PASSED - no false effects detected")
                test_passed = True
            elif significant_placebos <= len(placebo_results) * 0.05:  # Expected false positive rate
                print("✅ PLACEBO TEST PASSED - false positives within expected range")
                test_passed = True
            else:
                print("⚠️ PLACEBO TEST FAILED - too many false effects detected")
                test_passed = False
                
            placebo_results.append({'test_passed': test_passed})
        
        self.results['placebo_test'] = placebo_results
        return placebo_results
    
    def donut_hole_test(self, hole_size=None):
        """
        Test robustness by excluding observations very close to cutoff
        
        Parameters:
        -----------
        hole_size : float, optional
            Size of hole around cutoff to exclude. If None, uses 10% of optimal bandwidth
        """
        print("\n" + "="*60)
        print("ROBUSTNESS CHECK: DONUT HOLE TEST")
        print("="*60)
        
        if hole_size is None:
            optimal_bw = self.results.get('optimal_bandwidth', {}).get('bandwidth')
            if optimal_bw is None:
                optimal_bw = self.optimal_bandwidth_selection()
            hole_size = optimal_bw * 0.1
        
        print(f"Excluding observations within {hole_size:.3f} of cutoff")
        
        # Exclude observations in donut hole
        donut_data = self.data[np.abs(self.data['running_var_centered']) > hole_size].copy()
        
        print(f"Observations remaining: {len(donut_data):,} (excluded {len(self.data) - len(donut_data):,})")
        
        if len(donut_data) < 50:
            print("❌ Too few observations remaining for donut hole test")
            return {'insufficient_data': True}
        
        # Estimate effect on donut hole sample
        try:
            donut_result = self._estimate_rdd_effect(
                outcome=donut_data[self.outcome_var],
                running_var=donut_data['running_var_centered'],
                bandwidth=None,
                polynomial_order=1
            )
            
            # Compare with main result
            main_effect = self.results.get('rdd_effect', {}).get('treatment_effect', 0)
            
            difference = abs(donut_result['treatment_effect'] - main_effect)
            relative_difference = difference / abs(main_effect) if main_effect != 0 else float('inf')
            
            print(f"Donut hole results:")
            print(f"  Main effect: {main_effect:.3f}")
            print(f"  Donut effect: {donut_result['treatment_effect']:.3f}")
            print(f"  Difference: {difference:.3f}")
            print(f"  Relative difference: {relative_difference*100:.1f}%")
            
            if relative_difference < 0.2:  # Less than 20% difference
                print("✅ DONUT HOLE TEST PASSED - effect stable when excluding near-cutoff observations")
                test_passed = True
            else:
                print("⚠️ DONUT HOLE TEST QUESTIONABLE - effect sensitive to near-cutoff observations")
                test_passed = False
            
            donut_result.update({
                'hole_size': hole_size,
                'main_effect': main_effect,
                'difference': difference,
                'relative_difference': relative_difference,
                'test_passed': test_passed
            })
            
        except Exception as e:
            print(f"❌ Error in donut hole test: {e}")
            donut_result = {'error': str(e)}
        
        self.results['donut_hole_test'] = donut_result
        return donut_result
    
    def run_full_analysis(self, bandwidth=None, covariates=None, plot=True):
        """Run complete RDD analysis pipeline"""
        print("RUNNING COMPREHENSIVE RDD ANALYSIS")
        print("="*60)
        
        # Step 1: Check assumptions
        self.check_continuity_assumption(plot=plot)
        self.check_covariate_continuity(covariates=covariates)
        
        # Step 2: Bandwidth selection
        if bandwidth is None:
            bandwidth = self.optimal_bandwidth_selection()
        
        # Step 3: Main RDD estimation
        self.estimate_rdd_effect(bandwidth=bandwidth)
        
        # Step 4: Robustness checks
        self.sensitivity_analysis()
        self.placebo_test()
        self.donut_hole_test()
        
        # Step 5: Summary
        self.print_summary()
        
        # Step 6: Visualization
        if plot:
            self.plot_rdd()
        
        return self.results
    
    def print_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*60)
        print("RDD ANALYSIS SUMMARY")
        print("="*60)
        
        # Main effect
        if 'rdd_effect' in self.results:
            effect = self.results['rdd_effect']
            print(f"Treatment Effect:")
            print(f"  Estimate: {effect['treatment_effect']:.3f}")
            print(f"  Standard Error: {effect['standard_error']:.3f}")
            print(f"  p-value: {effect['p_value']:.3f}")
            print(f"  95% CI: [{effect['ci_lower']:.3f}, {effect['ci_upper']:.3f}]")
        
        # Assumption checks
        print(f"\nAssumption Checks:")
        if 'continuity' in self.assumption_checks:
            continuity_ok = self.assumption_checks['continuity'].get('assumption_satisfied', False)
            print(f"  Density Continuity: {'✅ SATISFIED' if continuity_ok else '⚠️ VIOLATED'}")
        
        if 'covariate_continuity' in self.assumption_checks:
            covar_ok = self.assumption_checks['covariate_continuity'].get('overall_satisfied', False)
            print(f"  Covariate Continuity: {'✅ SATISFIED' if covar_ok else '⚠️ VIOLATED'}")
        
        # Robustness checks
        print(f"\nRobustness Checks:")
        if 'placebo_test' in self.results:
            placebo_results = self.results['placebo_test']
            if placebo_results and isinstance(placebo_results[-1], dict) and 'test_passed' in placebo_results[-1]:
                placebo_passed = placebo_results[-1]['test_passed']
                print(f"  Placebo Test: {'✅ PASSED' if placebo_passed else '⚠️ FAILED'}")
        
        if 'donut_hole_test' in self.results:
            donut_passed = self.results['donut_hole_test'].get('test_passed', False)
            print(f"  Donut Hole Test: {'✅ PASSED' if donut_passed else '⚠️ FAILED'}")
        
        if 'sensitivity_analysis' in self.results:
            sensitivity_results = self.results['sensitivity_analysis']
            if sensitivity_results:
                effects = [r['treatment_effect'] for r in sensitivity_results]
                effect_range = max(effects) - min(effects)
                robust = effect_range / abs(np.mean(effects)) < 0.2 if np.mean(effects) != 0 else False
                print(f"  Sensitivity Test: {'✅ ROBUST' if robust else '⚠️ SENSITIVE'}")
    
    def plot_rdd(self, bandwidth=None, bins=50):
        """Create comprehensive RDD visualization"""
        print("\n" + "="*60)
        print("RDD VISUALIZATION")
        print("="*60)
        
        if bandwidth is None:
            bandwidth = self.results.get('rdd_effect', {}).get('bandwidth')
            if bandwidth is None:
                bandwidth = self.optimal_bandwidth_selection()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Scatter plot with regression lines
        ax1 = axes[0, 0]
        
        # Sample data for plotting (to avoid overcrowding)
        plot_data = self.data.sample(min(2000, len(self.data)), random_state=42)
        
        # Separate by treatment status
        control_data = plot_data[plot_data[self.treatment_var] == 0]
        treatment_data = plot_data[plot_data[self.treatment_var] == 1]
        
        # Scatter plots
        ax1.scatter(control_data['running_var_centered'], control_data[self.outcome_var], 
                   alpha=0.5, s=20, color='red', label='Control')
        ax1.scatter(treatment_data['running_var_centered'], treatment_data[self.outcome_var], 
                   alpha=0.5, s=20, color='blue', label='Treatment')
        
        # Regression lines
        self._add_regression_lines(ax1, bandwidth)
        
        # Cutoff line
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.8, label='Cutoff')
        
        ax1.set_xlabel(f'{self.running_var} (centered at cutoff)')
        ax1.set_ylabel(self.outcome_var)
        ax1.set_title('RDD: Outcome vs Running Variable')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Binned scatter plot
        ax2 = axes[0, 1]
        self._create_binned_plot(ax2, bins)
        
        # Plot 3: Density plot
        ax3 = axes[1, 0]
        self._plot_density_continuity(ax3)
        
        # Plot 4: Sensitivity analysis
        ax4 = axes[1, 1]
        if 'sensitivity_analysis' in self.results:
            self._plot_sensitivity_analysis(ax4)
        else:
            ax4.text(0.5, 0.5, 'Run sensitivity_analysis()\nto see results', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Sensitivity Analysis')
        
        plt.tight_layout()
        plt.show()
    
    def _add_regression_lines(self, ax, bandwidth):
        """Add fitted regression lines to RDD plot"""
        # Get data within bandwidth
        within_bw = self.data[np.abs(self.data['running_var_centered']) <= bandwidth]
        
        # Separate by treatment
        control_bw = within_bw[within_bw[self.treatment_var] == 0]
        treatment_bw = within_bw[within_bw[self.treatment_var] == 1]
        
        # Fit lines
        if len(control_bw) > 1:
            X_control = control_bw['running_var_centered'].values.reshape(-1, 1)
            y_control = control_bw[self.outcome_var].values
            reg_control = LinearRegression().fit(X_control, y_control)
            
            x_range_control = np.linspace(control_bw['running_var_centered'].min(), 0, 100)
            y_pred_control = reg_control.predict(x_range_control.reshape(-1, 1))
            ax.plot(x_range_control, y_pred_control, color='red', linewidth=2)
        
        if len(treatment_bw) > 1:
            X_treatment = treatment_bw['running_var_centered'].values.reshape(-1, 1)
            y_treatment = treatment_bw[self.outcome_var].values
            reg_treatment = LinearRegression().fit(X_treatment, y_treatment)
            
            x_range_treatment = np.linspace(0, treatment_bw['running_var_centered'].max(), 100)
            y_pred_treatment = reg_treatment.predict(x_range_treatment.reshape(-1, 1))
            ax.plot(x_range_treatment, y_pred_treatment, color='blue', linewidth=2)
    
    def _create_binned_plot(self, ax, bins):
        """Create binned scatter plot for cleaner visualization"""
        # Create bins
        running_var_range = (self.data['running_var_centered'].min(), self.data['running_var_centered'].max())
        bin_edges = np.linspace(running_var_range[0], running_var_range[1], bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate mean outcome in each bin
        bin_means = []
        bin_counts = []
        
        for i in range(len(bin_edges) - 1):
            in_bin = ((self.data['running_var_centered'] >= bin_edges[i]) & 
                     (self.data['running_var_centered'] < bin_edges[i + 1]))
            
            if in_bin.any():
                bin_mean = self.data[in_bin][self.outcome_var].mean()
                bin_count = in_bin.sum()
            else:
                bin_mean = np.nan
                bin_count = 0
            
            bin_means.append(bin_mean)
            bin_counts.append(bin_count)
        
        # Plot points, size based on number of observations
        sizes = [max(10, min(100, count/5)) for count in bin_counts]
        colors = ['red' if center < 0 else 'blue' for center in bin_centers]
        
        ax.scatter(bin_centers, bin_means, s=sizes, c=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.8)
        ax.set_xlabel(f'{self.running_var} (centered at cutoff)')
        ax.set_ylabel(f'Mean {self.outcome_var}')
        ax.set_title('RDD: Binned Scatter Plot')
        ax.grid(True, alpha=0.3)
    
    def _plot_density_continuity(self, ax=None):
        """Plot density of running variable around cutoff"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        bins = 50
        ax.hist(self.data['running_var_centered'], bins=bins, alpha=0.7, 
               color='skyblue', edgecolor='black')
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Cutoff')
        ax.set_xlabel(f'{self.running_var} (centered at cutoff)')
        ax.set_ylabel('Frequency')
        ax.set_title('Density Continuity Check')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if ax is None:
            plt.show()
    
    def _plot_sensitivity_analysis(self, ax):
        """Plot sensitivity analysis results"""
        if 'sensitivity_analysis' not in self.results:
            return
        
        sensitivity_results = self.results['sensitivity_analysis']
        if not sensitivity_results:
            return
        
        # Group by polynomial order
        poly_orders = sorted(set(r['polynomial_order'] for r in sensitivity_results))
        
        for poly_order in poly_orders:
            poly_results = [r for r in sensitivity_results if r['polynomial_order'] == poly_order]
            bandwidths = [r['bandwidth'] for r in poly_results]
            effects = [r['treatment_effect'] for r in poly_results]
            
            ax.plot(bandwidths, effects, 'o-', label=f'Polynomial Order {poly_order}')
        
        ax.set_xlabel('Bandwidth')
        ax.set_ylabel('Treatment Effect')
        ax.set_title('Sensitivity Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)























