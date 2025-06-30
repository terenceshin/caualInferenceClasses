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
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
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