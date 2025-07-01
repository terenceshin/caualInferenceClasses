"""
Regression Discontinuity Design implementation for causal inference.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
from typing import Dict, Any, Optional, List, Union, Tuple

from ..core.base import CausalInferenceBase
from ..core.exceptions import DataValidationError, EstimationError

warnings.filterwarnings('ignore')


class RegressionDiscontinuityDesign(CausalInferenceBase):
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
    cutoff : float
        The cutoff value for treatment assignment
    treatment_var : str, optional
        Name of the treatment indicator column. If None, will be created based on cutoff
    """
    
    def __init__(self, data: pd.DataFrame, outcome_var: str, running_var: str,
                 cutoff: float, treatment_var: Optional[str] = None):
        # RDD-specific attributes
        self.running_var = running_var
        self.cutoff = cutoff

        # Check if running variable exists before using it
        if self.running_var not in data.columns:
            raise DataValidationError(f"Running variable '{self.running_var}' not found in data")

        # Create treatment variable if not provided
        if treatment_var is None:
            self.treatment_var = 'treatment'
            data = data.copy()  # Don't modify original data
            data[self.treatment_var] = (data[self.running_var] >= self.cutoff).astype(int)
        else:
            self.treatment_var = treatment_var

        # Initialize base class after creating treatment variable
        super().__init__(data, outcome_var, self.treatment_var)
        
        # Center running variable around cutoff
        self.data['running_var_centered'] = self.data[self.running_var] - self.cutoff
        
        # Storage for results
        self.assumption_checks = {}
        
        # Validate RDD-specific data
        self._validate_rdd_data()

    def _validate_rdd_data(self) -> None:
        """Validate RDD-specific input data and variables"""
        required_cols = [self.running_var]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")
            
        # Check for missing values
        missing_data = self.data[required_cols + [self.outcome_var]].isnull().sum()
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

    def estimate(self) -> Dict[str, Any]:
        """
        Main estimation method - runs RDD estimation with optimal bandwidth.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing estimation results
        """
        return self.estimate_rdd_effect()
    
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
        summary_lines.append("Regression Discontinuity Design Results")
        summary_lines.append("=" * 50)
        
        if 'rdd_effect' in self.results_:
            effect = self.results_['rdd_effect']
            summary_lines.append(f"Treatment Effect: {effect['treatment_effect']:.3f}")
            summary_lines.append(f"Standard Error: {effect['standard_error']:.3f}")
            summary_lines.append(f"P-value: {effect['p_value']:.3f}")
            summary_lines.append(f"95% CI: [{effect['ci_lower']:.3f}, {effect['ci_upper']:.3f}]")
            summary_lines.append(f"Bandwidth: {effect['bandwidth']:.3f}")
            summary_lines.append(f"Observations: {effect['n_obs']}")
        
        return "\n".join(summary_lines)

    def check_continuity_assumption(self, plot: bool = True) -> Dict[str, Any]:
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

    def optimal_bandwidth_selection(self, method: str = 'imbens_kalyanaraman') -> float:
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
        
        if self.results_ is None:
            self.results_ = {}
        self.results_['optimal_bandwidth'] = {
            'bandwidth': bandwidth,
            'method': method,
            'observations_within': within_bandwidth
        }
        
        return bandwidth

    def _imbens_kalyanaraman_bandwidth(self) -> float:
        """Imbens-Kalyanaraman optimal bandwidth selection"""
        # Simplified implementation of IK bandwidth

        # Get data near cutoff for pilot estimation
        pilot_bandwidth = np.std(self.data['running_var_centered']) / 2

        pilot_data = self.data[np.abs(self.data['running_var_centered']) <= pilot_bandwidth].copy()

        if len(pilot_data) < 20:
            print("⚠️ Insufficient data for bandwidth selection, using rule of thumb")
            return np.std(self.data['running_var_centered']) / 4

        # Simple rule of thumb based on data characteristics
        n = len(self.data)
        range_running = self.data['running_var_centered'].max() - self.data['running_var_centered'].min()

        # IK-style bandwidth (simplified formula)
        bandwidth = 1.84 * (range_running / 4) * (n ** (-1/5))

        return max(bandwidth, range_running / 20)  # Minimum bandwidth

    def _cross_validation_bandwidth(self) -> float:
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

    def _predict_rdd(self, running_var_value: float, effect_result: Dict[str, Any], bandwidth: float) -> float:
        """Helper function to predict outcome for RDD model"""
        # Simplified prediction - in practice would use full model
        if running_var_value >= 0:
            return effect_result.get('intercept_right', 0) + effect_result.get('treatment_effect', 0)
        else:
            return effect_result.get('intercept_left', 0)

    def estimate_rdd_effect(self, bandwidth: Optional[float] = None,
                           polynomial_order: int = 1, kernel: str = 'triangular') -> Dict[str, Any]:
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

        if self.results_ is None:
            self.results_ = {}
        self.results_['rdd_effect'] = effect_result
        return effect_result

    def _estimate_rdd_effect(self, outcome: pd.Series, running_var: pd.Series,
                            bandwidth: Optional[float], polynomial_order: int = 1,
                            kernel: str = 'triangular') -> Dict[str, Any]:
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

    def run_full_analysis(self, bandwidth: Optional[float] = None,
                         covariates: Optional[List[str]] = None, plot: bool = True) -> Dict[str, Any]:
        """Run complete RDD analysis pipeline"""
        print("RUNNING COMPREHENSIVE RDD ANALYSIS")
        print("="*60)

        # Step 1: Check assumptions
        self.check_continuity_assumption(plot=plot)

        # Step 2: Bandwidth selection
        if bandwidth is None:
            bandwidth = self.optimal_bandwidth_selection()

        # Step 3: Main RDD estimation
        self.estimate_rdd_effect(bandwidth=bandwidth)

        # Step 4: Summary
        self.print_summary()

        # Step 5: Visualization
        if plot:
            self.plot_rdd()

        return self.results_

    def print_summary(self) -> None:
        """Print comprehensive analysis summary"""
        print("\n" + "="*60)
        print("RDD ANALYSIS SUMMARY")
        print("="*60)

        # Main effect
        if self.results_ and 'rdd_effect' in self.results_:
            effect = self.results_['rdd_effect']
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

    def _plot_density_continuity(self, ax=None) -> None:
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

    def plot_rdd(self, bandwidth: Optional[float] = None, bins: int = 50) -> None:
        """Create comprehensive RDD visualization"""
        print("\n" + "="*60)
        print("RDD VISUALIZATION")
        print("="*60)

        if bandwidth is None:
            bandwidth = self.results_.get('rdd_effect', {}).get('bandwidth')
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

        # Plot 4: Effect size
        ax4 = axes[1, 1]
        if self.results_ and 'rdd_effect' in self.results_:
            effect = self.results_['rdd_effect']
            ax4.bar(['Treatment Effect'], [effect['treatment_effect']],
                   yerr=[effect['standard_error']], capsize=10, alpha=0.7)
            ax4.set_ylabel('Effect Size')
            ax4.set_title('RDD Treatment Effect')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Run estimate_rdd_effect()\nto see results',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Treatment Effect')

        plt.tight_layout()
        plt.show()

    def _add_regression_lines(self, ax, bandwidth: float) -> None:
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

    def _create_binned_plot(self, ax, bins: int) -> None:
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
