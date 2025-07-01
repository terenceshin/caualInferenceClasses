"""
Propensity Score Matching implementation for causal inference.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
import warnings
from typing import Dict, Any, Optional, List, Union

from ..core.base import CausalInferenceBase
from ..core.exceptions import DataValidationError, EstimationError

warnings.filterwarnings('ignore')


class PropensityScoreMatching(CausalInferenceBase):
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
    
    def __init__(self, data: pd.DataFrame, outcome_var: str, treatment_var: str, 
                 covariates: List[str], unit_id: Optional[str] = None):
        # Initialize base class
        super().__init__(data, outcome_var, treatment_var)
        
        # PSM-specific attributes
        self.covariates = covariates
        self.unit_id = unit_id if unit_id else 'unit_id'
        
        # Create unit_id if not provided
        if self.unit_id not in self.data.columns:
            self.data[self.unit_id] = range(len(self.data))
        
        # Storage for results
        self.propensity_scores = None
        self.matched_data = None
        self.assumption_checks = {}
        
        # Validate PSM-specific data
        self._validate_psm_data()

    def _validate_psm_data(self) -> None:
        """Validate PSM-specific input data and variables"""
        required_cols = self.covariates
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise DataValidationError(f"Missing required covariate columns: {missing_cols}")
            
        # Check for missing values in key variables
        all_required = [self.outcome_var, self.treatment_var] + self.covariates
        missing_data = self.data[all_required].isnull().sum()
        if missing_data.any():
            print("⚠️  Missing values detected:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"  {col}: {count} missing values")
                
        print(f"✅ Data validation passed")
        print(f"  - Total observations: {len(self.data):,}")
        print(f"  - Treatment units: {(self.data[self.treatment_var] == 1).sum():,}")
        print(f"  - Control units: {(self.data[self.treatment_var] == 0).sum():,}")
        print(f"  - Covariates: {len(self.covariates)}")

    def estimate(self) -> Dict[str, Any]:
        """
        Main estimation method - runs full PSM analysis.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing estimation results
        """
        return self.run_full_analysis()
    
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
        summary_lines.append("Propensity Score Matching Results")
        summary_lines.append("=" * 50)
        
        if 'treatment_effect' in self.results_:
            effect = self.results_['treatment_effect']
            summary_lines.append(f"Average Treatment Effect: {effect['ate']:.3f}")
            summary_lines.append(f"Standard Error: {effect['standard_error']:.3f}")
            summary_lines.append(f"P-value: {effect['p_value']:.3f}")
            summary_lines.append(f"95% CI: [{effect['ci_lower']:.3f}, {effect['ci_upper']:.3f}]")
        
        return "\n".join(summary_lines)

    def estimate_propensity_scores(self, method: str = 'logistic', 
                                  include_interactions: bool = False, 
                                  polynomial_degree: int = 1) -> np.ndarray:
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
        
        if self.results_ is None:
            self.results_ = {}
        self.results_['propensity_estimation'] = {
            'method': method,
            'auc': auc_score,
            'extreme_scores': extreme_low + extreme_high,
            'feature_count': X.shape[1]
        }
        
        return propensity_scores

    def check_overlap(self, plot: bool = True) -> Dict[str, Any]:
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

    def check_balance_before_matching(self) -> Dict[str, Dict[str, float]]:
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

    def perform_matching(self, method: str = 'nearest', caliper: Optional[float] = None,
                        replacement: bool = False, ratio: int = 1,
                        random_state: int = 42) -> Optional[pd.DataFrame]:
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

    def _nearest_neighbor_matching(self, treatment_data: pd.DataFrame, control_data: pd.DataFrame,
                                  caliper: Optional[float], replacement: bool, ratio: int) -> List[pd.DataFrame]:
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

    def _caliper_matching(self, treatment_data: pd.DataFrame, control_data: pd.DataFrame,
                         caliper: float, replacement: bool, ratio: int) -> List[pd.DataFrame]:
        """Perform caliper matching"""
        return self._nearest_neighbor_matching(treatment_data, control_data, caliper, replacement, ratio)

    def _stratification_matching(self, treatment_data: pd.DataFrame, control_data: pd.DataFrame,
                               ratio: int) -> List[pd.DataFrame]:
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

    def _assess_match_quality(self) -> None:
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

    def check_balance_after_matching(self) -> Dict[str, Dict[str, float]]:
        """Check covariate balance after matching"""
        print("\n" + "="*60)
        print("COVARIATE BALANCE: AFTER MATCHING")
        print("="*60)

        if self.matched_data is None:
            print("No matched data available")
            return {}

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

    def estimate_treatment_effect(self, method: str = 'simple_difference') -> Dict[str, Any]:
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

        if self.results_ is None:
            self.results_ = {}
        self.results_['treatment_effect'] = results
        return results

    def run_full_analysis(self, matching_method: str = 'nearest',
                         caliper: Optional[float] = None, plot: bool = True) -> Dict[str, Any]:
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

            # Step 6: Summary
            self.print_summary()

        return self.results_

    def print_summary(self) -> None:
        """Print comprehensive analysis summary"""
        print("\n" + "="*60)
        print("PSM ANALYSIS SUMMARY")
        print("="*60)

        # Propensity score model
        if self.results_ and 'propensity_estimation' in self.results_:
            ps_results = self.results_['propensity_estimation']
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
        if self.results_ and 'treatment_effect' in self.results_:
            effect = self.results_['treatment_effect']
            print(f"\nTreatment Effect:")
            print(f"  ATE: {effect['ate']:.3f} (p={effect['p_value']:.3f})")
            print(f"  95% CI: [{effect['ci_lower']:.3f}, {effect['ci_upper']:.3f}]")
            print(f"  Effect size: {effect['effect_percentage']:.1f}% of control mean")

    def _plot_overlap(self, treatment_ps: np.ndarray, control_ps: np.ndarray) -> None:
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
