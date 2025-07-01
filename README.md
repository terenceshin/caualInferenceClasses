# Causal Inference Classes

A comprehensive Python library for causal inference analysis, providing implementations of popular methods including Difference-in-Differences (DiD), Propensity Score Matching (PSM), and Regression Discontinuity Design (RDD).

## 🚀 Features

- **Difference-in-Differences (DiD)**: Complete implementation with assumption testing and robustness checks
- **Propensity Score Matching (PSM)**: Multiple matching algorithms with balance assessment  
- **Regression Discontinuity Design (RDD)**: Sharp RDD with optimal bandwidth selection
- **Synthetic Data Generation**: Built-in functions to generate realistic test data
- **Comprehensive Testing**: Extensive test suite ensuring reliability
- **Rich Visualizations**: Publication-ready plots for all methods
- **Modular Design**: Clean, extensible architecture following best practices
- **Jupyter Integration**: Interactive notebooks with complete examples

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd causalInferenceClasses

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🎯 Quick Start

### Difference-in-Differences

```python
from causal_inference import DifferenceInDifferences, generate_did_data

# Generate synthetic data
data = generate_did_data(n_units=100, n_periods=10, treatment_period=6)

# Initialize DiD estimator
did = DifferenceInDifferences(
    data=data,
    outcome_var='outcome',
    unit_var='unit_id', 
    time_var='period',
    treatment_var='treated',
    post_var='post',
    treatment_start=6
)

# Run complete analysis
results = did.run_full_analysis()
print(did.summary())
```

### Propensity Score Matching

```python
from causal_inference import PropensityScoreMatching, generate_psm_data

# Generate synthetic data with selection bias
data = generate_psm_data(n_samples=1000, n_features=5, treatment_effect=2.5)

# Initialize PSM estimator
psm = PropensityScoreMatching(
    data=data,
    outcome_var='outcome',
    treatment_var='treated',
    covariates=['covariate_1', 'covariate_2', 'covariate_3', 'covariate_4', 'covariate_5']
)

# Run complete analysis
results = psm.run_full_analysis(matching_method='nearest', caliper=0.01)
print(psm.summary())
```

### Regression Discontinuity Design

```python
from causal_inference import RegressionDiscontinuityDesign, generate_rdd_data

# Generate synthetic RDD data
data = generate_rdd_data(n_obs=1000, cutoff=0.0, treatment_effect=3.0)

# Initialize RDD estimator
rdd = RegressionDiscontinuityDesign(
    data=data,
    outcome_var='outcome',
    running_var='running_var',
    cutoff=0.0
)

# Run complete analysis
results = rdd.run_full_analysis()
print(rdd.summary())
```

## 📚 Documentation & Examples

### Interactive Notebooks

The `examples/notebooks/` directory contains comprehensive Jupyter notebooks:

- **`01_difference_in_differences_example.ipynb`**: Complete DiD analysis walkthrough
  - Data generation and validation
  - Assumption testing (parallel trends, common shocks)
  - Multiple estimation methods
  - Robustness checks and placebo tests
  
- **`02_propensity_score_matching_example.ipynb`**: PSM analysis with multiple approaches
  - Propensity score estimation (logistic, random forest)
  - Overlap and balance assessment
  - Multiple matching methods (nearest neighbor, caliper, stratification)
  - Treatment effect estimation
  
- **`03_regression_discontinuity_design_example.ipynb`**: RDD analysis with robustness
  - Continuity assumption testing
  - Optimal bandwidth selection
  - Multiple specifications (linear, quadratic)
  - Sensitivity analysis

### Key Features by Method

#### Difference-in-Differences
- ✅ Basic 2x2 DiD estimation
- ✅ Regression-based DiD with controls
- ✅ Parallel trends testing
- ✅ Common shocks assumption checks
- ✅ Composition stability analysis
- ✅ Placebo tests
- ✅ Event study plots
- ✅ Robust standard errors

#### Propensity Score Matching
- ✅ Multiple propensity score models (logistic, random forest)
- ✅ Common support/overlap assessment
- ✅ Covariate balance testing (before/after)
- ✅ Multiple matching algorithms:
  - Nearest neighbor matching
  - Caliper matching  
  - Stratification matching
- ✅ Treatment effect estimation (simple difference, regression-adjusted)
- ✅ Match quality diagnostics

#### Regression Discontinuity Design
- ✅ Sharp RDD implementation
- ✅ Density continuity testing (McCrary test)
- ✅ Optimal bandwidth selection (Imbens-Kalyanaraman, cross-validation)
- ✅ Multiple specifications (linear, quadratic)
- ✅ Multiple kernels (triangular, uniform, epanechnikov)
- ✅ Sensitivity analysis
- ✅ Comprehensive visualization

## 🏗️ Project Structure

```
causalInferenceClasses/
├── causal_inference/           # Main package
│   ├── __init__.py            # Package initialization
│   ├── core/                  # Base classes and utilities
│   │   ├── base.py           # Abstract base class
│   │   └── exceptions.py     # Custom exceptions
│   ├── methods/               # Causal inference methods
│   │   ├── difference_in_differences.py
│   │   ├── propensity_score_matching.py
│   │   └── regression_discontinuity_design.py
│   ├── datasets/              # Data generation utilities
│   │   └── synthetic.py      # Synthetic data generators
│   └── utils/                 # Helper functions
│       └── validation.py     # Data validation utilities
├── examples/                  # Example notebooks and scripts
│   └── notebooks/            # Jupyter notebook examples
├── tests/                     # Comprehensive test suite
│   ├── test_did.py           # DiD tests
│   ├── test_psm.py           # PSM tests
│   └── test_rdd.py           # RDD tests
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## 🧪 Testing

The library includes a comprehensive test suite ensuring reliability:

```bash
# Run all tests
pytest tests/

# Run specific method tests
pytest tests/test_did.py -v
pytest tests/test_psm.py -v  
pytest tests/test_rdd.py -v

# Run with coverage report
pytest tests/ --cov=causal_inference --cov-report=html
```

**Test Coverage:**
- ✅ 28 tests covering all major functionality
- ✅ Data validation and error handling
- ✅ Method initialization and configuration
- ✅ Estimation accuracy and robustness
- ✅ Edge cases and boundary conditions

## 🎨 Design Principles

### Modular Architecture
- **Base Class**: Common interface for all causal inference methods
- **Method Classes**: Specialized implementations for each technique
- **Utility Modules**: Reusable components for data handling and validation
- **Clean Separation**: Clear boundaries between data, methods, and utilities

### User Experience
- **Consistent API**: Similar patterns across all methods
- **Rich Output**: Both human-readable summaries and programmatic access
- **Comprehensive Validation**: Extensive input checking and helpful error messages
- **Interactive Examples**: Jupyter notebooks for learning and exploration

### Code Quality
- **Type Hints**: Full type annotation for better IDE support
- **Documentation**: Comprehensive docstrings and examples
- **Testing**: Extensive test coverage with pytest
- **Standards**: Following PEP 8 and modern Python practices

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes with tests
4. **Ensure** all tests pass (`pytest tests/`)
5. **Commit** your changes (`git commit -m 'Add amazing feature'`)
6. **Push** to the branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/causalInferenceClasses.git
cd causalInferenceClasses

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this library in your research, please cite:

```bibtex
@software{causal_inference_classes,
  title={Causal Inference Classes: A Python Library for Causal Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/causalInferenceClasses}
}
```

## 🙏 Acknowledgments

- Inspired by the excellent work in the causal inference community
- Built with modern Python best practices and tools
- Designed for both academic research and practical applications
- Special thanks to the open-source community for foundational libraries

---

**Ready to start your causal inference journey?** 🚀 Check out the [example notebooks](examples/notebooks/) to see the library in action!
