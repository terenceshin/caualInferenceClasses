# Migration Guide: From main.py to Structured Library

This guide helps you transition from the old single-file structure (`main.py`) to the new organized library structure.

## What Changed

### Old Structure (Before)
```
causalInferenceClasses/
├── main.py                    # Everything in one file
├── DiD_walkthrough.ipynb      # Jupyter notebooks
├── other_notebooks.ipynb
└── ...
```

### New Structure (After)
```
causalInferenceClasses/
├── causal_inference/          # Main library package
│   ├── __init__.py
│   ├── core/                  # Base classes and utilities
│   ├── methods/               # Causal inference methods
│   ├── utils/                 # Helper functions
│   └── datasets/              # Data generation
├── examples/                  # Example scripts and notebooks
├── tests/                     # Test suite
├── setup.py                   # Package installation
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## How to Update Your Code

### 1. Import Changes

**Old way:**
```python
from main import DifferenceInDifferences
```

**New way:**
```python
from causal_inference.methods import DifferenceInDifferences
# or
from causal_inference import DifferenceInDifferences
```

### 2. Data Generation

**Old way:**
```python
# You had to write your own data generation functions
```

**New way:**
```python
from causal_inference.datasets import generate_did_data, generate_did_data_with_dates

# Generate synthetic data
data = generate_did_data(
    n_units=100,
    n_periods=10,
    treatment_period=6,
    treatment_effect=2.5,
    random_seed=42
)
```

### 3. Utility Functions

**New features available:**
```python
from causal_inference.utils import validate_panel_data, check_balance
from causal_inference.utils import plot_parallel_trends, plot_treatment_effects

# Validate your data
validation_results = validate_panel_data(
    data, unit_var='unit_id', time_var='period',
    outcome_var='outcome', treatment_var='treated'
)

# Check covariate balance
balance_results = check_balance(
    data, treatment_var='treated', 
    covariates=['age', 'income', 'education']
)
```

## Installation Options

### Option 1: Development Installation (Recommended)
```bash
cd causalInferenceClasses
pip install -e .
```

This allows you to import the library from anywhere and automatically picks up changes.

### Option 2: Direct Usage
Keep using the library directly from the folder by adding the path:
```python
import sys
sys.path.append('path/to/causalInferenceClasses')
from causal_inference.methods import DifferenceInDifferences
```

## Updated Workflow Example

Here's how to update a typical analysis:

### Old Workflow
```python
from main import DifferenceInDifferences

# Your data preparation code here...

did = DifferenceInDifferences(
    data=df,
    outcome_var='orders',
    unit_var='market',
    time_var='week',
    treatment_var='treatment_group',
    post_var='post_treatment',
    treatment_start=13
)

results = did.run_all_checks()
```

### New Workflow
```python
from causal_inference.methods import DifferenceInDifferences
from causal_inference.datasets import generate_did_data
from causal_inference.utils import validate_panel_data

# Generate or load your data
data = generate_did_data(n_units=100, n_periods=10, treatment_period=6)

# Validate data (optional but recommended)
validation = validate_panel_data(
    data, unit_var='unit_id', time_var='period',
    outcome_var='outcome', treatment_var='treated'
)

# Run analysis (same API as before!)
did = DifferenceInDifferences(
    data=data,
    outcome_var='outcome',
    unit_var='unit_id',
    time_var='period',
    treatment_var='treated',
    post_var='post',
    treatment_start=6
)

results = did.run_all_checks()
```

## Benefits of New Structure

1. **Modularity**: Each component has its own file and purpose
2. **Testability**: Comprehensive test suite ensures reliability
3. **Extensibility**: Easy to add new methods and utilities
4. **Documentation**: Better organized with clear API documentation
5. **Package Management**: Proper dependency management with requirements.txt
6. **Professional Structure**: Follows Python packaging best practices

## Backward Compatibility

The core API remains the same! Your existing `DifferenceInDifferences` usage should work with minimal changes - just update the import statements.

## Next Steps

1. **Update imports** in your existing notebooks/scripts
2. **Install the package** using `pip install -e .`
3. **Run tests** to ensure everything works: `pytest tests/`
4. **Explore new features** like synthetic data generation and validation utilities
5. **Consider migrating** your notebooks to the `examples/notebooks/` folder

## Need Help?

- Check the `examples/` folder for working examples
- Run the test suite to see expected behavior
- Look at the API documentation in each module's docstrings
