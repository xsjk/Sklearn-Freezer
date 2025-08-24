# Sklearn-Freezer

High-performance scikit-learn classifier compilation for ultra-fast single-sample inference.

## Overview

Sklearn-Freezer compiles scikit-learn classifiers' `predict` or `predict_proba` methods into optimized implementations, dramatically improving single-sample prediction performance for real-time applications.

**Disclaimer:** This is a simple optimization created to address performance bottlenecks from iterative scikit-learn model calls. The implementation is naive with limited model support. Currently only `predict_proba` for binary classification of `RandomForestClassifier` and `DecisionTreeClassifier` are supported.

## Installation

```bash
# Basic installation
pip install sklearn-freezer

# With optional backends
pip install sklearn-freezer[cython]  # Cython backend
pip install sklearn-freezer[c]       # C backend
pip install sklearn-freezer[all]     # All backends
```

## Quick Start

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sklearn_freezer as skf

# Train a model
X, y = make_classification(n_samples=1000, n_features=4, random_state=42, n_classes=2)
clf = RandomForestClassifier(random_state=42).fit(X, y)

# Compile for fast inference
compiled_func = skf.compile(clf.predict_proba, backend="c")

# Fast single-sample prediction
probability = compiled_func(*X[0])
```

## Compilation Caching

When you provide a `module_name` parameter to the compile function, the compiled module will be saved to the current working directory:

```python
# Compile and save with a specific name
compiled_func = skf.compile(clf.predict_proba, backend="c", module_name="my_compiled_model")
```

This enables efficient reuse of compiled models:

- If you compile again with the same `module_name`, the system first checks if the source code matches the existing file
- If the source code is identical, it directly imports the existing compiled module without recompiling
- If the source code differs, it recompiles the module with the updated code

## Compilation Backends

- **Python**: Pure Python implementation (baseline performance)
- **Cython**: Cython-compiled implementation (significant speedup over Python)
- **C**: Native C extension (maximum performance)

## Examples

- **Backend Comparison**: `example/simple_compile.py` - Compares performance across different compilation backends (Python, Cython, C)
- **Full Benchmark**: `example/benchmark.py` - Comprehensive comparison including original scikit-learn `predict_proba` methods vs compiled versions

## Performance

The benchmark compares five prediction approaches:

1. **Iterative clf.predict_proba**: Individual `clf.predict_proba([sample])` calls
2. **Compiled Python/Cython/C**: Using compiled backends
3. **Batch clf.predict_proba**: Single `clf.predict_proba(all_samples)` call

Performance ordering: iterative single-sample calls < compiled functions < batch processing.

Run benchmark: `python example/benchmark.py`

## Requirements

- Python 3.10+
- scikit-learn

**Optional dependencies:**

- `cython` (for Cython backend)
- `setuptools` (for C backend compilation)
