# Sklearn-Freezer

High-performance scikit-learn classifier compilation for ultra-fast single-sample inference.

## Overview

Sklearn-Freezer compiles scikit-learn classifiers' `predict` or `predict_proba` methods into optimized implementations, dramatically improving single-sample prediction performance for real-time applications.

**Disclaimer:** This is a simple optimization created to address performance bottlenecks from iterative scikit-learn model calls. The implementation is naive with limited model support. Currently only `predict_proba` for binary classification of `RandomForestClassifier` and `DecisionTreeClassifier` are supported.

## Compilation Backends

- **Python**: Pure Python implementation (baseline)
- **Cython**: Cython-compiled implementation (significant speedup)
- **C**: Native C extension (maximum speed)

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
from sklearn_freezer import compile_predict_proba

# Train and compile model
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
clf = RandomForestClassifier(random_state=42).fit(X, y)
compiled_func = compile_predict_proba(clf, backend="c")

# Fast single-sample prediction
probability = compiled_func(*X[0])
```

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

**Optional:** `cython`, `setuptools`
