# Sklearn-Freezer

High-performance scikit-learn classifier compilation for ultra-fast inference.

## Overview

Compiles scikit-learn `predict_proba` methods into optimized implementations for real-time applications.

**Support:** `RandomForestClassifier` and `DecisionTreeClassifier` binary classification only.

## Installation

```bash
pip install sklearn-freezer[all]  # Includes all backends
```

## Quick Start

```python
import sklearn_freezer as skf
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np

# Train model
X, y = make_classification(n_samples=1000, n_features=4, random_state=42, n_classes=2)
clf = RandomForestClassifier(random_state=42).fit(X, y)

# Compile for single fast inference
fast_predict = skf.compile(clf.predict_proba, backend="c")

# Fast prediction
probability = fast_predict(*X[0])
```

## Backends

- **python**: Pure Python (baseline)
- **cython**: Cython-compiled (faster)
- **c**: Native C extension (fastest)

## Batch Processing

```python
# Compile for fast batch inference (cython/c only)
batch_predict = skf.compile(clf.predict_proba, backend="c", batch_mode="numpy")

# Example batch input
X_batch = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64)

# Get probabilities using the compiled implementation
probabilities = batch_predict(X_batch)

# Get probabilities using the original scikit-learn method
# Both results should be identical, but the compiled implementation is faster
probabilities = clf.predict_proba(X_batch)[:, 1]
```

## Module Caching

```python
# Save compiled module for reuse
fast_predict = skf.compile(clf.predict_proba, backend="c", module_name="my_model")
# Subsequent calls with same module_name will reuse if source unchanged
```

## Examples

```bash
python example/basic_compiler_test.py      # Basic compilation test
python example/sklearn_model_benchmark.py  # Model performance benchmark
```

## Requirements

- Python 3.10+
- scikit-learn
- cython (for Cython backend)
- setuptools (for C backend)
- a C compiler (for C backends)
