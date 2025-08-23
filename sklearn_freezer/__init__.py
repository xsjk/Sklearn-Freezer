"""
Sklearn-Freezer: Optimize scikit-learn classifier inference through compilation.

This package provides tools to compile scikit-learn classifier predict/predict_proba
methods into optimized static C extensions for improved single-sample inference performance.
"""

from .models import compile_predict_proba, supported_models

__all__ = [
    "compile_predict_proba",
    "supported_models",
]
