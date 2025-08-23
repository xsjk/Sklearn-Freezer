import time
from typing import Callable

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn_freezer import compile_predict_proba
import numpy as np


def compile_backends(clf, backends=("python", "cython", "c")) -> dict[str, Callable]:
    compiled_funcs: dict[str, Callable] = {}
    for backend in backends:
        try:
            compiled_funcs[backend] = compile_predict_proba(clf, backend)
            print(f"✓ Successfully compiled with {backend} backend")
        except Exception as e:
            print(f"✗ Failed to compile with {backend} backend: {e}")
    return compiled_funcs


def compare_single_prediction(clf, compiled_funcs: dict[str, Callable], sample):
    original_proba = clf.predict_proba([sample])[0, 1]  # assume binary classification
    print(f"\nOriginal predict_proba result: {original_proba:.6f}")

    for backend, func in compiled_funcs.items():
        try:
            compiled_proba = func(*sample)
            diff = abs(original_proba - compiled_proba)
            print(f"{backend.capitalize():>7} backend result: {compiled_proba:.6f} (diff: {diff:.2e})")
        except Exception as e:
            print(f"{backend.capitalize():>7} backend failed: {e}")


def benchmark_performance(clf, compiled_funcs: dict[str, Callable], sample, n_iterations: int = 1000):
    print(f"\nPerformance Benchmark ({n_iterations} single predictions):")
    print("-" * 50)

    # Benchmark original by calling predict_proba repeatedly (reduced count to avoid very long runs)
    original_loop_count = max(1, n_iterations // 100)
    start_time = time.time()
    for _ in range(original_loop_count):
        clf.predict_proba([sample])
    original_loop_time = time.time() - start_time
    original_loop_per = original_loop_time / original_loop_count
    print(f"Original (loop x{original_loop_count}): {original_loop_time:.4f}s ({original_loop_per * 1000:.3f}ms per prediction)")

    # Benchmark original by doing a single batch call of size n_iterations
    X_batch = np.tile(sample, (n_iterations, 1))
    start_time = time.time()
    clf.predict_proba(X_batch)
    original_batch_time = time.time() - start_time
    original_batch_per = original_batch_time / n_iterations
    print(f"Original (batch {n_iterations}): {original_batch_time:.4f}s ({original_batch_per * 1000:.3f}ms per prediction)")

    # Benchmark compiled functions (per-call)
    for backend, func in compiled_funcs.items():
        try:
            start_time = time.time()
            for _ in range(n_iterations):
                func(*sample)
            compiled_time = time.time() - start_time
            compiled_per = compiled_time / n_iterations

            speedup_vs_loop = original_loop_per / compiled_per if compiled_per > 0 else float("inf")
            speedup_vs_batch = original_batch_per / compiled_per if compiled_per > 0 else float("inf")

            print(f"{backend.capitalize():>7}: {compiled_time:.4f}s ({compiled_per * 1000:.3f}ms per prediction, {speedup_vs_loop:.1f}x vs original-loop, {speedup_vs_batch:.1f}x vs original-batch)")
        except Exception as e:
            print(f"{backend.capitalize():>7}: Failed - {e}")


if __name__ == "__main__":
    # Generate sample data
    X, y = make_classification(random_state=42, n_samples=10000, n_features=8, n_informative=7, n_redundant=1, n_classes=2)

    for clf in (
        DecisionTreeClassifier(random_state=42, max_depth=10).fit(X, y),
        RandomForestClassifier(random_state=42, n_estimators=2, max_depth=2).fit(X, y),
        RandomForestClassifier(random_state=42, n_estimators=500, max_depth=10).fit(X, y),
    ):
        name = clf.__class__.__name__
        print(f"\n{'=' * 20} Benchmarking {name} {'=' * 20}")

        # Compile predict_proba for different backends
        compiled_funcs = compile_backends(clf)

        # Test single prediction
        sample = X[0]
        compare_single_prediction(clf, compiled_funcs, sample)

        # Benchmark performance
        benchmark_performance(clf, compiled_funcs, sample, n_iterations=100000)
