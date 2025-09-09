import os
import sys
import time
from typing import Callable

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sklearn_freezer as skf


def compile_backends(clf, save_module_prefix: str, backends: tuple = ("python", "cython", "c"), is_batch: bool = False) -> dict[str, Callable]:
    compiled_funcs: dict[str, Callable] = {}

    for backend in backends:
        try:
            if is_batch:
                compiled_funcs[backend] = skf.compile(clf.predict_proba, backend, module_name=f"{save_module_prefix}_{backend}_batch", batch_mode="numpy")
            else:
                compiled_funcs[backend] = skf.compile(clf.predict_proba, backend, module_name=f"{save_module_prefix}_{backend}")
            print(f"✓ Successfully compiled with {backend} backend{' (batch)' if is_batch else ''}")
        except Exception as e:
            print(f"✗ Failed to compile with {backend} backend: {e}")

    return compiled_funcs


def compare_single_prediction(clf, compiled_funcs: dict[str, Callable], sample: np.ndarray) -> None:
    original_proba = clf.predict_proba([sample])[0, 1]  # assume binary classification
    print(f"\nOriginal predict_proba result: {original_proba:.6f}")

    for backend, func in compiled_funcs.items():
        try:
            compiled_proba = func(*sample)
            diff = abs(original_proba - compiled_proba)
            print(f"{backend.capitalize():>7} backend result: {compiled_proba:.6f} (diff: {diff:.2e})")
        except Exception as e:
            print(f"{backend.capitalize():>7} backend failed: {e}")


def compare_batch_prediction(clf, compiled_funcs: dict[str, Callable], X_batch: np.ndarray) -> None:
    original_proba = clf.predict_proba(X_batch)[:, 1]  # assume binary classification
    print("\nBatch prediction comparison (first 5 results):")
    print(f"Original: {original_proba[:5]}")

    for backend, func in compiled_funcs.items():
        try:
            compiled_proba = func(X_batch)
            max_diff = np.max(np.abs(original_proba - compiled_proba))
            print(f"{backend.capitalize():>7}: {compiled_proba[:5]} (max diff: {max_diff:.2e})")

            if np.allclose(original_proba, compiled_proba, rtol=1e-10):
                print(f"{backend.capitalize():>7}: ✓ Results match within tolerance")
            else:
                print(f"{backend.capitalize():>7}: ✗ Results differ significantly")
        except Exception as e:
            print(f"{backend.capitalize():>7} backend failed: {e}")


def benchmark_single_performance(clf, compiled_funcs: dict[str, Callable], sample: np.ndarray, n_iterations: int = 1000) -> dict[str, float]:
    print(f"\nSingle Prediction Benchmark ({n_iterations} iterations):")
    print("-" * 60)

    results = {}

    # Benchmark original by calling predict_proba repeatedly (reduced count to avoid very long runs)
    original_loop_count = max(1, n_iterations // 100)
    start_time = time.perf_counter()
    for _ in range(original_loop_count):
        clf.predict_proba([sample])
    original_loop_time = time.perf_counter() - start_time
    original_loop_per = original_loop_time / original_loop_count
    results["original_loop"] = original_loop_per
    print(f"Original (loop x{original_loop_count}): {original_loop_time:.4f}s ({original_loop_per * 1000:.3f}ms per prediction)")

    # Benchmark original by doing a single batch call of size n_iterations
    X_batch = np.tile(sample, (n_iterations, 1))
    start_time = time.perf_counter()
    clf.predict_proba(X_batch)
    original_batch_time = time.perf_counter() - start_time
    original_batch_per = original_batch_time / n_iterations
    results["original_batch"] = original_batch_per
    print(f"Original (batch {n_iterations}): {original_batch_time:.4f}s ({original_batch_per * 1000:.3f}ms per prediction)")

    # Benchmark compiled functions (per-call)
    for backend, func in compiled_funcs.items():
        try:
            start_time = time.perf_counter()
            for _ in range(n_iterations):
                func(*sample)
            compiled_time = time.perf_counter() - start_time
            compiled_per = compiled_time / n_iterations
            results[backend] = compiled_per

            speedup_vs_loop = original_loop_per / compiled_per
            speedup_vs_batch = original_batch_per / compiled_per

            print(f"{backend.capitalize():>7}: {compiled_time:.4f}s ({compiled_per * 1000:.3f}ms per prediction, {speedup_vs_loop:.1f}x vs original-loop, {speedup_vs_batch:.1f}x vs original-batch)")
        except Exception as e:
            print(f"{backend.capitalize():>7}: Failed - {e}")
            results[backend] = float("inf")

    return results


def benchmark_batch_performance(clf, compiled_funcs: dict[str, Callable], X_batch: np.ndarray) -> dict[str, float]:
    print(f"\nBatch Prediction Benchmark ({X_batch.shape[0]} samples):")
    print("-" * 60)

    results = {}

    # Benchmark original sklearn model
    start_time = time.perf_counter()
    clf.predict_proba(X_batch)
    original_time = time.perf_counter() - start_time
    original_per_sample = original_time / X_batch.shape[0]
    results["original"] = original_per_sample
    print(f"Original: {original_time:.4f}s ({original_per_sample * 1000:.3f}ms per sample)")

    # Benchmark compiled batch functions
    for backend, func in compiled_funcs.items():
        try:
            start_time = time.perf_counter()
            func(X_batch)
            compiled_time = time.perf_counter() - start_time
            compiled_per_sample = compiled_time / X_batch.shape[0]
            results[backend] = compiled_per_sample

            speedup = original_per_sample / compiled_per_sample
            print(f"{backend.capitalize():>7}: {compiled_time:.4f}s ({compiled_per_sample * 1000:.3f}ms per sample, {speedup:.1f}x speedup)")
        except Exception as e:
            print(f"{backend.capitalize():>7}: Failed - {e}")
            results[backend] = float("inf")

    return results


def run_comprehensive_benchmark(clf, save_module_prefix: str, X_test: np.ndarray) -> None:
    name = clf.__class__.__name__
    print(f"\n{'=' * 20} Benchmarking {name} {'=' * 20}")

    # Single prediction benchmark
    print("\n" + "=" * 60)
    print("SINGLE PREDICTION BENCHMARK")
    print("=" * 60)

    single_compiled_funcs = compile_backends(clf, backends=("python", "cython", "c"), save_module_prefix=save_module_prefix, is_batch=False)
    sample = X_test[0]

    if single_compiled_funcs:
        compare_single_prediction(clf, single_compiled_funcs, sample)
        benchmark_single_performance(clf, single_compiled_funcs, sample, n_iterations=100000)

    # Batch prediction benchmark
    print("\n" + "=" * 60)
    print("BATCH PREDICTION BENCHMARK")
    print("=" * 60)

    batch_compiled_funcs = compile_backends(clf, backends=("cython", "c"), save_module_prefix=save_module_prefix, is_batch=True)

    # Use different batch sizes for testing
    batch_sizes = [1000, 100000, 1000000]

    for batch_size in batch_sizes:
        X_batch = X_test[np.random.randint(0, len(X_test), size=batch_size)]
        print(f"\n--- Batch Size: {batch_size} ---")

        if batch_compiled_funcs:
            compare_batch_prediction(clf, batch_compiled_funcs, X_batch)
            benchmark_batch_performance(clf, batch_compiled_funcs, X_batch)


# Generate sample data
X, y = make_classification(
    random_state=42,
    n_samples=100000,
    n_features=8,
    n_informative=7,
    n_redundant=1,
    n_classes=2,
)

# Split data for testing
X_train, X_test = X[:80000], X[80000:]
y_train, y_test = y[:80000], y[80000:]

# Benchmark different models
for save_module_prefix, clf in [
    ("decision_tree_depth_4", DecisionTreeClassifier(random_state=42, max_depth=4).fit(X_train, y_train)),
    ("decision_tree_depth_10", DecisionTreeClassifier(random_state=42, max_depth=10).fit(X_train, y_train)),
    ("random_forest_2_trees_depth_2", RandomForestClassifier(random_state=42, n_estimators=2, max_depth=2).fit(X_train, y_train)),
    ("random_forest_100_trees_depth_5", RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5).fit(X_train, y_train)),
]:
    run_comprehensive_benchmark(clf, save_module_prefix, X_test)
