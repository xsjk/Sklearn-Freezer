import os
import sys
import time
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn_freezer import get_compiler


def compile_square_functions(is_batch: bool = False) -> dict[str, Callable]:
    python_compiler = get_compiler("python")
    cython_compiler = get_compiler("cython")
    c_compiler = get_compiler("c")

    python_code = "def square(a): return a * a"
    cython_code = "cpdef double square(double a) noexcept nogil: return a * a"
    c_code = "double square(double a) { return a * a; }"

    funcs = {}

    try:
        if is_batch:
            funcs["cython"] = cython_compiler(cython_code, "square", batch_mode="numpy")
            funcs["c"] = c_compiler(c_code, "square", batch_mode="numpy")
        else:
            funcs["python"] = python_compiler(python_code, "square")
            funcs["cython"] = cython_compiler(cython_code, "square")
            funcs["c"] = c_compiler(c_code, "square")

        print(f"✓ All backends compiled{' (batch)' if is_batch else ''}")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")

    return funcs


def validate_correctness(funcs: dict[str, Callable], test_val: float = 3.0) -> bool:
    expected = test_val * test_val
    print(f"\nValidating correctness with test value: {test_val}")
    print(f"Expected result: {expected}")

    all_correct = True
    for backend, func in funcs.items():
        try:
            result = func(test_val)
            is_correct = abs(result - expected) < 1e-10
            all_correct &= is_correct
            status = "✓" if is_correct else "✗"
            print(f"{status} {backend.capitalize()}: {result}")
        except Exception as e:
            print(f"✗ {backend.capitalize()} failed: {e}")
            all_correct = False

    if all_correct:
        print("✓ All backends produce correct results")
    else:
        print("✗ Some backends produce incorrect results")

    return all_correct


def validate_batch_correctness(funcs: dict[str, Callable], test_array: np.ndarray) -> bool:
    assert test_array.ndim == 2 and test_array.shape[1] == 1, "Input must be shape (N, 1)"
    expected = test_array[:, 0] ** 2

    print(f"\nValidating batch correctness with array shape: {test_array.shape}")

    all_correct = True
    for backend, func in funcs.items():
        try:
            result = func(test_array)
            assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
            is_correct = np.allclose(result, expected)
            all_correct &= is_correct
            status = "✓" if is_correct else "✗"
            max_diff = np.max(np.abs(result - expected))
            print(f"{status} {backend.capitalize()}: max diff = {max_diff:.2e}")
        except Exception as e:
            print(f"✗ {backend.capitalize()} failed: {e}")
            all_correct = False

    print("✓ All batch backends produce correct results" if all_correct else "✗ Some batch backends produce incorrect results")
    return all_correct


def benchmark_scalar_performance(funcs: dict[str, Callable], iterations: int = 10_000_000, arg: float = 2.0) -> dict[str, float]:
    print(f"\nScalar Performance Benchmark ({iterations:,} iterations)")
    print("-" * 50)

    results = {}
    for backend, func in funcs.items():
        try:
            start = time.perf_counter()
            for _ in range(iterations):
                func(arg)
            elapsed = time.perf_counter() - start
            results[backend] = elapsed

            ops_per_sec = iterations / elapsed
            print(f"{backend.capitalize():>7}: {elapsed:.4f}s ({ops_per_sec:,.0f} ops/sec)")
        except Exception as e:
            print(f"{backend.capitalize():>7}: Failed - {e}")
            results[backend] = float("inf")

    return results


def benchmark_batch_performance(funcs: dict[str, Callable], test_array: np.ndarray) -> dict[str, float]:
    print(f"\nBatch Performance Benchmark (array shape: {test_array.shape})")
    print("-" * 50)

    results = {}

    # NumPy baseline
    start = time.perf_counter()
    (test_array**2)[:, 0]
    numpy_time = time.perf_counter() - start
    results["numpy"] = numpy_time

    elements_per_sec = test_array.shape[0] / numpy_time
    print(f"   NumPy: {numpy_time:.4f}s ({elements_per_sec:,.0f} elements/sec)")

    # Compiled functions
    for backend, func in funcs.items():
        try:
            start = time.perf_counter()
            func(test_array)
            elapsed = time.perf_counter() - start
            results[backend] = elapsed

            elements_per_sec = test_array.shape[0] / elapsed
            speedup = numpy_time / elapsed
            print(f"{backend.capitalize():>7}: {elapsed:.4f}s ({elements_per_sec:,.0f} elements/sec, {speedup:.2f}x vs NumPy)")
        except Exception as e:
            print(f"{backend.capitalize():>7}: Failed - {e}")
            results[backend] = float("inf")

    return results


print("=" * 70)
print("BASIC COMPILER COMPREHENSIVE TEST")
print("=" * 70)
print()

# Scalar mode test
print("=" * 40)
print("SCALAR MODE TEST")
print("=" * 40)

scalar_funcs = compile_square_functions(is_batch=False)

if scalar_funcs:
    scalar_correct = validate_correctness(scalar_funcs)
    if scalar_correct:
        benchmark_scalar_performance(scalar_funcs)
print()

# Batch mode test
print("=" * 40)
print("BATCH MODE TEST")
print("=" * 40)

batch_funcs = compile_square_functions(is_batch=True)

if batch_funcs:
    test_sizes = [1_000, 100_000, 10_000_000]

    for size in test_sizes:
        print(f"\n--- Testing with array size: {size:,} ---")
        test_array = np.random.rand(size, 1)

        batch_correct = validate_batch_correctness(batch_funcs, test_array)
        if batch_correct:
            benchmark_batch_performance(batch_funcs, test_array)
