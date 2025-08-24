import time

from sklearn_freezer.backend import supported as supported_backend

python_compile = supported_backend["python"]
cython_compile = supported_backend["cython"]
c_compile = supported_backend["c"]

f_python = python_compile("def f(a): return a * a", "f")
f_cython = cython_compile("def f(double a): return a * a", "f")
f_c = c_compile("double f(double a) { return a * a; }", "f")

assert f_python(3.0) == f_cython(3.0) == f_c(3.0) == 9.0

start = time.perf_counter()
for _ in range(10000000):
    f_python(2.0)
print("Python time:", time.perf_counter() - start)

start = time.perf_counter()
for _ in range(10000000):
    f_cython(2.0)
print("Cython time:", time.perf_counter() - start)

start = time.perf_counter()
for _ in range(10000000):
    f_c(2.0)
print("C time:", time.perf_counter() - start)
