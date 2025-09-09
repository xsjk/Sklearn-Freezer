from . import utils

CYTHON_PREAMBLE = """
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: autotestdict=False
"""

NUMPY_BATCH_FUNCTION_TEMPLATE = """
import numpy as np
cimport numpy as cnp
from cython import boundscheck, wraparound
from cython.parallel import prange

cnp.import_array()

@boundscheck(False)
@wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=1] {func_name}_batch(cnp.ndarray[cnp.float64_t, ndim=2] X):
    cdef cnp.float64_t[::1] y = np.empty(X.shape[0], dtype=np.float64)
    if X.shape[1] != {num_features}:
        raise ValueError("Input array must have {num_features} columns")
    for i in prange(X.shape[0], nogil=True):
        y[i] = {func_call}
    return np.asarray(y)
"""


def default_wrapper(code: str, func_name: str, module_name: str) -> tuple[str, str]:
    return code, func_name


def batch_wrapper_numpy(code: str, func_name: str, module_name: str) -> tuple[str, str]:
    num_args = utils.get_n_args_from_code(code, func_name, "double", "double")
    func_call = f"{func_name}({', '.join([f'X[i, {j}]' for j in range(num_args)])})"
    batch_code = NUMPY_BATCH_FUNCTION_TEMPLATE.format(func_name=func_name, num_features=num_args, func_call=func_call)
    code = f"{batch_code}\n\n{code}\n"
    return code, f"{func_name}_batch"
