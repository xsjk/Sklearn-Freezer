NUMPY_BATCH_FUNCTION_TEMPLATE = """
import numpy as np
def {func_name}_batch(arr: np.ndarray):
    if arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")
    if arr.dtype != np.float64:
        raise ValueError("Input array must be of type float64")
    return np.array([{func_name}(*arg) for arg in arr])
"""


def default_wrapper(code: str, func_name: str, module_name: str) -> tuple[str, str]:
    return code, func_name


def batch_wrapper_numpy(code: str, func_name: str, module_name: str) -> tuple[str, str]:
    batch_code = NUMPY_BATCH_FUNCTION_TEMPLATE.format(func_name=func_name)
    code = f"{batch_code}\n\n{code}\n"
    return code, f"{func_name}_batch"
