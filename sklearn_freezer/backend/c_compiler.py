import importlib
import os
import re
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from typing import Protocol

from . import utils


class FloatCallable(Protocol):
    def __call__(self, *args: float) -> float: ...


def silent_setup(**kwargs) -> None:
    from setuptools import setup

    with open(os.devnull, "w") as null, redirect_stdout(null), redirect_stderr(null):
        setup(**kwargs)


def get_n_args_from_code(code: str, func_name: str, return_type: str, arg_type: str) -> int:
    """
    Counts the number of arguments of a specific type in the parameter list of a given
    function name within the provided C code.
    """
    assert (m := re.search(rf"\b{return_type}\b\s+\b{func_name}\s*\(([^)]*)\)", code)) is not None
    param_list = m.group(1).strip()
    return len(re.findall(rf"\b{arg_type}\b", param_list))


# Ignore the warning about the function signature of _PyCFunctionFast differs from PyCFunction
C_PREAMBLE = """
#if defined(_MSC_VER)
#pragma warning(disable: 4113) 
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
#endif

#include <Python.h>
"""


def compile(code: str, func_name: str, module_name: str | None = None, reuse_output: bool = True) -> FloatCallable:
    """
    Compile the provided C code into a callable Python function.

    This function takes a string of C code, wraps it, and compiles it into a Python C extension.
    The specified function is then exported and can be called from Python.

    Parameters
    ----------
    code : str
        The C code to compile.
    func_name : str
        The name of the function to export.
    module_name : str, optional
        The name of the module to create. If None, a temporary module is generated;
        otherwise, the module is created in the current directory.
    reuse_output : bool, optional
        Determines whether to reuse the output PyObject for performance optimization.

    Returns
    -------
    Any
        The compiled function.

    Notes
    -----
    This module is optimized for ultra-low latency. It uses METH_FASTCALL to reduce overhead
    and omits argument validation and exception handling. Additionally, it reuses the output
    PyObject for performance, which makes the function non-thread-safe. If thread safety is
    required, set `reuse_output` to False.
    """
    try:
        # Import setuptools dynamically to avoid requiring its installation unless C backend is used
        from setuptools import Extension
    except ImportError:
        raise ImportError("setuptools is required for C compilation")

    # Assume the return type is double and argument types are double
    num_args = get_n_args_from_code(code, func_name, return_type="double", arg_type="double")

    code = f"{C_PREAMBLE}\n\n{code}\n\n"
    if reuse_output:
        code += f"static PyObject* {func_name}_cache;\n\n"
    code += "#define AsDouble(o) (((PyFloatObject*)(o))->ob_fval)\n"
    code += f"static PyObject* {func_name}_wrapper(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {{\n"
    code += f"    double result = {func_name}({', '.join(f'AsDouble(args[{i}])' for i in range(num_args))});\n"
    if reuse_output:
        code += f"    AsDouble({func_name}_cache) = result;\n"
        code += f"    Py_INCREF({func_name}_cache);\n"
        code += f"    return {func_name}_cache;\n"
    else:
        code += "    return PyFloat_FromDouble(result);\n"
    code += "}\n"
    code += "static PyMethodDef MyMethods[] = {\n"
    code += f'    {{ "{func_name}", {func_name}_wrapper, METH_FASTCALL, NULL }},\n'
    code += "    { NULL, NULL, 0, NULL }\n"
    code += "};\n"

    epilogue = 'static struct PyModuleDef {module_name} = {{ PyModuleDef_HEAD_INIT, "{module_name}", NULL, -1, MyMethods }};\n'
    epilogue += "PyMODINIT_FUNC PyInit_{module_name}(void) {{ \n"
    if reuse_output:
        epilogue += f"    {func_name}_cache = PyFloat_FromDouble(0.0);\n"
    epilogue += "    return PyModule_Create(&{module_name});\n"
    epilogue += "}}\n"

    if module_name is None:
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            dir_name, module_name, _ = utils.split_path(src_path := f.name)
            code += epilogue.format(module_name=module_name)
            f.write(code.encode())
        extension = Extension(module_name, sources=[src_path])
        silent_setup(name=module_name, packages=[], ext_modules=[extension], script_args=["build_ext", "-b", dir_name])
        module = utils.import_module_from_dir(module_name, dir_name)
        assert (module_path := module.__file__) is not None
        os.remove(src_path)
        os.remove(module_path)

    else:
        code += epilogue.format(module_name=module_name)
        src_path = f"{module_name}.c"
        with open(src_path, "w") as f:
            f.write(code)
        extension = Extension(module_name, sources=[src_path])
        silent_setup(name=module_name, packages=[], ext_modules=[extension], script_args=["build_ext", "--inplace"])
        module = importlib.import_module(module_name)

    return getattr(module, func_name)
