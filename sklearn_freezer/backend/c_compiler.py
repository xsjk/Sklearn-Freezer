import importlib.machinery
import os.path
import tempfile


def silent_setup(**kwargs):
    from contextlib import redirect_stderr, redirect_stdout
    from setuptools import setup

    with open(os.devnull, "w") as null, redirect_stdout(null), redirect_stderr(null):
        setup(**kwargs)


def c_compile(code: str, func_name: str, arg_names: list[str], module_name: str | None = None, reuse_output: bool = True):
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
    arg_names : list[str]
        A list of argument names for the function.
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
        from setuptools import Extension
    except ImportError:
        raise ImportError("setuptools is required for C compilation")

    code = f"#include <Python.h>\n\n{code}\n\n"
    if reuse_output:
        code += f"static PyObject* {func_name}_cache;\n\n"
    code += "#define AsDouble(o) (((PyFloatObject*)(o))->ob_fval)\n"
    code += f"static PyObject* {func_name}_wrapper(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {{\n"
    for i, name in enumerate(arg_names):
        code += f"    double {name} = AsDouble(args[{i}]);\n"
    if reuse_output:
        code += f"    AsDouble({func_name}_cache) = f({', '.join(arg_names)});\n"
        code += f"    Py_INCREF({func_name}_cache);\n"
        code += f"    return {func_name}_cache;\n"
    else:
        code += f"    return PyFloat_FromDouble(f({', '.join(arg_names)}));\n"
    code += "}\n"
    code += f'static PyMethodDef MyMethods[] = {{{{ "{func_name}", {func_name}_wrapper, METH_FASTCALL, NULL }}}};\n'

    epilogue = 'static struct PyModuleDef {module_name} = {{ PyModuleDef_HEAD_INIT, "{module_name}", NULL, 1, MyMethods }};\n'
    epilogue += "PyMODINIT_FUNC PyInit_{module_name}(void) {{ \n"
    if reuse_output:
        epilogue += f"    {func_name}_cache = PyFloat_FromDouble(0.0);\n"
    epilogue += "    return PyModule_Create(&{module_name});\n"
    epilogue += "}}\n"

    compile_flags = [
        "-Ofast",
        "-Wno-incompatible-pointer-types",  # the function signature of _PyCFunctionFast differs from PyCFunction
    ]

    if module_name is None:
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            src_path = f.name
            dirname, filename = os.path.split(src_path)
            module_name, _ = os.path.splitext(filename)
            f.write((code + epilogue.format(module_name=module_name)).encode())

        module = Extension(module_name, sources=[src_path], extra_compile_args=compile_flags)
        silent_setup(name=module_name, ext_modules=[module], script_args=["build_ext", "-b", dirname])
        importer = importlib.machinery.PathFinder
        assert (spec := importer.find_spec(module_name, [dirname])) is not None
        assert (loader := spec.loader) is not None
        assert (module := loader.create_module(spec)) is not None
        assert (module_path := module.__file__) is not None
        os.remove(src_path)
        os.remove(module_path)

    else:
        src_path = f"{module_name}.c"
        with open(src_path, "w") as f:
            f.write((code + epilogue.format(module_name=module_name)))
        module = Extension(module_name, sources=[src_path], extra_compile_args=compile_flags)
        silent_setup(name=module_name, ext_modules=[module], script_args=["build_ext", "--inplace"])
        module = importlib.import_module(module_name)

    return getattr(module, func_name)
