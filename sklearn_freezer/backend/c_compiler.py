from . import utils

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

DEFAULT_FUNCTION_TEMPLATE = """
{preamble}

{code}

#define AsDouble(o) (((PyFloatObject*)(o))->ob_fval)
static PyObject* {func_name}_wrapper(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {{
    return PyFloat_FromDouble({func_name}({args}));
}}

static PyMethodDef methods[] = {{
    {{ "{func_name}", {func_name}_wrapper, METH_FASTCALL, NULL }},
    {{ NULL, NULL, 0, NULL }}
}};

static struct PyModuleDef module = {{ PyModuleDef_HEAD_INIT, "{module_name}", NULL, -1, methods }};

PyMODINIT_FUNC PyInit_{module_name}(void) {{
    return PyModule_Create(&module);
}}
"""


def default_wrapper(code: str, func_name: str, module_name: str) -> tuple[str, str]:
    num_args = utils.get_n_args_from_code(code, func_name, "double", "double")
    code = DEFAULT_FUNCTION_TEMPLATE.format(
        preamble=C_PREAMBLE,
        code=code,
        func_name=func_name,
        num_args=num_args,
        args=", ".join(f"AsDouble(args[{i}])" for i in range(num_args)),
        module_name=module_name,
    )
    return code, func_name


NUMPY_BATCH_FUNCTION_TEMPLATE = """
{preamble}

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

{code}

static PyObject* {func_name}_wrapper(PyObject *self, PyObject *arg)
{{
    const PyArrayObject *arr = (const PyArrayObject *)arg;
    if (!PyArray_Check(arr)) {{
        PyErr_SetString(PyExc_TypeError, "Expected a numpy array");
        return NULL;
    }}

    if (PyArray_TYPE(arr) != NPY_DOUBLE) {{
        PyErr_SetString(PyExc_TypeError, "Expected a numpy array of doubles");
        return NULL;
    }}

    if (!PyArray_ISCONTIGUOUS(arr)) {{
        PyErr_SetString(PyExc_TypeError, "Expected a contiguous numpy array");
        return NULL;
    }}

    if (PyArray_NDIM(arr) != 2) {{
        PyErr_SetString(PyExc_TypeError, "Expected a 2-dimensional numpy array");
        return NULL;
    }}

    npy_intp rows = PyArray_DIM(arr, 0);
    npy_intp cols = PyArray_DIM(arr, 1);
    if (cols != {num_args}) {{
        PyErr_SetString(PyExc_TypeError, "Expected a 2-dimensional numpy array with {num_args} columns");
        return NULL;
    }}

    // Create output array
    PyObject *out = PyArray_SimpleNew(1, &rows, NPY_DOUBLE);
    if (!out) {{
        return NULL;
    }}

    PyArrayObject *out_arr = (PyArrayObject *)out;

    double *data = (double *)PyArray_DATA(arr);
    double *out_data = (double *)PyArray_DATA(out_arr);
    
    npy_intp i = 0;
    #pragma omp parallel for
    for (i = 0; i < rows; i++) {{
        double *input = data + i * cols;
        out_data[i] = {func_name}({args});
    }}
    return out;
}}

static PyMethodDef methods[] = {{
    {{ "{func_name}", {func_name}_wrapper, METH_O, NULL }},
    {{ NULL, NULL, 0, NULL }}
}};

static struct PyModuleDef module = {{ PyModuleDef_HEAD_INIT, "{module_name}", NULL, -1, methods }};

PyMODINIT_FUNC PyInit_{module_name}(void) {{
    // Initialize the numpy array API
    import_array();
    return PyModule_Create(&module);
}}

"""


def batch_wrapper_numpy(code: str, func_name: str, module_name: str) -> tuple[str, str]:
    num_args = utils.get_n_args_from_code(code, func_name, "double", "double")

    code = NUMPY_BATCH_FUNCTION_TEMPLATE.format(
        preamble=C_PREAMBLE,
        code=code,
        func_name=func_name,
        num_args=num_args,
        args=", ".join(f"input[{i}]" for i in range(num_args)),
        module_name=module_name,
    )
    return code, func_name
