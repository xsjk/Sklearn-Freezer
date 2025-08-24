from types import MethodType
from typing import Callable, Literal

from .backend import get_compiler
from .codegen import get_codegen


def compile(method: MethodType, backend: Literal["python", "c", "cython"], **kwargs) -> Callable:
    """Compile the method of a given classifier.

    This function takes a classifier and compiles its method
    into a more efficient form for the specified backend.

    Parameters
    ----------
    method : MethodType
        The method of the classifier whose implementation is to be compiled.
    backend : {"python", "c", "cython"}
        The backend used for compilation.
        This determines the compilation method used.
    kwargs : dict
        Additional keyword arguments to pass to the compilation function.

    Returns
    -------
    Callable[..., float]
        A callable that takes the features as input and returns the predicted probabilities.
    """
    # Load compilation backend
    compiler = get_compiler(backend)

    # Get code generator
    codegen = get_codegen(method, backend)

    # Compile the method
    code = codegen(func_name="f", **kwargs)
    compiled_func = compiler(code, func_name="f")
    return compiled_func
