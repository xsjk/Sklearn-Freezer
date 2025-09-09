import importlib
import pkgutil
import re
from types import ModuleType
from typing import Callable, Protocol

from . import core


class BackendCompiler(Protocol):
    def __call__(self, code: str, func_name: str, module_name: str | None = None, batch_mode: str | None = None, silent: bool = True, **kwargs) -> Callable: ...


supported: dict[str, ModuleType] = {}
for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if m := re.match(r"(.*)_compiler$", module_name):
        backend = m.group(1)
        module = importlib.import_module(f".{module_name}", package=__name__)
        supported[backend] = module


def get_include_dirs(batch_mode: str | None) -> list[str]:
    if batch_mode is None:
        return []
    match batch_mode:
        case "numpy":
            import numpy as np

            return [np.get_include()]
        case _:
            raise NotImplementedError(f"Batch mode '{batch_mode}' is not supported.")


def get_extra_compile_args(batch_mode: str | None) -> list[str]:
    from setuptools._distutils.ccompiler import get_default_compiler

    compiler = get_default_compiler()
    args = []
    match compiler:
        case "msvc":
            args.extend(["/O2", "/fp:fast", "/GL"])
            if batch_mode is not None:
                args.append("/openmp")
        case "unix":
            args.extend(["-O3"])
            if batch_mode is not None:
                args.append("-fopenmp")
        case _:
            raise NotImplementedError(f"Compiler '{compiler}' is not supported.")
    return args


def get_extra_link_args(batch_mode: str | None) -> list[str]:
    from setuptools._distutils.ccompiler import get_default_compiler

    compiler = get_default_compiler()
    args = []
    if compiler == "unix" and batch_mode is not None:
        args.append("-fopenmp")
    return args


def get_extension(backend: str) -> str:
    match backend:
        case "python":
            return "py"
        case "cython":
            return "pyx"
        case "c":
            return "c"
        case _:
            raise NotImplementedError(f"Backend '{backend}' is not supported.")


def get_compiler(backend: str) -> BackendCompiler:
    """Get the compiler function for a specific backend.

    Parameters
    ----------
    backend : str
        The name of the backend.

    Returns
    -------
    BackendCompiler
        The compiler function for the specified backend.
    """
    if backend not in supported:
        raise ValueError(f"Backend {backend} not supported. Supported backends are: {list(supported)}")

    module = supported[backend]
    extension = get_extension(backend)

    def compile(code: str, func_name: str, module_name: str | None = None, batch_mode: str | None = None, silent: bool = True, **kwargs) -> Callable:
        if batch_mode is None:
            wrapper = module.default_wrapper
        else:
            wrapper = getattr(module, f"batch_wrapper_{batch_mode}")

        include_dirs = get_include_dirs(batch_mode)
        extra_compile_args = get_extra_compile_args(batch_mode)
        extra_link_args = get_extra_link_args(batch_mode)

        return core.compile_function(
            code=code,
            func_name=func_name,
            module_name=module_name,
            extension=extension,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            wrapper=wrapper,
            silent=silent,
            **kwargs,
        )

    return compile
