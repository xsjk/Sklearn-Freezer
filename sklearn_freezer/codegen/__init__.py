import importlib
import inspect
import pkgutil
from functools import partial
from types import MethodType, ModuleType
from typing import Protocol


class CodeGen(Protocol):
    def __call__(self, func_name: str) -> str: ...


supported: dict[type, ModuleType] = {}
for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if not is_pkg:
        module = importlib.import_module(f".{module_name}", package=__name__)
        supported[module.TargetModel] = module


def get_impl(model: type) -> ModuleType:
    if model in supported:
        return supported[model]
    raise NotImplementedError(f"Model type {model.__name__} not supported yet. Supported models are {[t.__name__ for t in supported]}.")


def get_codegen(method: MethodType, backend: str) -> CodeGen:
    """Get the code generation function for a specific method and backend.

    Parameters
    ----------
    method : MethodType
        The method to generate code for.
    backend : str
        The backend to use for code generation.

    Returns
    -------
    CodeGen
        The code generation function for the specified method and backend.
    """
    if not inspect.ismethod(method):
        raise TypeError(f"Expected a method but got {type(method).__name__}")

    # Find the base model class that implements the method
    base_model: type = inspect._findclass(method)  # type: ignore

    # Get the implementation module for this model class
    impl: ModuleType = get_impl(base_model)

    # Get the code generation function for this method and backend
    codegen_impl = getattr(impl, f"generate_{method.__name__}_{backend}", None)
    if codegen_impl is None:
        raise NotImplementedError(f"Code generation for {method.__qualname__} with backend {backend} is not supported yet.")

    # Make a code generation function
    estimator = method.__self__
    n = getattr(estimator, "n_features_in_", None)
    if n is None:
        raise ValueError(f"Estimator {estimator} does not have n_features_in_ attribute.")
    arg_names = [f"x{i:0{len(str(n - 1))}d}" for i in range(n)]
    return partial(codegen_impl, estimator, arg_names)
