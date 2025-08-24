import importlib
import pkgutil
import re
from typing import Callable, Protocol


class BackendCompiler(Protocol):
    def __call__(self, code: str, func_name: str, module_name: str | None = None, **kwargs) -> Callable: ...


supported: dict[str, BackendCompiler] = {}
for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if m := re.match(r"(.*)_compiler$", module_name):
        backend = m.group(1)
        module = importlib.import_module(f".{module_name}", package=__name__)
        supported[backend] = getattr(module, "compile")


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
    return supported[backend]
