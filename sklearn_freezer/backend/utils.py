import importlib.machinery
import importlib.util
import os.path
import re
from contextlib import redirect_stderr, redirect_stdout
from importlib.abc import MetaPathFinder
from types import ModuleType


def split_path(src_path: str) -> tuple[str, str, str]:
    dir_name, file_name = os.path.split(src_path)
    module_name, ext = os.path.splitext(file_name)
    return dir_name, module_name, ext


def import_module_from_dir(module_name: str, directory: str = ".", importer: MetaPathFinder = importlib.machinery.PathFinder()) -> ModuleType:
    assert (spec := importer.find_spec(module_name, [directory])) is not None
    module = importlib.util.module_from_spec(spec)
    assert (loader := spec.loader) is not None
    loader.exec_module(module)
    return module


def get_n_args_from_code(code: str, func_name: str, return_type: str, arg_type: str) -> int:
    """
    Counts the number of arguments of a specific type in the parameter list of a given
    function name within the provided C or Cython code.
    """
    assert (m := re.search(rf"\b{return_type}\b\s+\b{func_name}\s*\(([^)]*)\)", code)) is not None
    param_list = m.group(1).strip()
    return len(re.findall(rf"\b{arg_type}\b", param_list))


def silent_setup(**kwargs) -> None:
    from setuptools import setup

    # Note: On Windows, MSVC output may not be fully suppressed by redirection.
    # This function only suppresses output where redirection works.
    with open(os.devnull, "w") as null, redirect_stdout(null), redirect_stderr(null):
        setup(**kwargs)
