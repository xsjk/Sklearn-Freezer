import importlib.machinery
import importlib.util
import os.path
from importlib.abc import MetaPathFinder


def split_path(src_path: str) -> tuple[str, str, str]:
    dir_name, file_name = os.path.split(src_path)
    module_name, ext = os.path.splitext(file_name)
    return dir_name, module_name, ext


def import_module_from_dir(module_name: str, directory: str = ".", importer: MetaPathFinder = importlib.machinery.PathFinder()):
    assert (spec := importer.find_spec(module_name, [directory])) is not None
    module = importlib.util.module_from_spec(spec)
    assert (loader := spec.loader) is not None
    loader.exec_module(module)
    return module
