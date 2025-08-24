import os
import tempfile
from typing import Callable

from . import utils

pyx_importer = None


# Import pyximport dynamically to avoid requiring its installation unless Cython backend is used
def get_pyx_importer():
    global pyx_importer
    if pyx_importer is None:
        try:
            import pyximport
        except ImportError:
            raise ImportError("cython is required for Cython compilation")

        _, pyx_importer = pyximport.install()
        assert pyx_importer is not None
    return pyx_importer


CYTHON_PREAMBLE = """
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: autotestdict=False
"""


def compile(code: str, func_name: str, module_name: str | None = None) -> Callable:
    """
    Compile the given Cython code.

    This function takes a string of Cython code and compiles it into a callable function
    using the provided function name.

    Parameters
    ----------
    code : str
        The Cython code to compile.
    func_name: str
        The name of the function to extract from the compiled module.
    module_name : str, optional
        The name of the module to create. When None, a temporary module is created.
        Otherwise, the module will be created in the current directory.

    Returns
    -------
    The compiled function.
    """
    code = f"{CYTHON_PREAMBLE}\n{code}"
    if module_name is None:
        # Create a temporary module
        with tempfile.NamedTemporaryFile(suffix=".pyx", delete=False) as f:
            f.write(code.encode())
            src_path = f.name

        # Import and clean up
        dir_name, module_name, _ = utils.split_path(src_path)
        try:
            module = utils.import_module_from_dir(module_name, dir_name, importer=get_pyx_importer())
        finally:
            os.remove(src_path)
    else:
        # Check if module with identical code already exists
        src_path = f"{module_name}.pyx"
        if os.path.exists(src_path):
            try:
                with open(src_path, "r") as f:
                    if f.read() == code:
                        module = utils.import_module_from_dir(module_name, importer=get_pyx_importer())
                        return getattr(module, func_name)
            except Exception:
                pass  # Fall through to writing the file

        # Write and import the module
        with open(src_path, "w") as f:
            f.write(code)
        module = utils.import_module_from_dir(module_name, importer=get_pyx_importer())

    return getattr(module, func_name)
