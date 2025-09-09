import importlib
import os.path
import tempfile
from types import ModuleType
from typing import Callable, Protocol
import importlib.util

from . import utils


def compile_module(
    module_name: str,
    sources: list[str],
    include_dirs: list[str] = [],
    extra_compile_args: list[str] = [],
    extra_link_args: list[str] = [],
    dir_name: str = ".",
    silent: bool = True,
) -> ModuleType:
    from setuptools import Extension, setup

    (utils.silent_setup if silent else setup)(
        name=module_name,
        packages=[],
        ext_modules=[
            Extension(
                module_name,
                sources=sources,
                include_dirs=include_dirs,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        ],
        script_args=["build_ext", "-b", dir_name],
    )

    return utils.import_module_from_dir(module_name, dir_name)


class CodeWrapper(Protocol):
    def __call__(self, code: str, func_name: str, module_name: str) -> tuple[str, str]: ...


def compile_function(
    code: str,
    func_name: str,
    extension: str,
    wrapper: CodeWrapper,
    module_name: str | None = None,
    include_dirs: list[str] = [],
    extra_compile_args: list[str] = [],
    extra_link_args: list[str] = [],
    silent: bool = True,
    **kwargs,
) -> Callable:
    if extension == "py":
        code, func_name = wrapper(code, func_name, module_name or "", **kwargs)
        exec(code, g := {})
        return g[func_name]
    elif extension == "pyx":
        if importlib.util.find_spec("Cython") is None:
            raise ImportError("Cython is required for Cython compilation")
    elif extension == "c":
        if importlib.util.find_spec("setuptools") is None:
            raise ImportError("setuptools is required for C compilation")
    else:
        raise NotImplementedError(f"Extension '{extension}' is not supported.")

    if module_name is None:
        # Create a temporary module
        with tempfile.NamedTemporaryFile(mode="w", suffix=f".{extension}", delete=False) as f:
            dir_name, module_name, _ = utils.split_path(src_path := f.name)
            code, func_name = wrapper(code, func_name, module_name, **kwargs)
            f.write(code)

        # Compile
        try:
            print("Compiling temporary module...", src_path)
            module = compile_module(
                module_name=module_name,
                sources=[src_path],
                include_dirs=include_dirs,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                dir_name=dir_name,
                silent=silent,
            )
            assert (module_path := module.__file__) is not None
            try:
                # Remove the compiled dynamic library
                os.remove(module_path)
            except PermissionError:  # On Windows, the running Python process may lock the file
                pass
        finally:
            os.remove(src_path)

    else:
        src_path = f"{module_name}.{extension}"
        code, func_name = wrapper(code, func_name, module_name, **kwargs)

        if os.path.exists(src_path):
            # Check if existing source file is identical
            try:
                with open(src_path, "r") as f:
                    if f.read() == code:
                        # Use existing module if code is identical
                        try:
                            module = importlib.import_module(module_name)
                            return getattr(module, func_name)
                        except ImportError:
                            # Module exists but can't be imported, recompile
                            pass
            except Exception:
                pass  # Fall through to writing the file

        # Write the C file and compile
        with open(src_path, "w") as f:
            f.write(code)

        module = compile_module(
            module_name=module_name,
            sources=[src_path],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            dir_name=".",
        )

    return getattr(module, func_name)
