import ast
from typing import Any


def python_compile(code: str, func_name: str) -> Any:
    """
    Compile Python code into a function.

    This function takes a string of Python code and compiles it into a callable function
    using the provided function name.

    Parameters
    ----------
    code : str
        The Python code to compile.
    func_name : str
        The name of the function to extract from the compiled code.

    Returns
    -------
    Any
        The compiled function.
    """
    exec(compile(ast.parse(code), filename="<string>", mode="exec"), g := {})
    return g[func_name]
