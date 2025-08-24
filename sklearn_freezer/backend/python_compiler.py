import ast
import builtins
from typing import Callable


def compile(code: str, func_name: str) -> Callable:
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
    exec(builtins.compile(ast.parse(code), filename="<string>", mode="exec"), g := {})
    return g[func_name]
