import ast
import builtins
from typing import Callable


def compile(code: str, func_name: str, module_name: str | None = None) -> Callable:
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
    module_name : str | None
        The name of the module to save the function code

    Returns
    -------
    Any
        The compiled function.
    """
    if module_name is not None:
        with open(f"{module_name}.py", "w") as f:
            f.write(code)
    exec(builtins.compile(ast.parse(code), filename="<string>", mode="exec"), g := {})
    return g[func_name]
