from typing import Callable

from sklearn.tree import DecisionTreeClassifier as TargetModel

from ..backend import c_compile, cython_compile, python_compile


def tree_to_code(return_formatter, if_formatter, indent_unit: str = "    "):
    """
    Factory function to create tree-to-code converters.

    Note
    ----
    This implementation extracts the probability of the last class, which is suitable for binary classification problems.
    For multi-class classification, modifications are required to handle all class probabilities.

    Parameters
    ----------
    return_formatter : callable
        Formatter for return statements.
    if_formatter : callable
        Formatter for if-else statements.
    indent_unit : str, default="    "
        Indentation unit for code formatting.

    Returns
    -------
    callable
        Function that converts a tree to code.
    """

    def _tree_to_code(t, names: list[str], initial_depth=0) -> str:
        if t.value.shape[2] > 2:
            raise NotImplementedError("Multi-class classification is not supported.")

        def recurse(n, depth):
            indent = indent_unit * depth
            if t.children_left[n] == t.children_right[n]:
                # Leaf node: return the probability of the second class, this should be used for binary classification
                total = t.value[n, 0, :].sum()
                value = t.value[n, 0, 1] / total
                return return_formatter(indent=indent, value=value)
            else:
                name = names[t.feature[n]]
                thr = t.threshold[n]
                left = recurse(t.children_left[n], depth + 1)
                right = recurse(t.children_right[n], depth + 1)
                return if_formatter(indent=indent, name=name, thr=thr, left=left, right=right)

        return recurse(0, initial_depth)

    return _tree_to_code


tree_to_python = tree_to_code(
    if_formatter="{indent}if {name} <= {thr}:\n{left}\n{indent}else:\n{right}".format,
    return_formatter="{indent}return {value}".format,
)

tree_to_c = tree_to_code(
    if_formatter="{indent}if ({name} <= {thr}) {{\n{left}\n{indent}}} else {{\n{right}\n{indent}}}".format,
    return_formatter="{indent}return {value};".format,
)


def compile_predict_proba_python(clf: TargetModel, names: list[str]) -> Callable[..., float]:
    code = "def f({args}):\n{code}".format(
        args=", ".join(names),
        code=tree_to_python(clf.tree_, names, initial_depth=1),
    )
    return python_compile(code, "f")


def compile_predict_proba_cython(clf: TargetModel, names: list[str]) -> Callable[..., float]:
    code = "def f({args}):\n{code}".format(
        args=", ".join(f"double {n}" for n in names),
        code=tree_to_python(clf.tree_, names, initial_depth=1),
    )
    return cython_compile(code, func_name="f")


def compile_predict_proba_c(clf: TargetModel, names: list[str]) -> Callable[..., float]:
    code = "inline double f({args}) {{\n{code}\n}}".format(
        args=", ".join(f"double {name}" for name in names),
        code=tree_to_c(clf.tree_, names, initial_depth=1),
    )
    return c_compile(code, func_name="f", arg_names=names, reuse_output=True)
