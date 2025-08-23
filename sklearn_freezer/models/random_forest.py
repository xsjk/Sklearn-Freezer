from typing import Callable

from sklearn.ensemble import RandomForestClassifier as TargetModel

from ..backend import c_compile, cython_compile, python_compile
from .decision_tree import tree_to_c, tree_to_python


def compile_predict_proba_python(clf: TargetModel, names: list[str]) -> Callable[..., float]:
    n_estimators = len(clf.estimators_)
    args = ", ".join(names)

    code = ""
    for i in range(n_estimators):
        code += f"def g{i}({args}) -> float:\n"
        code += tree_to_python(clf.estimators_[i].tree_, names, initial_depth=1) + "\n"
        code += "\n"

    code += f"def f({args}):\n"
    expr = " + ".join(f"g{i}({args})" for i in range(n_estimators))
    code += f"    return ({expr}) / {n_estimators}\n"

    return python_compile(code, "f")


def compile_predict_proba_cython(clf: TargetModel, names: list[str]) -> Callable[..., float]:
    n_estimators = len(clf.estimators_)
    args = ", ".join(f"double {n}" for n in names)

    code = ""
    for i in range(n_estimators):
        code += f"cdef double g{i}({args}) noexcept nogil:\n"
        code += tree_to_python(clf.estimators_[i].tree_, names, initial_depth=2) + "\n"
        code += "\n"

    code += f"cpdef double f({args}) noexcept nogil:\n"
    code += "    cdef double result = 0\n"
    for i in range(n_estimators):
        code += f"    result += g{i}({', '.join(names)})\n"
    code += f"    return result / {n_estimators}\n"
    return cython_compile(code, "f")


def compile_predict_proba_c(clf: TargetModel, names: list[str]) -> Callable[..., float]:
    n_estimators = len(clf.estimators_)
    args = ", ".join(f"double {n}" for n in names)

    code = ""
    for i in range(n_estimators):
        code += f"double g{i}({args}) {{\n"
        code += tree_to_c(clf.estimators_[i].tree_, names, initial_depth=1) + "\n"
        code += "}\n"
        code += "\n"

    code += f"double f({args}) {{\n"
    expr = " + ".join(f"g{i}({', '.join(names)})" for i in range(n_estimators))
    code += f"""    return ({expr}) / {n_estimators};\n"""
    code += "}\n"

    return c_compile(code, "f", arg_names=names)
