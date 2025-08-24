from sklearn.ensemble import RandomForestClassifier as TargetModel

from .decision_tree import tree_to_c, tree_to_python


def generate_predict_proba_python(clf: TargetModel, names: list[str], func_name: str) -> str:
    n_estimators = len(clf.estimators_)
    args = ", ".join(names)

    code = ""
    for i in range(n_estimators):
        code += f"def {func_name}{i}({args}) -> float:\n"
        code += tree_to_python(clf.estimators_[i].tree_, names, initial_depth=1) + "\n"
        code += "\n"

    code += f"def {func_name}({args}):\n"
    expr = " + ".join(f"{func_name}{i}({args})" for i in range(n_estimators))
    code += f"    return ({expr}) / {n_estimators}\n"

    return code


def generate_predict_proba_cython(clf: TargetModel, names: list[str], func_name: str) -> str:
    n_estimators = len(clf.estimators_)
    args = ", ".join(f"double {n}" for n in names)

    code = ""
    for i in range(n_estimators):
        code += f"cdef double {func_name}{i}({args}) noexcept nogil:\n"
        code += tree_to_python(clf.estimators_[i].tree_, names, initial_depth=2) + "\n"
        code += "\n"

    code += f"cpdef double {func_name}({args}) noexcept nogil:\n"
    code += "    cdef double result = 0\n"
    for i in range(n_estimators):
        code += f"    result += {func_name}{i}({', '.join(names)})\n"
    code += f"    return result / {n_estimators}\n"
    return code


def generate_predict_proba_c(clf: TargetModel, names: list[str], func_name: str) -> str:
    n_estimators = len(clf.estimators_)
    args = ", ".join(f"double {n}" for n in names)

    code = ""
    for i in range(n_estimators):
        code += f"double {func_name}{i}({args}) {{\n"
        code += tree_to_c(clf.estimators_[i].tree_, names, initial_depth=1) + "\n"
        code += "}\n"
        code += "\n"

    code += f"double {func_name}({args}) {{\n"
    expr = " + ".join(f"{func_name}{i}({', '.join(names)})" for i in range(n_estimators))
    code += f"""    return ({expr}) / {n_estimators};\n"""
    code += "}\n"
    return code
