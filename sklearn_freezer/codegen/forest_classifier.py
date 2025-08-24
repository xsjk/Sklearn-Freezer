from sklearn.ensemble._forest import ForestClassifier as TargetModel


def generate_predict_proba_python(clf: TargetModel, arg_names: list[str], func_name: str) -> str:
    from . import get_codegen

    n_estimators = len(clf.estimators_)
    args = ", ".join(arg_names)

    code = ""
    for i, e in enumerate(clf.estimators_):
        code += get_codegen(e.predict_proba, "python")(f"{func_name}{i}") + "\n"  # type: ignore

    code += f"def {func_name}({args}):\n"
    expr = " + ".join(f"{func_name}{i}({args})" for i in range(n_estimators))
    code += f"    return ({expr}) / {n_estimators}\n"

    return code


def generate_predict_proba_cython(clf: TargetModel, arg_names: list[str], func_name: str) -> str:
    from . import get_codegen

    n_estimators = len(clf.estimators_)
    args = ", ".join(f"double {n}" for n in arg_names)

    code = ""
    for i, e in enumerate(clf.estimators_):
        code += get_codegen(e.predict_proba, "cython")(f"{func_name}{i}") + "\n"  # type: ignore

    code += f"cpdef double {func_name}({args}) noexcept nogil:\n"
    code += "    cdef double result = 0\n"
    for i in range(n_estimators):
        code += f"    result += {func_name}{i}({', '.join(arg_names)})\n"
    code += f"    return result / {n_estimators}\n"
    return code


def generate_predict_proba_c(clf: TargetModel, arg_names: list[str], func_name: str) -> str:
    from . import get_codegen

    n_estimators = len(clf.estimators_)
    args = ", ".join(f"double {n}" for n in arg_names)

    code = ""
    for i, e in enumerate(clf.estimators_):
        code += get_codegen(e.predict_proba, "c")(f"{func_name}{i}") + "\n"  # type: ignore

    code += f"double {func_name}({args}) {{\n"
    expr = " + ".join(f"{func_name}{i}({', '.join(arg_names)})" for i in range(n_estimators))
    code += f"""    return ({expr}) / {n_estimators};\n"""
    code += "}\n"
    return code
