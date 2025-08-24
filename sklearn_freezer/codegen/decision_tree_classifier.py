from sklearn.tree import DecisionTreeClassifier as TargetModel


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
        assert t.value.ndim == 3
        if t.value.shape[2] > 2:
            raise NotImplementedError("Multi-class classification is not supported. Only binary classification is currently supported.")

        def recurse(n, depth):
            indent = indent_unit * depth
            if t.children_left[n] == t.children_right[n]:
                # Leaf node: return the probability of the second class (assumes classes are ordered as [negative_class, positive_class]).
                # This is standard for binary classification in scikit-learn, where the second class is usually the positive class.
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


def generate_predict_proba_python(clf: TargetModel, arg_names: list[str], func_name: str) -> str:
    return "def {func_name}({args}):\n{code}".format(
        func_name=func_name,
        args=", ".join(arg_names),
        code=tree_to_python(clf.tree_, arg_names, initial_depth=1),
    )


def generate_predict_proba_cython(clf: TargetModel, arg_names: list[str], func_name: str) -> str:
    return "cpdef double {func_name}({args}) noexcept nogil:\n{code}".format(
        func_name=func_name,
        args=", ".join(f"double {n}" for n in arg_names),
        code=tree_to_python(clf.tree_, arg_names, initial_depth=1),
    )


def generate_predict_proba_c(clf: TargetModel, arg_names: list[str], func_name: str) -> str:
    return "double {func_name}({args}) {{\n{code}\n}}".format(
        func_name=func_name,
        args=", ".join(f"double {n}" for n in arg_names),
        code=tree_to_c(clf.tree_, arg_names, initial_depth=1),
    )
