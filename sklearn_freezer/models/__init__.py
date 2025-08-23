from typing import Literal, Callable, Any

from . import decision_tree

supported_models = {m.target_model: m for m in (decision_tree,)}


def compile_predict_proba(clf: Any, backend: Literal["python", "c", "cython"]) -> Callable[..., float]:
    """Compile the predict_proba method of a given classifier.

    This function takes a classifier and compiles its predict_proba method
    into a more efficient form for the specified backend.

    Parameters
    ----------
    clf : Classifier
        The classifier whose predict_proba method is to be compiled.
    backend : {"python", "c", "cython"}
        The backend used for compilation.
        This determines the compilation method used.

    Returns
    -------
    Callable[..., float]
        A callable that takes the input data and returns the predicted probabilities.
    """

    if hasattr(clf, "feature_names_in_"):
        names = clf.feature_names_in_.tolist()
    else:
        n = clf.n_features_in_
        names = [f"x{i:0{len(str(n - 1))}d}" for i in range(n)]

    if type(clf) not in supported_models:
        raise NotImplementedError(f"Model type {type(clf)} not supported yet.")

    return supported_models[type(clf)].compile_predict_proba(clf, backend, names)
