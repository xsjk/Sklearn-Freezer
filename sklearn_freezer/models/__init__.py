import importlib
import pkgutil
from typing import Any, Callable, Literal

supported_models = {}
for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if not is_pkg:
        module = importlib.import_module(f".{module_name}", package=__name__)
        supported_models[module.TargetModel] = module


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

    compiler = getattr(supported_models[type(clf)], f"compile_predict_proba_{backend}", None)
    if compiler:
        return compiler(clf, names)
    raise NotImplementedError(f"Compilation {type(clf).__name__} with backend {backend} not supported yet.")
