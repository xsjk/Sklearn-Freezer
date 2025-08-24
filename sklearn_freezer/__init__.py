from typing import Any, Callable, Literal

from .backend import supported as supported_backend
from .codegen import supported as supported_codegen


def compile_predict_proba(clf: Any, backend: Literal["python", "c", "cython"], **kwargs) -> Callable:
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
    kwargs : dict
        Additional keyword arguments to pass to the compilation function.

    Returns
    -------
    Callable[..., float]
        A callable that takes the features as input and returns the predicted probabilities.
    """

    if hasattr(clf, "feature_names_in_"):
        names = clf.feature_names_in_.tolist()
    else:
        n = clf.n_features_in_
        names = [f"x{i:0{len(str(n - 1))}d}" for i in range(n)]

    # Load compilation backend
    if backend not in supported_backend:
        raise ValueError(f"Backend {backend} not supported. Supported backends are: {list(supported_backend)}")
    compiler = supported_backend[backend]

    # Load code generation backend
    clf_type = type(clf)
    if clf_type not in supported_codegen:
        raise NotImplementedError(f"Model type {clf_type.__name__} not supported yet. Supported models are {[t.__name__ for t in supported_codegen]}.")
    codegen = supported_codegen[clf_type]

    # Get code generation function
    generator = getattr(codegen, f"generate_predict_proba_{backend}", None)
    if generator is None:
        raise NotImplementedError(f"Code generation for {clf_type.__name__} with backend {backend} not supported yet.")

    code = generator(clf, names, func_name="f", **kwargs)
    compiled_func = compiler(code, func_name="f")
    return compiled_func
