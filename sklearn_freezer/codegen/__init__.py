import importlib
import pkgutil
from types import ModuleType

supported: dict[type, ModuleType] = {}
for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if not is_pkg:
        module = importlib.import_module(f".{module_name}", package=__name__)
        supported[module.TargetModel] = module
