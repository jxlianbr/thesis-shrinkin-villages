"""
Model definitions for the classification module.

Builds classifier instances from YAML configuration using dynamic imports.
Handles optional dependencies (xgboost) gracefully.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict


MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "dummy_most_frequent": "Dummy (Most Frequent)",
    "dummy_stratified": "Dummy (Stratified)",
    "logistic_regression": "Logistic Regression",
    "svm_linear": "SVM (Linear)",
    "svm_rbf": "SVM (RBF)",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "knn": "K-Nearest Neighbors",
    "mlp": "MLP Neural Network",
    "xgboost": "XGBoost",
}


def build_models(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build classifier instances from the ``models`` section of the config.

    Each enabled model entry is dynamically imported and instantiated
    with the configured parameters.  Models whose imports fail (e.g.
    xgboost not installed) are skipped with a warning.

    Args:
        cfg: Configuration dict.

    Returns:
        OrderedDict mapping *model_name* → {
            ``estimator``    : sklearn-compatible estimator instance,
            ``display_name`` : str for plots/tables,
            ``is_baseline``  : bool,
        }.
    """
    models_cfg = cfg["models"]
    models: Dict[str, Dict[str, Any]] = {}

    print("Building models ...")
    for name, mcfg in models_cfg.items():
        if not mcfg.get("enabled", False):
            print(f"  {name}: SKIPPED (disabled)")
            continue

        class_path = mcfg["class"]
        params = mcfg.get("params", {})

        try:
            cls = _import_class(class_path)
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"  {name}: SKIPPED ({exc})")
            continue

        estimator = cls(**params)
        display = MODEL_DISPLAY_NAMES.get(name, name)
        is_baseline = mcfg.get("is_baseline", False)

        models[name] = {
            "estimator": estimator,
            "display_name": display,
            "is_baseline": is_baseline,
        }
        tag = " [baseline]" if is_baseline else ""
        print(f"  {display}{tag}")

    print(f"  Total models: {len(models)}")
    return models


def _import_class(class_path: str) -> type:
    """Dynamically import a class from a dotted path string.

    Example:
        ``_import_class("sklearn.ensemble.RandomForestClassifier")``
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
