"""
Cross-validation engine for the classification module.

Runs Repeated Stratified K-Fold CV, collects per-fold predictions
and metrics, and provides a Leave-One-Out sensitivity check.
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
)


# ------------------------------------------------------------------ #
#  Metric helpers                                                     #
# ------------------------------------------------------------------ #

def _compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all classification metrics for a single fold."""
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": float(np.mean(y_true == y_pred)),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
    }


# ------------------------------------------------------------------ #
#  Main cross-validation loop                                        #
# ------------------------------------------------------------------ #

def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run Repeated Stratified K-Fold CV for every model.

    Args:
        X:      Feature matrix (n_samples × n_features).
        y:      Target vector (int codes 0/1/2).
        models: Dict from ``build_models`` (name → estimator info).
        cfg:    Configuration dict.

    Returns:
        Dict mapping *model_name* → {
            fold_metrics  : pd.DataFrame (rows=folds, cols=metric names),
            mean_metrics  : Dict[str, float],
            std_metrics   : Dict[str, float],
            all_y_true    : np.ndarray,
            all_y_pred    : np.ndarray,
            all_y_prob    : np.ndarray | None,
        }.
    """
    cv_cfg = cfg["cross_validation"]
    n_splits = cv_cfg["n_splits"]
    n_repeats = cv_cfg["n_repeats"]
    rs = cv_cfg["random_state"]

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=rs,
    )
    total_folds = n_splits * n_repeats
    print(f"Cross-validation: {n_splits}-fold x {n_repeats} repeats "
          f"= {total_folds} evaluations")

    X_arr = X.values
    y_arr = y.values

    results: Dict[str, Any] = {}

    for model_name, model_info in models.items():
        display = model_info["display_name"]
        print(f"  {display} ...", end=" ", flush=True)

        fold_rows: list[Dict[str, float]] = []
        all_y_true: list[np.ndarray] = []
        all_y_pred: list[np.ndarray] = []
        all_y_prob: list[np.ndarray] = []
        has_proba = hasattr(model_info["estimator"], "predict_proba")

        for train_idx, test_idx in rskf.split(X_arr, y_arr):
            estimator = deepcopy(model_info["estimator"])
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_train, y_train)

            y_pred = estimator.predict(X_test)
            fold_rows.append(_compute_fold_metrics(y_test, y_pred))

            all_y_true.append(y_test)
            all_y_pred.append(y_pred)

            if has_proba:
                try:
                    y_prob = estimator.predict_proba(X_test)
                    all_y_prob.append(y_prob)
                except Exception:
                    has_proba = False

        fold_df = pd.DataFrame(fold_rows)
        mean_m = fold_df.mean().to_dict()
        std_m = fold_df.std().to_dict()

        primary = cfg["evaluation"]["primary_metric"]
        print(f"{primary}={mean_m[primary]:.3f} ± {std_m[primary]:.3f}")

        results[model_name] = {
            "fold_metrics": fold_df,
            "mean_metrics": mean_m,
            "std_metrics": std_m,
            "all_y_true": np.concatenate(all_y_true),
            "all_y_pred": np.concatenate(all_y_pred),
            "all_y_prob": (
                np.concatenate(all_y_prob) if all_y_prob else None
            ),
        }

    return results


# ------------------------------------------------------------------ #
#  Leave-One-Out sensitivity check                                   #
# ------------------------------------------------------------------ #

def run_loo_check(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Run Leave-One-Out CV on selected models as a sensitivity check.

    Args:
        X:      Feature matrix.
        y:      Target vector.
        models: Subset of models to evaluate.
        cfg:    Configuration dict.

    Returns:
        Dict mapping *model_name* → metric dict.
    """
    loo = LeaveOneOut()
    X_arr = X.values
    y_arr = y.values
    n = len(y_arr)

    print(f"LOO sensitivity check ({n} folds) ...")
    results: Dict[str, Dict[str, float]] = {}

    for model_name, model_info in models.items():
        display = model_info["display_name"]
        print(f"  {display} ...", end=" ", flush=True)

        y_true_all: list[int] = []
        y_pred_all: list[int] = []

        for train_idx, test_idx in loo.split(X_arr, y_arr):
            estimator = deepcopy(model_info["estimator"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_arr[train_idx], y_arr[train_idx])
            pred = estimator.predict(X_arr[test_idx])
            y_true_all.append(int(y_arr[test_idx[0]]))
            y_pred_all.append(int(pred[0]))

        y_t = np.array(y_true_all)
        y_p = np.array(y_pred_all)
        metrics = _compute_fold_metrics(y_t, y_p)
        primary = cfg["evaluation"]["primary_metric"]
        print(f"{primary}={metrics[primary]:.3f}")
        results[model_name] = metrics

    return results
