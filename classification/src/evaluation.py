"""
Evaluation metrics for the classification module.

Computes balanced accuracy, F1 (weighted/macro), Cohen's kappa,
MCC, per-class precision/recall/F1, confusion matrices, ROC data,
and confidence intervals from fold-level metric distributions.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


# ------------------------------------------------------------------ #
#  Main evaluation entry point                                       #
# ------------------------------------------------------------------ #

def compute_all_metrics(
    cv_results: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for all models from CV results.

    Args:
        cv_results: Output of ``run_cross_validation``.
        cfg:        Configuration dict.

    Returns:
        Dict mapping *model_name* → {
            summary                  : Dict[str, float]  (mean ± std),
            per_class                : pd.DataFrame,
            confusion_matrix         : np.ndarray,
            confusion_matrix_norm    : np.ndarray,
            roc_data                 : Dict | None,
            confidence_intervals     : Dict[str, Tuple[float, float]],
        }.
    """
    class_labels = cfg["columns"]["class_labels"]
    class_codes = cfg["columns"]["class_codes"]
    ci_cfg = cfg["evaluation"]["confidence_interval"]

    print("Computing evaluation metrics ...")
    all_metrics: Dict[str, Any] = {}

    for model_name, res in cv_results.items():
        y_true = res["all_y_true"]
        y_pred = res["all_y_pred"]
        y_prob = res["all_y_prob"]

        # Summary (mean ± std from folds)
        summary = {}
        for metric_name in res["mean_metrics"]:
            summary[metric_name] = res["mean_metrics"][metric_name]
            summary[f"{metric_name}_std"] = res["std_metrics"][metric_name]

        # Per-class metrics from aggregated predictions
        report = classification_report(
            y_true, y_pred,
            labels=class_codes,
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        )
        per_class_rows = []
        for label in class_labels:
            r = report[label]
            per_class_rows.append({
                "class": label,
                "precision": r["precision"],
                "recall": r["recall"],
                "f1_score": r["f1-score"],
                "support": r["support"],
            })
        per_class = pd.DataFrame(per_class_rows)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_codes)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        # ROC data (one-vs-rest)
        roc_data = None
        if y_prob is not None:
            roc_data = _compute_roc_data(y_true, y_prob, class_codes)

        # Confidence intervals
        ci = {}
        if ci_cfg["enabled"]:
            ci = compute_confidence_intervals(
                res["fold_metrics"], alpha=ci_cfg["alpha"],
            )

        all_metrics[model_name] = {
            "summary": summary,
            "per_class": per_class,
            "confusion_matrix": cm,
            "confusion_matrix_norm": cm_norm,
            "roc_data": roc_data,
            "confidence_intervals": ci,
        }

    return all_metrics


# ------------------------------------------------------------------ #
#  Confidence intervals                                              #
# ------------------------------------------------------------------ #

def compute_confidence_intervals(
    fold_metrics: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute CIs from the distribution of fold-level metric values.

    Uses the percentile method: [alpha/2, 1-alpha/2] quantiles.

    Args:
        fold_metrics: DataFrame with one row per fold, columns = metrics.
        alpha:        Significance level (default 0.05 → 95 % CI).

    Returns:
        Dict mapping *metric_name* → (lower, upper).
    """
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    ci: Dict[str, Tuple[float, float]] = {}
    for col in fold_metrics.columns:
        vals = fold_metrics[col].dropna()
        ci[col] = (float(vals.quantile(lower_q)), float(vals.quantile(upper_q)))
    return ci


# ------------------------------------------------------------------ #
#  Model comparison table                                            #
# ------------------------------------------------------------------ #

def build_model_comparison_table(
    all_metrics: Dict[str, Any],
    models: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build the main model comparison summary table.

    Args:
        all_metrics: Output of ``compute_all_metrics``.
        models:      Model info dict from ``build_models``.
        cfg:         Configuration dict.

    Returns:
        DataFrame sorted by primary metric descending.
    """
    primary = cfg["evaluation"]["primary_metric"]
    rows: list[Dict[str, Any]] = []

    for name, met in all_metrics.items():
        row: Dict[str, Any] = {
            "model": name,
            "display_name": models[name]["display_name"],
            "is_baseline": models[name]["is_baseline"],
        }
        for metric in cfg["evaluation"]["metrics"]:
            row[metric] = met["summary"].get(metric, np.nan)
            row[f"{metric}_std"] = met["summary"].get(f"{metric}_std", np.nan)
        ci = met.get("confidence_intervals", {})
        if primary in ci:
            row[f"{primary}_ci_lower"] = ci[primary][0]
            row[f"{primary}_ci_upper"] = ci[primary][1]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(primary, ascending=False).reset_index(drop=True)
    print("Model comparison table built.")
    return df


# ------------------------------------------------------------------ #
#  ROC helper                                                        #
# ------------------------------------------------------------------ #

def _compute_roc_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_codes: list[int],
) -> Dict[int, Dict[str, Any]]:
    """Compute one-vs-rest ROC curves and AUC for each class."""
    y_bin = label_binarize(y_true, classes=class_codes)
    roc: Dict[int, Dict[str, Any]] = {}
    for i, code in enumerate(class_codes):
        if y_prob.shape[1] > i:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc[code] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
    return roc
