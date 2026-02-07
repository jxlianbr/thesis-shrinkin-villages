"""
Feature leakage analysis for the classification module.

Identifies features that leak target information because the
target (shrinkage_class) is derived from elderly_ratio thresholds.
Produces multiple feature subsets for comparative experiments.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats


def run_leakage_analysis(
    data: Dict[str, Any], cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyse target–feature correlations and build experiment feature sets.

    Computes ANOVA F-statistics between each feature and the target to
    quantify leakage empirically.  Generates feature subsets for each
    experiment defined in the configuration.

    Args:
        data: Dict from ``load_classification_data`` (keys: X, y, …).
        cfg:  Configuration dict.

    Returns:
        Dict with keys:
            experiments      – Dict[str, list[str] | None] from config
            leakage_scores   – pd.DataFrame (feature, f_statistic, p_value, is_leaky)
            primary_experiment – str
            feature_sets     – Dict[str, pd.DataFrame] (experiment → X subset)
    """
    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"]
    leak_cfg = cfg["leakage"]

    print("Running feature leakage analysis ...")

    # --- 1. Compute ANOVA F-statistic per feature ---
    leaky_set = set(leak_cfg["leaky_features"])
    partial_set = set(leak_cfg.get("partially_leaky_features", []))

    records: list[Dict[str, Any]] = []
    for col in feature_names:
        groups = [X.loc[y == c, col].dropna().values for c in sorted(y.unique())]
        if all(len(g) > 1 for g in groups):
            f_stat, p_val = stats.f_oneway(*groups)
        else:
            f_stat, p_val = np.nan, np.nan
        records.append({
            "feature": col,
            "f_statistic": f_stat,
            "p_value": p_val,
            "is_leaky": col in leaky_set,
            "is_partially_leaky": col in partial_set,
        })

    leakage_scores = (
        pd.DataFrame(records)
        .sort_values("f_statistic", ascending=False)
        .reset_index(drop=True)
    )

    # Print top leaky features
    top = leakage_scores.head(10)
    print("  Top 10 features by ANOVA F-statistic:")
    for _, row in top.iterrows():
        tag = " [LEAKY]" if row["is_leaky"] else (
            " [partial]" if row["is_partially_leaky"] else ""
        )
        print(f"    {row['feature']:30s}  F={row['f_statistic']:10.2f}  "
              f"p={row['p_value']:.2e}{tag}")

    # --- 2. Build experiment feature sets ---
    experiments = leak_cfg["experiments"]
    primary = leak_cfg["primary_experiment"]
    feature_sets: Dict[str, pd.DataFrame] = {}

    for exp_name, drop_cols in experiments.items():
        if drop_cols is None:
            feature_sets[exp_name] = X.copy()
        else:
            keep = [c for c in feature_names if c not in drop_cols]
            feature_sets[exp_name] = X[keep].copy()
        n_feat = feature_sets[exp_name].shape[1]
        marker = " <- PRIMARY" if exp_name == primary else ""
        print(f"  Experiment '{exp_name}': {n_feat} features{marker}")

    return {
        "experiments": experiments,
        "leakage_scores": leakage_scores,
        "primary_experiment": primary,
        "feature_sets": feature_sets,
    }
