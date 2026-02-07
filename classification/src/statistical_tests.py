"""
Statistical tests for the classification module.

Performs Friedman test (non-parametric comparison across multiple
classifiers) and post-hoc Nemenyi test for pairwise comparisons.
Also computes paired t-tests between the best model and baselines.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats


def run_statistical_tests(
    cv_results: Dict[str, Any],
    models: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare classifier performance statistically.

    1. **Friedman test**: are there significant differences among classifiers?
    2. **Nemenyi post-hoc**: which pairs differ significantly?
    3. **Paired t-tests**: is the best model significantly better than baselines?

    Args:
        cv_results: Output of ``run_cross_validation``.
        models:     Model info dict.
        cfg:        Configuration dict.

    Returns:
        Dict with keys:
            friedman             – Dict (statistic, p_value)
            nemenyi              – pd.DataFrame | None  (pairwise p-values)
            paired_t_tests       – list[Dict]
            significant_findings – list[str]
    """
    primary = cfg["evaluation"]["primary_metric"]
    print(f"Statistical tests (metric: {primary}) ...")

    # Build fold-level metric matrix (rows=folds, cols=models)
    model_names = list(cv_results.keys())
    fold_matrix = pd.DataFrame({
        name: cv_results[name]["fold_metrics"][primary]
        for name in model_names
    })

    results: Dict[str, Any] = {
        "friedman": {"statistic": np.nan, "p_value": np.nan},
        "nemenyi": None,
        "paired_t_tests": [],
        "significant_findings": [],
    }

    # --- 1. Friedman test ---
    if len(model_names) >= 3:
        groups = [fold_matrix[name].values for name in model_names]
        try:
            stat, p_val = stats.friedmanchisquare(*groups)
            results["friedman"] = {"statistic": float(stat), "p_value": float(p_val)}
            print(f"  Friedman: chi2={stat:.2f}, p={p_val:.4f}")
            if p_val < 0.05:
                results["significant_findings"].append(
                    f"Friedman test significant (p={p_val:.4f}): "
                    "classifiers differ in performance."
                )
        except Exception as exc:
            print(f"  Friedman test failed: {exc}")

    # --- 2. Nemenyi post-hoc (pairwise Wilcoxon as approximation) ---
    if results["friedman"]["p_value"] < 0.05 and len(model_names) >= 3:
        print("  Running pairwise Wilcoxon signed-rank tests (Bonferroni) ...")
        n_tests = len(model_names) * (len(model_names) - 1) // 2
        pairwise: list[Dict[str, Any]] = []
        for i, name_a in enumerate(model_names):
            for name_b in model_names[i + 1:]:
                try:
                    stat_w, p_w = stats.wilcoxon(
                        fold_matrix[name_a].values,
                        fold_matrix[name_b].values,
                        alternative="two-sided",
                    )
                    p_adjusted = min(p_w * n_tests, 1.0)  # Bonferroni
                    pairwise.append({
                        "model_a": name_a,
                        "model_b": name_b,
                        "statistic": float(stat_w),
                        "p_value": float(p_w),
                        "p_adjusted": float(p_adjusted),
                        "significant": p_adjusted < 0.05,
                    })
                except Exception:
                    pairwise.append({
                        "model_a": name_a,
                        "model_b": name_b,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "p_adjusted": np.nan,
                        "significant": False,
                    })
        results["nemenyi"] = pd.DataFrame(pairwise)
        n_sig = sum(1 for r in pairwise if r["significant"])
        print(f"  {n_sig}/{len(pairwise)} pairwise comparisons significant "
              "(Bonferroni alpha=0.05)")

    # --- 3. Paired t-tests (best vs baselines) ---
    best_name = _find_best_non_baseline(cv_results, models, primary)
    if best_name:
        baselines = [n for n in model_names if models[n]["is_baseline"]]
        for bname in baselines:
            t_stat, p_val = stats.ttest_rel(
                fold_matrix[best_name].values,
                fold_matrix[bname].values,
            )
            sig = p_val < 0.05
            results["paired_t_tests"].append({
                "model_a": best_name,
                "model_b": bname,
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "significant": sig,
            })
            display_a = models[best_name]["display_name"]
            display_b = models[bname]["display_name"]
            status = "YES" if sig else "no"
            print(f"  Paired t-test: {display_a} vs {display_b}: "
                  f"t={t_stat:.2f}, p={p_val:.4f} -> significant={status}")
            if sig:
                results["significant_findings"].append(
                    f"{display_a} significantly outperforms "
                    f"{display_b} (p={p_val:.4f})."
                )

    return results


def _find_best_non_baseline(
    cv_results: Dict[str, Any],
    models: Dict[str, Dict[str, Any]],
    metric: str,
) -> str | None:
    """Return name of the best non-baseline model."""
    best_name, best_score = None, -np.inf
    for name, res in cv_results.items():
        if models[name]["is_baseline"]:
            continue
        score = res["mean_metrics"].get(metric, -np.inf)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name
