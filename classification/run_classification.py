"""
Main entry point for the classification pipeline.

Trains and evaluates multiple classifiers on the preprocessed
shrinking-villages dataset (65 units, 30 features, 3 classes).

Usage:
    python classification/run_classification.py
    python classification/run_classification.py --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Force a non-interactive backend before pyplot is imported by the src modules.
# This pipeline only writes figures to disk; the default Tk backend can crash on
# interpreter teardown ("Tcl_AsyncDelete: async handler deleted by the wrong
# thread") in headless/batch runs.
import matplotlib
matplotlib.use("Agg")

from classification.src.utils import (
    load_config,
    ensure_output_dirs,
    save_table,
    setup_plot_style,
    write_json,
)
from classification.src.data_loader import load_classification_data
from classification.src.leakage_analysis import run_leakage_analysis
from classification.src.feature_selection import run_feature_selection
from classification.src.model_definitions import build_models
from classification.src.cross_validation import run_cross_validation, run_loo_check
from classification.src.evaluation import (
    compute_all_metrics,
    build_model_comparison_table,
)
from classification.src.feature_importance import run_feature_importance
from classification.src.statistical_tests import run_statistical_tests
from classification.src.visualization import (
    plot_confusion_matrices,
    plot_model_comparison,
    plot_cv_boxplots,
    plot_roc_curves,
    plot_feature_importance,
    plot_shap_summary,
    plot_leakage_comparison,
    plot_pca_variance,
    plot_selectkbest,
)
from classification.src.report_generator import generate_report


def main(config_path: str = "classification/config/classification_config.yaml") -> None:
    """Run the full classification pipeline."""
    t0 = time.time()

    print("=" * 60)
    print("  Classification Pipeline - Shrinking Villages")
    print("=" * 60)

    cfg = load_config(config_path)
    ensure_output_dirs(cfg)
    setup_plot_style(cfg)

    # Collect everything for the report
    summary: Dict[str, Any] = {}

    # ----------------------------------------------------------------
    # Step 1: Load Data
    # ----------------------------------------------------------------
    print("\n--- Step 1: Load Data ---")
    data = load_classification_data(cfg)
    summary["n_samples"] = data["n_samples"]
    summary["n_features_total"] = data["n_features"]
    summary["class_distribution"] = data["class_distribution"]

    # ----------------------------------------------------------------
    # Step 2: Feature Leakage Analysis
    # ----------------------------------------------------------------
    print("\n--- Step 2: Feature Leakage Analysis ---")
    leakage = run_leakage_analysis(data, cfg)
    primary_exp = leakage["primary_experiment"]
    primary_X = leakage["feature_sets"][primary_exp]

    summary["primary_experiment"] = primary_exp
    summary["n_features_primary"] = primary_X.shape[1]
    summary["experiment_feature_counts"] = {
        name: Xdf.shape[1] for name, Xdf in leakage["feature_sets"].items()
    }

    # Save leakage scores
    save_table(leakage["leakage_scores"], "leakage_feature_scores", cfg)

    # ----------------------------------------------------------------
    # Step 3: Feature Selection Diagnostics
    # ----------------------------------------------------------------
    print("\n--- Step 3: Feature Selection Diagnostics ---")
    fs_data = {"X": primary_X, "y": data["y"]}
    selection = run_feature_selection(fs_data, cfg)

    if not selection["kbest_rankings"].empty:
        save_table(selection["kbest_rankings"], "selectkbest_scores", cfg)
    if not selection["pca_variance"].empty:
        save_table(selection["pca_variance"], "pca_explained_variance", cfg)

    summary["n_components_95"] = selection.get("n_components_95", 0)

    # ----------------------------------------------------------------
    # Step 4: Build Models
    # ----------------------------------------------------------------
    print("\n--- Step 4: Build Models ---")
    models = build_models(cfg)

    # ----------------------------------------------------------------
    # Step 5: Cross-Validation (all experiments)
    # ----------------------------------------------------------------
    print("\n--- Step 5: Cross-Validation ---")
    all_experiment_cv: Dict[str, Any] = {}
    all_experiment_tables: Dict[str, pd.DataFrame] = {}

    for exp_name, X_exp in leakage["feature_sets"].items():
        marker = " <- PRIMARY" if exp_name == primary_exp else ""
        print(f"\n  Experiment: {exp_name} ({X_exp.shape[1]} features){marker}")
        cv_res = run_cross_validation(
            X_exp, data["y"], models, cfg, groups=data.get("groups"),
        )
        all_experiment_cv[exp_name] = cv_res

        # Quick evaluation for comparison table
        exp_metrics = compute_all_metrics(cv_res, cfg)
        exp_table = build_model_comparison_table(exp_metrics, models, cfg)
        all_experiment_tables[exp_name] = exp_table

    primary_cv = all_experiment_cv[primary_exp]

    # Save experiment comparison
    exp_rows: list[Dict[str, Any]] = []
    primary_metric_name = cfg["evaluation"]["primary_metric"]
    for exp_name, table in all_experiment_tables.items():
        for _, row in table.iterrows():
            exp_rows.append({
                "experiment": exp_name,
                "model": row["display_name"],
                primary_metric_name: row[primary_metric_name],
            })
    exp_comparison = pd.DataFrame(exp_rows)
    save_table(exp_comparison, "leakage_experiment_results", cfg)

    # ----------------------------------------------------------------
    # Step 6: Evaluate (primary experiment)
    # ----------------------------------------------------------------
    print("\n--- Step 6: Evaluate (Primary Experiment) ---")
    all_metrics = compute_all_metrics(primary_cv, cfg)
    comparison_table = build_model_comparison_table(all_metrics, models, cfg)

    save_table(comparison_table, "model_comparison", cfg)
    summary["comparison_table"] = comparison_table

    # Per-class metrics
    per_class_rows: list[Dict[str, Any]] = []
    for model_name, met in all_metrics.items():
        for _, row in met["per_class"].iterrows():
            per_class_rows.append({
                "model": models[model_name]["display_name"],
                **row.to_dict(),
            })
    save_table(pd.DataFrame(per_class_rows), "per_class_metrics", cfg)

    # Fold-level metrics
    fold_rows: list[Dict[str, Any]] = []
    for model_name, res in primary_cv.items():
        fm = res["fold_metrics"].copy()
        fm.insert(0, "model", models[model_name]["display_name"])
        fold_rows.append(fm)
    save_table(pd.concat(fold_rows, ignore_index=True), "fold_level_metrics", cfg)

    # Best model
    best_name = _find_best(all_metrics, models, cfg)
    summary["best_model_name"] = best_name
    summary["best_model_display"] = models[best_name]["display_name"]
    summary["best_model_metrics"] = all_metrics[best_name]["summary"]
    print(f"\n  Best model: {models[best_name]['display_name']}")

    # ----------------------------------------------------------------
    # Step 7: Feature Importance
    # ----------------------------------------------------------------
    print("\n--- Step 7: Feature Importance ---")
    importance = run_feature_importance(
        primary_X, data["y"], primary_cv, models, cfg,
    )

    if not importance["permutation"].empty:
        save_table(importance["permutation"], "feature_importance_permutation", cfg)
    for tname, df_imp in importance.get("tree_based", {}).items():
        save_table(df_imp, f"feature_importance_tree_{tname}", cfg)
    if not importance["consensus_ranking"].empty:
        save_table(importance["consensus_ranking"], "feature_importance_consensus", cfg)

    # ----------------------------------------------------------------
    # Step 8: Statistical Tests
    # ----------------------------------------------------------------
    print("\n--- Step 8: Statistical Tests ---")
    stat_results = run_statistical_tests(primary_cv, models, cfg)
    summary["statistical_tests"] = {
        "friedman": stat_results["friedman"],
        "paired_t_tests": stat_results["paired_t_tests"],
    }
    summary["significant_findings"] = stat_results["significant_findings"]

    if stat_results["nemenyi"] is not None:
        save_table(stat_results["nemenyi"], "statistical_tests_pairwise", cfg)

    # ----------------------------------------------------------------
    # Step 9: LOO Sensitivity Check
    # ----------------------------------------------------------------
    loo_results = None
    if cfg["cross_validation"]["loo_check"]["enabled"]:
        print("\n--- Step 9: LOO Sensitivity Check ---")
        loo_model_names = cfg["cross_validation"]["loo_check"]["models"]
        loo_models = {
            k: v for k, v in models.items() if k in loo_model_names
        }
        if loo_models:
            loo_results = run_loo_check(primary_X, data["y"], loo_models, cfg)
            loo_df = pd.DataFrame(loo_results).T
            loo_df.index.name = "model"
            save_table(loo_df, "loo_results", cfg)

    # ----------------------------------------------------------------
    # Step 10: Visualization
    # ----------------------------------------------------------------
    print("\n--- Step 10: Visualization ---")
    plot_confusion_matrices(all_metrics, models, cfg)
    plot_model_comparison(comparison_table, cfg)
    plot_cv_boxplots(primary_cv, models, cfg)
    plot_roc_curves(all_metrics, best_name, models, cfg)
    plot_feature_importance(importance, cfg)
    plot_shap_summary(importance, primary_X, cfg)
    plot_leakage_comparison(all_experiment_tables, cfg)
    plot_pca_variance(selection, cfg)
    plot_selectkbest(selection.get("kbest_rankings", pd.DataFrame()), cfg)

    # ----------------------------------------------------------------
    # Step 11: Save Models
    # ----------------------------------------------------------------
    print("\n--- Step 11: Save Models ---")
    _save_models(models, primary_X, data["y"], cfg)

    # ----------------------------------------------------------------
    # Step 12: Generate Report
    # ----------------------------------------------------------------
    print("\n--- Step 12: Generate Report ---")
    generate_report(summary, cfg)

    # Done
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  Classification pipeline complete.")
    print(f"  Best model: {models[best_name]['display_name']}")
    pm = cfg["evaluation"]["primary_metric"]
    best_score = all_metrics[best_name]["summary"][pm]
    print(f"  {pm}: {best_score:.3f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Outputs: {cfg['output']['base_dir']}")
    print("=" * 60)


# ------------------------------------------------------------------ #
#  Helpers                                                           #
# ------------------------------------------------------------------ #

def _find_best(
    all_metrics: Dict[str, Any],
    models: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> str:
    """Return name of the best non-baseline model."""
    primary = cfg["evaluation"]["primary_metric"]
    best_name, best_score = "", -np.inf
    for name, met in all_metrics.items():
        if models[name]["is_baseline"]:
            continue
        score = met["summary"].get(primary, -np.inf)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def _save_models(
    models: Dict[str, Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
) -> None:
    """Retrain all models on full data and save with joblib."""
    import warnings
    from copy import deepcopy

    models_dir = Path(cfg["output"]["models_dir"])

    for name, info in models.items():
        est = deepcopy(info["estimator"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X.values, y.values)
        path = models_dir / f"{name}_final.joblib"
        joblib.dump(est, path)
        print(f"  Saved: {path}")

    # Best model info
    primary = cfg["evaluation"]["primary_metric"]
    best_name = _find_best(
        {n: {"summary": {primary: r["mean_metrics"].get(primary, -1)}}
         for n, r in {}. items()},
        models, cfg,
    ) if False else ""  # noqa — handled below

    info_path = models_dir / "best_model_info.json"
    # Simple approach: re-identify from saved models
    write_json(str(info_path), {
        "note": "Best model determined from comparison_table in reports/",
        "models_saved": list(models.keys()),
    })
    print(f"  Saved: {info_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classification pipeline for shrinking villages",
    )
    parser.add_argument(
        "--config",
        default="classification/config/classification_config.yaml",
        help="Path to classification config YAML",
    )
    args = parser.parse_args()
    main(args.config)
