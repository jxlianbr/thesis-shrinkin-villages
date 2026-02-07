"""
Visualization for the classification module.

Produces thesis-quality figures: confusion matrices, model comparison
bar charts, ROC curves, CV score boxplots, feature importance plots,
and leakage experiment comparisons.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from classification.src.utils import save_figure


# ------------------------------------------------------------------ #
#  Confusion matrices                                                #
# ------------------------------------------------------------------ #

def plot_confusion_matrices(
    all_metrics: Dict[str, Any],
    models: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> list[Path]:
    """Plot normalised confusion matrix for each model."""
    class_labels = cfg["columns"]["class_labels"]
    saved: list[Path] = []

    for model_name, met in all_metrics.items():
        display = models[model_name]["display_name"]
        cm_norm = met["confusion_matrix_norm"]

        fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_single"])
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels,
            vmin=0, vmax=1, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - {display}")
        saved.append(save_figure(fig, f"confusion_matrix_{model_name}", cfg))

    return saved


# ------------------------------------------------------------------ #
#  Model comparison bar chart                                        #
# ------------------------------------------------------------------ #

def plot_model_comparison(
    comparison_table: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Path:
    """Bar chart comparing models on the primary metric with CI error bars."""
    primary = cfg["evaluation"]["primary_metric"]
    ci_lower_col = f"{primary}_ci_lower"
    ci_upper_col = f"{primary}_ci_upper"

    df = comparison_table.sort_values(primary, ascending=True).copy()
    colors = [
        "#9E9E9E" if bl else cfg["plot"]["class_colors"]["shrinking"]
        for bl in df["is_baseline"]
    ]

    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_wide"])

    y_pos = range(len(df))
    bars = ax.barh(y_pos, df[primary], color=colors, edgecolor="white", height=0.6)

    # Error bars from CI (skip rows with NaN)
    if ci_lower_col in df.columns and ci_upper_col in df.columns:
        xerr_lower = np.clip(df[primary].values - df[ci_lower_col].values, 0, None)
        xerr_upper = np.clip(df[ci_upper_col].values - df[primary].values, 0, None)
        # Replace NaN with 0
        xerr_lower = np.nan_to_num(xerr_lower, nan=0.0)
        xerr_upper = np.nan_to_num(xerr_upper, nan=0.0)
        xerr = np.array([xerr_lower, xerr_upper])
        ax.errorbar(
            df[primary], y_pos, xerr=xerr,
            fmt="none", ecolor="black", capsize=3,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display_name"])
    ax.set_xlabel(primary.replace("_", " ").title())
    ax.set_title("Model Comparison")
    ax.axvline(x=1/3, color="red", linestyle="--", alpha=0.5, label="Random baseline (0.33)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)

    return save_figure(fig, "model_comparison", cfg)


# ------------------------------------------------------------------ #
#  CV boxplots                                                       #
# ------------------------------------------------------------------ #

def plot_cv_boxplots(
    cv_results: Dict[str, Any],
    models: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Path:
    """Boxplots of fold-level balanced accuracy across models."""
    primary = cfg["evaluation"]["primary_metric"]

    data_frames = []
    for name, res in cv_results.items():
        fold_df = res["fold_metrics"][[primary]].copy()
        fold_df["model"] = models[name]["display_name"]
        data_frames.append(fold_df)

    plot_df = pd.concat(data_frames, ignore_index=True)

    # Sort by median
    order = (
        plot_df.groupby("model")[primary]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_wide"])
    sns.boxplot(
        data=plot_df, x=primary, y="model", order=order,
        palette="viridis", ax=ax,
    )
    ax.axvline(x=1/3, color="red", linestyle="--", alpha=0.5, label="Random (0.33)")
    ax.set_xlabel(primary.replace("_", " ").title())
    ax.set_ylabel("")
    ax.set_title(f"Cross-Validation {primary.replace('_', ' ').title()} Distribution")
    ax.legend(loc="lower right")

    return save_figure(fig, "cv_boxplots", cfg)


# ------------------------------------------------------------------ #
#  ROC curves                                                        #
# ------------------------------------------------------------------ #

def plot_roc_curves(
    all_metrics: Dict[str, Any],
    best_model_name: str,
    models: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Path | None:
    """One-vs-rest ROC curves for the best model (3-class)."""
    met = all_metrics.get(best_model_name)
    if met is None or met["roc_data"] is None:
        print("  ROC curves: skipped (no probability data)")
        return None

    class_labels = cfg["columns"]["class_labels"]
    class_colors = cfg["plot"]["class_colors"]
    roc_data = met["roc_data"]
    display = models[best_model_name]["display_name"]

    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_single"])
    for code, label in enumerate(class_labels):
        if code in roc_data:
            rd = roc_data[code]
            color = class_colors.get(label, None)
            ax.plot(rd["fpr"], rd["tpr"], label=f"{label} (AUC={rd['auc']:.2f})",
                    color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (One-vs-Rest) - {display}")
    ax.legend(loc="lower right")

    return save_figure(fig, "roc_curves_best", cfg)


# ------------------------------------------------------------------ #
#  Feature importance                                                #
# ------------------------------------------------------------------ #

def plot_feature_importance(
    importance_data: Dict[str, Any],
    cfg: Dict[str, Any],
) -> list[Path]:
    """Horizontal bar charts for permutation and tree-based importance."""
    saved: list[Path] = []
    max_display = cfg["feature_importance"]["shap"].get("max_display", 15)

    # Permutation
    perm = importance_data.get("permutation")
    if perm is not None and not perm.empty:
        df = perm.head(max_display).iloc[::-1]
        fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_single"])
        ax.barh(df["feature"], df["importance_mean"],
                xerr=df["importance_std"], color="#FF9800", edgecolor="white")
        ax.set_xlabel("Importance (Δ balanced accuracy)")
        ax.set_title("Permutation Feature Importance")
        saved.append(save_figure(fig, "permutation_importance", cfg))

    # Tree-based
    for tname, df_imp in importance_data.get("tree_based", {}).items():
        df = df_imp.head(max_display).iloc[::-1]
        fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_single"])
        ax.barh(df["feature"], df["importance"], color="#4CAF50", edgecolor="white")
        ax.set_xlabel("Gini Importance")
        ax.set_title(f"Feature Importance - {tname.replace('_', ' ').title()}")
        saved.append(save_figure(fig, f"tree_importance_{tname}", cfg))

    return saved


def plot_shap_summary(
    importance_data: Dict[str, Any],
    X: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Path | None:
    """SHAP beeswarm summary plot."""
    shap_vals = importance_data.get("shap_values")
    if shap_vals is None:
        return None

    try:
        import shap
    except ImportError:
        return None

    max_display = cfg["feature_importance"]["shap"].get("max_display", 15)
    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_single"])
    plt.sca(ax)
    shap.summary_plot(
        shap_vals, X, max_display=max_display, show=False,
        plot_type="bar",
    )
    ax.set_title("SHAP Feature Importance (mean |SHAP|)")
    return save_figure(fig, "shap_summary", cfg)


# ------------------------------------------------------------------ #
#  Leakage experiment comparison                                     #
# ------------------------------------------------------------------ #

def plot_leakage_comparison(
    experiment_results: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
) -> Path:
    """Grouped bar chart comparing model performance across experiments."""
    primary = cfg["evaluation"]["primary_metric"]
    rows: list[Dict[str, Any]] = []
    for exp_name, comp_table in experiment_results.items():
        for _, row in comp_table.iterrows():
            rows.append({
                "experiment": exp_name,
                "model": row["display_name"],
                primary: row[primary],
            })
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_wide"])
    sns.barplot(
        data=plot_df, x="model", y=primary, hue="experiment",
        palette="viridis", ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel(primary.replace("_", " ").title())
    ax.set_title("Performance Across Feature Leakage Experiments")
    ax.legend(title="Experiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    return save_figure(fig, "leakage_comparison", cfg)


# ------------------------------------------------------------------ #
#  PCA scree plot                                                    #
# ------------------------------------------------------------------ #

def plot_pca_variance(
    pca_data: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Path | None:
    """Scree plot of PCA explained variance."""
    pca_df = pca_data.get("pca_variance")
    if pca_df is None or pca_df.empty:
        return None

    n95 = pca_data.get("n_components_95", 0)
    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_single"])
    ax.bar(pca_df["component"], pca_df["explained_variance"],
           color="#2196F3", alpha=0.7, label="Individual")
    ax.plot(pca_df["component"], pca_df["cumulative"],
            "r-o", markersize=4, label="Cumulative")
    if n95 > 0:
        ax.axvline(x=n95, color="green", linestyle="--", alpha=0.7,
                   label=f"95% variance (n={n95})")
    ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")
    ax.legend()

    return save_figure(fig, "pca_variance", cfg)


# ------------------------------------------------------------------ #
#  SelectKBest                                                       #
# ------------------------------------------------------------------ #

def plot_selectkbest(
    kbest_rankings: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Path | None:
    """Horizontal bar chart of ANOVA F-scores."""
    if kbest_rankings.empty:
        return None

    k = cfg["feature_selection"]["select_k_best"]["k"]
    df = kbest_rankings.head(k).iloc[::-1]

    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_single"])
    ax.barh(df["feature"], df["score"], color="#9C27B0", edgecolor="white")
    ax.set_xlabel("ANOVA F-Score")
    ax.set_title(f"Top-{k} Features by SelectKBest")

    return save_figure(fig, "selectkbest_scores", cfg)
