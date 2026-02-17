"""
Distribution analysis for the EDA module.

Produces histogram grids for remote-sensing and demographic features,
boxplots by prefecture, and a distribution statistics table
(skewness, kurtosis, Shapiro-Wilk test).
"""
from __future__ import annotations

import math
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import seaborn as sns

from eda.src.utils import get_existing_columns, save_figure, save_table


def run_distribution_analysis(
    df: pd.DataFrame, cfg: Dict[str, Any], output_dir: str,
) -> Dict[str, Any]:
    """
    Analyse and visualise feature distributions.

    Outputs:
        figures/distributions_rs.png
        figures/distributions_demo.png
        figures/boxplots_by_prefecture.png
        tables/distribution_statistics.csv

    Returns:
        Summary dict with per-feature normality flags and distribution highlights.
    """
    print("Running distribution analysis...")

    # Remote sensing features (including GLCM texture)
    rs_cols = get_existing_columns(df,
        cfg["columns"].get("spectral", [])
        + cfg["columns"].get("indices", [])
        + cfg["columns"].get("nightlights", [])
        + cfg["columns"].get("texture", [])
    )

    lulc_cols = get_existing_columns(df, cfg["columns"].get("lulc", []))
    demo_cols = get_existing_columns(df, cfg["columns"].get("demographic", []))
    osm_cols = get_existing_columns(df, cfg["columns"].get("osm", []))

    # --- Histogram grids ---
    if rs_cols:
        _plot_histogram_grid(df, rs_cols, "Remote Sensing Features",
                             "distributions_rs", cfg)
    if lulc_cols:
        _plot_histogram_grid(df, lulc_cols, "LULC Class Fractions (Dynamic World)",
                             "distributions_lulc", cfg)
    if demo_cols:
        _plot_histogram_grid(df, demo_cols, "Demographic Features",
                             "distributions_demo", cfg)

    # --- Boxplots by prefecture ---
    key_features = get_existing_columns(
        df, ["NDVI", "NDBI", "viirs_mean", "pop_total", "age_65_plus"],
    )
    if key_features and "pref_name" in df.columns:
        _plot_boxplots_by_prefecture(df, key_features, cfg)

    # --- Distribution statistics table ---
    all_cols = rs_cols + lulc_cols + demo_cols + osm_cols
    stats_records = []
    for col in all_cols:
        series = df[col].dropna()
        if len(series) < 8:
            continue
        skew = float(series.skew())
        kurt = float(series.kurtosis())

        # Shapiro-Wilk on a random sample (max 5000 for performance)
        sample = series.sample(min(len(series), 5000), random_state=42)
        _, shapiro_p = sp_stats.shapiro(sample)

        # Suggest transform
        transform = "none"
        if abs(skew) > 2:
            transform = "log1p" if series.min() >= 0 else "robust_scale"
        elif abs(skew) > 1:
            transform = "sqrt" if series.min() >= 0 else "robust_scale"

        stats_records.append({
            "feature": col,
            "n_valid": len(series),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "shapiro_p": round(float(shapiro_p), 6),
            "normal_5pct": shapiro_p > 0.05,
            "suggested_transform": transform,
        })

    if stats_records:
        stats_df = pd.DataFrame(stats_records).set_index("feature")
        save_table(stats_df, "distribution_statistics", cfg)

    # --- Summary ---
    non_normal = [r["feature"] for r in stats_records if not r["normal_5pct"]]
    needs_transform = [r["feature"] for r in stats_records
                       if r["suggested_transform"] != "none"]
    summary = {
        "non_normal_features": non_normal,
        "features_needing_transform": needs_transform,
        "n_analysed": len(stats_records),
    }
    print(f"  {len(non_normal)}/{len(stats_records)} features non-normal (p<0.05).")
    return summary


def _plot_histogram_grid(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    filename: str,
    cfg: Dict[str, Any],
) -> None:
    """Plot a grid of histograms with KDE overlay."""
    n = len(columns)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(columns):
        ax = axes_flat[i]
        data = df[col].dropna()
        ax.hist(data, bins=50, alpha=0.7, edgecolor="white", density=True)
        if len(data) > 10:
            data.plot.kde(ax=ax, color="red", linewidth=1.5)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("")

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=cfg["plot"]["title_size"], y=1.01)
    fig.tight_layout()
    save_figure(fig, filename, cfg)


def _plot_boxplots_by_prefecture(
    df: pd.DataFrame,
    features: list[str],
    cfg: Dict[str, Any],
) -> None:
    """Side-by-side boxplots for key features grouped by prefecture."""
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    pref_colors = cfg["plot"].get("prefecture_colors", {})

    for i, col in enumerate(features):
        ax = axes[i]
        sns.boxplot(
            data=df, x="pref_name", y=col, hue="pref_name", ax=ax,
            palette=pref_colors if pref_colors else None,
            width=0.5, legend=False,
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        if i > 0:
            ax.set_ylabel("")

    fig.suptitle("Feature Distributions by Prefecture",
                 fontsize=cfg["plot"]["title_size"], y=1.02)
    fig.tight_layout()
    save_figure(fig, "boxplots_by_prefecture", cfg)
