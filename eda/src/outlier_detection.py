"""
Outlier detection for the EDA module.

Identifies outliers using the IQR method, produces boxplots
highlighting outliers, and exports summary and observation-level tables.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda.src.utils import get_numeric_columns, save_figure, save_table


def run_outlier_detection(
    df: pd.DataFrame, cfg: Dict[str, Any], output_dir: str,
) -> Dict[str, Any]:
    """
    Detect and visualise outliers in numeric features.

    Outputs:
        figures/outlier_boxplots.png
        tables/outlier_summary.csv
        tables/outlier_observations.csv

    Returns:
        Summary dict with total_outliers, most_affected_features, most_affected_units.
    """
    print("Running outlier detection...")
    num_cols = get_numeric_columns(df, cfg)

    method = cfg["analysis"]["outlier_method"]
    iqr_factor = cfg["analysis"]["outlier_iqr_factor"]

    # --- Detect outliers per feature ---
    feature_summaries = []
    all_observations = []

    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outlier_rows = df.loc[outlier_mask & df[col].notna()]
        n_outliers = len(outlier_rows)

        feature_summaries.append({
            "feature": col,
            "n_valid": int(len(series)),
            "outlier_count": n_outliers,
            "outlier_pct": round(n_outliers / len(series) * 100, 2),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
        })

        # Record individual outlier observations
        for _, row in outlier_rows.iterrows():
            val = row[col]
            all_observations.append({
                "unit_id": row.get("unit_id", None),
                "month": row.get("month", None),
                "feature": col,
                "value": round(float(val), 4),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
                "bound_exceeded": "lower" if val < lower else "upper",
            })

    # Save tables
    if feature_summaries:
        summary_df = pd.DataFrame(feature_summaries).set_index("feature")
        save_table(summary_df, "outlier_summary", cfg)

    if all_observations:
        obs_df = pd.DataFrame(all_observations)
        save_table(obs_df.set_index("unit_id") if "unit_id" in obs_df.columns else obs_df,
                   "outlier_observations", cfg)

    # --- Boxplot figure ---
    _plot_outlier_boxplots(df, num_cols, cfg)

    # --- Summary ---
    total = sum(s["outlier_count"] for s in feature_summaries)
    most_affected_features = sorted(
        feature_summaries, key=lambda x: x["outlier_pct"], reverse=True,
    )[:5]
    most_affected_units = []
    if all_observations:
        obs_df_tmp = pd.DataFrame(all_observations)
        if "unit_id" in obs_df_tmp.columns:
            unit_counts = obs_df_tmp["unit_id"].value_counts().head(5)
            most_affected_units = [
                {"unit_id": uid, "outlier_count": int(cnt)}
                for uid, cnt in unit_counts.items()
            ]

    summary = {
        "total_outlier_observations": total,
        "features_with_outliers": sum(1 for s in feature_summaries if s["outlier_count"] > 0),
        "most_affected_features": [
            {"feature": s["feature"], "outlier_pct": s["outlier_pct"]}
            for s in most_affected_features
        ],
        "most_affected_units": most_affected_units,
        "method": f"IQR × {iqr_factor}",
    }
    print(f"  {total} outlier observations across {summary['features_with_outliers']} features.")
    return summary


def _plot_outlier_boxplots(
    df: pd.DataFrame,
    columns: list[str],
    cfg: Dict[str, Any],
) -> None:
    """Grid of boxplots for all numeric features, highlighting outliers."""
    n = len(columns)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(columns):
        ax = axes_flat[i]
        data = df[col].dropna()
        ax.boxplot(data, vert=True, widths=0.5,
                   flierprops={"marker": "o", "markersize": 2, "alpha": 0.5})
        ax.set_title(col, fontsize=9)
        ax.set_xticks([])

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions with Outliers (IQR Method)",
                 fontsize=cfg["plot"]["title_size"], y=1.01)
    fig.tight_layout()
    save_figure(fig, "outlier_boxplots", cfg)
