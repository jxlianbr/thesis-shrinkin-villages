"""
Missing data analysis for the EDA module.

Produces a heatmap of missing percentages per feature × prefecture,
a time-series of missing counts for spectral features, and a summary table.
"""
from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eda.src.utils import (
    get_existing_columns,
    get_numeric_columns,
    save_figure,
    save_table,
)


def run_missing_analysis(
    df: pd.DataFrame, cfg: Dict[str, Any], output_dir: str,
) -> Dict[str, Any]:
    """
    Analyse and visualise missing data patterns.

    Outputs:
        figures/missing_heatmap.png
        figures/missing_by_time.png
        tables/missing_data_summary.csv

    Returns:
        Summary dict with overall_missing_pct, columns_above_threshold, glcm_note.
    """
    print("Running missing data analysis...")
    num_cols = get_numeric_columns(df, cfg)
    threshold = cfg["analysis"]["missing_threshold"]

    # --- Per-column missing summary ---
    records = []
    for col in num_cols:
        n_miss = int(df[col].isna().sum())
        pct_miss = round(n_miss / len(df) * 100, 2)
        by_pref = {}
        if "pref_name" in df.columns:
            for pref, grp in df.groupby("pref_name"):
                by_pref[pref] = round(grp[col].isna().mean() * 100, 2)
        records.append({
            "feature": col,
            "missing_count": n_miss,
            "missing_pct": pct_miss,
            "flagged": pct_miss > threshold * 100,
            **{f"missing_pct_{k}": v for k, v in by_pref.items()},
        })
    summary_df = pd.DataFrame(records).set_index("feature")
    save_table(summary_df, "missing_data_summary", cfg)

    # --- Missing heatmap (feature × prefecture) ---
    _plot_missing_heatmap(df, num_cols, cfg)

    # --- Missing by time (spectral features only) ---
    spectral_cols = get_existing_columns(
        df,
        cfg["columns"].get("spectral", []) + cfg["columns"].get("indices", []),
    )
    if spectral_cols and "month_dt" in df.columns:
        _plot_missing_by_time(df, spectral_cols, cfg)

    # --- Build summary ---
    cols_above = summary_df.loc[
        summary_df["missing_pct"] > threshold * 100
    ].index.tolist()

    overall_pct = round(df[num_cols].isna().mean().mean() * 100, 2)

    summary = {
        "overall_missing_pct": overall_pct,
        "columns_above_threshold": cols_above,
        "glcm_note": cfg["analysis"]["glcm_nan_note"],
        "threshold_pct": threshold * 100,
    }
    print(f"  Overall missing: {overall_pct}%  |  "
          f"Columns above {threshold*100}%: {len(cols_above)}")
    return summary


def _plot_missing_heatmap(
    df: pd.DataFrame,
    num_cols: list[str],
    cfg: Dict[str, Any],
) -> None:
    """Heatmap of missing % per feature, grouped by prefecture."""
    if "pref_name" not in df.columns:
        return

    pref_groups = df.groupby("pref_name")
    miss_by_pref = pd.DataFrame({
        pref: grp[num_cols].isna().mean() * 100
        for pref, grp in pref_groups
    })
    # Add overall column
    miss_by_pref["Overall"] = df[num_cols].isna().mean() * 100

    fig, ax = plt.subplots(figsize=(8, max(6, len(num_cols) * 0.35)))
    sns.heatmap(
        miss_by_pref,
        annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Missing %"},
    )
    ax.set_title("Missing Data by Feature and Prefecture")
    ax.set_ylabel("")

    save_figure(fig, "missing_heatmap", cfg)


def _plot_missing_by_time(
    df: pd.DataFrame,
    spectral_cols: list[str],
    cfg: Dict[str, Any],
) -> None:
    """Line plot of monthly missing count for spectral/index features."""
    monthly_miss = df.groupby("month_dt")[spectral_cols].apply(
        lambda x: x.isna().sum()
    )

    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_wide"])
    for col in spectral_cols:
        ax.plot(monthly_miss.index, monthly_miss[col], label=col, alpha=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Missing Count")
    ax.set_title("Missing Spectral/Index Values Over Time (Cloud Cover Effect)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "missing_by_time", cfg)
