"""
Correlation analysis for the EDA module.

Produces full correlation heatmap, RS-demographic focused heatmap,
and tables of correlation matrices and high-correlation pairs.
"""
from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eda.src.utils import get_existing_columns, get_numeric_columns, save_figure, save_table


def run_correlation_analysis(
    df: pd.DataFrame, cfg: Dict[str, Any], output_dir: str,
) -> Dict[str, Any]:
    """
    Compute and visualise correlations between features.

    Outputs:
        figures/correlation_heatmap.png
        figures/rs_demo_correlation.png
        tables/correlation_matrix.csv
        tables/high_correlations.csv
        tables/rs_demographic_correlations.csv

    Returns:
        Summary dict with highly_correlated_pairs and top RS-demo associations.
    """
    print("Running correlation analysis...")
    threshold = cfg["analysis"]["correlation_threshold"]

    # All numeric columns
    num_cols = get_numeric_columns(df, cfg)

    # --- Full correlation matrix ---
    corr = df[num_cols].corr()
    save_table(corr, "correlation_matrix", cfg)

    # --- Heatmap ---
    _plot_correlation_heatmap(corr, "Feature Correlation Matrix",
                              "correlation_heatmap", cfg)

    # --- High correlations ---
    high_pairs = _extract_high_correlations(corr, threshold)
    if high_pairs:
        high_df = pd.DataFrame(high_pairs)
        save_table(high_df.set_index("pair"), "high_correlations", cfg)

    # --- RS × Demographic focused analysis ---
    rs_cols = get_existing_columns(df,
        cfg["columns"].get("spectral", [])
        + cfg["columns"].get("indices", [])
        + cfg["columns"].get("nightlights", [])
        + cfg["columns"].get("texture", [])
        + cfg["columns"].get("osm", [])
    )
    demo_cols = get_existing_columns(df, cfg["columns"].get("demographic", []))

    rs_demo_summary: list[dict] = []
    if rs_cols and demo_cols:
        rs_demo_corr = df[rs_cols + demo_cols].corr().loc[rs_cols, demo_cols]
        save_table(rs_demo_corr, "rs_demographic_correlations", cfg)
        _plot_rs_demo_heatmap(rs_demo_corr, cfg)

        # Top associations
        for rs_col in rs_cols:
            for demo_col in demo_cols:
                r = rs_demo_corr.loc[rs_col, demo_col]
                if abs(r) > 0.3:
                    rs_demo_summary.append({
                        "rs_feature": rs_col,
                        "demo_feature": demo_col,
                        "correlation": round(float(r), 4),
                    })
        rs_demo_summary.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    summary = {
        "n_features": len(num_cols),
        "highly_correlated_pairs": high_pairs[:10] if high_pairs else [],
        "top_rs_demo_associations": rs_demo_summary[:10],
        "correlation_threshold": threshold,
    }
    print(f"  {len(high_pairs)} pairs above |r|>{threshold}.")
    return summary


def _plot_correlation_heatmap(
    corr: pd.DataFrame, title: str, filename: str, cfg: Dict[str, Any],
) -> None:
    """Full correlation matrix heatmap with upper triangle masked."""
    mask = np.triu(np.ones_like(corr, dtype=bool))
    figsize = cfg["plot"]["figsize_heatmap"]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 7},
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, filename, cfg)


def _plot_rs_demo_heatmap(corr: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    """Focused heatmap: RS features (rows) vs demographic features (columns)."""
    fig, ax = plt.subplots(figsize=(max(8, corr.shape[1] * 1.2),
                                    max(6, corr.shape[0] * 0.5)))
    sns.heatmap(
        corr, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 9},
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Remote Sensing × Demographic Correlations")
    ax.set_xlabel("Demographic Features")
    ax.set_ylabel("Remote Sensing Features")
    fig.tight_layout()
    save_figure(fig, "rs_demo_correlation", cfg)


def _extract_high_correlations(
    corr: pd.DataFrame, threshold: float,
) -> list[Dict[str, Any]]:
    """Extract feature pairs with |r| above threshold."""
    pairs = []
    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r = corr.loc[c1, c2]
            if abs(r) >= threshold:
                pairs.append({
                    "pair": f"{c1} -- {c2}",
                    "feature_1": c1,
                    "feature_2": c2,
                    "correlation": round(float(r), 4),
                    "abs_correlation": round(abs(float(r)), 4),
                })
    pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return pairs
