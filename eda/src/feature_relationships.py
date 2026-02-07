"""
Feature relationship analysis for the EDA module.

Produces scatter plots examining key RS-demographic relationships
at the unit level (averaged across months).
"""
from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda.src.utils import get_existing_columns, save_figure


def run_feature_relationships(
    df: pd.DataFrame, cfg: Dict[str, Any], output_dir: str,
) -> Dict[str, Any]:
    """
    Visualise key remote-sensing vs demographic relationships.

    Uses per-unit means (averaged across months) to create scatter plots
    colored by prefecture.

    Outputs:
        figures/ndvi_vs_population.png
        figures/viirs_vs_population.png

    Returns:
        Summary dict with key_relationships.
    """
    print("Running feature relationship analysis...")

    # Aggregate to unit level (mean across months)
    group_cols = ["unit_id"]
    if "pref_name" in df.columns:
        group_cols.append("pref_name")

    unit_means = df.groupby(group_cols).mean(numeric_only=True).reset_index()

    pref_colors = cfg["plot"].get("prefecture_colors", {})
    relationships = []

    # --- NDVI vs Population ---
    if "NDVI" in unit_means.columns and "pop_total" in unit_means.columns:
        r = _plot_scatter(
            unit_means, "NDVI", "pop_total",
            "Mean NDVI vs Total Population (per unit)",
            "ndvi_vs_population", pref_colors, cfg,
        )
        relationships.append({"x": "NDVI", "y": "pop_total", "r": r})

    # --- VIIRS vs Population ---
    if "viirs_mean" in unit_means.columns and "pop_total" in unit_means.columns:
        r = _plot_scatter(
            unit_means, "viirs_mean", "pop_total",
            "Mean VIIRS Night Lights vs Total Population (per unit)",
            "viirs_vs_population", pref_colors, cfg,
        )
        relationships.append({"x": "viirs_mean", "y": "pop_total", "r": r})

    # --- NDBI vs Population ---
    if "NDBI" in unit_means.columns and "pop_total" in unit_means.columns:
        r = _plot_scatter(
            unit_means, "NDBI", "pop_total",
            "Mean NDBI vs Total Population (per unit)",
            "ndbi_vs_population", pref_colors, cfg,
        )
        relationships.append({"x": "NDBI", "y": "pop_total", "r": r})

    # --- VIIRS vs Elderly Ratio ---
    if "viirs_mean" in unit_means.columns and "age_65_plus" in unit_means.columns:
        # Compute elderly ratio if pop_total available
        if "pop_total" in unit_means.columns:
            mask = unit_means["pop_total"] > 0
            unit_means.loc[mask, "elderly_ratio"] = (
                unit_means.loc[mask, "age_65_plus"] / unit_means.loc[mask, "pop_total"]
            )
            if "elderly_ratio" in unit_means.columns:
                r = _plot_scatter(
                    unit_means, "viirs_mean", "elderly_ratio",
                    "Mean VIIRS Night Lights vs Elderly Ratio (per unit)",
                    "viirs_vs_elderly_ratio", pref_colors, cfg,
                )
                relationships.append({"x": "viirs_mean", "y": "elderly_ratio", "r": r})

    summary = {"key_relationships": relationships}
    print(f"  {len(relationships)} relationship plots generated.")
    return summary


def _plot_scatter(
    unit_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    filename: str,
    pref_colors: Dict[str, str],
    cfg: Dict[str, Any],
) -> float | None:
    """
    Scatter plot with points colored by prefecture and a linear fit line.

    Returns Pearson r value.
    """
    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_single"])

    valid = unit_df[[x_col, y_col]].dropna()
    r_value = None

    if "pref_name" in unit_df.columns:
        for pref in sorted(unit_df["pref_name"].dropna().unique()):
            mask = unit_df["pref_name"] == pref
            subset = unit_df.loc[mask].dropna(subset=[x_col, y_col])
            color = pref_colors.get(pref, None)
            ax.scatter(
                subset[x_col], subset[y_col],
                label=pref, color=color, alpha=0.7, s=40, edgecolors="white",
                linewidths=0.5,
            )
        ax.legend()
    else:
        ax.scatter(valid[x_col], valid[y_col], alpha=0.7, s=40)

    # Linear fit
    if len(valid) >= 3:
        coeffs = np.polyfit(valid[x_col], valid[y_col], 1)
        fit_x = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
        ax.plot(fit_x, np.polyval(coeffs, fit_x), "r--", alpha=0.6, linewidth=1.5)
        r_value = round(float(valid[x_col].corr(valid[y_col])), 4)
        ax.text(
            0.05, 0.95, f"r = {r_value}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, filename, cfg)

    return r_value
