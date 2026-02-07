"""
Temporal analysis for the EDA module.

Produces monthly trend plots, seasonal pattern boxplots,
temporal coverage tables, and per-unit linear trend slopes.
"""
from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eda.src.utils import get_existing_columns, save_figure, save_table


def run_temporal_analysis(
    df: pd.DataFrame, cfg: Dict[str, Any], output_dir: str,
) -> Dict[str, Any]:
    """
    Analyse temporal patterns in the feature data.

    Outputs:
        figures/temporal_trends.png
        figures/seasonal_patterns.png
        tables/temporal_coverage.csv
        tables/unit_trends.csv

    Returns:
        Summary dict with coverage_summary, seasonal_amplitude, trend_directions.
    """
    print("Running temporal analysis...")
    if "month_dt" not in df.columns:
        print("  WARNING: month_dt column missing, skipping temporal analysis.")
        return {"skipped": True, "reason": "no month_dt column"}

    key_features = get_existing_columns(df, ["NDVI", "NDBI", "viirs_mean"])

    # --- Temporal trends ---
    if key_features and "pref_name" in df.columns:
        _plot_temporal_trends(df, key_features, cfg)

    # --- Seasonal patterns ---
    seasonal_cols = get_existing_columns(df, ["NDVI", "NDBI"])
    if seasonal_cols and "month_num" in df.columns:
        _plot_seasonal_patterns(df, seasonal_cols, cfg)

    # --- Temporal coverage ---
    coverage_df = _compute_temporal_coverage(df, cfg)
    save_table(coverage_df, "temporal_coverage", cfg)

    # --- Unit-level trends ---
    trends_df = _compute_unit_trends(df, key_features)
    if trends_df is not None:
        save_table(trends_df, "unit_trends", cfg)

    # --- Summary ---
    seasonal_amp = {}
    if "month_num" in df.columns:
        for col in get_existing_columns(df, ["NDVI", "NDBI"]):
            monthly = df.groupby("month_num")[col].mean()
            seasonal_amp[col] = round(float(monthly.max() - monthly.min()), 4)

    trend_dirs = {}
    if trends_df is not None:
        for col in [c for c in trends_df.columns if c.endswith("_slope")]:
            mean_slope = trends_df[col].mean()
            trend_dirs[col.replace("_slope", "")] = (
                "increasing" if mean_slope > 0 else "decreasing"
            )

    summary = {
        "coverage_summary": {
            "mean_months_per_unit": round(float(coverage_df["n_valid_months"].mean()), 1),
            "min_months": int(coverage_df["n_valid_months"].min()),
            "units_with_gaps": int((coverage_df["gap_count"] > 0).sum()),
        },
        "seasonal_amplitude": seasonal_amp,
        "trend_directions": trend_dirs,
    }
    print(f"  Mean coverage: {summary['coverage_summary']['mean_months_per_unit']} months/unit.")
    return summary


def _plot_temporal_trends(
    df: pd.DataFrame,
    features: list[str],
    cfg: Dict[str, Any],
) -> None:
    """Monthly mean trends with ±1 std shading, faceted by prefecture."""
    prefectures = sorted(df["pref_name"].unique())
    n_feat = len(features)
    n_pref = len(prefectures)
    pref_colors = cfg["plot"].get("prefecture_colors", {})

    fig, axes = plt.subplots(n_feat, 1, figsize=(14, 4 * n_feat), sharex=True)
    if n_feat == 1:
        axes = [axes]

    for i, col in enumerate(features):
        ax = axes[i]
        for pref in prefectures:
            pref_df = df[df["pref_name"] == pref]
            monthly = pref_df.groupby("month_dt")[col].agg(["mean", "std"])
            color = pref_colors.get(pref, None)
            ax.plot(monthly.index, monthly["mean"], label=pref, color=color)
            ax.fill_between(
                monthly.index,
                monthly["mean"] - monthly["std"],
                monthly["mean"] + monthly["std"],
                alpha=0.15, color=color,
            )
        ax.set_ylabel(col)
        ax.legend(loc="upper right")
        if i == 0:
            ax.set_title("Monthly Temporal Trends by Prefecture")

    axes[-1].set_xlabel("Month")
    fig.tight_layout()
    save_figure(fig, "temporal_trends", cfg)


def _plot_seasonal_patterns(
    df: pd.DataFrame,
    features: list[str],
    cfg: Dict[str, Any],
) -> None:
    """Monthly boxplots (1-12) aggregated across years to show seasonality."""
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(features):
        ax = axes[i]
        sns.boxplot(
            data=df, x="month_num", y=col, ax=ax,
            color="steelblue", width=0.6, fliersize=2,
        )
        ax.set_xlabel("Month")
        ax.set_title(f"Seasonal Pattern: {col}")
        if i > 0:
            ax.set_ylabel("")

    fig.suptitle("Seasonal Patterns (Aggregated Across Years)",
                 fontsize=cfg["plot"]["title_size"], y=1.02)
    fig.tight_layout()
    save_figure(fig, "seasonal_patterns", cfg)


def _compute_temporal_coverage(
    df: pd.DataFrame, cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Per-unit temporal coverage statistics."""
    # Use NDVI as the primary indicator of valid RS observations
    indicator_col = "NDVI" if "NDVI" in df.columns else df.select_dtypes("number").columns[0]

    records = []
    for unit_id, grp in df.groupby("unit_id"):
        valid = grp[indicator_col].notna()
        n_total = len(grp)
        n_valid = int(valid.sum())
        months_sorted = grp["month_dt"].sort_values()
        first = str(months_sorted.iloc[0].date()) if len(months_sorted) > 0 else None
        last = str(months_sorted.iloc[-1].date()) if len(months_sorted) > 0 else None

        # Count gaps in the valid observation sequence
        valid_months = grp.loc[valid, "month_dt"].sort_values()
        gap_count = 0
        if len(valid_months) > 1:
            diffs = valid_months.diff().dt.days
            gap_count = int((diffs > 35).sum())  # >35 days = gap

        records.append({
            "unit_id": unit_id,
            "n_total_months": n_total,
            "n_valid_months": n_valid,
            "valid_pct": round(n_valid / n_total * 100, 1) if n_total > 0 else 0,
            "first_month": first,
            "last_month": last,
            "gap_count": gap_count,
        })

    return pd.DataFrame(records).set_index("unit_id")


def _compute_unit_trends(
    df: pd.DataFrame, features: list[str],
) -> pd.DataFrame | None:
    """Compute per-unit linear trend slope (OLS) for key features."""
    if not features or "unit_id" not in df.columns or "month_dt" not in df.columns:
        return None

    records = []
    for unit_id, grp in df.groupby("unit_id"):
        row: Dict[str, Any] = {"unit_id": unit_id}
        grp = grp.sort_values("month_dt")
        # Time in fractional years
        t = (grp["month_dt"] - grp["month_dt"].min()).dt.days.values.astype(float)
        if len(t) < 3 or t[-1] == 0:
            continue
        t = t / t[-1]  # Normalize to [0, 1]

        for col in features:
            valid = grp[col].notna()
            if valid.sum() < 3:
                row[f"{col}_slope"] = np.nan
                continue
            y = grp.loc[valid, col].values.astype(float)
            t_valid = t[valid.values]
            # Simple OLS slope
            coeffs = np.polyfit(t_valid, y, 1)
            row[f"{col}_slope"] = round(float(coeffs[0]), 6)

        records.append(row)

    if not records:
        return None
    return pd.DataFrame(records).set_index("unit_id")
