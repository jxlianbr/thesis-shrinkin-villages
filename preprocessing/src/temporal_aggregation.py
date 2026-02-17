"""
Temporal aggregation for the preprocessing module.

Collapses the panel-format DataFrame (unit x month) into a
cross-sectional DataFrame (one row per unit) by computing
temporal summary statistics for remote-sensing features.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def aggregate_to_cross_section(
    df: pd.DataFrame, cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Collapse panel data (4680 rows) to cross-section (65 rows).

    For RS features: computes mean, std, and optionally trend slope
    and seasonal amplitude per unit.
    For demographic/OSM features: takes first value (static per unit).

    Args:
        df: Panel-format DataFrame with month_dt column.
        cfg: Preprocessing configuration dict.

    Returns:
        Cross-sectional DataFrame with one row per unit.
    """
    print("Aggregating to cross-section (panel -> 1 row per unit)...")
    agg_cfg = cfg["temporal_aggregation"]
    demo_cols = [c for c in cfg["demographic_features"] if c in df.columns]
    osm_cols = [c for c in cfg["osm_features"] if c in df.columns]
    terrain_cols = [c for c in cfg.get("terrain_features", []) if c in df.columns]
    lulc_cols = [c for c in cfg.get("lulc_features", []) if c in df.columns]
    mean_std_cols = [c for c in agg_cfg["mean_std_features"] if c in df.columns]
    trend_cols = [c for c in agg_cfg["trend_features"] if c in df.columns]
    seasonal_cols = [c for c in agg_cfg["seasonal_features"] if c in df.columns]

    records: list[Dict[str, Any]] = []

    for unit_id, grp in df.groupby("unit_id"):
        row: Dict[str, Any] = {"unit_id": unit_id}

        # Carry through pref_name
        if "pref_name" in grp.columns:
            row["pref_name"] = grp["pref_name"].iloc[0]

        # --- Mean + Std for RS features ---
        for col in mean_std_cols:
            series = grp[col].dropna()
            row[f"{col}_mean"] = float(series.mean()) if len(series) > 0 else np.nan
            row[f"{col}_std"] = float(series.std()) if len(series) > 1 else 0.0

        # --- Trend slope (OLS) ---
        if trend_cols and "month_dt" in grp.columns:
            grp_sorted = grp.sort_values("month_dt")
            t = (grp_sorted["month_dt"] - grp_sorted["month_dt"].min()).dt.days.values.astype(float)
            if len(t) >= 3 and t[-1] > 0:
                t_norm = t / t[-1]  # Normalize to [0, 1]
                for col in trend_cols:
                    valid = grp_sorted[col].notna()
                    if valid.sum() >= 3:
                        y = grp_sorted.loc[valid, col].values.astype(float)
                        t_valid = t_norm[valid.values]
                        coeffs = np.polyfit(t_valid, y, 1)
                        row[f"{col}_slope"] = round(float(coeffs[0]), 6)
                    else:
                        row[f"{col}_slope"] = np.nan

        # --- Seasonal amplitude ---
        if seasonal_cols and "month_num" in grp.columns:
            for col in seasonal_cols:
                monthly_means = grp.groupby("month_num")[col].mean()
                if len(monthly_means.dropna()) >= 2:
                    amp = float(monthly_means.max() - monthly_means.min())
                    row[f"{col}_seasonal_amp"] = round(amp, 6)
                else:
                    row[f"{col}_seasonal_amp"] = np.nan

        # --- Static features (demographics, OSM, terrain, LULC): take first value ---
        for col in demo_cols + osm_cols + terrain_cols + lulc_cols:
            row[col] = grp[col].iloc[0]

        records.append(row)

    result = pd.DataFrame(records)
    print(f"  Aggregated: {len(df)} rows -> {len(result)} rows")
    print(f"  Columns: {len(result.columns)} "
          f"(RS aggregates + static demographics/OSM/terrain/LULC)")
    return result
