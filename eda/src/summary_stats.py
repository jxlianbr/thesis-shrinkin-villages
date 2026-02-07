"""
Descriptive statistics computation for the EDA module.

Produces basic statistics (describe + skew/kurtosis) and
per-prefecture grouped statistics.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from scipy import stats as sp_stats

from eda.src.utils import get_numeric_columns, save_table


def run_summary_stats(
    df: pd.DataFrame, cfg: Dict[str, Any], output_dir: str,
) -> Dict[str, Any]:
    """
    Compute and save descriptive statistics.

    Outputs:
        tables/basic_statistics.csv
        tables/statistics_by_prefecture.csv

    Returns:
        Summary dict with row_count, column_count, unit_count, month_range,
        and per-column stats highlights.
    """
    print("Running summary statistics...")
    num_cols = get_numeric_columns(df, cfg)

    # --- Basic statistics ---
    desc = df[num_cols].describe().T
    desc["skew"] = df[num_cols].skew()
    desc["kurtosis"] = df[num_cols].kurtosis()
    desc["missing_count"] = df[num_cols].isna().sum()
    desc["missing_pct"] = (df[num_cols].isna().mean() * 100).round(2)
    save_table(desc, "basic_statistics", cfg)

    # --- Statistics by prefecture ---
    if "pref_name" in df.columns:
        grouped = df.groupby("pref_name")[num_cols].agg(["mean", "std", "min", "max", "count"])
        # Flatten multi-level columns
        grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
        save_table(grouped, "statistics_by_prefecture", cfg)

    # --- Build summary dict ---
    month_range = None
    if "month" in df.columns:
        month_range = {"min": str(df["month"].min()), "max": str(df["month"].max())}

    summary = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "numeric_feature_count": len(num_cols),
        "unit_count": int(df["unit_id"].nunique()) if "unit_id" in df.columns else None,
        "month_range": month_range,
        "prefectures": list(df["pref_name"].unique()) if "pref_name" in df.columns else [],
        "most_variable_features": desc["std"].nlargest(5).to_dict(),
        "most_skewed_features": desc["skew"].abs().nlargest(5).to_dict(),
    }
    print(f"  {len(num_cols)} numeric features analysed.")
    return summary
