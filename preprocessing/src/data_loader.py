"""
Data loading for the preprocessing module.

Reads the features table from Parquet and adds derived temporal columns.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd


SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


def load_features_table(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load the features table from Parquet and add temporal columns.

    Adds: month_dt (datetime), year, month_num, season.

    Args:
        cfg: Preprocessing configuration dict.

    Returns:
        DataFrame with original + derived temporal columns.
    """
    path = cfg["data"]["features_table"]
    print(f"Loading features table from {path}...")
    df = pd.read_parquet(path)
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    if "month" in df.columns:
        df["month_dt"] = pd.to_datetime(df["month"], format="%Y-%m")
        df["year"] = df["month_dt"].dt.year
        df["month_num"] = df["month_dt"].dt.month
        df["season"] = df["month_num"].map(SEASON_MAP)
        print(f"  Time range: {df['month'].min()} to {df['month'].max()}")

    if "unit_id" in df.columns:
        print(f"  Unique units: {df['unit_id'].nunique()}")

    return df
