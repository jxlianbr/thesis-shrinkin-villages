"""
Data loading and schema validation for the EDA module.

Reads the features table from Parquet, validates expected columns,
and adds derived temporal columns (year, month_num, season).
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
    Load the features table from Parquet and add derived temporal columns.

    Adds columns: month_dt (datetime), year (int), month_num (int), season (str).

    Args:
        cfg: EDA configuration dict.

    Returns:
        DataFrame with all original columns plus derived temporal columns.
    """
    path = cfg["data"]["features_table"]
    print(f"Loading features table from {path}...")
    df = pd.read_parquet(path)
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    # Parse month string (e.g. "2020-01") to datetime and derive temporal columns
    if "month" in df.columns:
        df["month_dt"] = pd.to_datetime(df["month"], format="%Y-%m")
        df["year"] = df["month_dt"].dt.year
        df["month_num"] = df["month_dt"].dt.month
        df["season"] = df["month_num"].map(SEASON_MAP)
        print(f"  Time range: {df['month'].min()} to {df['month'].max()}")

    if "unit_id" in df.columns:
        print(f"  Unique units: {df['unit_id'].nunique()}")

    if "pref_name" in df.columns:
        for pref, count in df["pref_name"].value_counts().items():
            n_units = df.loc[df["pref_name"] == pref, "unit_id"].nunique()
            print(f"  {pref}: {count} rows ({n_units} units)")

    return df


def validate_schema(df: pd.DataFrame, cfg: Dict[str, Any]) -> list[str]:
    """
    Check that all expected columns from config are present in the DataFrame.

    Args:
        df: Features DataFrame.
        cfg: EDA configuration dict.

    Returns:
        List of warning messages (empty if all columns present).
    """
    warnings: list[str] = []
    col_cfg = cfg["columns"]

    all_expected: list[str] = []
    for group_name in ("identifiers", "spectral", "indices", "texture",
                       "nightlights", "osm", "demographic"):
        group_cols = col_cfg.get(group_name, [])
        all_expected.extend(group_cols)

    missing = [c for c in all_expected if c not in df.columns]
    if missing:
        warnings.append(f"Missing expected columns: {missing}")
        print(f"  WARNING: Missing columns: {missing}")

    extra = [c for c in df.columns if c not in all_expected
             and c not in col_cfg.get("meta_only", [])
             and c not in ("month_dt", "year", "month_num", "season")]
    if extra:
        warnings.append(f"Extra columns not in config: {extra}")
        print(f"  INFO: Extra columns: {extra}")

    # Type checks for numeric columns
    numeric_expected = []
    for group_name in ("spectral", "indices", "texture", "nightlights", "osm", "demographic"):
        numeric_expected.extend(col_cfg.get(group_name, []))

    for col in numeric_expected:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            warnings.append(f"Column '{col}' expected numeric but is {df[col].dtype}")
            print(f"  WARNING: Column '{col}' is {df[col].dtype}, expected numeric")

    if not warnings:
        print("  Schema validation: all expected columns present and correct types.")

    return warnings
