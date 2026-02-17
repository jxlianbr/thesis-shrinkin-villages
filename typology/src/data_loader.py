"""
Data loading for the typology module.

Loads the preprocessed cross-sectional data, raw monthly panel data,
boundary geometries, and preprocessing metadata needed for indicator
compilation and analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


# Season mapping for monthly data
SEASON_MAP = {12: "DJF", 1: "DJF", 2: "DJF",
              3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA",
              9: "SON", 10: "SON", 11: "SON"}


def load_typology_data(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load all input datasets for the typology pipeline.

    Args:
        cfg: Typology configuration dict.

    Returns:
        Dict with keys: scaled_df, X_scaled, y_labels, y_codes,
        identifiers, panel_df, boundaries, feature_metadata,
        preprocessing_report, n_units, n_months.
    """
    result: Dict[str, Any] = {}

    # --- 1. Classification-ready data (65 x 34, scaled) ---
    cr_path = cfg["data"]["classification_ready_path"]
    print(f"  Loading cross-sectional data: {cr_path}")
    scaled_df = pd.read_parquet(cr_path)
    result["scaled_df"] = scaled_df

    id_cols = cfg["columns"]["identifiers"]
    target_col = cfg["columns"]["target"]
    target_code_col = cfg["columns"]["target_code"]

    result["identifiers"] = scaled_df[id_cols].copy()
    result["y_labels"] = scaled_df[target_col].copy()
    result["y_codes"] = scaled_df[target_code_col].copy()

    exclude = set(id_cols + [target_col, target_code_col])
    feat_cols = [c for c in scaled_df.select_dtypes(include="number").columns
                 if c not in exclude]
    result["X_scaled"] = scaled_df[feat_cols].copy()
    result["n_units"] = len(scaled_df)

    print(f"    {result['n_units']} units, {len(feat_cols)} scaled features")

    # --- 2. Raw monthly panel data (8450 x 30) ---
    panel_path = cfg["data"]["features_table_path"]
    print(f"  Loading raw panel data: {panel_path}")
    panel_df = pd.read_parquet(panel_path)

    # Add temporal columns
    if "month" in panel_df.columns:
        panel_df["month_dt"] = pd.to_datetime(panel_df["month"])
        panel_df["year"] = panel_df["month_dt"].dt.year
        panel_df["month_num"] = panel_df["month_dt"].dt.month
        panel_df["season"] = panel_df["month_num"].map(SEASON_MAP)

    result["panel_df"] = panel_df
    result["n_months"] = panel_df.groupby("unit_id").size().mean()
    print(f"    {len(panel_df)} rows, mean {result['n_months']:.1f} months/unit")

    # --- 3. Boundary geometries ---
    boundaries = _load_boundaries(cfg)
    result["boundaries"] = boundaries

    # --- 4. Feature metadata ---
    meta_path = cfg["data"]["feature_metadata_path"]
    if Path(meta_path).exists():
        result["feature_metadata"] = pd.read_csv(meta_path)
        print(f"  Loaded feature metadata: {meta_path}")
    else:
        result["feature_metadata"] = None
        print(f"  WARNING: Feature metadata not found: {meta_path}")

    # --- 5. Preprocessing report (scaler params) ---
    report_path = cfg["data"]["preprocessing_report_path"]
    if Path(report_path).exists():
        with open(report_path, "r", encoding="utf-8") as f:
            result["preprocessing_report"] = json.load(f)
        print(f"  Loaded preprocessing report: {report_path}")
    else:
        result["preprocessing_report"] = None
        print(f"  WARNING: Preprocessing report not found: {report_path}")

    # --- 6. Classification results (optional) ---
    cr_results_path = cfg["data"].get("classification_results_path", "")
    if cr_results_path and Path(cr_results_path).exists():
        result["classification_results"] = pd.read_parquet(cr_results_path)
        print(f"  Loaded classification results: {cr_results_path}")
    else:
        result["classification_results"] = None
        print(f"  NOTE: Classification results not found -- "
              f"built-up change indicators will be skipped")

    return result


def _load_boundaries(cfg: Dict[str, Any]) -> Any:
    """Load boundary GeoPackage, returning None if unavailable."""
    try:
        import geopandas as gpd
    except ImportError:
        print("  WARNING: geopandas not installed -- spatial features disabled")
        return None

    bounds_path = cfg["data"]["boundaries_path"]
    if not Path(bounds_path).exists():
        print(f"  WARNING: Boundaries file not found: {bounds_path}")
        return None

    gdf = gpd.read_file(bounds_path)
    print(f"  Loaded boundaries: {bounds_path} ({len(gdf)} units)")
    return gdf
