"""
Dynamic World LULC class fraction extraction.

Extracts mean probability fractions for 9 land-cover classes
per administrative unit from GOOGLE/DYNAMICWORLD/V1.

Classes: water, trees, grass, flooded_vegetation, crops,
         shrub_and_scrub, built, bare, snow_and_ice

Features per unit: dw_{class}_frac (9 columns)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import ee
import pandas as pd

from data_preprocessing.gee_monthly import (
    _export_fc_to_asset,
    _download_table_asset_csv,
    _retry_with_backoff,
)

# Dynamic World V1 class band names (probability layers)
DW_CLASSES: List[str] = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

# Output column names (prefixed + suffixed for clarity)
LULC_FEATURES: List[str] = [f"dw_{cls}_frac" for cls in DW_CLASSES]


def run_lulc_extraction(
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Extract Dynamic World LULC fractions for all study-area units.

    Computes a multi-year composite of mean class probabilities,
    then reduces to unit-level mean fractions.

    Args:
        cfg: Full project configuration dict.

    Returns:
        DataFrame with one row per unit and 9 dw_*_frac columns.
    """
    # --- Authenticate / initialize ---
    cloud_project = (cfg.get("gee", {}).get("cloud_project") or "").strip()
    if cloud_project and cloud_project != "YOUR_GCP_PROJECT_ID" and not cloud_project.startswith("YOUR_"):
        ee.Initialize(project=cloud_project)
    else:
        ee.Initialize()

    # --- Export settings ---
    asset_folder = (cfg.get("gee", {}).get("export_asset_folder") or "").rstrip("/")
    max_retries = int(cfg.get("gee", {}).get("max_retries", 3))

    # --- Load boundaries ---
    unit_level = cfg.get("run_mode", {}).get("unit_level", "mura")
    unit_id_field = cfg["data"]["unit_id_field"]

    if unit_level == "mura":
        boundaries_asset_id = cfg["gee"]["boundaries_asset_id_mura"]
    elif unit_level == "aza":
        boundaries_asset_id = cfg["gee"]["boundaries_asset_id_aza"]
    else:
        raise ValueError("run_mode.unit_level must be 'mura' or 'aza'")

    fc = ee.FeatureCollection(boundaries_asset_id)

    # Filter study-area prefectures
    prefs = cfg.get("study_area", {}).get("prefectures", [])
    if prefs:
        fc = fc.filter(ee.Filter.inList("pref_name", prefs))

    # Subsample for golden runs
    unit_sample_n = int(cfg.get("run_mode", {}).get("unit_sample_n") or 0)
    if unit_sample_n > 0:
        fc = fc.sort(unit_id_field).limit(unit_sample_n)

    # --- Load AOI for satellite filtering ---
    aoi_cfg = cfg.get("aoi", {})
    aoi_mode = aoi_cfg.get("mode", "full")
    aoi_geometry = None

    if aoi_cfg:
        aoi_asset_key = f"aoi_{aoi_mode}_asset_id"
        aoi_asset_id = aoi_cfg.get(aoi_asset_key)
        if aoi_asset_id:
            try:
                aoi_fc = ee.FeatureCollection(aoi_asset_id)
                aoi_geometry = aoi_fc.geometry()
            except Exception:
                pass

    if aoi_geometry is None:
        aoi_geometry = fc.geometry()

    # --- Skip-existing check ---
    out_dir = cfg.get("project", {}).get("outputs_dir", "outputs")
    cache_dir = os.path.join(out_dir, "gee", "lulc")
    cache_csv = os.path.join(cache_dir, f"lulc_features_{unit_level}.csv")

    skip_existing = bool(cfg.get("run_mode", {}).get("skip_existing_month_csv", True))
    if skip_existing and os.path.exists(cache_csv):
        print(f"  [lulc] Skipping (cached): {cache_csv}")
        df = pd.read_csv(cache_csv, dtype={unit_id_field: "string", "unit_code": "string"})
        return df

    print("Extracting LULC class fractions from Dynamic World V1...")

    # --- Settings ---
    start = cfg["time"]["start"]
    end = cfg["time"]["end"]
    lulc_scale = int(cfg.get("features", {}).get("lulc_scale", 100))

    os.makedirs(cache_dir, exist_ok=True)
    raw_selectors = [unit_id_field, "unit_level", "pref_name", "unit_code"] + DW_CLASSES

    # --- Process year-by-year to avoid GEE memory limits ---
    # Dynamic World V1 availability starts 2015-06-27 (tied to S2)
    start_year = int(start[:4])
    end_year = int(end[:4])
    yearly_dfs: List[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        year_csv = os.path.join(cache_dir, f"lulc_{unit_level}_{year}.csv")

        # Skip years already exported
        if os.path.exists(year_csv):
            print(f"  [lulc] {year}: cached, skipping")
            yr_df = pd.read_csv(year_csv, dtype={unit_id_field: "string", "unit_code": "string"})
            yearly_dfs.append(yr_df)
            continue

        print(f"  [lulc] {year}: computing composite...")

        yr_start = f"{year}-01-01"
        yr_end = f"{year + 1}-01-01"

        dw_year = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterDate(yr_start, yr_end)
            .filterBounds(aoi_geometry)
            .select(DW_CLASSES)
        )

        dw_composite = dw_year.mean()

        reduced = dw_composite.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=lulc_scale,
        )

        description = f"lulc_{unit_level}_{year}"
        temp_asset_id = f"{asset_folder}/temp_{description}"

        def _do_export(red: ee.FeatureCollection = reduced,
                       aid: str = temp_asset_id,
                       desc: str = description) -> None:
            _export_fc_to_asset(red, asset_id=aid, description=f"temp_{desc}")

        _retry_with_backoff(_do_export, max_retries=max_retries)

        _download_table_asset_csv(
            cfg={},
            asset_id=temp_asset_id,
            out_csv_path=year_csv,
            selectors=raw_selectors,
            max_retries=max_retries,
        )

        # Clean up temp asset
        try:
            ee.data.deleteAsset(temp_asset_id)
        except Exception:
            pass

        yr_df = pd.read_csv(year_csv, dtype={unit_id_field: "string", "unit_code": "string"})
        yearly_dfs.append(yr_df)
        print(f"  [lulc] {year}: done ({len(yr_df)} units)")

    # --- Average across years ---
    all_years = pd.concat(yearly_dfs, ignore_index=True)
    id_cols = [c for c in [unit_id_field, "unit_level", "pref_name", "unit_code"] if c in all_years.columns]
    df = all_years.groupby(id_cols, as_index=False)[DW_CLASSES].mean()

    # --- Rename to dw_*_frac ---
    rename_map = {cls: f"dw_{cls}_frac" for cls in DW_CLASSES}
    df = df.rename(columns=rename_map)

    keep_cols = [c for c in id_cols + LULC_FEATURES if c in df.columns]
    df = df[keep_cols]

    # Save final averaged result
    df.to_csv(cache_csv, index=False)

    n_units = len(df)
    print(f"  [lulc] Extracted {len(LULC_FEATURES)} class fractions for {n_units} units")
    for feat in LULC_FEATURES:
        if feat in df.columns:
            vals = df[feat].dropna()
            if len(vals) > 0:
                print(f"    {feat}: min={vals.min():.4f}, mean={vals.mean():.4f}, max={vals.max():.4f}")

    return df


if __name__ == "__main__":
    import yaml

    config_path = "config/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = run_lulc_extraction(cfg)
    print(f"\nResult: {len(df)} rows x {len(df.columns)} columns")
    print(df.head())
