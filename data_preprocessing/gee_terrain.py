"""
DEM/Terrain feature extraction from Copernicus GLO-30.

Extracts static terrain features per administrative unit:
- elevation_mean, elevation_std
- slope_mean, slope_std
- aspect_sin_mean, aspect_cos_mean  (circular-mean decomposition)
- tri_mean (Terrain Ruggedness Index, stdev proxy)

Source: ee.ImageCollection("COPERNICUS/DEM/GLO30"), band "DEM"
"""
from __future__ import annotations

import math
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

# Output column names
TERRAIN_FEATURES: List[str] = [
    "elevation_mean",
    "elevation_std",
    "slope_mean",
    "slope_std",
    "aspect_sin_mean",
    "aspect_cos_mean",
    "tri_mean",
]


def run_terrain_extraction(
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Extract terrain features for all study-area units.

    Computes zonal statistics from Copernicus DEM GLO-30 and
    derived terrain layers (slope, aspect, TRI).

    Args:
        cfg: Full project configuration dict.

    Returns:
        DataFrame with one row per unit and 7 terrain feature columns.
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

    # --- Skip-existing check ---
    out_dir = cfg.get("project", {}).get("outputs_dir", "outputs")
    cache_dir = os.path.join(out_dir, "gee", "terrain")
    cache_csv = os.path.join(cache_dir, f"terrain_features_{unit_level}.csv")

    skip_existing = bool(cfg.get("run_mode", {}).get("skip_existing_month_csv", True))
    if skip_existing and os.path.exists(cache_csv):
        print(f"  [terrain] Skipping (cached): {cache_csv}")
        df = pd.read_csv(cache_csv, dtype={unit_id_field: "string", "unit_code": "string"})
        return df

    print("Extracting terrain features from Copernicus DEM GLO-30...")

    # --- DEM source ---
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic()

    # --- Terrain derivatives ---
    slope = ee.Terrain.slope(dem).rename("slope")

    aspect = ee.Terrain.aspect(dem)  # degrees, 0=N clockwise
    aspect_rad = aspect.multiply(math.pi / 180.0)
    aspect_sin = aspect_rad.sin().rename("aspect_sin")
    aspect_cos = aspect_rad.cos().rename("aspect_cos")

    # TRI: standard deviation of elevation in a 3x3 window (good proxy)
    tri = dem.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(1),  # 3x3 kernel (radius=1 pixel)
    ).rename("tri")

    # --- Stack all terrain bands ---
    terrain_stack = (
        dem.rename("elevation")
        .addBands(slope)
        .addBands(aspect_sin)
        .addBands(aspect_cos)
        .addBands(tri)
    )

    # --- Reduce to regions ---
    terrain_scale = int(cfg.get("features", {}).get("terrain_scale", 30))

    # Combined mean+stdDev reducer
    reducer = ee.Reducer.mean().combine(
        reducer2=ee.Reducer.stdDev(),
        sharedInputs=True,
    )

    reduced = terrain_stack.reduceRegions(
        collection=fc,
        reducer=reducer,
        scale=terrain_scale,
    )

    # --- Export to temp asset, download CSV ---
    # Selectors: unit metadata + terrain stats
    raw_selectors = [
        unit_id_field, "unit_level", "pref_name", "unit_code",
        # Mean columns
        "elevation_mean", "slope_mean", "aspect_sin_mean", "aspect_cos_mean", "tri_mean",
        # StdDev columns
        "elevation_stdDev", "slope_stdDev", "aspect_sin_stdDev", "aspect_cos_stdDev", "tri_stdDev",
    ]

    description = f"terrain_{unit_level}"
    temp_asset_id = f"{asset_folder}/temp_{description}"

    def _do_export() -> None:
        _export_fc_to_asset(reduced, asset_id=temp_asset_id, description=f"temp_{description}")

    _retry_with_backoff(_do_export, max_retries=max_retries)

    os.makedirs(cache_dir, exist_ok=True)
    _download_table_asset_csv(
        cfg={},
        asset_id=temp_asset_id,
        out_csv_path=cache_csv,
        selectors=raw_selectors,
        max_retries=max_retries,
    )

    # Clean up temp asset
    try:
        ee.data.deleteAsset(temp_asset_id)
    except Exception:
        pass

    # --- Post-process: rename stdDev -> std, select final columns ---
    df = pd.read_csv(cache_csv, dtype={unit_id_field: "string", "unit_code": "string"})

    rename_map = {
        "elevation_stdDev": "elevation_std",
        "slope_stdDev": "slope_std",
        "aspect_sin_stdDev": "aspect_sin_std",
        "aspect_cos_stdDev": "aspect_cos_std",
        "tri_stdDev": "tri_std",
    }
    df = df.rename(columns=rename_map)

    # Keep only the 7 planned features + identifiers
    keep_cols = [unit_id_field, "unit_level", "pref_name", "unit_code"] + TERRAIN_FEATURES
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Overwrite cache with cleaned version
    df.to_csv(cache_csv, index=False)

    n_units = len(df)
    print(f"  [terrain] Extracted {len(TERRAIN_FEATURES)} features for {n_units} units")
    for feat in TERRAIN_FEATURES:
        if feat in df.columns:
            vals = df[feat].dropna()
            if len(vals) > 0:
                print(f"    {feat}: min={vals.min():.2f}, mean={vals.mean():.2f}, max={vals.max():.2f}")

    return df


if __name__ == "__main__":
    import yaml

    config_path = "config/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = run_terrain_extraction(cfg)
    print(f"\nResult: {len(df)} rows x {len(df.columns)} columns")
    print(df.head())
