"""
Upload AOI geometries to Google Earth Engine as assets.

Creates FeatureCollection assets from local AOI GeoPackages:
- projects/ee-brodnow77/assets/aoi_full
- projects/ee-brodnow77/assets/aoi_golden

Usage:
    python admin_demographics/upload_aoi_to_gee.py [config_path]

Prerequisites:
    - Run build_aoi.py first to create local AOI files
    - Authenticate with GEE: earthengine authenticate
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import ee
import geopandas as gpd
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gdf_to_ee_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    """Convert GeoDataFrame to EE FeatureCollection."""
    # Ensure WGS84
    gdf = gdf.to_crs("EPSG:4326")

    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry.__geo_interface__
        props = {k: v for k, v in row.items() if k != "geometry"}
        features.append(ee.Feature(ee.Geometry(geom), props))

    return ee.FeatureCollection(features)


def wait_for_task(task: ee.batch.Task, poll_interval: int = 30, timeout: int = 600) -> bool:
    """Wait for GEE task to complete."""
    start = time.time()
    while True:
        status = task.status()
        state = status.get("state", "UNKNOWN")

        if state == "COMPLETED":
            print(f"  Task completed: {task.id}")
            return True
        elif state in ("FAILED", "CANCELLED"):
            error = status.get("error_message", "Unknown error")
            print(f"  Task failed: {error}")
            return False

        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"  Task timeout after {timeout}s")
            return False

        print(f"  Task state: {state}, elapsed: {int(elapsed)}s")
        time.sleep(poll_interval)


def upload_aoi_to_gee(
    local_path: str,
    asset_id: str,
    description: str,
    overwrite: bool = True,
) -> bool:
    """
    Upload local AOI GeoPackage to GEE as a FeatureCollection asset.

    Args:
        local_path: Path to local GeoPackage file
        asset_id: Target GEE asset ID (e.g., projects/ee-brodnow77/assets/aoi_full)
        description: Human-readable description for the task
        overwrite: If True, delete existing asset before upload

    Returns:
        True if upload succeeded, False otherwise
    """
    if not Path(local_path).exists():
        print(f"Error: Local file not found: {local_path}")
        return False

    print(f"Uploading {local_path} to {asset_id}")

    # Check if asset exists
    try:
        existing = ee.data.getAsset(asset_id)
        if existing and overwrite:
            print(f"  Deleting existing asset: {asset_id}")
            ee.data.deleteAsset(asset_id)
        elif existing:
            print(f"  Asset already exists (skipping): {asset_id}")
            return True
    except ee.EEException:
        pass  # Asset doesn't exist

    # Load and convert
    gdf = gpd.read_file(local_path)
    fc = gdf_to_ee_fc(gdf)

    # Export to asset
    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description=description,
        assetId=asset_id,
    )
    task.start()
    print(f"  Started task: {task.id}")

    return wait_for_task(task)


def main(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)

    # Initialize GEE
    cloud_project = cfg["gee"]["cloud_project"]
    ee.Initialize(project=cloud_project)
    print(f"Initialized GEE with project: {cloud_project}")

    # AOI paths from config
    aoi_cfg = cfg.get("aoi", {})
    aoi_full_path = aoi_cfg.get("aoi_full_path", "admin_demographics/aoi/aoi_full.gpkg")
    aoi_golden_path = aoi_cfg.get("aoi_golden_path", "admin_demographics/aoi/aoi_golden.gpkg")
    aoi_full_asset = aoi_cfg.get("aoi_full_asset_id", "projects/ee-brodnow77/assets/aoi_full")
    aoi_golden_asset = aoi_cfg.get("aoi_golden_asset_id", "projects/ee-brodnow77/assets/aoi_golden")

    # Upload AOI_FULL
    print("\n=== Uploading AOI_FULL ===")
    success_full = upload_aoi_to_gee(
        local_path=aoi_full_path,
        asset_id=aoi_full_asset,
        description="aoi_full_upload",
    )

    # Upload AOI_GOLDEN
    print("\n=== Uploading AOI_GOLDEN ===")
    success_golden = upload_aoi_to_gee(
        local_path=aoi_golden_path,
        asset_id=aoi_golden_asset,
        description="aoi_golden_upload",
    )

    # Summary
    print("\n=== Summary ===")
    print(f"AOI_FULL:   {'SUCCESS' if success_full else 'FAILED'} -> {aoi_full_asset}")
    print(f"AOI_GOLDEN: {'SUCCESS' if success_golden else 'FAILED'} -> {aoi_golden_asset}")

    if success_full and success_golden:
        print("\nAll uploads completed successfully.")
    else:
        print("\nSome uploads failed. Check logs above.")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    main(config_path)
