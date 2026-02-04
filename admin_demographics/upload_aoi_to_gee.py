"""
Upload AOI geometries and boundary assets to Google Earth Engine.

Creates FeatureCollection assets from local GeoPackages:

AOI Assets (single dissolved geometry for filterBounds):
- projects/ee-brodnow77/assets/aoi_full
- projects/ee-brodnow77/assets/aoi_golden

Boundary Assets (individual polygons with unit_id for reduceRegions):
- projects/ee-brodnow77/assets/mura_jis_shp
- projects/ee-brodnow77/assets/aza_shp

Usage:
    python admin_demographics/upload_aoi_to_gee.py [config_path]

Prerequisites:
    - Run build_aoi.py first to create local AOI files
    - Authenticate with GEE: earthengine authenticate
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import ee
import geopandas as gpd
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gdf_to_ee_fc(gdf: gpd.GeoDataFrame, simplify_tolerance: float = 0.0) -> ee.FeatureCollection:
    """Convert GeoDataFrame to EE FeatureCollection.

    Args:
        gdf: GeoDataFrame to convert
        simplify_tolerance: If > 0, simplify geometry using Douglas-Peucker algorithm.
                           Value is in CRS units (degrees for WGS84, ~0.001 = ~100m)
    """
    # Ensure WGS84
    gdf = gdf.to_crs("EPSG:4326")

    # Simplify geometry if tolerance specified (helps with GEE upload limits)
    if simplify_tolerance > 0:
        gdf = gdf.copy()
        gdf["geometry"] = gdf["geometry"].simplify(
            tolerance=simplify_tolerance,
            preserve_topology=True
        )

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
    simplify_tolerance: float = 0.0,
) -> bool:
    """
    Upload local AOI GeoPackage to GEE as a FeatureCollection asset.

    Args:
        local_path: Path to local GeoPackage file
        asset_id: Target GEE asset ID (e.g., projects/ee-brodnow77/assets/aoi_full)
        description: Human-readable description for the task
        overwrite: If True, delete existing asset before upload
        simplify_tolerance: If > 0, simplify geometry to reduce vertex count.
                           Value is in degrees (~0.001 = ~100m). Use for large AOIs
                           that exceed GEE's 10MB payload limit.

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

    if simplify_tolerance > 0:
        print(f"  Simplifying geometry with tolerance={simplify_tolerance}")

    fc = gdf_to_ee_fc(gdf, simplify_tolerance=simplify_tolerance)

    # Export to asset
    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description=description,
        assetId=asset_id,
    )
    task.start()
    print(f"  Started task: {task.id}")

    return wait_for_task(task)


def upload_boundaries_to_gee(
    local_path: str,
    asset_id: str,
    description: str,
    prefectures: Optional[List[str]] = None,
    overwrite: bool = True,
    simplify_tolerance: float = 0.0,
    timeout: int = 900,
    chunk_size: int = 1000,
) -> bool:
    """
    Upload boundary GeoPackage to GEE preserving unit attributes.

    This uploads individual polygons (not dissolved) so they can be used
    with reduceRegions() for zonal statistics. For large datasets, it uploads
    in chunks and merges them in GEE.

    Args:
        local_path: Path to local GeoPackage file
        asset_id: Target GEE asset ID
        description: Human-readable description for the task
        prefectures: List of prefectures to filter (e.g., ["Aomori", "Akita"])
        overwrite: If True, delete existing asset before upload
        simplify_tolerance: If > 0, simplify geometries to reduce vertex count.
                           Value is in degrees (~0.0001 = ~10m).
        timeout: Task timeout in seconds (default 900s for large uploads)
        chunk_size: Max features per chunk (default 1000)

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

    # Load boundaries
    gdf = gpd.read_file(local_path)
    original_count = len(gdf)

    # Filter by prefectures if specified
    if prefectures and "pref_name" in gdf.columns:
        gdf = gdf[gdf["pref_name"].isin(prefectures)]
        print(f"  Filtered to {len(gdf)} features (from {original_count}) for prefectures: {prefectures}")

    if len(gdf) == 0:
        print(f"  Error: No features after filtering")
        return False

    if simplify_tolerance > 0:
        print(f"  Simplifying geometry with tolerance={simplify_tolerance}")
        gdf = gdf.copy()
        gdf = gdf.to_crs("EPSG:4326")
        gdf["geometry"] = gdf["geometry"].simplify(
            tolerance=simplify_tolerance,
            preserve_topology=True
        )

    # Check if we need chunked upload
    if len(gdf) > chunk_size:
        return _upload_boundaries_chunked(
            gdf=gdf,
            asset_id=asset_id,
            description=description,
            timeout=timeout,
            chunk_size=chunk_size,
        )

    # Direct upload for small datasets
    fc = gdf_to_ee_fc(gdf, simplify_tolerance=0.0)  # Already simplified above

    # Export to asset
    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description=description,
        assetId=asset_id,
    )
    task.start()
    print(f"  Started task: {task.id}")
    print(f"  Uploading {len(gdf)} features...")

    return wait_for_task(task, timeout=timeout)


def _upload_boundaries_chunked(
    gdf: gpd.GeoDataFrame,
    asset_id: str,
    description: str,
    timeout: int,
    chunk_size: int,
) -> bool:
    """
    Upload large FeatureCollection in chunks, then merge in GEE.

    Strategy:
    1. Split GeoDataFrame into chunks of chunk_size features
    2. Upload each chunk as a temporary asset
    3. Merge all chunks in GEE into final asset
    4. Delete temporary assets
    """
    import math

    n_chunks = math.ceil(len(gdf) / chunk_size)
    print(f"  Large dataset ({len(gdf)} features): uploading in {n_chunks} chunks of {chunk_size}")

    # Generate chunk asset IDs
    base_asset_id = asset_id.rsplit("/", 1)
    asset_folder = base_asset_id[0] if len(base_asset_id) > 1 else "projects/ee-brodnow77/assets"
    asset_name = base_asset_id[1] if len(base_asset_id) > 1 else asset_id.split("/")[-1]

    chunk_asset_ids = []
    tasks = []

    # Upload each chunk
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(gdf))
        chunk_gdf = gdf.iloc[start_idx:end_idx]

        chunk_asset_id = f"{asset_folder}/{asset_name}_chunk_{i}"
        chunk_asset_ids.append(chunk_asset_id)

        # Delete existing chunk asset if present
        try:
            ee.data.deleteAsset(chunk_asset_id)
        except ee.EEException:
            pass

        fc = gdf_to_ee_fc(chunk_gdf, simplify_tolerance=0.0)

        task = ee.batch.Export.table.toAsset(
            collection=fc,
            description=f"{description}_chunk_{i}",
            assetId=chunk_asset_id,
        )
        task.start()
        tasks.append((task, chunk_asset_id, i))
        print(f"    Chunk {i+1}/{n_chunks}: {len(chunk_gdf)} features -> {chunk_asset_id}")

    # Wait for all chunk uploads
    print(f"  Waiting for {n_chunks} chunk uploads...")
    all_success = True
    for task, chunk_id, i in tasks:
        if not wait_for_task(task, timeout=timeout, poll_interval=15):
            print(f"    Chunk {i} failed")
            all_success = False

    if not all_success:
        print("  Error: Some chunk uploads failed")
        return False

    # Merge chunks in GEE
    print(f"  Merging {n_chunks} chunks into final asset...")
    chunk_fcs = [ee.FeatureCollection(cid) for cid in chunk_asset_ids]
    merged_fc = ee.FeatureCollection(chunk_fcs).flatten()

    # Export merged to final asset
    merge_task = ee.batch.Export.table.toAsset(
        collection=merged_fc,
        description=f"{description}_merged",
        assetId=asset_id,
    )
    merge_task.start()
    print(f"  Started merge task: {merge_task.id}")

    if not wait_for_task(merge_task, timeout=timeout):
        print("  Error: Merge task failed")
        return False

    # Cleanup: delete chunk assets
    print(f"  Cleaning up {n_chunks} temporary chunk assets...")
    for chunk_id in chunk_asset_ids:
        try:
            ee.data.deleteAsset(chunk_id)
        except ee.EEException:
            pass

    print(f"  Successfully uploaded {len(gdf)} features")
    return True


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

    # Upload AOI_FULL (with simplification to avoid GEE 10MB payload limit)
    # Tolerance of 0.001 degrees ≈ 100m - acceptable for clipping satellite imagery
    print("\n=== Uploading AOI_FULL ===")
    success_full = upload_aoi_to_gee(
        local_path=aoi_full_path,
        asset_id=aoi_full_asset,
        description="aoi_full_upload",
        simplify_tolerance=0.001,  # ~100m simplification for large dissolved geometry
    )

    # Upload AOI_GOLDEN (no simplification needed - smaller geometry)
    print("\n=== Uploading AOI_GOLDEN ===")
    success_golden = upload_aoi_to_gee(
        local_path=aoi_golden_path,
        asset_id=aoi_golden_asset,
        description="aoi_golden_upload",
        simplify_tolerance=0.0,  # No simplification for smaller golden AOI
    )

    # --- Upload Boundary Assets (for reduceRegions) ---
    prefectures = cfg.get("study_area", {}).get("prefectures", ["Aomori", "Akita"])

    # Upload MURA boundaries (65 features - no simplification needed)
    mura_path = cfg["data"]["boundaries_path"]
    mura_asset = cfg["gee"]["boundaries_asset_id_mura"]

    print("\n=== Uploading MURA Boundaries ===")
    success_mura = upload_boundaries_to_gee(
        local_path=mura_path,
        asset_id=mura_asset,
        description="mura_boundaries_upload",
        prefectures=prefectures,
        simplify_tolerance=0.0005,  # ~50m - complex geometries still exceed 10MB limit
    )

    # Upload AZA boundaries (9K features, may need simplification)
    aza_path = "admin_demographics/boundaries/aza.gpkg"
    aza_asset = cfg["gee"]["boundaries_asset_id_aza"]

    print("\n=== Uploading AZA Boundaries ===")
    if Path(aza_path).exists():
        success_aza = upload_boundaries_to_gee(
            local_path=aza_path,
            asset_id=aza_asset,
            description="aza_boundaries_upload",
            prefectures=prefectures,
            simplify_tolerance=0.0001,  # ~10m to reduce 1.3M vertices
            timeout=1200,  # Allow more time for large upload
        )
    else:
        print(f"  Skipping AZA upload: {aza_path} not found")
        success_aza = True  # Not a failure if file doesn't exist

    # Summary
    print("\n=== Summary ===")
    print(f"AOI_FULL:        {'SUCCESS' if success_full else 'FAILED'} -> {aoi_full_asset}")
    print(f"AOI_GOLDEN:      {'SUCCESS' if success_golden else 'FAILED'} -> {aoi_golden_asset}")
    print(f"MURA Boundaries: {'SUCCESS' if success_mura else 'FAILED'} -> {mura_asset}")
    print(f"AZA Boundaries:  {'SUCCESS' if success_aza else 'FAILED'} -> {aza_asset}")

    all_success = success_full and success_golden and success_mura and success_aza
    if all_success:
        print("\nAll uploads completed successfully.")
    else:
        print("\nSome uploads failed. Check logs above.")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    main(config_path)
