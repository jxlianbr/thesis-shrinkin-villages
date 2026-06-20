"""
OSM building footprint processing for label generation.

This module downloads OpenStreetMap building footprints and computes
built-up area features for each administrative unit.

Features computed:
- osm_built_area: Total building footprint area (m²) per unit
- osm_building_count: Number of buildings per unit
- osm_built_ratio: Building area / unit area (built-up density)
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import pandas as pd


def _get_prefecture_bbox(pref_name: str) -> tuple[float, float, float, float]:
    """
    Get bounding box for a Japanese prefecture.

    Returns (min_lon, min_lat, max_lon, max_lat) in WGS84.
    """
    # Approximate bounding boxes for Tohoku prefectures
    PREFECTURE_BBOX = {
        "Aomori": (139.3, 40.2, 141.7, 41.6),
        "Akita": (139.5, 38.9, 140.9, 40.5),
        "Iwate": (140.6, 38.7, 142.1, 40.5),
        "Yamagata": (139.5, 37.7, 140.6, 39.2),
        "Miyagi": (140.2, 37.8, 141.7, 39.0),
        "Fukushima": (139.2, 36.8, 141.0, 37.9),
    }
    if pref_name not in PREFECTURE_BBOX:
        raise ValueError(f"Unknown prefecture: {pref_name}. Known: {list(PREFECTURE_BBOX.keys())}")
    return PREFECTURE_BBOX[pref_name]


def download_osm_buildings_overpass(
    prefectures: list[str],
    output_path: str,
    timeout: int = 600,
) -> gpd.GeoDataFrame:
    """
    Download OSM building footprints via Overpass API.

    Args:
        prefectures: List of prefecture names (e.g., ["Aomori", "Akita"])
        output_path: Path to save the GeoPackage
        timeout: Overpass API timeout in seconds

    Returns:
        GeoDataFrame with building polygons
    """
    all_buildings = []

    for pref in prefectures:
        print(f"Downloading OSM buildings for {pref}...")
        bbox = _get_prefecture_bbox(pref)
        min_lon, min_lat, max_lon, max_lat = bbox

        # Overpass QL query for buildings
        # Using south,west,north,east format
        query = f"""
[out:json][timeout:{timeout}];
(
  way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
);
out body;
>;
out skel qt;
"""
        url = "https://overpass-api.de/api/interpreter"
        data = query.encode("utf-8")

        # Overpass rejects the default Python-urllib agent with HTTP 406;
        # send a descriptive User-Agent (per Overpass usage policy) plus Accept.
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "shrinking-villages-thesis/1.0 (academic research)",
            "Accept": "application/json",
        }
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)

        max_retries = 3
        result = None
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=timeout + 60) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break  # success
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 30 * (2 ** attempt)  # 30s, 60s, 120s
                    print(f"  Attempt {attempt + 1}/{max_retries} failed for {pref}: {e}")
                    print(f"  Retrying in {delay}s...")
                    time.sleep(delay)
                    # Rebuild request (urlopen consumes it)
                    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
                else:
                    print(f"  Warning: Failed to download {pref} after {max_retries} attempts: {e}")

        if result is None:
            continue

        # Parse OSM JSON to GeoDataFrame
        buildings_gdf = _parse_osm_json_to_gdf(result, pref)
        if buildings_gdf is not None and len(buildings_gdf) > 0:
            all_buildings.append(buildings_gdf)
            print(f"  Downloaded {len(buildings_gdf)} buildings for {pref}")

    if not all_buildings:
        raise RuntimeError("No buildings downloaded from any prefecture")

    # Combine all prefectures
    combined = gpd.GeoDataFrame(pd.concat(all_buildings, ignore_index=True), crs="EPSG:4326")

    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_file(output_path, driver="GPKG")
    print(f"Saved {len(combined)} buildings to {output_path}")

    return combined


def _parse_osm_json_to_gdf(osm_json: dict, pref_name: str) -> gpd.GeoDataFrame | None:
    """
    Parse Overpass JSON response to GeoDataFrame.

    Converts ways to polygons using node coordinates.
    """
    from shapely.geometry import Polygon

    elements = osm_json.get("elements", [])
    if not elements:
        return None

    # Build node lookup
    nodes = {}
    for el in elements:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    # Build polygons from ways
    features = []
    for el in elements:
        if el["type"] == "way" and "nodes" in el:
            coords = []
            for node_id in el["nodes"]:
                if node_id in nodes:
                    coords.append(nodes[node_id])
            if len(coords) >= 4:  # Need at least 4 points for a closed polygon
                try:
                    poly = Polygon(coords)
                    if poly.is_valid and poly.area > 0:
                        features.append({
                            "geometry": poly,
                            "osm_id": el["id"],
                            "building": el.get("tags", {}).get("building", "yes"),
                            "pref_name": pref_name,
                        })
                except Exception:
                    pass

    if not features:
        return None

    return gpd.GeoDataFrame(features, crs="EPSG:4326")


def load_osm_buildings(path: str) -> gpd.GeoDataFrame:
    """Load OSM buildings from a GeoPackage file."""
    return gpd.read_file(path)


def compute_osm_features(
    buildings_gdf: gpd.GeoDataFrame,
    units_gdf: gpd.GeoDataFrame,
    unit_id_field: str = "unit_id",
) -> pd.DataFrame:
    """
    Compute OSM-derived features for each administrative unit.

    Args:
        buildings_gdf: GeoDataFrame with building polygons
        units_gdf: GeoDataFrame with unit boundaries
        unit_id_field: Name of the unit ID column

    Returns:
        DataFrame with columns:
        - unit_id
        - osm_built_area: Total building footprint area (m²)
        - osm_building_count: Number of buildings
        - osm_built_ratio: Building area / unit area
    """
    print(f"Computing OSM features for {len(units_gdf)} units...")

    # Ensure both are in the same CRS (use a projected CRS for area calculations)
    # JGD2011 / Japan Plane Rectangular CS VII (EPSG:6675) covers Tohoku
    target_crs = "EPSG:6675"

    buildings_proj = buildings_gdf.to_crs(target_crs)
    units_proj = units_gdf.to_crs(target_crs)

    # Compute unit areas
    units_proj["unit_area_m2"] = units_proj.geometry.area

    # Spatial join: find which buildings intersect each unit
    joined = gpd.sjoin(buildings_proj, units_proj[[unit_id_field, "geometry", "unit_area_m2"]], how="inner", predicate="intersects")

    # Compute building areas (intersection area, not full building if partially inside)
    results = []
    for unit_id, group in joined.groupby(unit_id_field):
        # Get the unit geometry for clipping
        unit_geom = units_proj[units_proj[unit_id_field] == unit_id].geometry.iloc[0]
        unit_area = units_proj[units_proj[unit_id_field] == unit_id]["unit_area_m2"].iloc[0]

        # Clip buildings to unit boundary and sum areas
        clipped_areas = []
        for _, building in group.iterrows():
            try:
                clipped = building.geometry.intersection(unit_geom)
                if not clipped.is_empty:
                    clipped_areas.append(clipped.area)
            except Exception:
                pass

        total_built_area = sum(clipped_areas)
        building_count = len(group)
        built_ratio = total_built_area / unit_area if unit_area > 0 else 0

        results.append({
            unit_id_field: unit_id,
            "osm_built_area": total_built_area,
            "osm_building_count": building_count,
            "osm_built_ratio": built_ratio,
        })

    # Add units with no buildings (zero values)
    all_unit_ids = set(units_gdf[unit_id_field].tolist())
    covered_unit_ids = {r[unit_id_field] for r in results}
    for unit_id in all_unit_ids - covered_unit_ids:
        results.append({
            unit_id_field: unit_id,
            "osm_built_area": 0.0,
            "osm_building_count": 0,
            "osm_built_ratio": 0.0,
        })

    df = pd.DataFrame(results)
    print(f"Computed OSM features: {len(df)} units, {df['osm_building_count'].sum()} total buildings")
    return df


def run_osm_labels(
    cfg: Dict[str, Any],
    units_gdf: gpd.GeoDataFrame,
    output_dir: str,
) -> pd.DataFrame:
    """
    Main entry point for OSM label pipeline.

    Handles downloading (if needed) and computing features.

    Args:
        cfg: Configuration dictionary
        units_gdf: GeoDataFrame with unit boundaries
        output_dir: Directory for OSM data cache

    Returns:
        DataFrame with OSM features per unit
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check for pre-downloaded buildings
    osm_path = cfg.get("labels", {}).get("osm_buildings_path")
    if osm_path and Path(osm_path).exists():
        print(f"Loading cached OSM buildings from {osm_path}")
        buildings_gdf = load_osm_buildings(osm_path)
    else:
        # Download from Overpass API
        prefectures = cfg.get("study_area", {}).get("prefectures", ["Aomori", "Akita"])
        osm_path = f"{output_dir}/osm_buildings.gpkg"
        buildings_gdf = download_osm_buildings_overpass(prefectures, osm_path)

    # Compute features
    unit_id_field = cfg.get("data", {}).get("unit_id_field", "unit_id")
    osm_features = compute_osm_features(buildings_gdf, units_gdf, unit_id_field)

    # Save features
    osm_features_path = f"{output_dir}/osm_features.csv"
    osm_features.to_csv(osm_features_path, index=False)
    print(f"Saved OSM features to {osm_features_path}")

    return osm_features
