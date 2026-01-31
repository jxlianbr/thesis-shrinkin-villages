"""
Build AOI geometries for the shrinking villages pipeline.

Creates two AOI geometries from processed boundaries:
- AOI_FULL: Dissolved union of all Aomori + Akita boundaries
- AOI_GOLDEN: Union of selected subset (~10 units) for validation

Usage:
    python admin_demographics/build_aoi.py [config_path]

Outputs:
    - admin_demographics/aoi/aoi_full.gpkg
    - admin_demographics/aoi/aoi_golden.gpkg
    - admin_demographics/aoi/aoi_provenance.json
"""
from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import yaml
from shapely.validation import make_valid


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_aoi_full(
    boundaries_path: str,
    prefectures: List[str],
) -> gpd.GeoDataFrame:
    """
    Build AOI_FULL by dissolving all boundaries for target prefectures.

    Args:
        boundaries_path: Path to processed boundaries GeoPackage
        prefectures: List of prefecture names to include (e.g., ["Aomori", "Akita"])

    Returns:
        GeoDataFrame with single dissolved geometry
    """
    gdf = gpd.read_file(boundaries_path)

    # Filter to target prefectures
    gdf = gdf[gdf["pref_name"].isin(prefectures)].copy()

    if len(gdf) == 0:
        raise ValueError(f"No boundaries found for prefectures: {prefectures}")

    # Dissolve all polygons into single geometry
    dissolved = gdf.dissolve()

    # Fix any invalid geometries
    dissolved["geometry"] = dissolved["geometry"].apply(
        lambda g: make_valid(g) if g is not None else g
    )

    # Add metadata columns
    dissolved["aoi_type"] = "full"
    dissolved["prefectures"] = ",".join(prefectures)
    dissolved["source_unit_count"] = len(gdf)

    return dissolved[["aoi_type", "prefectures", "source_unit_count", "geometry"]]


def build_aoi_golden(
    boundaries_path: str,
    golden_unit_ids: List[str],
) -> gpd.GeoDataFrame:
    """
    Build AOI_GOLDEN by unioning selected boundary units.

    Args:
        boundaries_path: Path to processed boundaries GeoPackage
        golden_unit_ids: List of unit_id values to include

    Returns:
        GeoDataFrame with single dissolved geometry
    """
    gdf = gpd.read_file(boundaries_path)

    # Filter to golden units
    golden = gdf[gdf["unit_id"].isin(golden_unit_ids)].copy()

    if len(golden) == 0:
        raise ValueError(f"No boundaries found for unit_ids: {golden_unit_ids}")

    missing = set(golden_unit_ids) - set(golden["unit_id"].values)
    if missing:
        print(f"Warning: Missing units from boundaries: {missing}")

    # Dissolve selected polygons into single geometry
    dissolved = golden.dissolve()

    # Fix any invalid geometries
    dissolved["geometry"] = dissolved["geometry"].apply(
        lambda g: make_valid(g) if g is not None else g
    )

    # Add metadata columns
    dissolved["aoi_type"] = "golden"
    dissolved["unit_ids"] = ",".join(golden_unit_ids)
    dissolved["source_unit_count"] = len(golden)

    return dissolved[["aoi_type", "unit_ids", "source_unit_count", "geometry"]]


def write_provenance(
    output_path: str,
    aoi_full_path: str,
    aoi_golden_path: str,
    boundaries_path: str,
    prefectures: List[str],
    golden_unit_ids: List[str],
    full_unit_count: int,
    golden_unit_count: int,
) -> None:
    """Write provenance metadata JSON."""
    provenance = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_boundaries": str(boundaries_path),
        "aoi_full": {
            "path": str(aoi_full_path),
            "prefectures": prefectures,
            "unit_count": full_unit_count,
        },
        "aoi_golden": {
            "path": str(aoi_golden_path),
            "unit_ids": golden_unit_ids,
            "unit_count": golden_unit_count,
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)


def main(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)

    # Paths from config
    boundaries_path = cfg["data"]["boundaries_path"]
    prefectures = cfg.get("study_area", {}).get("prefectures", ["Aomori", "Akita"])

    # AOI config section
    aoi_cfg = cfg.get("aoi", {})
    aoi_dir = Path("admin_demographics/aoi")
    aoi_dir.mkdir(parents=True, exist_ok=True)

    aoi_full_path = aoi_cfg.get("aoi_full_path", str(aoi_dir / "aoi_full.gpkg"))
    aoi_golden_path = aoi_cfg.get("aoi_golden_path", str(aoi_dir / "aoi_golden.gpkg"))

    golden_unit_ids = aoi_cfg.get("golden_unit_ids", [
        # Aomori (5 units)
        "mura:Aomori:02423",
        "mura:Aomori:02387",
        "mura:Aomori:02201",
        "mura:Aomori:02405",
        "mura:Aomori:02443",
        # Akita (5 units)
        "mura:Akita:05303",
        "mura:Akita:05202",
        "mura:Akita:05363",
        "mura:Akita:05215",
        "mura:Akita:05207",
    ])

    print(f"Building AOIs from: {boundaries_path}")
    print(f"Target prefectures: {prefectures}")
    print(f"Golden unit IDs: {len(golden_unit_ids)} units")

    # Build AOI_FULL
    print("\nBuilding AOI_FULL...")
    aoi_full = build_aoi_full(boundaries_path, prefectures)
    aoi_full.to_file(aoi_full_path, layer="aoi_full", driver="GPKG")
    print(f"  Wrote: {aoi_full_path}")
    print(f"  Source units: {aoi_full['source_unit_count'].iloc[0]}")

    # Build AOI_GOLDEN
    print("\nBuilding AOI_GOLDEN...")
    aoi_golden = build_aoi_golden(boundaries_path, golden_unit_ids)
    aoi_golden.to_file(aoi_golden_path, layer="aoi_golden", driver="GPKG")
    print(f"  Wrote: {aoi_golden_path}")
    print(f"  Source units: {aoi_golden['source_unit_count'].iloc[0]}")

    # Write provenance
    provenance_path = str(aoi_dir / "aoi_provenance.json")
    write_provenance(
        output_path=provenance_path,
        aoi_full_path=aoi_full_path,
        aoi_golden_path=aoi_golden_path,
        boundaries_path=boundaries_path,
        prefectures=prefectures,
        golden_unit_ids=golden_unit_ids,
        full_unit_count=int(aoi_full["source_unit_count"].iloc[0]),
        golden_unit_count=int(aoi_golden["source_unit_count"].iloc[0]),
    )
    print(f"\nWrote provenance: {provenance_path}")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    main(config_path)
