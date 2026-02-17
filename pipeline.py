from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(paths: list[str]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)

    out_dir = cfg["project"]["outputs_dir"]
    ensure_dirs([
        out_dir,
        f"{out_dir}/logs",
        f"{out_dir}/intermediate",
        f"{out_dir}/final",
    ])

    manifest: Dict[str, Any] = {
        "project": cfg["project"],
        "time": cfg["time"],
        "features": cfg["features"],
        "started_utc": _utc_now(),
        "steps": [],
    }

    # 1) Data acquisition hooks (validate required local inputs exist)
    boundaries_path = cfg["data"]["boundaries_path"]
    demographics_path = cfg["data"]["demographics_path"]

    if not Path(boundaries_path).exists():
        raise FileNotFoundError(f"Missing boundaries file: {boundaries_path}")
    if not Path(demographics_path).exists():
        raise FileNotFoundError(f"Missing demographics file: {demographics_path}")

    # Validate AOI files if AOI config is present
    aoi_cfg = cfg.get("aoi", {})
    aoi_mode = aoi_cfg.get("mode", "full")
    aoi_path = None

    if aoi_cfg:
        aoi_path_key = f"aoi_{aoi_mode}_path"
        aoi_path = aoi_cfg.get(aoi_path_key)
        if aoi_path and not Path(aoi_path).exists():
            print(f"Warning: AOI file not found: {aoi_path}. Run build_aoi.py first.")
            print("Continuing without AOI validation (will use fallback in GEE processing)")
        elif aoi_path:
            print(f"Using AOI ({aoi_mode}): {aoi_path}")

    manifest["aoi"] = {
        "mode": aoi_mode,
        "path": aoi_path,
    }

    manifest["steps"].append({"step": "data_acquisition_hooks", "status": "ok", "ts_utc": _utc_now()})

    # 2) Optical preprocessing + monthly composites (GEE)
    gee_enabled = bool(cfg["gee"]["enabled"])
    features_df: Optional[pd.DataFrame] = None

    if gee_enabled:
        from data_preprocessing.gee_monthly import run_gee_monthly_feature_export

        features_df = run_gee_monthly_feature_export(cfg)
    else:
        raise RuntimeError("This Task-1 backbone expects GEE enabled for scalable preprocessing.")

    # 3) Optional VIIRS aggregation is handled inside the same GEE export function when enabled in config
    manifest["steps"].append({"step": "optional_viirs_aggregation", "status": "ok", "ts_utc": _utc_now()})

    # 4) Feature computation (NDVI/NDBI + optional local GLCM)
    # NDVI/NDBI are computed in GEE. GLCM is computed locally for performance.
    compute_glcm_local = bool(cfg.get("features", {}).get("compute_glcm_local", False))

    if compute_glcm_local and features_df is not None:
        print("Computing GLCM texture features locally...")
        from data_preprocessing.compute_glcm_local import run_local_glcm

        features_df = run_local_glcm(
            cfg=cfg,
            features_df=features_df,
            boundaries_path=boundaries_path,
            rasters_dir=f"{out_dir}/rasters",
        )
        manifest["steps"].append({
            "step": "local_glcm_computation",
            "status": "ok",
            "ts_utc": _utc_now(),
            "metrics": cfg.get("features", {}).get("glcm_metrics", []),
        })

    manifest["steps"].append({"step": "feature_computation", "status": "ok", "ts_utc": _utc_now()})

    # 4a) Export map rasters for AOI_GOLDEN (optional)
    export_map_rasters = cfg.get("outputs", {}).get("export_map_rasters", False)
    if export_map_rasters:
        print("Exporting map rasters for visualization...")
        from data_preprocessing.gee_monthly import export_map_rasters as do_export_map_rasters

        map_rasters_dir = cfg.get("outputs", {}).get("map_rasters_dir", f"{out_dir}/rasters/map_figures")
        months_override = cfg.get("run_mode", {}).get("months_override")

        do_export_map_rasters(
            cfg=cfg,
            output_dir=map_rasters_dir,
            months=months_override,
        )
        manifest["steps"].append({
            "step": "map_raster_export",
            "status": "ok",
            "ts_utc": _utc_now(),
            "output_dir": map_rasters_dir,
        })

    # 4b) OSM label features (building footprints)
    include_osm_labels = cfg.get("labels", {}).get("source") == "osm"
    if include_osm_labels and features_df is not None:
        print("Computing OSM building features...")
        import geopandas as gpd
        from data_preprocessing.osm_labels import run_osm_labels

        units_gdf = gpd.read_file(boundaries_path)
        osm_df = run_osm_labels(cfg, units_gdf, f"{out_dir}/osm")

        unit_id_field = cfg["data"]["unit_id_field"]
        features_df = features_df.merge(
            osm_df[[unit_id_field, "osm_built_area", "osm_building_count", "osm_built_ratio"]],
            on=unit_id_field,
            how="left"
        )
        manifest["steps"].append({
            "step": "osm_label_computation",
            "status": "ok",
            "ts_utc": _utc_now(),
            "source": "OpenStreetMap building footprints",
        })

    # 4c) Terrain features (DEM/slope/aspect/TRI -- static per unit)
    compute_terrain = bool(cfg.get("features", {}).get("compute_terrain", False))
    if compute_terrain and features_df is not None:
        print("Extracting terrain features from Copernicus DEM...")
        from data_preprocessing.gee_terrain import run_terrain_extraction, TERRAIN_FEATURES

        terrain_df = run_terrain_extraction(cfg)

        unit_id_field = cfg["data"]["unit_id_field"]
        terrain_cols = [c for c in TERRAIN_FEATURES if c in terrain_df.columns]
        merge_cols = [unit_id_field] + terrain_cols

        features_df = features_df.merge(
            terrain_df[merge_cols],
            on=unit_id_field,
            how="left",
        )
        manifest["steps"].append({
            "step": "terrain_feature_extraction",
            "status": "ok",
            "ts_utc": _utc_now(),
            "source": "COPERNICUS/DEM/GLO30",
            "features": terrain_cols,
        })

    # 4d) LULC class fractions (Dynamic World -- static composite per unit)
    compute_lulc = bool(cfg.get("features", {}).get("compute_lulc", False))
    if compute_lulc and features_df is not None:
        print("Extracting LULC class fractions from Dynamic World...")
        from data_preprocessing.gee_lulc import run_lulc_extraction, LULC_FEATURES

        lulc_df = run_lulc_extraction(cfg)

        unit_id_field = cfg["data"]["unit_id_field"]
        lulc_cols = [c for c in LULC_FEATURES if c in lulc_df.columns]
        merge_cols = [unit_id_field] + lulc_cols

        features_df = features_df.merge(
            lulc_df[merge_cols],
            on=unit_id_field,
            how="left",
        )
        manifest["steps"].append({
            "step": "lulc_extraction",
            "status": "ok",
            "ts_utc": _utc_now(),
            "source": "GOOGLE/DYNAMICWORLD/V1",
            "features": lulc_cols,
        })

    # 5) Aggregation to village/sub-municipal units
    # In this backbone, aggregation is done in GEE via reduceRegions; table is already at unit level.
    manifest["steps"].append({"step": "aggregation_to_units", "status": "ok", "ts_utc": _utc_now()})

    # 6) Demographic join (local, deterministic join by unit_id)
    demo = pd.read_csv(demographics_path, dtype={"unit_code": "string"})
    unit_id_right = cfg["data"]["demographics_unit_id_field"]

    if features_df is None:
        raise RuntimeError("features_df not produced.")

    # Normalize join keys (defensive)
    demo[unit_id_right] = demo[unit_id_right].astype(str).str.strip()
    unit_id_left = cfg["data"]["unit_id_field"]
    features_df[unit_id_left] = features_df[unit_id_left].astype(str).str.strip()

    unit_level = cfg.get("run_mode", {}).get("unit_level", "mura")

    if unit_level == "mura":
        # direct join: mura:<PrefName>:<5-digit municipality code>
        unit_id_left = cfg["data"]["unit_id_field"]
        merged = features_df.merge(
            demo,
            left_on=unit_id_left,
            right_on=unit_id_right,
            how="left",
            validate="m:1",
        )

    elif unit_level == "aza":
        # demographics are municipality-level -> map aza -> municipality using first 5 digits of its code
        # aza unit_id example: aza:Aomori:022010010  -> muni_code = 02201 -> mura_unit_id = mura:Aomori:02201
        parts = features_df["unit_id"].str.split(":", n=2, expand=True)
        pref = parts[1].astype(str).str.strip()
        code = parts[2].astype(str).str.replace(r"\D", "", regex=True)
        muni_code = code.str[:5].str.zfill(5)

        features_df["_demo_join_id"] = "mura:" + pref + ":" + muni_code

        merged = features_df.merge(
            demo,
            left_on="_demo_join_id",
            right_on=unit_id_right,
            how="left",
            validate="m:1",
            suffixes=("", "_demo"),
        )

    else:
        raise ValueError("run_mode.unit_level must be 'mura' or 'aza'")

    # Hard fail if the join is effectively empty
    demo_cols = [c for c in merged.columns if c.startswith(("pop_", "age_", "households_"))]
    if demo_cols and merged[demo_cols].isna().mean().max() > 0.95:
        raise RuntimeError("Demographics join failed: >95% of demographic columns are NaN.")

    features_df = merged

    manifest["steps"].append({
        "step": "gee_optical_preprocessing_monthly_composites",
        "status": "ok",
        "ts_utc": _utc_now(),
        "sources": {
            "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
            "viirs_optional": bool(cfg.get("features", {}).get("include_viirs", False)),
        },
    })

    # drop join helper column if present
    if "_demo_join_id" in features_df.columns:
        features_df = features_df.drop(columns=["_demo_join_id"])

    # collapse duplicated pref_name columns produced by merge
    if "pref_name_x" in features_df.columns:
        features_df = features_df.rename(columns={"pref_name_x": "pref_name"})
    if "pref_name_y" in features_df.columns:
        features_df = features_df.drop(columns=["pref_name_y"])
    # final safety: enforce unique unit-month
    if "month" in features_df.columns:
        features_df = features_df.drop_duplicates(subset=["unit_id", "month"], keep="first")


    # 7) Export final features table
    out_csv = cfg["outputs"]["features_table_csv"]
    out_parquet = cfg["outputs"]["features_table_parquet"]
    out_manifest = cfg["outputs"]["run_manifest_json"]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    #Parquet stability: enforce str types for unit_id and pref_name
    features_df[unit_id_left] = features_df[unit_id_left].astype("string").str.strip()
    features_df["unit_code"] = features_df["unit_code"].astype("string").str.strip()
    if unit_level == "mura":
        features_df["unit_code"] = features_df["unit_code"].str.zfill(5)

    features_df.to_csv(out_csv, index=False)
    features_df.to_parquet(out_parquet, index=False)

    manifest["finished_utc"] = _utc_now()
    manifest["row_count"] = int(len(features_df))
    write_json(out_manifest, manifest)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_parquet}")
    print(f"Wrote: {out_manifest}")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    main(config_path)
