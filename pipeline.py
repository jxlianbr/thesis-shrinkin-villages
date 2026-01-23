from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


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

    manifest["steps"].append({"step": "data_acquisition_hooks", "status": "ok", "ts_utc": _utc_now()})

    # 2) Optical preprocessing + monthly composites (GEE)
    gee_enabled = bool(cfg["gee"]["enabled"])
    features_df: Optional[pd.DataFrame] = None

    if gee_enabled:
        from data_preprocessing.gee_monthly import run_gee_monthly_feature_export

        features_df = run_gee_monthly_feature_export(cfg)
        manifest["steps"].append({"step": "gee_optical_preprocessing_monthly_composites", "status": "ok", "ts_utc": _utc_now()})
    else:
        raise RuntimeError("This Task-1 backbone expects GEE enabled for scalable preprocessing.")

    # 3) Optional VIIRS aggregation is handled inside the same GEE export function when enabled in config
    manifest["steps"].append({"step": "optional_viirs_aggregation", "status": "ok", "ts_utc": _utc_now()})

    # 4) Feature computation (NDVI/NDBI + optional GLCM)
    # In this backbone, NDVI/NDBI are computed in GEE (scalable). Optional local GLCM can be added later.
    manifest["steps"].append({"step": "feature_computation", "status": "ok", "ts_utc": _utc_now()})

    # 5) Aggregation to village/sub-municipal units
    # In this backbone, aggregation is done in GEE via reduceRegions; table is already at unit level.
    manifest["steps"].append({"step": "aggregation_to_units", "status": "ok", "ts_utc": _utc_now()})

    # 6) Demographic join (local, deterministic join by unit_id)
    demo = pd.read_csv(demographics_path)
    unit_id_left = cfg["data"]["unit_id_field"]
    unit_id_right = cfg["data"]["demographics_unit_id_field"]

    if features_df is None:
        raise RuntimeError("features_df not produced.")

    merged = features_df.merge(demo, left_on=unit_id_left, right_on=unit_id_right, how="left")

    manifest["steps"].append({"step": "demographic_join", "status": "ok", "ts_utc": _utc_now()})

    # 7) Export final features table
    out_csv = cfg["outputs"]["features_table_csv"]
    out_parquet = cfg["outputs"]["features_table_parquet"]
    out_manifest = cfg["outputs"]["run_manifest_json"]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    merged.to_parquet(out_parquet, index=False)

    manifest["finished_utc"] = _utc_now()
    manifest["row_count"] = int(len(merged))
    write_json(out_manifest, manifest)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_parquet}")
    print(f"Wrote: {out_manifest}")


if __name__ == "__main__":
    main()
