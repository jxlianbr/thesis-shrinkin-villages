from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
import os
import time
import urllib.request

import ee
import pandas as pd


def _wait_for_task(task: ee.batch.Task, poll_s: int = 15) -> None:
    while True:
        status = task.status()
        state = status.get("state")
        if state == "COMPLETED":
            return
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"EE task {state}: {status}")
        time.sleep(poll_s)


def _export_fc_to_asset(fc: ee.FeatureCollection, asset_id: str, description: str) -> None:
    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description=description,
        assetId=asset_id,
    )
    task.start()
    _wait_for_task(task)


def _download_table_asset_csv(asset_id: str, out_csv_path: str) -> None:
    download_id = ee.data.getTableDownloadId(
        {
            "table": asset_id,
            "format": "CSV",
            "filename": os.path.basename(out_csv_path),
        }
    )
    url = ee.data.makeTableDownloadUrl(download_id)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    urllib.request.urlretrieve(url, out_csv_path)


def _ym_list(start: str, end: str) -> List[str]:
    # start/end format: YYYY-MM-DD
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    out: List[str] = []
    y, m = s.year, s.month
    while (y < e.year) or (y == e.year and m <= e.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out


def run_gee_monthly_feature_export(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Scalable preprocessing + monthly composites + zonal aggregation in Google Earth Engine.
    Exports each month as an EE Asset table and downloads CSVs locally, then concatenates to a DataFrame.
    """

    # --- Authenticate/initialize ---
    cloud_project = (cfg.get("gee", {}).get("cloud_project") or "").strip()
    if cloud_project and cloud_project != "YOUR_GCP_PROJECT_ID" and not cloud_project.startswith("YOUR_"):
        ee.Initialize(project=cloud_project)
    else:
        ee.Initialize()

    # --- Export behavior (NO GCS) ---
    export_target = (cfg.get("gee", {}).get("export_target") or "asset").strip().lower()
    asset_folder = (cfg.get("gee", {}).get("export_asset_folder") or "").rstrip("/")
    keep_assets = bool(cfg.get("gee", {}).get("keep_export_assets", True))

    if export_target not in {"asset", "drive"}:
        raise ValueError("cfg.gee.export_target must be 'asset' or 'drive' (GCS disabled).")
    if export_target == "asset" and not asset_folder:
        raise ValueError("cfg.gee.export_asset_folder must be set when export_target='asset'.")

    # --- Load boundaries ---
    unit_level = cfg.get("run_mode", {}).get("unit_level", "mura")

    if unit_level == "mura":
        boundaries_asset_id = cfg["gee"]["boundaries_asset_id_mura"]
    elif unit_level == "aza":
        boundaries_asset_id = cfg["gee"]["boundaries_asset_id_aza"]
    else:
        raise ValueError("run_mode.unit_level must be 'mura' or 'aza'")

    fc = ee.FeatureCollection(boundaries_asset_id)
    unit_id_field = cfg["data"]["unit_id_field"]

    # --- Collections ---
    start = cfg["time"]["start"]
    end = cfg["time"]["end"]

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(fc)
    )

    # Basic S2 mask via SCL (keeps non-cloud, non-shadow)
    def mask_s2(img):
        scl = img.select("SCL")
        keep = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
        return img.updateMask(keep)

    s2 = s2.map(mask_s2)

    # NDVI/NDBI on S2 bands (B8 NIR, B4 red, B11 SWIR)
    def add_indices(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndbi = img.normalizedDifference(["B11", "B8"]).rename("NDBI")
        return img.addBands([ndvi, ndbi])

    if cfg["features"]["compute_ndvi"] or cfg["features"]["compute_ndbi"]:
        s2 = s2.map(add_indices)

    include_viirs = bool(cfg["features"]["include_viirs"])
    viirs = None
    if include_viirs:
        viirs = (
            ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
            .filterDate(start, end)
            .filterBounds(fc)
            .select(["avg_rad"], ["VIIRS_avg_rad"])
        )

    # --- Monthly composite + zonal stats ---
    scale_m = int(cfg["gee"]["scale_m"])
    ym_list = _ym_list(start, end)

    def month_range(ym: str):
        y, m = map(int, ym.split("-"))
        start_m = ee.Date.fromYMD(y, m, 1)
        end_m = start_m.advance(1, "month")
        return start_m, end_m

    rows: List[pd.DataFrame] = []

    for ym in ym_list:
        start_m, end_m = month_range(ym)

        # Composite
        comp = s2.filterDate(start_m, end_m).median()

        # Select consistent bands/features
        bands = ["B2", "B3", "B4", "B8", "B11"]
        select_list = bands[:]
        if cfg["features"]["compute_ndvi"]:
            select_list.append("NDVI")
        if cfg["features"]["compute_ndbi"]:
            select_list.append("NDBI")

        comp = comp.select(select_list)

        # Reduce to boundaries (mean per band/index)
        reduced = comp.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=scale_m,
        )

        # Optional VIIRS monthly join (mean radiance)
        if include_viirs and viirs is not None:
            vimg = viirs.filterDate(start_m, end_m).mean()
            vreduced = vimg.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=500,
            )
            join = ee.Join.inner()
            filt = ee.Filter.equals(leftField="system:index", rightField="system:index")
            joined = join.apply(reduced, vreduced, filt)

            def merge_props(f):
                left = ee.Feature(f.get("primary"))
                right = ee.Feature(f.get("secondary"))
                return left.copyProperties(right, right.propertyNames())

            reduced = ee.FeatureCollection(joined.map(merge_props))

        # Add month column
        reduced = reduced.map(lambda f: ee.Feature(f).set({"month": ym}))

        # ---- Export + download (NO GCS) ----
        description = f"features_{unit_level}_{ym}".replace("-", "_")
        out_csv_path = os.path.join("outputs", "features_tables", f"{description}.csv")

        if export_target == "asset":
            asset_id = f"{asset_folder}/{description}"
            _export_fc_to_asset(reduced, asset_id=asset_id, description=description)
            _download_table_asset_csv(asset_id, out_csv_path=out_csv_path)
            if not keep_assets:
                ee.data.deleteAsset(asset_id)

        elif export_target == "drive":
            # Exports to Google Drive; this does not auto-download.
            task = ee.batch.Export.table.toDrive(
                collection=reduced,
                description=description,
                fileNamePrefix=description,
                fileFormat="CSV",
            )
            task.start()
            _wait_for_task(task)
            raise RuntimeError(
                "Exported to Google Drive. Download manually from Drive or implement Drive API download."
            )

        df = pd.read_csv(out_csv_path)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    if unit_id_field not in out.columns:
        raise RuntimeError(
            f"Expected '{unit_id_field}' in exported table; ensure boundary FC has that property."
        )

    return out
