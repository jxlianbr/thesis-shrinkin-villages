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
        overwrite=True,
    )
    task.start()
    _wait_for_task(task)

def _download_table_asset_csv(cfg: Dict[str, Any],asset_id: str, out_csv_path: str, selectors: List[str] | None = None) -> None:
    """
    Download an EE TABLE asset (FeatureCollection) as CSV without using GCS/Drive.
    If selectors is provided, only those properties are included (excludes .geo).
    """
    fc = ee.FeatureCollection(asset_id)
    
    prefs = cfg.get("study_area", {}).get("prefectures", [])
    if prefs:
        fc = fc.filter(ee.Filter.inList("pref_name", prefs))

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    filename = os.path.splitext(os.path.basename(out_csv_path))[0]

    url = fc.getDownloadURL(
        filetype="CSV",
        selectors=selectors,
        filename=filename,
    )
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

def _read_month_csv(path: str, unit_level: str, unit_id_field: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={unit_id_field: "string", "unit_code": "string", "month": "string"},
    )

    df[unit_id_field] = df[unit_id_field].astype("string").str.strip()
    df["unit_code"] = df["unit_code"].astype("string").str.strip()

    if unit_level == "mura":
        df["unit_code"] = df["unit_code"].str.zfill(5)
    elif unit_level == "aza":
        df["unit_code"] = df["unit_code"].str.replace(r"\D", "", regex=True)
    else:
        raise ValueError("unit_level must be 'mura' or 'aza'")

    if "month" in df.columns:
        df["month"] = df["month"].astype("string").str.strip()

    # enforce 1 row per unit-month
    if unit_id_field in df.columns and "month" in df.columns:
        df = df.drop_duplicates(subset=[unit_id_field, "month"], keep="first")

    return df

def _join_props_by_index_keep_all(
    primary: ee.FeatureCollection,
    secondary: ee.FeatureCollection,
    match_key: str,
) -> ee.FeatureCollection:
    """
    Keep ALL features from `primary` and (if a match exists) copy properties from `secondary`.
    Joins on system:index produced by reduceRegions for the same input FC.
    """
    join = ee.Join.saveFirst(match_key)
    filt = ee.Filter.equals(leftField="system:index", rightField="system:index")
    joined = join.apply(primary, secondary, filt)

    def merge(f):
        f = ee.Feature(f)
        props = ee.Dictionary(
            ee.Algorithms.If(
                f.get(match_key),
                ee.Feature(f.get(match_key)).toDictionary(),
                ee.Dictionary({}),
            )
        )
        return f.set(props)
        
    return ee.FeatureCollection(joined.map(merge))


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

    if export_target == "asset":
        if "\\" in asset_folder:
            raise ValueError("cfg.gee.export_asset_folder must use forward slashes '/', not backslashes.")
        if not asset_folder.startswith("projects/") or "/assets/" not in asset_folder:
            raise ValueError("cfg.gee.export_asset_folder must be like: projects/<project-id>/assets/<folder>")

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

    # --- run_mode knobs ---
    fast_dev = bool(cfg.get("run_mode", {}).get("fast_dev", False))
    months_override = cfg.get("run_mode", {}).get("months_override")
    unit_sample_n = int(cfg.get("run_mode", {}).get("unit_sample_n") or 0)
    skip_existing = bool(cfg.get("run_mode", {}).get("skip_existing_month_csv", True))
    landsat8_enabled = bool(cfg.get("gee", {}).get("landsat8_enabled", False))
    landsat8_collection_id = cfg.get("gee", {}).get("landsat8_collection_id") or "LANDSAT/LC08/C02/T1_L2"

    # Deterministic sampling for fast iteration
    if unit_sample_n > 0:   
        fc = fc.sort(unit_id_field).limit(unit_sample_n)

    # --- Collections ---
    start = cfg["time"]["start"]
    end = cfg["time"]["end"]

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(fc)
    )

    cmax = cfg.get("gee", {}).get("cloudy_pixel_percentage_max")
    if cmax is not None:
        s2 = s2.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", float(cmax)))


    # Basic S2 mask via SCL (keeps non-cloud, non-shadow)
    def mask_s2(img):
        scl = img.select("SCL")
        keep = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
        return img.updateMask(keep)

    s2 = s2.map(mask_s2)

    def mask_l8(img):
        qa = img.select("QA_PIXEL")
        # Mask out: dilated cloud(1), cirrus(2), cloud(3), cloud shadow(4), snow(5), water(7)
        mask = (
            qa.bitwiseAnd(1 << 1).eq(0)  # cloud
            .And(qa.bitwiseAnd(1 << 2).eq(0))  # cloud shadow
            .And(qa.bitwiseAnd(1 << 3).eq(0))  # snow
            .And(qa.bitwiseAnd(1 << 4).eq(0))
            .And(qa.bitwiseAnd(1 << 5).eq(0))  # water
            .And(qa.bitwiseAnd(1 << 7).eq(0))  # cirrus
        )
        return img.updateMask(mask)
    
    def prep_l8(img):
    #Select SR bands and rename to stable names for merging.
    #(No unit claims here; goal is consistent numeric ranges and stable schema.)
        sr = img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6"]).rename(
            ["L8_B2", "L8_B3", "L8_B4", "L8_B5", "L8_B6"]
        )
        if cfg["features"].get("compute_ndvi"):
            ndvi = sr.normalizedDifference(["L8_B5", "L8_B4"]).rename("L8_NDVI")
            sr = sr.addBands(ndvi)
        if cfg["features"].get("compute_ndbi"):
            ndbi = sr.normalizedDifference(["L8_B6", "L8_B5"]).rename("L8_NDBI")
            sr = sr.addBands(ndbi)
        return sr

    l8 = None
    if landsat8_enabled:
        l8 = (
            ee.ImageCollection(landsat8_collection_id)
            .filterDate(start, end)
            .filterBounds(fc)
        )
        #Lightweight access check (fails fast if misconfigured / no permission)
        _ = l8.limit(1).size().getInfo()

        # apply masking + band prep (renames to L8_* and optionally adds L8_NDVI, L8_NDBI)
        l8 = l8.map(mask_l8).map(prep_l8)

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
    ym_list = months_override if months_override else _ym_list(start, end)


    def month_range(ym: str):
        y, m = map(int, ym.split("-"))
        start_m = ee.Date.fromYMD(y, m, 1)
        end_m = start_m.advance(1, "month")
        return start_m, end_m

    rows: List[pd.DataFrame] = []

    for ym in ym_list:
        start_m, end_m = month_range(ym)

        # Composite
        # Monthly subset
        month_s2 = s2.filterDate(start_m, end_m)

        # If empty, skip the month (prevents "Image has no bands" on select())
        n_s2 = month_s2.size().getInfo()
        if n_s2 == 0:
            print(f"[WARN] {ym}: Sentinel-2 empty after filters; skipping month.")
            continue

        # Composite
        comp = month_s2.median()

        # Extra safety: composite can still end up with 0 bands in edge cases
        n_bands = comp.bandNames().size().getInfo()
        if n_bands == 0:
            print(f"[WARN] {ym}: composite has 0 bands; skipping month.")
            continue    

        # Select consistent bands/features
        bands = ["B2", "B3", "B4", "B8", "B11"]
        select_list = bands[:]
        if cfg["features"]["compute_ndvi"]:
            select_list.append("NDVI")
        if cfg["features"]["compute_ndbi"]:
            select_list.append("NDBI")

        comp = comp.select(select_list)
        comp = comp.resample("bilinear")

        # Build a stacked monthly composite (S2 + optional L8) and reduce once
        img_stack = comp

        if landsat8_enabled and l8 is not None:
            month_l8 = l8.filterDate(start_m, end_m)
            n_l8 = month_l8.size().getInfo()
            if n_l8 == 0:
                print(f"[WARN] {ym}: Landsat 8 empty after filters; skipping L8 for this month.")
            else:
                l8_comp = month_l8.median().resample("bilinear")
                img_stack = img_stack.addBands(l8_comp)

            l8_comp = month_l8.median().resample("bilinear")
            img_stack = img_stack.addBands(l8_comp)

        reduced = img_stack.reduceRegions(
            collection = fc,
            reducer = ee.Reducer.mean(),
            scale = 10, # hard-force 10 m sampling
        )

        # Optional VIIRS monthly join (mean radiance)
        if include_viirs and viirs is not None:
            vimg = viirs.filterDate(start_m, end_m).mean()
            vreduced = vimg.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean().setOutputs(["viirs_mean"]),
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
        description = f"features_{ym}"
        out_csv_path = f"outputs/gee/monthly/features_{unit_level}_{ym}.csv"

        if skip_existing and os.path.exists(out_csv_path):
            rows.append(_read_month_csv(out_csv_path, unit_level, unit_id_field))
            continue

        selectors = [
            unit_id_field,
            "unit_level",
            "pref_name",
            "unit_code",
            "month",
            "B2", "B3", "B4", "B8", "B11",
        ]
        if cfg["features"]["compute_ndvi"]:
            selectors.append("NDVI")
        if cfg["features"]["compute_ndbi"]:
            selectors.append("NDBI")
        if include_viirs:
            selectors.append("viirs_mean")
        if landsat8_enabled:
            selectors += ["L8_B2", "L8_B3", "L8_B4", "L8_B5", "L8_B6"]
            if cfg["features"]["compute_ndvi"]:
                selectors.append("L8_NDVI")
            if cfg["features"]["compute_ndbi"]:
                selectors.append("L8_NDBI")

        # Ensure output dir exists
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

        if fast_dev:
             # Direct download of the computed FeatureCollection (no EE batch export tasks).
            url = reduced.getDownloadURL(filetype="CSV", selectors=selectors, filename=description)
            urllib.request.urlretrieve(url, out_csv_path)

        elif export_target == "asset":
            asset_id = f"{asset_folder}/{description}"
            _export_fc_to_asset(reduced, asset_id=asset_id, description=description)
            _download_table_asset_csv(cfg, asset_id, out_csv_path=out_csv_path, selectors=selectors)
            if not keep_assets:
                ee.data.deleteAsset(asset_id)

        elif export_target == "drive":
            task = ee.batch.Export.table.toDrive(
                collection=reduced,
                description=description,
                fileNamePrefix=description,
                fileFormat="CSV",
            )
            task.start()
            _wait_for_task(task)
            raise RuntimeError("Exported to Google Drive. Download manually from Drive or implement Drive API download.")
        
        rows.append(_read_month_csv(out_csv_path, unit_level, unit_id_field))

    if not rows:
        raise RuntimeError("No monthly data was exported; check logs for issues.")

    out = pd.concat(rows, ignore_index=True)

    if unit_id_field not in out.columns:
        raise RuntimeError(
            f"Expected '{unit_id_field}' in exported table; ensure boundary FC has that property."
        )

    return out
