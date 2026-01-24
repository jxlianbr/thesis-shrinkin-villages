from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import ee
import pandas as pd


def _ym_list(start: str, end: str) -> List[str]:
    # start/end format: YYYY-MM-DD
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    out = []
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
    Returns a unit-level feature table as a Pandas DataFrame (downloaded from GCS).
    """
    import ee
    from google.cloud import storage

    # --- Authenticate/initialize ---
    # If you already authenticated via `earthengine authenticate`, Initialize uses stored credentials.
    cloud_project = cfg["gee"]["cloud_project"]
    ee.Initialize(project=cloud_project)

    # --- Load boundaries (must be uploaded as an Earth Engine Asset OR ingested from GCS) ---
    # Backbone assumption for Task 1: boundaries are stored as an EE FeatureCollection asset.
    # Set this in config by replacing the placeholder below.
    # Example: "users/<username>/villages_fc"
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
    # Sentinel-2 SR (surface reflectance) + Landsat 8/9 SR are typical for analysis.
    # Cloud masking uses QA/SCL layers available in these collections.
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
        # keep: vegetation(4), not_veg(5), water(6), unclassified(7), snow(11) optionally
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

    rows = []

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
            scale=scale_m
        )

        # Optional VIIRS monthly join (mean radiance)
        if include_viirs and viirs is not None:
            vimg = viirs.filterDate(start_m, end_m).mean()
            vreduced = vimg.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=500
            )
            # Join by system:index (stable for same FC)
            join = ee.Join.inner()
            filt = ee.Filter.equals(leftField="system:index", rightField="system:index")
            joined = join.apply(reduced, vreduced, filt)

            def merge_props(f):
                left = ee.Feature(f.get("primary"))
                right = ee.Feature(f.get("secondary"))
                return left.copyProperties(right, right.propertyNames())

            reduced = ee.FeatureCollection(joined.map(merge_props))

        # Add month column
        def add_month(f):
            return ee.Feature(f).set({"month": ym})

        reduced = reduced.map(add_month)

        # Export to GCS as CSV (deterministic object name)
        bucket = cfg["gee"]["gcs_bucket"]
        prefix = cfg["gee"]["export_prefix"]
        obj_name = f"{prefix}/features_{ym}.csv"

        task = ee.batch.Export.table.toCloudStorage(
            collection=reduced,
            description=f"export_features_{ym}",
            bucket=bucket,
            fileNamePrefix=obj_name.replace(".csv", ""),
            fileFormat="CSV"
        )
        task.start()

        # Poll until completion (keeps “one-run” semantics)
        while task.active():
            pass

        status = task.status()
        if status.get("state") != "COMPLETED":
            raise RuntimeError(f"GEE export failed for {ym}: {status}")

        # Download from GCS
        client = storage.Client(project=cloud_project)
        b = client.bucket(bucket)
        blob = b.blob(f"{obj_name}")
        csv_bytes = blob.download_as_bytes()
        df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))

        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    # Ensure unit_id column exists (comes from FC properties)
    if unit_id_field not in out.columns:
        raise RuntimeError(f"Expected '{unit_id_field}' in exported table; ensure boundary FC has that property.")

    return out
