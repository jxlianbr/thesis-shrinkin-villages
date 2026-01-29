from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Sequence
import os
import time
import urllib.request

import ee
import pandas as pd


# ---------------------------
# EE task helpers
# ---------------------------

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


def _download_table_asset_csv(
    cfg: Dict[str, Any],
    asset_id: str,
    out_csv_path: str,
    selectors: List[str] | None = None,
) -> None:
    """
    Download an EE TABLE asset (FeatureCollection) as CSV without using GCS/Drive.
    If selectors is provided, only those properties are included (excludes .geo).
    Optionally filters by cfg.study_area.prefectures if provided.
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


# ---------------------------
# Local CSV helpers
# ---------------------------

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


def _assert_month_features_not_all_nan(df: pd.DataFrame, ym: str, required_cols: Sequence[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ym}: missing required feature columns: {missing}")
    if df[list(required_cols)].isna().all().all():
        raise RuntimeError(f"{ym}: all required features are NaN (all-masked / empty export).")


# ---------------------------
# GEE image helpers
# ---------------------------

def _masked_constant_image(band_names: List[str]) -> ee.Image:
    """Fully-masked constant image with the requested bands (keeps stable schema)."""
    zeros = ee.Image.constant([0] * len(band_names)).rename(band_names)
    return zeros.updateMask(ee.Image.constant(0))


def _add_glcm_texture(
    img: ee.Image,
    src_band: str,
    out_prefix: str,
    size: int,
    metrics: List[str],
) -> ee.Image:
    """
    Adds selected GLCM texture bands (computed from `src_band`) to `img`.
    GLCM requires integer inputs -> quantize to 8-bit.

    Output band names:
      f"{out_prefix}_{metric}" e.g. "S2_NDBI_contrast"
    """
    b = img.select([src_band])
    name_upper = src_band.upper()

    # Quantization ranges:
    # - Indices: [-1, 1]
    # - Landsat scaled SR bands: roughly [-0.2, 1.0]
    # - Sentinel-2 SR bands (as stored): roughly [0, 10000]
    if "NDVI" in name_upper or "NDBI" in name_upper:
        q = b.clamp(-1, 1).unitScale(-1, 1).multiply(255).toUint8()
    elif name_upper.startswith("L8_"):
        q = b.clamp(-0.2, 1.0).unitScale(-0.2, 1.0).multiply(255).toUint8()
    else:
        q = b.clamp(0, 10000).unitScale(0, 10000).multiply(255).toUint8()

    tex = q.glcmTexture(size=size)

    # Map user-friendly names -> EE suffixes
    metric_map = {
        "entropy": "ent",
        "homogeneity": "idm",
        # pass-throughs that already match EE names
        "contrast": "contrast",
        "asm": "asm",
        "corr": "corr",
        "var": "var",
        "diss": "diss",
        "inertia": "inertia",
        "shade": "shade",
        "prom": "prom",
        "savg": "savg",
        "svar": "svar",
        "sent": "sent",
        "dvar": "dvar",
        "dent": "dent",
        "imcorr1": "imcorr1",
        "imcorr2": "imcorr2",
        "maxcorr": "maxcorr",
    }
    
    ee_suffixes: List[str] = []
    for m in metrics:
        if m not in metric_map:
            raise ValueError(f"Unknown GLCM metric '{m}'. Supported: {sorted(metric_map.keys())}")
        ee_suffixes.append(metric_map[m])

    src_names = [f"{src_band}_{s}" for s in ee_suffixes]
    out_names = [f"{out_prefix}_{m}" for m in metrics]  # keep your chosen names in output

    tex_sel = tex.select(src_names).rename(out_names)
    return img.addBands(tex_sel)


# ---------------------------
# Main entry
# ---------------------------

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

    # Filter study-area prefectures at the FEATURE level
    prefs = cfg.get("study_area", {}).get("prefectures", [])
    if prefs:
        fc = fc.filter(ee.Filter.inList("pref_name", prefs))

    # --- run_mode knobs ---
    fast_dev = bool(cfg.get("run_mode", {}).get("fast_dev", False))
    months_override = cfg.get("run_mode", {}).get("months_override")
    unit_sample_n = int(cfg.get("run_mode", {}).get("unit_sample_n") or 0)
    skip_existing = bool(cfg.get("run_mode", {}).get("skip_existing_month_csv", True))
    strict_no_empty = bool(cfg.get("run_mode", {}).get("strict_no_empty_month", True))

    # Sensor enablement
    landsat8_enabled = bool(cfg.get("gee", {}).get("landsat8_enabled", False))
    landsat8_collection_id = cfg.get("gee", {}).get("landsat8_collection_id") or "LANDSAT/LC08/C02/T1_L2"

    # GLCM / indices settings
    compute_ndvi = bool(cfg.get("features", {}).get("compute_ndvi", True))
    compute_ndbi = bool(cfg.get("features", {}).get("compute_ndbi", True))

    compute_glcm = bool(cfg.get("features", {}).get("compute_glcm", False))
    glcm_size = int(cfg.get("features", {}).get("glcm_size", 3))
    glcm_metrics = cfg.get("features", {}).get("glcm_metrics") or ["contrast", "entropy", "homogeneity"]
    glcm_source_s2 = cfg.get("features", {}).get("glcm_source_s2") or ("NDBI" if compute_ndbi else "B8")
    glcm_source_l8 = cfg.get("features", {}).get("glcm_source_l8") or ("L8_NDBI" if compute_ndbi else "L8_B5")

    # Deterministic sampling for fast iteration
    if unit_sample_n > 0:
        fc = fc.sort(unit_id_field).limit(unit_sample_n)

    # --- Collections ---
    start = cfg["time"]["start"]
    end = cfg["time"]["end"]

    # Sentinel-2 L2A (already atmospherically corrected)
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(fc)
    )

    cmax = cfg.get("gee", {}).get("cloudy_pixel_percentage_max")
    if cmax is not None:
        s2 = s2.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", float(cmax)))

    # Sentinel-2 mask via SCL: keep vegetation + bare + water
    def mask_s2(img: ee.Image) -> ee.Image:
        scl = img.select("SCL")
        keep = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
        return img.updateMask(keep)

    s2 = s2.map(mask_s2)

    def add_s2_indices(img: ee.Image) -> ee.Image:
        out = img
        if compute_ndvi:
            out = out.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
        if compute_ndbi:
            out = out.addBands(img.normalizedDifference(["B11", "B8"]).rename("NDBI"))
        return out

    if compute_ndvi or compute_ndbi:
        s2 = s2.map(add_s2_indices)

    # Landsat 8 Collection 2 L2 SR: QA_PIXEL mask + scale SR bands before indices
    l8 = None
    if landsat8_enabled:
        l8 = (
            ee.ImageCollection(landsat8_collection_id)
            .filterDate(start, end)
            .filterBounds(fc)
        )
        _ = l8.limit(1).size().getInfo()  # fail fast if inaccessible

        def mask_l8(img: ee.Image) -> ee.Image:
            qa = img.select("QA_PIXEL")
            # Bits: 1=dilated cloud, 2=cirrus, 3=cloud, 4=cloud shadow, 5=snow, 7=water
            m = (
                qa.bitwiseAnd(1 << 1).eq(0)
                .And(qa.bitwiseAnd(1 << 2).eq(0))
                .And(qa.bitwiseAnd(1 << 3).eq(0))
                .And(qa.bitwiseAnd(1 << 4).eq(0))
                .And(qa.bitwiseAnd(1 << 5).eq(0))
                .And(qa.bitwiseAnd(1 << 7).eq(0))
            )
            return img.updateMask(m)

        def prep_l8(img: ee.Image) -> ee.Image:
            sr = (
                img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6"])
                .multiply(0.0000275)
                .add(-0.2)
                .rename(["L8_B2", "L8_B3", "L8_B4", "L8_B5", "L8_B6"])
            )
            out = sr
            if compute_ndvi:
                out = out.addBands(sr.normalizedDifference(["L8_B5", "L8_B4"]).rename("L8_NDVI"))
            if compute_ndbi:
                out = out.addBands(sr.normalizedDifference(["L8_B6", "L8_B5"]).rename("L8_NDBI"))
            return out

        l8 = l8.map(mask_l8).map(prep_l8)

    # VIIRS monthly
    include_viirs = bool(cfg.get("features", {}).get("include_viirs", False))
    viirs = None
    if include_viirs:
        viirs = (
            ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
            .filterDate(start, end)
            .filterBounds(fc)
            .select(["avg_rad"], ["VIIRS_avg_rad"])
        )

    # --- Monthly composite + zonal stats ---
    ym_list = months_override if months_override else _ym_list(start, end)

    def month_range(ym: str):
        y, m = map(int, ym.split("-"))
        start_m = ee.Date.fromYMD(y, m, 1)
        end_m = start_m.advance(1, "month")
        return start_m, end_m

    # Precompute placeholder band names for stable schema (L8 only)
    l8_placeholder_bands: List[str] = []
    if landsat8_enabled:
        l8_placeholder_bands = ["L8_B2", "L8_B3", "L8_B4", "L8_B5", "L8_B6"]
        if compute_ndvi:
            l8_placeholder_bands.append("L8_NDVI")
        if compute_ndbi:
            l8_placeholder_bands.append("L8_NDBI")
        if compute_glcm:
            l8_placeholder_bands += [f"{glcm_source_l8}_{m}" for m in glcm_metrics]  # e.g. L8_NDBI_contrast

    rows: List[pd.DataFrame] = []

    for ym in ym_list:
        start_m, end_m = month_range(ym)

        # --- Sentinel-2 monthly composite ---
        month_s2 = s2.filterDate(start_m, end_m)
        n_s2 = month_s2.size().getInfo()
        if n_s2 == 0:
            print(f"[WARN] {ym}: Sentinel-2 empty after filters; skipping month.")
            continue

        comp = month_s2.median()

        s2_bands = ["B2", "B3", "B4", "B8", "B11"]
        select_list = s2_bands[:]
        if compute_ndvi:
            select_list.append("NDVI")
        if compute_ndbi:
            select_list.append("NDBI")

        comp = comp.select(select_list).resample("bilinear")

        if compute_glcm:
            comp = _add_glcm_texture(
                comp,
                src_band=glcm_source_s2,
                out_prefix=f"S2_{glcm_source_s2}",
                size=glcm_size,
                metrics=list(glcm_metrics),
            )

        img_stack = comp

        # --- Landsat 8 monthly composite (optional) ---
        if landsat8_enabled:
            if l8 is None:
                raise RuntimeError("landsat8_enabled=True but L8 collection is None")

            month_l8 = l8.filterDate(start_m, end_m)
            n_l8 = month_l8.size().getInfo()

            if n_l8 > 0:
                l8_comp = month_l8.median().resample("bilinear")
                if compute_glcm:
                    l8_comp = _add_glcm_texture(
                        l8_comp,
                        src_band=glcm_source_l8,
                        out_prefix=glcm_source_l8,
                        size=glcm_size,
                        metrics=list(glcm_metrics),
                    )
                img_stack = img_stack.addBands(l8_comp)
            else:
                print(f"[WARN] {ym}: Landsat-8 empty after filters; writing null L8 bands for this month.")
                if l8_placeholder_bands:
                    img_stack = img_stack.addBands(_masked_constant_image(l8_placeholder_bands))

        # --- Reduce to boundaries (mean per band/index/texture) ---
        reduced = img_stack.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=10,  # harmonized 10 m sampling
        )

        # --- Optional VIIRS monthly join (mean radiance) ---
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

        reduced = reduced.map(lambda f: ee.Feature(f).set({"month": ym}))

        # ---- Export + download (NO GCS) ----
        description = f"features_{ym}"
        out_csv_path = f"outputs/gee/monthly/features_{unit_level}_{ym}.csv"

        if skip_existing and os.path.exists(out_csv_path):
            df = _read_month_csv(out_csv_path, unit_level, unit_id_field)
            rows.append(df)
            continue

        selectors = [
            unit_id_field,
            "unit_level",
            "pref_name",
            "unit_code",
            "month",
            "B2", "B3", "B4", "B8", "B11",
        ]
        if compute_ndvi:
            selectors.append("NDVI")
        if compute_ndbi:
            selectors.append("NDBI")
        if compute_glcm:
            selectors += [f"S2_{glcm_source_s2}_{m}" for m in glcm_metrics]

        if include_viirs:
            selectors.append("viirs_mean")

        if landsat8_enabled:
            selectors += ["L8_B2", "L8_B3", "L8_B4", "L8_B5", "L8_B6"]
            if compute_ndvi:
                selectors.append("L8_NDVI")
            if compute_ndbi:
                selectors.append("L8_NDBI")
            if compute_glcm:
                selectors += [f"{glcm_source_l8}_{m}" for m in glcm_metrics]

        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

        if fast_dev:
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

        df = _read_month_csv(out_csv_path, unit_level, unit_id_field)

        if strict_no_empty:
            req: List[str] = []
            if compute_ndvi:
                req.append("NDVI")
            if compute_ndbi:
                req.append("NDBI")
            if compute_glcm:
                req += [f"S2_{glcm_source_s2}_{m}" for m in glcm_metrics]
            _assert_month_features_not_all_nan(df, ym, req)

        rows.append(df)

    if not rows:
        raise RuntimeError("No monthly data was exported; check logs for issues.")

    out = pd.concat(rows, ignore_index=True)

    if unit_id_field not in out.columns:
        raise RuntimeError(
            f"Expected '{unit_id_field}' in exported table; ensure boundary FC has that property."
        )

    return out
