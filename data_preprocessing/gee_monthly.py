from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Sequence, TypeVar
import glob
import os
import random
import re
import time
import urllib.request

import ee
import pandas as pd

T = TypeVar("T")


def _retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> T:
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry (should take no arguments)
        max_retries: Maximum number of attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter: If True, add randomness to delay to avoid thundering herd

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries fail
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                if jitter:
                    delay = delay * (0.5 + random.random())
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)

    raise last_exception  # type: ignore[misc]


# ---------------------------
# EE task helpers
# ---------------------------

def _wait_for_task(task: ee.batch.Task, poll_s: int = 30) -> None:
    """Wait for an EE task to complete, polling every poll_s seconds."""
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
    max_retries: int = 3,
) -> None:
    """
    Download an EE TABLE asset (FeatureCollection) as CSV without using GCS/Drive.
    If selectors is provided, only those properties are included (excludes .geo).
    Optionally filters by cfg.study_area.prefectures if provided.
    Uses retry with exponential backoff for reliability.
    """
    fc = ee.FeatureCollection(asset_id)

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    filename = os.path.splitext(os.path.basename(out_csv_path))[0]

    def _get_and_download():
        url = fc.getDownloadURL(
            filetype="CSV",
            selectors=selectors,
            filename=filename,
        )
        urllib.request.urlretrieve(url, out_csv_path)

    _retry_with_backoff(_get_and_download, max_retries=max_retries)


def _export_viirs_for_month(
    viirs: ee.ImageCollection,
    fc: ee.FeatureCollection,
    start_m: ee.Date,
    end_m: ee.Date,
    ym: str,
    unit_level: str,
    out_csv_path: str,
    asset_folder: str,
    unit_id_field: str,
    max_retries: int = 3,
) -> None:
    """
    Export VIIRS radiance separately from S2 features.

    This is done as a separate export to avoid the expensive GEE join operation.
    The results are merged locally in Python for better performance.
    """
    print(f"  Exporting VIIRS for {ym}...")
    vimg = viirs.filterDate(start_m, end_m).mean()
    vreduced = vimg.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean().setOutputs(["viirs_mean"]),
        scale=500,
    )
    vreduced = vreduced.map(lambda f: ee.Feature(f).set({"month": ym, "unit_level": unit_level}))

    selectors = [unit_id_field, "month", "viirs_mean"]
    description = f"viirs_{ym}"
    temp_asset_id = f"{asset_folder}/temp_{description}"

    def _do_export():
        _export_fc_to_asset(vreduced, asset_id=temp_asset_id, description=f"temp_{description}")

    _retry_with_backoff(_do_export, max_retries=max_retries)
    _download_table_asset_csv({}, temp_asset_id, out_csv_path, selectors=selectors, max_retries=max_retries)

    # Clean up temp asset
    try:
        ee.data.deleteAsset(temp_asset_id)
    except Exception:
        pass


def _batch_feature_collection(fc: ee.FeatureCollection, batch_size: int) -> List[ee.FeatureCollection]:
    """
    Split a FeatureCollection into batches for parallel/sequential processing.
    Returns list of FeatureCollections, each with at most batch_size features.

    Args:
        fc: Input FeatureCollection to split
        batch_size: Maximum features per batch

    Returns:
        List of FeatureCollections (batches)
    """
    total = fc.size().getInfo()
    print(f"Splitting {total} features into batches of {batch_size}...")

    batches: List[ee.FeatureCollection] = []
    for start in range(0, total, batch_size):
        batch_list = fc.toList(batch_size, start)
        batch_fc = ee.FeatureCollection(batch_list)
        batches.append(batch_fc)

    print(f"Created {len(batches)} batches")
    return batches


def _get_completed_batches(out_dir: str, ym: str, unit_level: str) -> set[int]:
    """
    Check which batch CSV files already exist for a given month.

    Args:
        out_dir: Directory containing batch CSVs
        ym: Year-month string (e.g., "2020-01")
        unit_level: Unit level (mura/aza)

    Returns:
        Set of completed batch indices
    """
    pattern = os.path.join(out_dir, f"features_{unit_level}_{ym}_batch*.csv")
    existing = glob.glob(pattern)
    completed: set[int] = set()
    for f in existing:
        match = re.search(r"_batch(\d+)\.csv$", f)
        if match:
            completed.add(int(match.group(1)))
    return completed


def _merge_batch_csvs(out_dir: str, ym: str, unit_level: str, final_path: str) -> None:
    """
    Merge all batch CSV files for a month into a single CSV.

    Args:
        out_dir: Directory containing batch CSVs
        ym: Year-month string
        unit_level: Unit level (mura/aza)
        final_path: Path for merged output CSV
    """
    pattern = os.path.join(out_dir, f"features_{unit_level}_{ym}_batch*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No batch files found for {ym}")

    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(final_path, index=False)
    print(f"Merged {len(files)} batch files into {final_path}")


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
    glcm_scale: int | None = None,
) -> ee.Image:
    """
    Adds selected GLCM texture bands (computed from `src_band`) to `img`.
    GLCM requires integer inputs -> quantize to 8-bit.

    Args:
        img: Input image containing src_band
        src_band: Band name to compute GLCM on
        out_prefix: Prefix for output band names
        size: GLCM kernel size (e.g., 3 for 3x3)
        metrics: List of texture metrics to compute
        glcm_scale: Optional scale in meters for GLCM computation.
                    If provided and > native scale, reprojects to coarser
                    resolution before computing GLCM (improves performance).

    Output band names:
      f"{out_prefix}_{metric}" e.g. "S2_NDBI_contrast"
    """
    b = img.select([src_band])
    name_upper = src_band.upper()

    # If glcm_scale is provided, reproject to coarser resolution for efficiency
    if glcm_scale is not None and glcm_scale > 10:
        b = b.reproject(crs=img.projection(), scale=glcm_scale)

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
# Single month processing (for parallel execution)
# ---------------------------

def _process_single_month(ym: str, ctx: Dict[str, Any]) -> pd.DataFrame | None:
    """
    Process a single month's data: composite, reduceRegions, export, and return DataFrame.

    This function is designed to be called from a ThreadPoolExecutor for parallel processing.
    All required context (collections, settings, etc.) is passed via the ctx dict.

    Args:
        ym: Year-month string (e.g., "2020-01")
        ctx: Dictionary containing all shared context (s2, viirs, fc, batches, settings, etc.)

    Returns:
        DataFrame with processed features for the month, or None on failure
    """
    # Unpack context
    s2 = ctx["s2"]
    viirs = ctx["viirs"]
    fc = ctx["fc"]
    batches = ctx["batches"]
    unit_level = ctx["unit_level"]
    unit_id_field = ctx["unit_id_field"]
    scale_m = ctx["scale_m"]
    selectors = ctx["selectors"]
    asset_folder = ctx["asset_folder"]
    export_target = ctx["export_target"]
    max_retries = ctx["max_retries"]
    compute_ndvi = ctx["compute_ndvi"]
    compute_ndbi = ctx["compute_ndbi"]
    compute_mndwi = ctx["compute_mndwi"]
    compute_glcm = ctx["compute_glcm"]
    glcm_source_s2 = ctx["glcm_source_s2"]
    glcm_size = ctx["glcm_size"]
    glcm_metrics = ctx["glcm_metrics"]
    glcm_scale = ctx["glcm_scale"]
    include_viirs = ctx["include_viirs"]
    skip_existing = ctx["skip_existing"]
    strict_no_empty = ctx["strict_no_empty"]
    cfg = ctx["cfg"]

    # Parse year-month
    y, m = map(int, ym.split("-"))
    start_m = ee.Date.fromYMD(y, m, 1)
    end_m = start_m.advance(1, "month")

    # Output paths
    out_csv_path = f"outputs/gee/monthly/features_{unit_level}_{ym}.csv"
    out_dir = os.path.dirname(out_csv_path)

    # Check if already exists
    if skip_existing and os.path.exists(out_csv_path):
        print(f"[{ym}] Skipping (already exists)")
        return _read_month_csv(out_csv_path, unit_level, unit_id_field)

    print(f"[{ym}] Starting processing...")

    # --- Sentinel-2 monthly composite ---
    month_s2 = s2.filterDate(start_m, end_m)
    has_s2 = month_s2.size().gt(0)

    s2_bands = ["B2", "B3", "B4", "B8", "B11"]
    select_list = s2_bands[:]
    if compute_ndvi:
        select_list.append("NDVI")
    if compute_ndbi:
        select_list.append("NDBI")
    if compute_mndwi:
        select_list.append("MNDWI")

    # If a month has no images, use a masked placeholder so the export still runs
    empty_s2 = _masked_constant_image(select_list)
    comp = ee.Image(ee.Algorithms.If(has_s2, month_s2.median(), empty_s2))
    comp = comp.select(select_list).resample("bilinear")

    if compute_glcm:
        out_prefix = f"S2_{glcm_source_s2}"
        glcm_out_bands = [f"{out_prefix}_{m}" for m in glcm_metrics]
        comp = ee.Image(
            ee.Algorithms.If(
                has_s2,
                _add_glcm_texture(
                    comp,
                    src_band=glcm_source_s2,
                    out_prefix=out_prefix,
                    size=glcm_size,
                    metrics=list(glcm_metrics),
                    glcm_scale=glcm_scale,
                ),
                _masked_constant_image(select_list + glcm_out_bands),
            )
        )

    img_stack = comp

    os.makedirs(out_dir, exist_ok=True)

    # --- Helper to process a single batch of features (S2 only, no VIIRS) ---
    def _process_batch(batch_fc: ee.FeatureCollection) -> ee.FeatureCollection:
        """Reduce image stack to a batch of boundary features (S2 only)."""
        reduced = img_stack.reduceRegions(
            collection=batch_fc,
            reducer=ee.Reducer.mean(),
            scale=scale_m,
        )
        # Add metadata (VIIRS is exported separately and merged locally)
        reduced = reduced.map(lambda f: ee.Feature(f).set({"month": ym, "unit_level": unit_level}))
        return reduced

    # --- Helper to export a FeatureCollection to CSV via asset ---
    def _export_batch_to_csv(batch_reduced: ee.FeatureCollection, batch_csv_path: str, batch_desc: str) -> None:
        """Export a batch using async asset export (more reliable than getDownloadURL)."""
        temp_asset_id = f"{asset_folder}/temp_{batch_desc}"

        def _do_export():
            _export_fc_to_asset(batch_reduced, asset_id=temp_asset_id, description=f"temp_{batch_desc}")

        _retry_with_backoff(_do_export, max_retries=max_retries)
        _download_table_asset_csv(cfg, temp_asset_id, out_csv_path=batch_csv_path, selectors=selectors, max_retries=max_retries)

        # Clean up temp asset
        try:
            ee.data.deleteAsset(temp_asset_id)
        except Exception:
            pass  # Ignore cleanup errors

    # --- Process using pre-computed batches ---
    completed_batches = _get_completed_batches(out_dir, ym, unit_level)
    print(f"[{ym}] {len(batches)} batches, {len(completed_batches)} already completed")

    for batch_idx, batch_fc in enumerate(batches):
        if batch_idx in completed_batches:
            print(f"[{ym}] Batch {batch_idx + 1}/{len(batches)}: skipping (already exists)")
            continue

        print(f"[{ym}] Batch {batch_idx + 1}/{len(batches)}: processing...")
        batch_csv_path = os.path.join(out_dir, f"features_{unit_level}_{ym}_batch{batch_idx:03d}.csv")
        batch_desc = f"features_{ym}_batch{batch_idx:03d}"

        batch_reduced = _process_batch(batch_fc)

        if export_target == "drive":
            raise RuntimeError("Drive export not supported with batching. Use export_target='asset'.")

        _export_batch_to_csv(batch_reduced, batch_csv_path, batch_desc)

    # Merge all batch CSVs into final monthly CSV (only if we have multiple batches)
    if len(batches) > 1:
        _merge_batch_csvs(out_dir, ym, unit_level, out_csv_path)

        # Clean up batch files after successful merge
        for batch_file in glob.glob(os.path.join(out_dir, f"features_{unit_level}_{ym}_batch*.csv")):
            try:
                os.remove(batch_file)
            except OSError:
                pass
    else:
        # Single batch - just rename the batch file to final name
        single_batch_path = os.path.join(out_dir, f"features_{unit_level}_{ym}_batch000.csv")
        if os.path.exists(single_batch_path) and not os.path.exists(out_csv_path):
            os.rename(single_batch_path, out_csv_path)

    # --- Export VIIRS separately and merge locally ---
    if include_viirs and viirs is not None:
        viirs_csv_path = os.path.join(out_dir, f"viirs_{unit_level}_{ym}.csv")

        # Export VIIRS if not already done
        if not os.path.exists(viirs_csv_path):
            _export_viirs_for_month(
                viirs=viirs,
                fc=fc,
                start_m=start_m,
                end_m=end_m,
                ym=ym,
                unit_level=unit_level,
                out_csv_path=viirs_csv_path,
                asset_folder=asset_folder,
                unit_id_field=unit_id_field,
                max_retries=max_retries,
            )

        # Merge VIIRS into S2 features locally
        print(f"[{ym}] Merging VIIRS data locally...")
        s2_df = pd.read_csv(out_csv_path)
        viirs_df = pd.read_csv(viirs_csv_path)
        merged = s2_df.merge(
            viirs_df[[unit_id_field, "viirs_mean"]],
            on=unit_id_field,
            how="left"
        )
        merged.to_csv(out_csv_path, index=False)

    df = _read_month_csv(out_csv_path, unit_level, unit_id_field)

    if strict_no_empty:
        req: List[str] = []
        if compute_ndvi:
            req.append("NDVI")
        if compute_ndbi:
            req.append("NDBI")
        if compute_mndwi:
            req.append("MNDWI")
        if compute_glcm:
            req += [f"S2_{glcm_source_s2}_{m}" for m in glcm_metrics]
        _assert_month_features_not_all_nan(df, ym, req)

    return df


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

    # --- Load AOI for satellite image filtering ---
    # AOI provides a single geometry for efficient filterBounds on image collections
    aoi_cfg = cfg.get("aoi", {})
    aoi_mode = aoi_cfg.get("mode", "full")
    aoi_geometry = None

    if aoi_cfg:
        aoi_asset_key = f"aoi_{aoi_mode}_asset_id"
        aoi_asset_id = aoi_cfg.get(aoi_asset_key)
        if aoi_asset_id:
            try:
                aoi_fc = ee.FeatureCollection(aoi_asset_id)
                aoi_geometry = aoi_fc.geometry()
                print(f"Loaded AOI ({aoi_mode}): {aoi_asset_id}")
            except Exception as e:
                print(f"Warning: Could not load AOI asset {aoi_asset_id}: {e}")
                print("Falling back to boundary FC geometry for filterBounds")

    # Fallback: use boundary FC geometry if AOI not available
    if aoi_geometry is None:
        aoi_geometry = fc.geometry()

    # --- run_mode knobs ---
    fast_dev = bool(cfg.get("run_mode", {}).get("fast_dev", False))
    months_override = cfg.get("run_mode", {}).get("months_override")
    unit_sample_n = int(cfg.get("run_mode", {}).get("unit_sample_n") or 0)
    skip_existing = bool(cfg.get("run_mode", {}).get("skip_existing_month_csv", True))
    strict_no_empty = bool(cfg.get("run_mode", {}).get("strict_no_empty_month", True))

    # GLCM / indices settings
    compute_ndvi = bool(cfg.get("features", {}).get("compute_ndvi", True))
    compute_ndbi = bool(cfg.get("features", {}).get("compute_ndbi", True))
    compute_mndwi = bool(cfg.get("features", {}).get("compute_mndwi", False))

    compute_glcm = bool(cfg.get("features", {}).get("compute_glcm", False))
    glcm_size = int(cfg.get("features", {}).get("glcm_size", 3))
    glcm_metrics = cfg.get("features", {}).get("glcm_metrics") or ["contrast", "entropy", "homogeneity"]
    glcm_source_s2 = cfg.get("features", {}).get("glcm_source_s2") or ("NDBI" if compute_ndbi else "B8")
    glcm_source_l8 = cfg.get("features", {}).get("glcm_source_l8") or ("L8_NDBI" if compute_ndbi else "L8_B5")
    glcm_scale = cfg.get("features", {}).get("glcm_scale")  # None = use native scale
    if glcm_scale is not None:
        glcm_scale = int(glcm_scale)

    # Batching settings to avoid GEE computation timeouts
    use_batching = bool(cfg.get("gee", {}).get("use_batching", True))
    batch_size = int(cfg.get("gee", {}).get("batch_size", 500))
    max_retries = int(cfg.get("gee", {}).get("max_retries", 3))

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
        .filterBounds(aoi_geometry)  # Use AOI geometry for efficient filtering
        # Keep only bands needed downstream (cuts median cost dramatically)
        .select(["B2", "B3", "B4", "B8", "B11", "SCL"])
    )

    cmax = cfg.get("gee", {}).get("cloudy_pixel_percentage_max")
    if cmax is not None:
        s2 = s2.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", float(cmax)))

    # Sentinel-2 mask via SCL: keep vegetation + bare + water
    def mask_s2(img: ee.Image) -> ee.Image:
        scl = img.select("SCL")
        keep = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(11))  # keep vegetation + bare + water + snow
        # After masking, drop SCL and any other unuse bands; preserve time metadata
        return (
            img.updateMask(keep)
            .select(["B2", "B3", "B4", "B8", "B11"])
            .copyProperties(img, img.propertyNames())
        )

    s2 = s2.map(mask_s2)

    def add_s2_indices(img: ee.Image) -> ee.Image:
        out = img
        if compute_ndvi:
            out = out.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI"))
        if compute_ndbi:
            out = out.addBands(img.normalizedDifference(["B11", "B8"]).rename("NDBI"))
        if compute_mndwi:
            out = out.addBands(img.normalizedDifference(["B3", "B11"]).rename("MNDWI"))
        return out

    if compute_ndvi or compute_ndbi or compute_mndwi:
        s2 = s2.map(add_s2_indices)

    
    # VIIRS monthly
    include_viirs = bool(cfg.get("features", {}).get("include_viirs", False))
    viirs = None
    if include_viirs:
        viirs = (
            ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
            .filterDate(start, end)
            .filterBounds(aoi_geometry)  # Use AOI geometry for efficient filtering
            .select(["avg_rad"], ["VIIRS_avg_rad"])
        )

    # --- Monthly composite + zonal stats ---
    ym_list = months_override if months_override else _ym_list(start, end)

    # Pre-compute batches ONCE before the loop (boundaries don't change per month)
    if use_batching:
        batches = _batch_feature_collection(fc, batch_size)
    else:
        batches = [fc]  # Single batch containing all features

    # Hoist scale_m outside inner functions to avoid repeated config lookups
    scale_m = int(cfg.get("gee", {}).get("scale_m", 10))

    # Parallel processing configuration
    max_parallel_months = int(cfg.get("gee", {}).get("max_parallel_months", 6))

    # Build selectors list (shared by all months)
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
    if compute_mndwi:
        selectors.append("MNDWI")
    if compute_glcm:
        selectors += [f"S2_{glcm_source_s2}_{m}" for m in glcm_metrics]

    # Package shared context for the month processing function
    month_context = {
        "s2": s2,
        "viirs": viirs,
        "fc": fc,
        "batches": batches,
        "aoi_geometry": aoi_geometry,
        "aoi_mode": aoi_mode,
        "unit_level": unit_level,
        "unit_id_field": unit_id_field,
        "scale_m": scale_m,
        "selectors": selectors,
        "asset_folder": asset_folder,
        "export_target": export_target,
        "max_retries": max_retries,
        "compute_ndvi": compute_ndvi,
        "compute_ndbi": compute_ndbi,
        "compute_mndwi": compute_mndwi,
        "compute_glcm": compute_glcm,
        "glcm_source_s2": glcm_source_s2,
        "glcm_size": glcm_size,
        "glcm_metrics": glcm_metrics,
        "glcm_scale": glcm_scale,
        "include_viirs": include_viirs,
        "skip_existing": skip_existing,
        "strict_no_empty": strict_no_empty,
        "cfg": cfg,
    }

    rows: List[pd.DataFrame] = []

    # Process months in parallel using ThreadPoolExecutor
    print(f"\nProcessing {len(ym_list)} months with {max_parallel_months} parallel workers...")

    with ThreadPoolExecutor(max_workers=max_parallel_months) as executor:
        future_to_ym = {
            executor.submit(_process_single_month, ym, month_context): ym
            for ym in ym_list
        }

        for future in as_completed(future_to_ym):
            ym = future_to_ym[future]
            try:
                df = future.result()
                if df is not None:
                    rows.append(df)
                print(f"[OK] Completed: {ym}")
            except Exception as e:
                print(f"[FAIL] Failed: {ym} - {e}")

    if not rows:
        raise RuntimeError("No monthly data was exported; check logs for issues.")

    out = pd.concat(rows, ignore_index=True)

    if unit_id_field not in out.columns:
        raise RuntimeError(
            f"Expected '{unit_id_field}' in exported table; ensure boundary FC has that property."
        )

    return out


def export_map_rasters(
    cfg: Dict[str, Any],
    output_dir: str,
    months: List[str] | None = None,
) -> None:
    """
    Export visualization-ready rasters for AOI_GOLDEN.

    This function exports NDVI and NDBI composites as GeoTIFFs for map figures.
    Rasters are clipped to the golden AOI for visualization purposes.

    Args:
        cfg: Configuration dictionary
        output_dir: Directory to save raster files
        months: List of year-month strings to export (None = use config months_override or first/last month)
    """
    # Initialize EE
    cloud_project = (cfg.get("gee", {}).get("cloud_project") or "").strip()
    if cloud_project and not cloud_project.startswith("YOUR_"):
        ee.Initialize(project=cloud_project)
    else:
        ee.Initialize()

    os.makedirs(output_dir, exist_ok=True)

    # Get golden AOI
    aoi_cfg = cfg.get("aoi", {})
    aoi_golden_asset = aoi_cfg.get("aoi_golden_asset_id")

    if not aoi_golden_asset:
        print("Warning: No AOI_GOLDEN asset configured. Skipping map raster export.")
        return

    try:
        aoi_fc = ee.FeatureCollection(aoi_golden_asset)
        aoi_bounds = aoi_fc.geometry().bounds()
        print(f"Loaded AOI_GOLDEN: {aoi_golden_asset}")
    except Exception as e:
        print(f"Error loading AOI_GOLDEN {aoi_golden_asset}: {e}")
        return

    # Determine months to export
    if months is None:
        months_override = cfg.get("run_mode", {}).get("months_override")
        if months_override:
            months = months_override
        else:
            # Default: first and last month from time range
            start = cfg["time"]["start"]
            end = cfg["time"]["end"]
            all_months = _ym_list(start, end)
            months = [all_months[0], all_months[-1]] if len(all_months) > 1 else all_months

    scale = cfg.get("features", {}).get("glcm_scale", 30)
    cmax = cfg.get("gee", {}).get("cloudy_pixel_percentage_max", 80)

    for ym in months:
        print(f"Exporting map rasters for {ym}...")

        # Parse year-month
        y, m = map(int, ym.split("-"))
        start_date = ee.Date.fromYMD(y, m, 1)
        end_date = start_date.advance(1, "month")

        # Get Sentinel-2 collection
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(aoi_bounds)
            .select(["B2", "B3", "B4", "B8", "B11", "SCL"])
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", float(cmax)))
        )

        # Apply cloud mask
        def mask_s2(img: ee.Image) -> ee.Image:
            scl = img.select("SCL")
            keep = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(11))
            return img.updateMask(keep).select(["B2", "B3", "B4", "B8", "B11"])

        s2 = s2.map(mask_s2)

        # Create median composite
        composite = s2.median()

        # Add NDVI and NDBI
        ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndbi = composite.normalizedDifference(["B11", "B8"]).rename("NDBI")

        # Export each band
        for band_name, band_img in [("NDVI", ndvi), ("NDBI", ndbi)]:
            out_path = os.path.join(output_dir, f"{band_name}_{ym}.tif")

            if os.path.exists(out_path):
                print(f"  Skipping {band_name} (already exists)")
                continue

            try:
                url = band_img.getDownloadURL({
                    "name": f"{band_name}_{ym}",
                    "scale": scale,
                    "region": aoi_bounds,
                    "format": "GEO_TIFF",
                })
                urllib.request.urlretrieve(url, out_path)
                print(f"  Exported: {out_path}")
            except Exception as e:
                print(f"  Failed to export {band_name}: {e}")

    print(f"Map raster export complete. Files in: {output_dir}")
