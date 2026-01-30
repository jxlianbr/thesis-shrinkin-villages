"""
Local GLCM texture computation for shrinking villages analysis.

This module computes GLCM (Gray Level Co-occurrence Matrix) texture features
locally using scikit-image, avoiding the expensive per-pixel computation in GEE.

The approach:
1. Download monthly NDBI raster tiles from GEE (clipped to study area)
2. For each unit boundary, extract pixels and compute GLCM texture
3. Merge GLCM features with the main features table

This is much faster than computing GLCM in GEE because:
- We only process pixels within unit boundaries (not entire image)
- Computation is parallelized locally
- Results can be cached
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    import ee
except ImportError:
    ee = None

try:
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    from rasterio.crs import CRS
except ImportError:
    rasterio = None

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    graycomatrix = None
    graycoprops = None


def _check_dependencies() -> None:
    """Check that required packages are installed."""
    missing = []
    if rasterio is None:
        missing.append("rasterio")
    if graycomatrix is None:
        missing.append("scikit-image")
    if missing:
        raise ImportError(
            f"Missing required packages for local GLCM: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


def _quantize_to_uint8(arr: np.ndarray, vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    """
    Quantize array to 8-bit unsigned integer for GLCM computation.

    Args:
        arr: Input array (e.g., NDBI values in range [-1, 1])
        vmin: Minimum expected value
        vmax: Maximum expected value

    Returns:
        Quantized array as uint8
    """
    # Clip to expected range
    arr = np.clip(arr, vmin, vmax)
    # Scale to 0-255
    scaled = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return scaled


def compute_glcm_metrics(
    arr: np.ndarray,
    metrics: List[str] = ["contrast", "homogeneity", "entropy"],
    distances: List[int] = [1],
    angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
) -> Dict[str, float]:
    """
    Compute GLCM texture metrics for a 2D array.

    Args:
        arr: 2D uint8 array
        metrics: List of metrics to compute (contrast, homogeneity, entropy, etc.)
        distances: Pixel distances for GLCM
        angles: Angles in radians for GLCM

    Returns:
        Dictionary of metric_name -> value (averaged across angles)
    """
    if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return {m: np.nan for m in metrics}

    # Handle masked arrays
    if np.ma.isMaskedArray(arr):
        if arr.mask.all():
            return {m: np.nan for m in metrics}
        arr = arr.filled(0)

    # Ensure we have valid data
    if np.isnan(arr).all() or (arr == 0).all():
        return {m: np.nan for m in metrics}

    try:
        # Compute GLCM
        glcm = graycomatrix(
            arr,
            distances=distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )

        results = {}
        for metric in metrics:
            if metric == "entropy":
                # Entropy is not in graycoprops, compute manually
                # Average across distances and angles
                entropy_vals = []
                for d_idx in range(len(distances)):
                    for a_idx in range(len(angles)):
                        p = glcm[:, :, d_idx, a_idx]
                        p = p[p > 0]  # Avoid log(0)
                        entropy_vals.append(-np.sum(p * np.log2(p)))
                results[metric] = np.mean(entropy_vals)
            else:
                # Use scikit-image's graycoprops
                vals = graycoprops(glcm, metric)
                results[metric] = np.mean(vals)  # Average across distances and angles

        return results
    except Exception:
        return {m: np.nan for m in metrics}


def _process_unit_glcm(
    unit_id: str,
    geometry,
    raster_path: str,
    metrics: List[str],
    src_band_idx: int = 0,
) -> Dict[str, Any]:
    """
    Process GLCM for a single unit geometry.

    Args:
        unit_id: Unit identifier
        geometry: Shapely geometry of the unit boundary
        raster_path: Path to the raster file
        metrics: List of GLCM metrics to compute
        src_band_idx: Band index to use (0-based)

    Returns:
        Dictionary with unit_id and GLCM metric values
    """
    result = {"unit_id": unit_id}

    try:
        with rasterio.open(raster_path) as src:
            # Mask raster to unit geometry
            out_image, out_transform = rasterio_mask(
                src,
                [geometry],
                crop=True,
                nodata=np.nan,
                filled=True
            )

            # Get the band we need
            band_data = out_image[src_band_idx]

            # Skip if too small or all nodata
            if band_data.size < 9 or np.isnan(band_data).all():
                for m in metrics:
                    result[f"S2_NDBI_{m}"] = np.nan
                return result

            # Quantize to uint8 (assuming NDBI in [-1, 1])
            valid_mask = ~np.isnan(band_data)
            band_uint8 = np.zeros_like(band_data, dtype=np.uint8)
            band_uint8[valid_mask] = _quantize_to_uint8(band_data[valid_mask])

            # Compute GLCM metrics
            glcm_results = compute_glcm_metrics(band_uint8, metrics=metrics)

            for m, v in glcm_results.items():
                result[f"S2_NDBI_{m}"] = v

    except Exception as e:
        # Return NaN for all metrics on error
        for m in metrics:
            result[f"S2_NDBI_{m}"] = np.nan

    return result


def download_monthly_raster_from_gee(
    cfg: Dict[str, Any],
    ym: str,
    band: str,
    out_path: str,
    scale: int = 30,
) -> str:
    """
    Download a monthly composite raster from GEE for local GLCM computation.

    Args:
        cfg: Configuration dictionary
        ym: Year-month string (e.g., "2020-01")
        band: Band name to download (e.g., "NDBI")
        out_path: Output file path
        scale: Resolution in meters

    Returns:
        Path to downloaded raster
    """
    if ee is None:
        raise ImportError("earthengine-api required for downloading rasters from GEE")

    # Initialize EE
    cloud_project = (cfg.get("gee", {}).get("cloud_project") or "").strip()
    if cloud_project and not cloud_project.startswith("YOUR_"):
        ee.Initialize(project=cloud_project)
    else:
        ee.Initialize()

    # Get study area bounds
    prefs = cfg.get("study_area", {}).get("prefectures", [])
    unit_level = cfg.get("run_mode", {}).get("unit_level", "mura")

    if unit_level == "mura":
        boundaries_asset = cfg["gee"]["boundaries_asset_id_mura"]
    else:
        boundaries_asset = cfg["gee"]["boundaries_asset_id_aza"]

    fc = ee.FeatureCollection(boundaries_asset)
    if prefs:
        fc = fc.filter(ee.Filter.inList("pref_name", prefs))

    bounds = fc.geometry().bounds()

    # Parse year-month
    y, m = map(int, ym.split("-"))
    start_date = ee.Date.fromYMD(y, m, 1)
    end_date = start_date.advance(1, "month")

    # Get Sentinel-2 monthly median composite
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(fc)
        .select(["B8", "B11"])  # Bands needed for NDBI
    )

    # Cloud masking via SCL would require SCL band, simplify for now
    cmax = cfg.get("gee", {}).get("cloudy_pixel_percentage_max", 80)
    s2 = s2.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", float(cmax)))

    # Compute NDBI
    def add_ndbi(img):
        ndbi = img.normalizedDifference(["B11", "B8"]).rename("NDBI")
        return img.addBands(ndbi)

    s2 = s2.map(add_ndbi)
    composite = s2.select(band).median()

    # Get download URL
    url = composite.getDownloadURL({
        "name": f"ndbi_{ym}",
        "scale": scale,
        "region": bounds,
        "format": "GEO_TIFF",
    })

    # Download
    import urllib.request
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    urllib.request.urlretrieve(url, out_path)

    return out_path


def run_local_glcm(
    cfg: Dict[str, Any],
    features_df: pd.DataFrame,
    boundaries_path: str,
    rasters_dir: str,
    months: Optional[List[str]] = None,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Compute GLCM texture features locally and merge with features table.

    Args:
        cfg: Configuration dictionary
        features_df: DataFrame with unit-level features (from GEE export)
        boundaries_path: Path to boundaries GeoPackage
        rasters_dir: Directory for downloaded/cached rasters
        months: List of year-month strings to process (or None for all in features_df)
        max_workers: Number of parallel workers

    Returns:
        Updated features_df with GLCM columns added
    """
    _check_dependencies()

    # Get GLCM settings from config
    metrics = cfg.get("features", {}).get("glcm_metrics", ["contrast", "entropy", "homogeneity"])
    glcm_scale = cfg.get("features", {}).get("glcm_scale", 30)
    unit_level = cfg.get("run_mode", {}).get("unit_level", "mura")

    # Load boundaries
    boundaries = gpd.read_file(boundaries_path)

    # Filter by prefectures if specified
    prefs = cfg.get("study_area", {}).get("prefectures", [])
    if prefs and "pref_name" in boundaries.columns:
        boundaries = boundaries[boundaries["pref_name"].isin(prefs)]

    # Get months to process
    if months is None:
        months = features_df["month"].unique().tolist() if "month" in features_df.columns else []

    if not months:
        print("No months to process for GLCM")
        return features_df

    os.makedirs(rasters_dir, exist_ok=True)

    all_glcm_results = []

    for ym in months:
        print(f"Processing GLCM for {ym}...")

        # Check for cached raster or download
        raster_path = os.path.join(rasters_dir, f"ndbi_{ym}.tif")

        if not os.path.exists(raster_path):
            print(f"  Downloading NDBI raster for {ym}...")
            try:
                download_monthly_raster_from_gee(
                    cfg, ym, "NDBI", raster_path, scale=glcm_scale
                )
            except Exception as e:
                print(f"  Failed to download raster for {ym}: {e}")
                continue

        # Process each unit
        unit_results = []

        # Use parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, row in boundaries.iterrows():
                unit_id = row.get("unit_id", f"unit_{idx}")
                geom = row.geometry
                future = executor.submit(
                    _process_unit_glcm,
                    unit_id,
                    geom,
                    raster_path,
                    metrics,
                )
                futures[future] = unit_id

            for future in as_completed(futures):
                result = future.result()
                result["month"] = ym
                unit_results.append(result)

        all_glcm_results.extend(unit_results)
        print(f"  Processed {len(unit_results)} units for {ym}")

    if not all_glcm_results:
        print("No GLCM results computed")
        return features_df

    # Create GLCM DataFrame
    glcm_df = pd.DataFrame(all_glcm_results)

    # Merge with features_df
    glcm_cols = [c for c in glcm_df.columns if c.startswith("S2_NDBI_")]
    merge_cols = ["unit_id", "month"] + glcm_cols

    result_df = features_df.merge(
        glcm_df[merge_cols],
        on=["unit_id", "month"],
        how="left",
    )

    print(f"Added GLCM columns: {glcm_cols}")

    return result_df


if __name__ == "__main__":
    # Test/example usage
    import yaml

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Load existing features
    features_path = cfg["outputs"]["features_table_csv"]
    if os.path.exists(features_path):
        features_df = pd.read_csv(features_path)

        result = run_local_glcm(
            cfg=cfg,
            features_df=features_df,
            boundaries_path=cfg["data"]["boundaries_path"],
            rasters_dir="outputs/rasters",
        )

        # Save updated features
        result.to_csv(features_path, index=False)
        print(f"Updated {features_path} with GLCM features")
    else:
        print(f"Features file not found: {features_path}")
