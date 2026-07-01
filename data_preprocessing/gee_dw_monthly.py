"""
Monthly Dynamic World bare-fraction extraction.

Computes per-aza mean probability of the 'bare' land-cover class from
GOOGLE/DYNAMICWORLD/V1 for every calendar month in the study window.
Output is a panel: one row per (unit_id, month).

The static dw_bare_frac column that gee_lulc.py broadcasts to every month
will be overwritten by these values in pipeline.py step 4e, so
temporal_aggregation can compute a genuine dw_bare_frac_slope per aza.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, List

import ee
import pandas as pd

from data_preprocessing.gee_monthly import (
    _export_fc_to_asset,
    _download_table_asset_csv,
    _retry_with_backoff,
)

# DW V1 availability starts 2015-06-27 (tied to Sentinel-2)
_DW_START = "2015-06"


def _month_range(start: str, end: str) -> List[str]:
    """Return 'YYYY-MM' strings from start to end (inclusive)."""
    d   = date.fromisoformat(start[:10]).replace(day=1)
    end_d = date.fromisoformat(end[:10]).replace(day=1)
    months: List[str] = []
    while d <= end_d:
        months.append(d.strftime("%Y-%m"))
        d = d.replace(month=1, year=d.year + 1) if d.month == 12 \
            else d.replace(month=d.month + 1)
    return months


def run_dw_monthly_extraction(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract monthly DW bare fraction per aza for every month in the study window.

    Parameters
    ----------
    cfg : project config dict (same object passed to all pipeline steps)

    Returns
    -------
    pd.DataFrame  columns: unit_id, month, dw_bare_frac  (one row per aza per month)
        Months with no DW imagery are absent (not filled with NaN rows).
    """
    # --- GEE init ---
    cloud_project = (cfg.get("gee", {}).get("cloud_project") or "").strip()
    if cloud_project and not cloud_project.startswith("YOUR_"):
        ee.Initialize(project=cloud_project)
    else:
        ee.Initialize()

    unit_level    = cfg.get("run_mode", {}).get("unit_level", "mura")
    unit_id_field = cfg["data"]["unit_id_field"]
    asset_key     = f"boundaries_asset_id_{unit_level}"
    fc = ee.FeatureCollection(cfg["gee"][asset_key])

    prefs = cfg.get("study_area", {}).get("prefectures", [])
    if prefs:
        fc = fc.filter(ee.Filter.inList("pref_name", prefs))
    sample_n = int(cfg.get("run_mode", {}).get("unit_sample_n") or 0)
    if sample_n > 0:
        fc = fc.sort(unit_id_field).limit(sample_n)

    out_dir    = cfg.get("project", {}).get("outputs_dir", "outputs")
    cache_dir  = os.path.join(out_dir, "gee", "dw_monthly")
    os.makedirs(cache_dir, exist_ok=True)

    asset_folder = (cfg.get("gee", {}).get("export_asset_folder") or "").rstrip("/")
    max_retries  = int(cfg.get("gee", {}).get("max_retries", 3))
    lulc_scale   = int(cfg.get("features", {}).get("lulc_scale", 100))
    skip_existing = bool(cfg.get("run_mode", {}).get("skip_existing_month_csv", True))

    months = [m for m in _month_range(cfg["time"]["start"], cfg["time"]["end"])
              if m >= _DW_START]

    monthly_dfs: List[pd.DataFrame] = []
    n_skipped = 0

    print(f"  [dw_monthly] {len(months)} months to process "
          f"({unit_level}, scale={lulc_scale}m) ...")

    for ym in months:
        csv_path = os.path.join(cache_dir, f"dw_{unit_level}_{ym}.csv")

        if skip_existing and os.path.exists(csv_path):
            df_m = pd.read_csv(csv_path, dtype={unit_id_field: "string"})
            if not df_m.empty:
                monthly_dfs.append(df_m)
            n_skipped += 1
            continue

        yr, mo  = ym.split("-")
        m_start = f"{ym}-01"
        m_end   = (f"{int(yr)+1}-01-01" if int(mo) == 12
                   else f"{yr}-{int(mo)+1:02d}-01")

        dw_col = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterDate(m_start, m_end)
            .select(["bare"])
        )

        try:
            n_imgs = dw_col.size().getInfo()
        except Exception as exc:
            print(f"  [dw_monthly] {ym}: image count failed ({exc}), skipping")
            continue

        if n_imgs == 0:
            print(f"  [dw_monthly] {ym}: 0 DW images, skipped")
            continue

        print(f"  [dw_monthly] {ym}: {n_imgs} images -> reduceRegions ...")

        dw_mean = dw_col.mean()   # mean bare probability over all images that month
        reduced = dw_mean.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean().setOutputs(["bare"]),
            scale=lulc_scale,
        ).map(lambda f: ee.Feature(f).set({"month": ym}))

        description  = f"dw_monthly_{unit_level}_{ym}"
        temp_asset   = f"{asset_folder}/temp_{description}"

        def _do_export(red=reduced, aid=temp_asset, desc=description):
            _export_fc_to_asset(red, asset_id=aid, description=f"temp_{desc}")

        _retry_with_backoff(_do_export, max_retries=max_retries)

        raw_csv = csv_path + ".raw"
        _download_table_asset_csv(
            cfg={}, asset_id=temp_asset, out_csv_path=raw_csv,
            selectors=[unit_id_field, "month", "bare"],
            max_retries=max_retries,
        )

        try:
            ee.data.deleteAsset(temp_asset)
        except Exception:
            pass

        df_m = pd.read_csv(raw_csv, dtype={unit_id_field: "string"})
        df_m = df_m.rename(columns={"bare": "dw_bare_frac"})
        df_m.to_csv(csv_path, index=False)
        os.remove(raw_csv)

        monthly_dfs.append(df_m)
        print(f"  [dw_monthly] {ym}: {len(df_m)} rows saved")

    if n_skipped:
        print(f"  [dw_monthly] {n_skipped}/{len(months)} months loaded from cache")

    if not monthly_dfs:
        print("  [dw_monthly] WARNING: no data returned")
        return pd.DataFrame(columns=[unit_id_field, "month", "dw_bare_frac"])

    panel = (
        pd.concat(monthly_dfs, ignore_index=True)
        [[unit_id_field, "month", "dw_bare_frac"]]
        .assign(**{
            unit_id_field:   lambda d: d[unit_id_field].astype(str).str.strip(),
            "month":          lambda d: d["month"].astype(str).str.strip(),
            "dw_bare_frac":   lambda d: pd.to_numeric(d["dw_bare_frac"], errors="coerce"),
        })
    )
    valid = panel["dw_bare_frac"].notna().sum()
    print(f"  [dw_monthly] Panel: {len(panel):,} rows, "
          f"{valid:,} non-null ({valid / max(len(panel), 1) * 100:.1f}%)")
    return panel
