"""
Spatial analysis for the EDA module.

Produces choropleth maps of key features using the municipality boundaries
GeoPackage. Falls back gracefully if boundaries file is unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda.src.utils import get_existing_columns, save_figure


def run_spatial_analysis(
    df: pd.DataFrame, cfg: Dict[str, Any], output_dir: str,
) -> Dict[str, Any]:
    """
    Create choropleth maps of feature means per unit.

    Requires geopandas and a boundaries GeoPackage.

    Outputs:
        figures/spatial_ndvi_mean.png
        figures/spatial_pop_total.png
        figures/spatial_viirs_mean.png

    Returns:
        Summary dict with per-prefecture spatial statistics.
    """
    print("Running spatial analysis...")

    boundaries_path = cfg["data"].get("boundaries_path", "")
    if not boundaries_path or not Path(boundaries_path).exists():
        print(f"  WARNING: Boundaries file not found: {boundaries_path}")
        print("  Skipping spatial analysis (choropleth maps).")
        return {"skipped": True, "reason": f"boundaries not found: {boundaries_path}"}

    try:
        import geopandas as gpd
    except ImportError:
        print("  WARNING: geopandas not installed, skipping spatial analysis.")
        return {"skipped": True, "reason": "geopandas not installed"}

    # Load boundaries
    gdf = gpd.read_file(boundaries_path)
    print(f"  Loaded {len(gdf)} unit boundaries.")

    # Compute per-unit means (aggregate across months)
    unit_means = df.groupby("unit_id").mean(numeric_only=True).reset_index()

    # Merge with boundaries
    merged = gdf.merge(unit_means, on="unit_id", how="left")

    # Plot choropleths for key features
    feature_map = {
        "NDVI": ("Mean NDVI", "YlGn", "spatial_ndvi_mean"),
        "viirs_mean": ("Mean VIIRS Night Lights", "YlOrRd", "spatial_viirs_mean"),
        "pop_total": ("Total Population", "Blues", "spatial_pop_total"),
    }

    for col, (title, cmap, fname) in feature_map.items():
        if col in merged.columns:
            _plot_choropleth(merged, col, title, fname, cmap, cfg)

    # Per-prefecture summary
    pref_summary = {}
    if "pref_name" in df.columns:
        for pref, grp in df.groupby("pref_name"):
            pref_summary[pref] = {
                "n_units": int(grp["unit_id"].nunique()),
                "mean_ndvi": round(float(grp["NDVI"].mean()), 4) if "NDVI" in grp else None,
                "mean_viirs": round(float(grp["viirs_mean"].mean()), 4) if "viirs_mean" in grp else None,
                "mean_pop": round(float(grp["pop_total"].mean()), 1) if "pop_total" in grp else None,
            }

    summary = {
        "n_units_mapped": int(len(merged)),
        "by_prefecture": pref_summary,
    }
    print(f"  Mapped {len(merged)} units.")
    return summary


def _plot_choropleth(
    gdf: "gpd.GeoDataFrame",
    column: str,
    title: str,
    filename: str,
    cmap: str,
    cfg: Dict[str, Any],
) -> None:
    """Plot a single choropleth map."""
    fig, ax = plt.subplots(1, 1, figsize=cfg["plot"]["figsize_single"])
    gdf.plot(
        column=column,
        cmap=cmap,
        legend=True,
        legend_kwds={"label": column, "shrink": 0.7},
        ax=ax,
        edgecolor="gray",
        linewidth=0.3,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    save_figure(fig, filename, cfg)
