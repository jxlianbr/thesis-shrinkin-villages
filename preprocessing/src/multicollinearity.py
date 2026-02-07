"""
Multicollinearity resolution for the preprocessing module.

Drops highly correlated features based on EDA findings,
keeping one representative per correlated cluster.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def resolve_multicollinearity(
    df: pd.DataFrame, cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Drop highly correlated features defined in config.

    For temporal-aggregated features, drops both _mean and _std variants.

    Args:
        df: Cross-sectional DataFrame.
        cfg: Preprocessing configuration dict.

    Returns:
        DataFrame with correlated features removed.
    """
    print("Resolving multicollinearity...")
    mc_cfg = cfg["multicollinearity"]
    n_before = len(df.columns)

    to_drop: list[str] = []

    # Static demographic features to drop directly
    for col in mc_cfg.get("drop_correlated", []):
        if col in df.columns:
            to_drop.append(col)

    # Spectral features: drop _mean and _std variants
    for col in mc_cfg.get("drop_spectral_correlated", []):
        for suffix in ("_mean", "_std"):
            full_name = f"{col}{suffix}"
            if full_name in df.columns:
                to_drop.append(full_name)

    # OSM features to drop directly
    for col in mc_cfg.get("drop_osm_correlated", []):
        if col in df.columns:
            to_drop.append(col)

    # Texture features: drop _mean and _std variants
    for col in mc_cfg.get("drop_texture_correlated", []):
        for suffix in ("_mean", "_std"):
            full_name = f"{col}{suffix}"
            if full_name in df.columns:
                to_drop.append(full_name)

    if to_drop:
        df = df.drop(columns=to_drop)

    n_after = len(df.columns)
    print(f"  Dropped {n_before - n_after} correlated columns:")
    for col in sorted(to_drop):
        print(f"    - {col}")
    print(f"  Remaining: {n_after} columns")
    return df
