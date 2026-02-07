"""
Feature transformation for the preprocessing module.

Applies log1p transform to skewed features and
RobustScaler to all numeric features.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def transform_features(
    df: pd.DataFrame, cfg: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply log1p transforms and scaling to numeric features.

    For temporal-aggregated features, log1p is applied to the _mean variant.
    Identifiers and target columns are not transformed.

    Args:
        df: Cross-sectional DataFrame.
        cfg: Preprocessing configuration dict.

    Returns:
        Tuple of (transformed DataFrame, metadata dict with transform details).
    """
    print("Transforming features...")
    transform_cfg = cfg["transform"]
    identifiers = cfg.get("identifiers", [])
    target_cols = [cfg["target"]["name"], "shrinkage_code"]

    # Identify numeric feature columns (exclude identifiers and target)
    exclude = set(identifiers + target_cols)
    feature_cols = [c for c in df.columns if c not in exclude
                    and pd.api.types.is_numeric_dtype(df[c])]

    # --- Step 1: log1p transform ---
    log1p_base = transform_cfg.get("log1p_features", [])
    log1p_applied: list[str] = []

    for base_name in log1p_base:
        # Check for _mean variant (temporal-aggregated) or raw name (static)
        candidates = [f"{base_name}_mean", base_name]
        for col in candidates:
            if col in feature_cols and col in df.columns:
                # Ensure non-negative before log1p
                min_val = df[col].min()
                if min_val < 0:
                    # Shift to make non-negative
                    df[col] = df[col] - min_val
                df[col] = np.log1p(df[col])
                log1p_applied.append(col)
                break

    print(f"  log1p applied to {len(log1p_applied)} features: {log1p_applied}")

    # --- Step 2: Scaling ---
    scaler_type = transform_cfg.get("scaler", "robust")
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    # Scale only numeric feature columns
    scaled_cols = [c for c in feature_cols if c in df.columns]
    if scaled_cols:
        df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

    print(f"  {scaler_type.capitalize()}Scaler applied to {len(scaled_cols)} features.")

    # --- Build metadata ---
    metadata: Dict[str, Any] = {
        "scaler_type": scaler_type,
        "log1p_features": log1p_applied,
        "scaled_features": scaled_cols,
        "n_features": len(scaled_cols),
    }

    # Store scaler parameters for reproducibility
    if hasattr(scaler, "center_"):
        metadata["scaler_center"] = {
            col: round(float(v), 6)
            for col, v in zip(scaled_cols, scaler.center_)
        }
        metadata["scaler_scale"] = {
            col: round(float(v), 6)
            for col, v in zip(scaled_cols, scaler.scale_)
        }

    return df, metadata
