"""
Target variable builder for the preprocessing module.

Creates the 3-class shrinkage label from elderly ratio thresholds.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def build_target(
    df: pd.DataFrame, cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Create the shrinkage classification target from elderly_ratio.

    Thresholds are configurable in YAML:
    - elderly_ratio < stable_threshold -> "stable" (0)
    - stable_threshold <= elderly_ratio < shrinking_threshold -> "shrinking" (1)
    - elderly_ratio >= shrinking_threshold -> "severely_shrinking" (2)

    Args:
        df: Cross-sectional DataFrame with elderly_ratio column.
        cfg: Preprocessing configuration dict.

    Returns:
        DataFrame with shrinkage_class (str) and shrinkage_code (int) columns.
    """
    print("Building target variable...")
    target_cfg = cfg["target"]
    thresholds = target_cfg["thresholds"]
    labels = target_cfg["labels"]
    col_name = target_cfg["name"]

    stable_thresh = thresholds["stable"]
    shrinking_thresh = thresholds["shrinking"]

    if "elderly_ratio" not in df.columns:
        raise ValueError("elderly_ratio column required but not found. "
                         "Run feature_engineering first.")

    conditions = [
        df["elderly_ratio"] < stable_thresh,
        (df["elderly_ratio"] >= stable_thresh) & (df["elderly_ratio"] < shrinking_thresh),
        df["elderly_ratio"] >= shrinking_thresh,
    ]
    codes = [0, 1, 2]

    df[col_name] = np.select(conditions, labels, default=labels[-1])
    df["shrinkage_code"] = np.select(conditions, codes, default=codes[-1])

    # Print class distribution
    print(f"  Thresholds: stable < {stable_thresh}, "
          f"shrinking [{stable_thresh}, {shrinking_thresh}), "
          f"severely_shrinking >= {shrinking_thresh}")
    print(f"  Class distribution:")
    for label, code in zip(labels, codes):
        count = int((df["shrinkage_code"] == code).sum())
        pct = count / len(df) * 100
        print(f"    {label} ({code}): {count} units ({pct:.1f}%)")
        if count < 10:
            print(f"    WARNING: Class '{label}' has fewer than 10 samples. "
                  f"Consider adjusting thresholds in config.")

    return df
