"""
Feature engineering for the preprocessing module.

Derives new features from existing demographic columns:
elderly ratio, aging index, youth ratio, household size.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def engineer_features(
    df: pd.DataFrame, cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute derived demographic features.

    Args:
        df: Cross-sectional DataFrame (one row per unit).
        cfg: Preprocessing configuration dict.

    Returns:
        DataFrame with additional derived columns.
    """
    print("Engineering derived features...")
    derived = cfg.get("derived_features", {})
    n_before = len(df.columns)

    if "elderly_ratio" in derived and "age_65_plus" in df.columns and "pop_total" in df.columns:
        df["elderly_ratio"] = df["age_65_plus"] / df["pop_total"].clip(lower=1)
        print(f"  elderly_ratio: mean={df['elderly_ratio'].mean():.4f}, "
              f"range=[{df['elderly_ratio'].min():.4f}, {df['elderly_ratio'].max():.4f}]")

    if "aging_index" in derived and "age_65_plus" in df.columns and "age_u15" in df.columns:
        # Clip age_u15 to min 1 to avoid division by zero
        df["aging_index"] = df["age_65_plus"] / df["age_u15"].clip(lower=1)
        print(f"  aging_index: mean={df['aging_index'].mean():.2f}")

    if "youth_ratio" in derived and "age_u15" in df.columns and "pop_total" in df.columns:
        df["youth_ratio"] = df["age_u15"] / df["pop_total"].clip(lower=1)
        print(f"  youth_ratio: mean={df['youth_ratio'].mean():.4f}")

    if "household_size" in derived and "pop_total" in df.columns and "households_total" in df.columns:
        df["household_size"] = df["pop_total"] / df["households_total"].clip(lower=1)
        print(f"  household_size: mean={df['household_size'].mean():.2f}")

    n_after = len(df.columns)
    print(f"  Added {n_after - n_before} derived features.")
    return df
