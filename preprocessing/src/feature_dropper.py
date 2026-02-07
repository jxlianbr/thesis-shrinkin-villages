"""
Feature dropper for the preprocessing module.

Removes meta-only columns and other unusable features
before aggregation.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def drop_unusable_features(
    df: pd.DataFrame, cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Drop meta-only and other unusable columns.

    Args:
        df: Raw features DataFrame (panel format).
        cfg: Preprocessing configuration dict.

    Returns:
        DataFrame with unusable columns removed.
    """
    print("Dropping unusable features...")
    n_before = len(df.columns)

    to_drop: list[str] = []
    for group_key, cols in cfg["drop"].items():
        existing = [c for c in cols if c in df.columns]
        to_drop.extend(existing)

    if to_drop:
        df = df.drop(columns=to_drop)

    n_after = len(df.columns)
    print(f"  Dropped {n_before - n_after} columns: {to_drop}")
    print(f"  Remaining: {n_after} columns")
    return df
