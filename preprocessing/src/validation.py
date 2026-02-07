"""
Output validation for the preprocessing module.

Checks the final dataset for correctness: shape, NaN, class balance,
constant features, etc.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def validate_output(
    df: pd.DataFrame, cfg: Dict[str, Any],
) -> list[str]:
    """
    Validate the final classification-ready DataFrame.

    Args:
        df: Final preprocessed DataFrame.
        cfg: Preprocessing configuration dict.

    Returns:
        List of warning/error messages (empty if all checks pass).
    """
    print("Validating output...")
    warnings: list[str] = []
    identifiers = cfg.get("identifiers", [])
    target_cols = [cfg["target"]["name"], "shrinkage_code"]

    # --- Row count ---
    if len(df) != df["unit_id"].nunique():
        warnings.append(f"Row count ({len(df)}) != unique unit count "
                        f"({df['unit_id'].nunique()})")
    print(f"  Rows: {len(df)}")

    # --- NaN check ---
    feature_cols = [c for c in df.columns if c not in identifiers]
    nan_counts = df[feature_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        warnings.append(f"NaN found in {len(nan_cols)} columns: "
                        f"{nan_cols.to_dict()}")
        print(f"  WARNING: NaN in {len(nan_cols)} columns")
    else:
        print(f"  NaN check: PASS (no missing values)")

    # --- Target check ---
    target_name = cfg["target"]["name"]
    if target_name not in df.columns:
        warnings.append(f"Target column '{target_name}' not found")
    else:
        n_classes = df[target_name].nunique()
        expected_labels = cfg["target"]["labels"]
        actual_labels = sorted(df[target_name].unique())
        if n_classes != len(expected_labels):
            warnings.append(f"Expected {len(expected_labels)} classes, "
                            f"found {n_classes}: {actual_labels}")
        print(f"  Target classes: {n_classes} ({actual_labels})")

    # --- Constant feature check ---
    numeric_cols = [c for c in df.columns
                    if c not in set(identifiers + target_cols)
                    and pd.api.types.is_numeric_dtype(df[c])]
    constant = [c for c in numeric_cols if df[c].std() == 0]
    if constant:
        warnings.append(f"Constant features (zero variance): {constant}")
        print(f"  WARNING: {len(constant)} constant features: {constant}")
    else:
        print(f"  Constant feature check: PASS")

    # --- Feature count ---
    n_features = len(numeric_cols)
    print(f"  Feature columns: {n_features}")
    print(f"  Total columns: {len(df.columns)}")

    if warnings:
        print(f"  WARNINGS: {len(warnings)}")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(f"  All validation checks PASSED.")

    return warnings
