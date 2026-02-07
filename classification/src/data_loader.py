"""
Data loading for the classification module.

Reads classification_ready.parquet and splits it into feature matrix X,
target vector y, and identifier columns.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def load_classification_data(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load the preprocessed dataset and separate features, target, identifiers.

    Args:
        cfg: Configuration dict.

    Returns:
        Dict with keys:
            df            – full DataFrame (65 × 34)
            X             – pd.DataFrame of numeric features (65 × 30)
            y             – pd.Series of target codes (int 0/1/2)
            y_labels      – pd.Series of target labels (str)
            identifiers   – pd.DataFrame (unit_id, pref_name)
            feature_names – list[str] of feature column names
            n_samples     – int
            n_features    – int
            class_distribution – Dict[str, int]
    """
    path = cfg["data"]["input_path"]
    print(f"Loading classification data from {path} ...")
    df = pd.read_parquet(path)
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Separate columns
    id_cols = cfg["columns"]["identifiers"]
    target_col = cfg["columns"]["target"]
    target_code_col = cfg["columns"]["target_code"]

    identifiers = df[id_cols].copy()
    y_labels = df[target_col].copy()
    y = df[target_code_col].copy()

    exclude = set(id_cols + [target_col, target_code_col])
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()

    # Class distribution
    class_dist = y_labels.value_counts().to_dict()
    print(f"  Features: {len(feature_cols)}")
    print(f"  Class distribution:")
    for label in cfg["columns"]["class_labels"]:
        count = class_dist.get(label, 0)
        pct = count / len(y) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")

    return {
        "df": df,
        "X": X,
        "y": y,
        "y_labels": y_labels,
        "identifiers": identifiers,
        "feature_names": feature_cols,
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "class_distribution": class_dist,
    }
