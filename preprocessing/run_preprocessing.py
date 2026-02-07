"""
Main entry point for the preprocessing pipeline.

Transforms the panel-format feature table (65 units x 72 months)
into a classification-ready cross-sectional dataset (65 rows).

Usage:
    python preprocessing/run_preprocessing.py
    python preprocessing/run_preprocessing.py --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from preprocessing.src.utils import load_config, ensure_output_dirs, write_json
from preprocessing.src.data_loader import load_features_table
from preprocessing.src.feature_dropper import drop_unusable_features
from preprocessing.src.temporal_aggregation import aggregate_to_cross_section
from preprocessing.src.feature_engineering import engineer_features
from preprocessing.src.target_builder import build_target
from preprocessing.src.multicollinearity import resolve_multicollinearity
from preprocessing.src.transformer import transform_features
from preprocessing.src.validation import validate_output


def main(config_path: str = "preprocessing/config/preprocessing_config.yaml") -> None:
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("  Preprocessing Pipeline — Shrinking Villages")
    print("=" * 60)

    cfg = load_config(config_path)
    ensure_output_dirs(cfg)

    report: dict = {"steps": []}

    # Step 1: Load
    print("\n--- Step 1: Load Data ---")
    df = load_features_table(cfg)
    report["steps"].append({
        "step": "load", "rows": len(df), "columns": len(df.columns),
    })

    # Step 2: Drop unusable features
    print("\n--- Step 2: Drop Unusable Features ---")
    df = drop_unusable_features(df, cfg)
    report["steps"].append({
        "step": "drop_features", "columns_remaining": len(df.columns),
    })

    # Step 3: Temporal aggregation (panel -> cross-section)
    print("\n--- Step 3: Temporal Aggregation ---")
    df = aggregate_to_cross_section(df, cfg)
    report["steps"].append({
        "step": "temporal_aggregation",
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
    })

    # Step 4: Feature engineering
    print("\n--- Step 4: Feature Engineering ---")
    df = engineer_features(df, cfg)
    report["steps"].append({
        "step": "feature_engineering", "columns": len(df.columns),
    })

    # Step 5: Build target variable
    print("\n--- Step 5: Build Target ---")
    df = build_target(df, cfg)
    target_name = cfg["target"]["name"]
    report["steps"].append({
        "step": "build_target",
        "class_distribution": df[target_name].value_counts().to_dict(),
    })

    # Step 6: Resolve multicollinearity
    print("\n--- Step 6: Resolve Multicollinearity ---")
    df = resolve_multicollinearity(df, cfg)
    report["steps"].append({
        "step": "multicollinearity", "columns_remaining": len(df.columns),
    })

    # Step 7: Transform & scale
    print("\n--- Step 7: Transform & Scale ---")
    df, transform_meta = transform_features(df, cfg)
    report["steps"].append({
        "step": "transform",
        "log1p_applied": transform_meta["log1p_features"],
        "scaler": transform_meta["scaler_type"],
        "n_scaled_features": transform_meta["n_features"],
    })

    # Step 8: Validate
    print("\n--- Step 8: Validate ---")
    warnings = validate_output(df, cfg)
    report["steps"].append({
        "step": "validation", "warnings": warnings,
    })

    # Step 9: Export
    print("\n--- Step 9: Export ---")
    _export(df, transform_meta, report, cfg)

    print("\n" + "=" * 60)
    print("  Preprocessing complete.")
    print(f"  Output: {cfg['output']['parquet_path']}")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("=" * 60)


def _export(
    df: pd.DataFrame,
    transform_meta: dict,
    report: dict,
    cfg: dict,
) -> None:
    """Save all outputs: parquet, CSV, JSON report, feature metadata."""
    out = cfg["output"]

    # Parquet
    df.to_parquet(out["parquet_path"], index=False)
    print(f"  Saved: {out['parquet_path']}")

    # CSV
    df.to_csv(out["csv_path"], index=False)
    print(f"  Saved: {out['csv_path']}")

    # Preprocessing report JSON
    report["final_shape"] = {"rows": len(df), "columns": len(df.columns)}
    report["final_columns"] = list(df.columns)
    report["transform_metadata"] = transform_meta
    write_json(out["report_path"], report)
    print(f"  Saved: {out['report_path']}")

    # Feature metadata CSV
    identifiers = set(cfg.get("identifiers", []))
    target_cols = {cfg["target"]["name"], "shrinkage_code"}
    log1p_set = set(transform_meta.get("log1p_features", []))

    meta_rows = []
    for col in df.columns:
        role = "identifier" if col in identifiers else (
            "target" if col in target_cols else "feature"
        )
        meta_rows.append({
            "column": col,
            "role": role,
            "dtype": str(df[col].dtype),
            "log1p_applied": col in log1p_set,
            "scaled": role == "feature" and pd.api.types.is_numeric_dtype(df[col]),
        })

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(out["metadata_path"], index=False)
    print(f"  Saved: {out['metadata_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess features table for classification",
    )
    parser.add_argument(
        "--config",
        default="preprocessing/config/preprocessing_config.yaml",
        help="Path to preprocessing config YAML",
    )
    args = parser.parse_args()
    main(args.config)
