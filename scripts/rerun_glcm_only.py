"""
Re-run GLCM computation on an existing features table.

This script loads an existing features table (from a previous pipeline run)
and re-computes only the GLCM texture features, then saves the updated table.

Usage:
    python scripts/rerun_glcm_only.py [config_path] [features_parquet_path]

Examples:
    # Use defaults from config
    python scripts/rerun_glcm_only.py config/config.yaml

    # Specify features file explicitly
    python scripts/rerun_glcm_only.py config/config.yaml outputs/final/features_table.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path so we can import data_preprocessing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config/config.yaml", features_path: str = None) -> None:
    cfg = load_config(config_path)

    # Determine features file path
    if features_path is None:
        features_path = cfg["outputs"]["features_table_parquet"]

    if not Path(features_path).exists():
        print(f"Error: Features file not found: {features_path}")
        sys.exit(1)

    print(f"Loading features from: {features_path}")
    features_df = pd.read_parquet(features_path)
    print(f"Loaded {len(features_df)} rows")

    # Check existing GLCM columns
    glcm_cols = [c for c in features_df.columns if c.startswith("S2_NDBI_")]
    if glcm_cols:
        print(f"Existing GLCM columns: {glcm_cols}")
        print("Dropping existing GLCM columns before recomputation...")
        features_df = features_df.drop(columns=glcm_cols)

    # Import and run GLCM
    from data_preprocessing.compute_glcm_local import run_local_glcm

    boundaries_path = cfg["data"]["boundaries_path"]
    out_dir = cfg["project"]["outputs_dir"]
    rasters_dir = f"{out_dir}/rasters"

    print(f"\nRecomputing GLCM texture features...")
    print(f"  Boundaries: {boundaries_path}")
    print(f"  Rasters dir: {rasters_dir}")

    features_df = run_local_glcm(
        cfg=cfg,
        features_df=features_df,
        boundaries_path=boundaries_path,
        rasters_dir=rasters_dir,
    )

    # Check new GLCM columns
    new_glcm_cols = [c for c in features_df.columns if c.startswith("S2_NDBI_")]
    print(f"\nNew GLCM columns: {new_glcm_cols}")

    # Save updated table
    csv_path = cfg["outputs"]["features_table_csv"]
    parquet_path = cfg["outputs"]["features_table_parquet"]

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    features_df.to_parquet(parquet_path, index=False)
    features_df.to_csv(csv_path, index=False)

    print(f"\nSaved updated features table:")
    print(f"  {parquet_path}")
    print(f"  {csv_path}")

    # Show GLCM stats
    for col in new_glcm_cols:
        non_null = features_df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(features_df)} non-null values")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    features_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(config_path, features_path)
