"""
Main entry point for the EDA pipeline.

Usage:
    python eda/run_eda.py
    python eda/run_eda.py --config path/to/eda_config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'eda.src' is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from eda.src.utils import ensure_output_dirs, load_eda_config, setup_plot_style
from eda.src.data_loader import load_features_table, validate_schema
from eda.src.summary_stats import run_summary_stats
from eda.src.missing_data import run_missing_analysis
from eda.src.distributions import run_distribution_analysis
from eda.src.correlations import run_correlation_analysis
from eda.src.temporal_analysis import run_temporal_analysis
from eda.src.spatial_analysis import run_spatial_analysis
from eda.src.outlier_detection import run_outlier_detection
from eda.src.feature_relationships import run_feature_relationships
from eda.src.report_generator import generate_report


def main(config_path: str = "eda/config/eda_config.yaml") -> None:
    """Run the full EDA pipeline."""
    print("=" * 60)
    print("  EDA Pipeline — Shrinking Villages Feature Table")
    print("=" * 60)

    cfg = load_eda_config(config_path)
    ensure_output_dirs(cfg)
    setup_plot_style(cfg)

    output_dir = cfg["output"]["base_dir"]

    # Load and validate
    print("\n--- Data Loading ---")
    df = load_features_table(cfg)
    warnings = validate_schema(df, cfg)

    # Run analyses
    summary: dict = {"schema_warnings": warnings}

    print("\n--- Summary Statistics ---")
    summary["summary_stats"] = run_summary_stats(df, cfg, output_dir)

    print("\n--- Missing Data Analysis ---")
    summary["missing_data"] = run_missing_analysis(df, cfg, output_dir)

    print("\n--- Distribution Analysis ---")
    summary["distributions"] = run_distribution_analysis(df, cfg, output_dir)

    print("\n--- Correlation Analysis ---")
    summary["correlations"] = run_correlation_analysis(df, cfg, output_dir)

    print("\n--- Temporal Analysis ---")
    summary["temporal"] = run_temporal_analysis(df, cfg, output_dir)

    print("\n--- Spatial Analysis ---")
    summary["spatial"] = run_spatial_analysis(df, cfg, output_dir)

    print("\n--- Outlier Detection ---")
    summary["outliers"] = run_outlier_detection(df, cfg, output_dir)

    print("\n--- Feature Relationships ---")
    summary["feature_relationships"] = run_feature_relationships(df, cfg, output_dir)

    # Generate report
    print("\n--- Report Generation ---")
    generate_report(summary, cfg, output_dir)

    print("\n" + "=" * 60)
    print("  EDA complete. Outputs in:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on features table")
    parser.add_argument(
        "--config", default="eda/config/eda_config.yaml",
        help="Path to EDA config YAML",
    )
    args = parser.parse_args()
    main(args.config)
