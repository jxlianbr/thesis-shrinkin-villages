"""
Golden sample reproducibility test.

This test runs the full pipeline with a minimal configuration and validates
that all expected outputs are produced correctly.

Expected:
- 10 units x 2 months = 20 rows
- All feature columns present (NDVI, NDBI, GLCM, OSM, VIIRS)
- No all-NaN feature columns
- Runtime < 10 minutes

Usage:
    python -m pytest tests/test_golden_sample.py -v
    python tests/test_golden_sample.py  # Direct run
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


GOLDEN_SAMPLE_CONFIG = "config/config_golden_sample.yaml"
OUTPUT_DIR = Path("outputs/golden_sample")
EXPECTED_ROWS = 20  # 10 units x 2 months


def run_pipeline() -> subprocess.CompletedProcess:
    """Run the pipeline with golden sample config."""
    return subprocess.run(
        [sys.executable, "pipeline.py", GOLDEN_SAMPLE_CONFIG],
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
    )


def test_pipeline_runs_successfully():
    """Pipeline should complete without errors."""
    result = run_pipeline()
    assert result.returncode == 0, f"Pipeline failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_output_files_exist():
    """All expected output files should be created."""
    # Skip if pipeline hasn't run
    if not OUTPUT_DIR.exists():
        run_pipeline()

    assert (OUTPUT_DIR / "features_table.csv").exists(), "Missing features_table.csv"
    assert (OUTPUT_DIR / "features_table.parquet").exists(), "Missing features_table.parquet"
    assert (OUTPUT_DIR / "run_manifest.json").exists(), "Missing run_manifest.json"


def test_row_count():
    """Features table should have expected number of rows."""
    csv_path = OUTPUT_DIR / "features_table.csv"
    if not csv_path.exists():
        run_pipeline()

    df = pd.read_csv(csv_path)
    assert len(df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(df)}"


def test_required_columns():
    """Features table should have all required columns."""
    csv_path = OUTPUT_DIR / "features_table.csv"
    if not csv_path.exists():
        run_pipeline()

    df = pd.read_csv(csv_path)

    required_cols = [
        "unit_id",
        "month",
        "NDVI",
        "NDBI",
        "viirs_mean",
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_spectral_bands_present():
    """Sentinel-2 spectral bands should be present."""
    csv_path = OUTPUT_DIR / "features_table.csv"
    if not csv_path.exists():
        run_pipeline()

    df = pd.read_csv(csv_path)

    spectral_bands = ["B2", "B3", "B4", "B8", "B11"]
    for band in spectral_bands:
        assert band in df.columns, f"Missing spectral band: {band}"


def test_no_all_nan_features():
    """Feature columns should not be entirely NaN."""
    csv_path = OUTPUT_DIR / "features_table.csv"
    if not csv_path.exists():
        run_pipeline()

    df = pd.read_csv(csv_path)

    feature_cols = ["NDVI", "NDBI", "B2", "B3", "B4", "B8", "B11"]
    for col in feature_cols:
        if col in df.columns:
            assert not df[col].isna().all(), f"Column {col} is all NaN"


def test_osm_features_present():
    """OSM building features should be computed."""
    csv_path = OUTPUT_DIR / "features_table.csv"
    if not csv_path.exists():
        run_pipeline()

    df = pd.read_csv(csv_path)

    osm_cols = ["osm_built_area", "osm_building_count", "osm_built_ratio"]
    for col in osm_cols:
        assert col in df.columns, f"Missing OSM column: {col}"


def test_glcm_features_present():
    """GLCM texture features should be computed (if enabled)."""
    csv_path = OUTPUT_DIR / "features_table.csv"
    if not csv_path.exists():
        run_pipeline()

    df = pd.read_csv(csv_path)

    # GLCM columns have format: S2_NDBI_contrast, S2_NDBI_entropy, etc.
    glcm_cols = [col for col in df.columns if col.startswith("S2_NDBI_")]

    # At least one GLCM column should exist if compute_glcm_local is enabled
    # This is a soft check - GLCM may not be computed in all configs
    if glcm_cols:
        for col in glcm_cols:
            assert not df[col].isna().all(), f"GLCM column {col} is all NaN"


def test_manifest_valid():
    """Run manifest should be valid JSON with expected fields."""
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    if not manifest_path.exists():
        run_pipeline()

    with open(manifest_path) as f:
        manifest = json.load(f)

    assert "project" in manifest, "Missing 'project' in manifest"
    assert "started_utc" in manifest, "Missing 'started_utc' in manifest"
    assert "finished_utc" in manifest, "Missing 'finished_utc' in manifest"
    assert "row_count" in manifest, "Missing 'row_count' in manifest"
    assert manifest["row_count"] == EXPECTED_ROWS, f"Manifest row_count mismatch: {manifest['row_count']} != {EXPECTED_ROWS}"


def test_unique_unit_month_pairs():
    """Each unit-month combination should appear exactly once."""
    csv_path = OUTPUT_DIR / "features_table.csv"
    if not csv_path.exists():
        run_pipeline()

    df = pd.read_csv(csv_path)

    duplicates = df.duplicated(subset=["unit_id", "month"], keep=False)
    assert not duplicates.any(), f"Found duplicate unit-month pairs: {df[duplicates][['unit_id', 'month']]}"


if __name__ == "__main__":
    # Run as standalone script
    print("Running golden sample pipeline...")
    result = run_pipeline()

    if result.returncode != 0:
        print("Pipeline FAILED:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    print("Pipeline completed. Running validations...")

    # Run all tests
    tests = [
        test_output_files_exist,
        test_row_count,
        test_required_columns,
        test_spectral_bands_present,
        test_no_all_nan_features,
        test_osm_features_present,
        test_glcm_features_present,
        test_manifest_valid,
        test_unique_unit_month_pairs,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__} - {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__} - {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
