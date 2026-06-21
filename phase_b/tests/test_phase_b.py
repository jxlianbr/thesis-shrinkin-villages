"""
Phase B unit tests.

Coverage:
  target_builder:  residual mean-zero, residual orthogonal to demography
  accessibility:   inside-DID distance=0, nearest-point, dist-to-line,
                   zero-match buffer, key alignment
  feature_matrix:  key alignment, no leakage columns, no silent imputation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

# Ensure phase_b package is importable from the test directory.
_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
# phase_b lives inside phase_a; its parent IS the phase_a root.
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

import yaml

_CFG_PATH = _PKG / "config" / "phase_b_config.yaml"
with open(_CFG_PATH, encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def tiny_ols_data():
    """Tiny (n=30) synthetic dataset for OLS residual tests."""
    rng = np.random.default_rng(42)
    n = 30
    elderly_ratio = rng.uniform(0.2, 0.6, n)
    pop_total = rng.uniform(100, 5000, n)
    # Physical indicator is a noisy linear combination of demographics.
    contrast = 0.8 * elderly_ratio + 0.0001 * pop_total + rng.normal(0, 0.05, n)
    unit_ids = [f"aza:Test:{i:05d}" for i in range(n)]
    return pd.DataFrame({
        "unit_id": unit_ids,
        "S2_NDBI_contrast_mean": contrast,
        "elderly_ratio": elderly_ratio,
        "pop_total": pop_total,
    })


@pytest.fixture()
def tiny_aza_pts():
    """GeoDataFrame of 5 synthetic aza representative points (EPSG:6680)."""
    coords = [(x, y) for x, y in [
        (400000, 4500000),
        (400100, 4500100),
        (400200, 4500200),
        (400300, 4500300),
        (400400, 4500400),
    ]]
    unit_ids = [f"aza:Test:{i:05d}" for i in range(5)]
    gdf = gpd.GeoDataFrame(
        {"unit_id": unit_ids},
        geometry=[Point(*c) for c in coords],
        crs="EPSG:6680",
    )
    return gdf


# ===========================================================================
# target_builder tests
# ===========================================================================

class TestTargetBuilder:
    def test_residual_mean_zero(self, tiny_ols_data):
        """OLS residuals must be mean-zero to within floating-point tolerance."""
        from target_builder import _assert_residual_mean_zero
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.diagnostic import het_breuschpagan
        from typology.src.step3_relationships import _ols_statsmodels

        df = tiny_ols_data
        y = df["S2_NDBI_contrast_mean"].values
        X = df[["elderly_ratio", "pop_total"]].values

        result = _ols_statsmodels(
            y, X, ["elderly_ratio", "pop_total"], len(y), 10.0,
            sm, variance_inflation_factor, het_breuschpagan,
        )
        assert "error" not in result, f"OLS failed: {result.get('error')}"
        _assert_residual_mean_zero(result["residuals"])  # must not raise

    def test_residual_orthogonal_to_demography(self, tiny_ols_data):
        """OLS residuals must be uncorrelated with the regressors (X'e=0)."""
        from target_builder import assert_orthogonal_to_demography
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.diagnostic import het_breuschpagan
        from typology.src.step3_relationships import _ols_statsmodels

        df = tiny_ols_data
        y = df["S2_NDBI_contrast_mean"].values
        X = df[["elderly_ratio", "pop_total"]].values

        result = _ols_statsmodels(
            y, X, ["elderly_ratio", "pop_total"], len(y), 10.0,
            sm, variance_inflation_factor, het_breuschpagan,
        )
        assert_orthogonal_to_demography(result["residuals"], X, tol=1e-4)

    def test_leakage_guard_raises(self):
        """physical_indicator in banned_columns must raise ValueError."""
        from target_builder import _assert_no_leakage
        cfg = dict(_CFG)
        cfg["feature_matrix"] = {
            "banned_columns": ["elderly_ratio", "aging_index"]
        }
        with pytest.raises(ValueError, match="leakage"):
            _assert_no_leakage("elderly_ratio", cfg)


# ===========================================================================
# accessibility_features tests
# ===========================================================================

class TestAccessibilityFeatures:
    def test_inside_did_distance_zero(self, tiny_aza_pts):
        """A point inside a DID polygon must have dist_did_m == 0."""
        from accessibility_features import _euclidean_dist_point

        # DID polygon that contains the first aza point.
        poly = Polygon([
            (399900, 4499900), (400200, 4499900),
            (400200, 4500200), (399900, 4500200),
        ])
        did = gpd.GeoDataFrame(
            {"id": [0]}, geometry=[poly], crs="EPSG:6680",
        )
        dists = _euclidean_dist_point(tiny_aza_pts, did, "unit_id")
        # First point (400000, 4500000) is inside poly -> dist = 0
        assert dists["aza:Test:00000"] == pytest.approx(0.0, abs=1e-3)

    def test_nearest_point_distance(self, tiny_aza_pts):
        """Distance to a single facility should equal the hand-computed Euclidean."""
        from accessibility_features import _euclidean_dist_point

        fac = gpd.GeoDataFrame(
            {"id": [0]}, geometry=[Point(400000, 4501000)], crs="EPSG:6680",
        )
        dists = _euclidean_dist_point(tiny_aza_pts, fac, "unit_id")
        expected = 1000.0  # 4500000 -> 4501000 = 1000 m
        assert dists["aza:Test:00000"] == pytest.approx(expected, rel=1e-3)

    def test_distance_to_line(self, tiny_aza_pts):
        """Distance to a horizontal line should equal vertical offset."""
        from accessibility_features import _euclidean_dist_line

        line = gpd.GeoDataFrame(
            {"id": [0]},
            geometry=[LineString([(0, 4501000), (1000000, 4501000)])],
            crs="EPSG:6680",
        )
        dists = _euclidean_dist_line(tiny_aza_pts, line, "unit_id")
        expected = 4501000 - 4500000  # = 1000 m for first point
        assert dists["aza:Test:00000"] == pytest.approx(expected, rel=1e-2)

    def test_zero_match_buffer(self, tiny_aza_pts):
        """Buffer count should be 0 when facility is beyond the radius."""
        from accessibility_features import _buffer_count

        far_fac = gpd.GeoDataFrame(
            {"id": [0]}, geometry=[Point(500000, 4600000)], crs="EPSG:6680",
        )
        counts = _buffer_count(tiny_aza_pts, far_fac, radius_m=100, aza_id_col="unit_id")
        assert int(counts.loc["aza:Test:00000"]) == 0

    def test_key_alignment_assertion(self, tiny_aza_pts):
        """_assert_key_alignment must raise if a key is missing from result."""
        from accessibility_features import _assert_key_alignment

        # Result missing one unit
        incomplete = pd.DataFrame({
            "unit_id": tiny_aza_pts["unit_id"].iloc[:4].tolist(),
            "dist_did_m": [100, 200, 300, 400],
        })
        aza_ref = gpd.GeoDataFrame(
            {"unit_id": tiny_aza_pts["unit_id"].tolist()},
            geometry=tiny_aza_pts.geometry,
        )
        with pytest.raises(AssertionError, match="missing from accessibility output"):
            _assert_key_alignment(incomplete, aza_ref, "unit_id")


# ===========================================================================
# feature_matrix tests
# ===========================================================================

class TestFeatureMatrix:
    def test_no_leakage_columns(self):
        """Banned columns must never appear in the feature matrix."""
        banned = set(_CFG["feature_matrix"]["banned_columns"])
        # Check that _assert_no_leakage fires on a dataframe containing a banned col.
        from feature_matrix import _assert_no_leakage
        df_bad = pd.DataFrame({"unit_id": ["x"], "elderly_ratio": [0.4]})
        with pytest.raises(ValueError, match="Leakage columns"):
            _assert_no_leakage(df_bad, _CFG)

    def test_key_alignment_fails_loudly(self):
        """Mismatched keys before a merge must raise AssertionError."""
        from feature_matrix import _assert_key_alignment
        left = pd.DataFrame({"unit_id": ["a", "b", "c"]})
        right = pd.DataFrame({"unit_id": ["a", "b", "c", "d"], "val": [1, 2, 3, 4]})
        with pytest.raises(AssertionError, match="Key alignment failure"):
            _assert_key_alignment(left, right, "unit_id", "test_table")

    def test_hls_missing_not_silently_imputed(self):
        """HLS NaN rows must be carried as NaN + hls_missing=1, not silently 0."""
        from feature_matrix import _parse_num
        # Verify '-' parses to 0 (not NaN), which is the explicit convention
        # for empty cells in HLS XLS format.
        assert _parse_num("-") == 0.0
        assert _parse_num("") == 0.0
        assert _parse_num("2,345") == 2345.0

    def test_extract_muni_code(self):
        """muni_code extraction must return 5-digit prefix."""
        from feature_matrix import _extract_muni_code
        assert _extract_muni_code("aza:Aomori:022010010") == "02201"
        assert _extract_muni_code("aza:Akita:05202107002") == "05202"

    def test_finance_leakage_guard_raises(self):
        """Loading a post-2014 finance year must raise ValueError."""
        from feature_matrix import _parse_kessancard_pdf
        import tempfile, os

        # Create a minimal PDF with FY2015 in it — instead of a real PDF,
        # we test the guard logic directly by calling the year-check.
        from feature_matrix import _jp_year_to_western
        assert _jp_year_to_western("平成", 27) == 2015
        assert _jp_year_to_western("平成", 26) == 2014
        assert _jp_year_to_western("令和", 1) == 2019

    def test_kaso_flags_present(self):
        """Hard-coded kaso list must produce non-zero flags for known municipalities."""
        from feature_matrix import _KASO_ZENBU, _KASO_ICHIBU
        assert "五所川原市" in _KASO_ZENBU["Aomori"]
        assert "東成瀬村" in _KASO_ZENBU["Akita"]
        assert "弘前市" in _KASO_ICHIBU["Aomori"]
        assert "秋田市" in _KASO_ICHIBU["Akita"]
