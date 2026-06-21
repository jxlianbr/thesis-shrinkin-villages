"""
Phase B — Option-3 decoupling target.

Regresses the chosen aza physical-trajectory indicator on demography
(elderly_ratio + pop_total) via OLS with iterative VIF pruning, then
takes the per-aza residual as the Phase B prediction target.

Reuses the OLS/VIF/residual logic from typology/src/step3_relationships.py
by importing _ols_statsmodels directly (Phase A is NOT modified).

The residual is orthogonal to the demographic regressors by OLS construction,
satisfying the decoupling requirement in Option 3.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Phase A import — add phase_a to sys.path at import time using config.
# We resolve the path relative to this file so the module works from any cwd.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_CFG_PATH = _HERE / "config" / "phase_b_config.yaml"

def _load_cfg() -> Dict[str, Any]:
    with open(_CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)

_cfg = _load_cfg()
_PHASE_A_ROOT = Path(_cfg["phase_a_root"])
if str(_PHASE_A_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHASE_A_ROOT))

# Now import the Phase A OLS helper (read-only; Phase A file is not modified).
from typology.src.step3_relationships import _ols_statsmodels  # noqa: E402


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_target(cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Compute the Option-3 residual target for all aza units.

    Steps:
    1. Load Phase A classification_ready_aza.csv.
    2. Run OLS: physical_indicator ~ demographic_regressors (with VIF pruning).
    3. Persist residuals alongside unit_id.

    Returns
    -------
    pd.DataFrame
        Columns: [aza_id_col, 'physical_residual'].
        One row per aza unit that survived NaN filtering.
    """
    if cfg is None:
        cfg = _load_cfg()

    phase_a_root = Path(cfg["phase_a_root"])
    ready_path = phase_a_root / cfg["phase_a"]["classification_ready"]
    tgt_cfg = cfg["target"]
    aza_id_col = cfg["aza_id_col"]
    phys_col = tgt_cfg["physical_indicator"]
    demo_cols = list(tgt_cfg["demographic_regressors"])
    vif_threshold = float(tgt_cfg.get("vif_threshold", 10.0))
    out_col = tgt_cfg["output_col"]

    print(f"Loading Phase A classification-ready data: {ready_path}")
    df = pd.read_csv(ready_path, encoding="utf-8")
    _assert_columns(df, [aza_id_col, phys_col] + demo_cols, ready_path)

    # Leakage guard: ensure physical_indicator is not a demographic ratio.
    _assert_no_leakage(phys_col, cfg)

    # Drop rows where any required column is NaN.
    required = [aza_id_col, phys_col] + demo_cols
    mask = df[required].notna().all(axis=1)
    n_before = len(df)
    df_valid = df.loc[mask].copy().reset_index(drop=True)
    n_dropped = n_before - len(df_valid)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with NaN in required columns.")

    y = df_valid[phys_col].values
    X = df_valid[demo_cols].values
    X_names = list(demo_cols)
    n = len(y)
    print(f"  OLS: {phys_col} ~ {' + '.join(demo_cols)}  (n={n})")

    # Import statsmodels here (required for _ols_statsmodels).
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.diagnostic import het_breuschpagan
    except ImportError as exc:
        raise RuntimeError(
            "statsmodels is required for target_builder. "
            "Install with: pip install statsmodels"
        ) from exc

    result = _ols_statsmodels(
        y, X, X_names, n, vif_threshold,
        sm, variance_inflation_factor, het_breuschpagan,
    )

    if "error" in result:
        raise RuntimeError(f"OLS failed: {result['error']}")

    residuals = result["residuals"]
    _assert_residual_mean_zero(residuals)

    # Build output frame.
    out = df_valid[[aza_id_col]].copy()
    out[out_col] = residuals
    out[f"{phys_col}_fitted"] = result["fitted"]
    out[phys_col] = y  # original, for reference

    print(f"  Residuals: mean={float(np.mean(residuals)):.6f}, "
          f"std={float(np.std(residuals)):.4f}")
    print(f"  OLS R2={result['r_squared']:.3f}, "
          f"adj_R2={result['adj_r_squared']:.3f}")

    # Persist
    out_path = Path(cfg["phase_b_root"]) / cfg["output"]["residuals"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"  Saved residuals -> {out_path}")

    return out


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def _assert_columns(
    df: pd.DataFrame, cols: list[str], source: Path,
) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns {missing} in {source}. "
            f"Available: {list(df.columns)}"
        )


def _assert_no_leakage(physical_col: str, cfg: Dict[str, Any]) -> None:
    banned = set(cfg["feature_matrix"]["banned_columns"])
    if physical_col in banned:
        raise ValueError(
            f"physical_indicator '{physical_col}' is in the banned "
            f"(leakage) column list. Choose a remote-sensing feature."
        )


def _assert_residual_mean_zero(residuals: np.ndarray, tol: float = 1e-6) -> None:
    """OLS residuals must have mean zero by construction."""
    mean = float(np.mean(residuals))
    if abs(mean) > tol:
        raise AssertionError(
            f"Residual mean {mean:.2e} exceeds tolerance {tol}. "
            "This indicates an OLS fit error."
        )


def assert_orthogonal_to_demography(
    residuals: np.ndarray,
    demo_matrix: np.ndarray,
    tol: float = 1e-6,
) -> None:
    """
    Verify residuals are orthogonal to the demographic regressor matrix.

    OLS guarantees X'e = 0, so all correlations should be near zero.
    Called in tests; not in the main pipeline.
    """
    for i in range(demo_matrix.shape[1]):
        col = demo_matrix[:, i]
        valid = ~(np.isnan(col) | np.isnan(residuals))
        r = float(np.corrcoef(residuals[valid], col[valid])[0, 1])
        if abs(r) > tol:
            raise AssertionError(
                f"Residuals correlated with demographic col {i}: r={r:.4f}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_target_builder(cfg_path: Path | None = None) -> pd.DataFrame:
    """Load config, build target, return residual DataFrame."""
    if cfg_path is not None:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = _load_cfg()
    return build_target(cfg)


if __name__ == "__main__":
    result = run_target_builder()
    print(result.head())
