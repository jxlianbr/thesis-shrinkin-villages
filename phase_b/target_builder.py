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
# Trajectory target (Task 1b / Task 2 of the gate fix)
# ---------------------------------------------------------------------------

# Demographic columns that must never be used as the physical target.
# Checked in _assert_not_demographic below.
# NOTE: we do NOT check against cfg["feature_matrix"]["banned_columns"] here
# because that list now includes EO/RS columns (to prevent them being used as
# predictors), but the physical target is supposed to be EO-derived — checking
# against the full banned list would incorrectly reject valid EO targets.
_DEMOGRAPHIC_LEAKAGE_COLS: frozenset[str] = frozenset({
    "elderly_ratio", "aging_index", "youth_ratio",
    "shrinkage_class", "shrinkage_code",
    "age_65_plus", "age_u15", "pop_total",
    "pop_male", "pop_female", "households_total",
})


def _compute_glcm_contrast_slope(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute S2_NDBI_contrast_slope per aza from the monthly EO panel.

    Imports and calls compute_ols_slope() from preprocessing.src.temporal_aggregation
    — the same function that produces NDBI_slope / NDVI_slope / MNDWI_slope in
    Phase A preprocessing — so time-normalisation and min-obs handling are
    guaranteed to be identical.

    Phase A outputs are NOT modified; the slope lives only in this result.

    Returns
    -------
    pd.DataFrame  [aza_id_col, 'S2_NDBI_contrast_slope']
    """
    from preprocessing.src.temporal_aggregation import compute_ols_slope  # noqa: PLC0415

    phase_a_root = Path(cfg["phase_a_root"])
    ft_path = phase_a_root / cfg["phase_a"]["eo_trajectory"]
    aza_id_col = cfg["aza_id_col"]
    indicator = "S2_NDBI_contrast"
    slope_col = f"{indicator}_slope"

    print(f"  Loading monthly panel: {ft_path}")
    ft = pd.read_parquet(ft_path, columns=[aza_id_col, "month", indicator])
    n_aza = ft[aza_id_col].nunique()
    print(f"  Panel: {len(ft):,} rows, {n_aza} aza units, "
          f"{ft[indicator].notna().sum():,} non-null {indicator} observations.")

    records = []
    n_null = 0
    for unit_id, grp in ft.groupby(aza_id_col):
        slope = compute_ols_slope(grp[indicator], grp["month"], min_obs=3)
        records.append({aza_id_col: unit_id, slope_col: slope})
        if np.isnan(slope):
            n_null += 1

    df = pd.DataFrame(records)
    n_valid = len(df) - n_null
    s = df[slope_col].dropna()
    print(f"  {slope_col}: {n_valid}/{len(df)} aza non-null "
          f"({n_null} below min_obs=3 -> NaN-in-y, dropped from CV).")
    print(f"  Distribution: mean={s.mean():.4f}, std={s.std():.4f}, "
          f"min={s.min():.4f}, max={s.max():.4f}")
    if n_null > 0.10 * len(df):
        print(f"  WARNING: {n_null/len(df)*100:.1f}% of aza have NaN slope. "
              f"Check GLCM coverage in the monthly panel.")
    return df


def build_trajectory_target(cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Option-3 trajectory target: OLS residual of S2_NDBI_contrast_slope on
    2015 demographic baseline (elderly_ratio + pop_total).

    Saves to target_trajectory.parquet; does NOT overwrite the level target.
    """
    if cfg is None:
        cfg = _load_cfg()

    phase_a_root = Path(cfg["phase_a_root"])
    aza_id_col = cfg["aza_id_col"]
    tgt_cfg = cfg["target"]
    demo_cols = list(tgt_cfg["demographic_regressors"])
    vif_threshold = float(tgt_cfg.get("vif_threshold", 10.0))
    out_col = tgt_cfg["output_col"]                           # "physical_residual"
    phys_col = tgt_cfg["physical_indicator_trajectory"]       # "S2_NDBI_contrast_slope"

    _assert_not_demographic(phys_col)

    # ---- Step 1: slope per aza -------------------------------------------
    print(f"Step 1: computing {phys_col} from monthly panel...")
    slope_df = _compute_glcm_contrast_slope(cfg)

    # ---- Step 2: 2015 demographic baseline --------------------------------
    ready_path = phase_a_root / cfg["phase_a"]["classification_ready"]
    print(f"\nStep 2: loading 2015 demographic baseline: {ready_path.name}")
    base = pd.read_csv(ready_path, encoding="utf-8")
    _assert_columns(base, [aza_id_col] + demo_cols, ready_path)
    # classification_ready_aza.csv is built from 2015 census (tblT000848C02/C05).
    # The 2020 vintage does not exist in this pipeline. 2015 confirmed.
    print(f"  Demographic regressors: {demo_cols}  [2015 census vintage confirmed]")

    # ---- Step 3: join and filter ------------------------------------------
    df = base[[aza_id_col] + demo_cols].merge(slope_df, on=aza_id_col, how="inner")
    required = [aza_id_col, phys_col] + demo_cols
    mask = df[required].notna().all(axis=1)
    n_before = len(df)
    df_valid = df.loc[mask].copy().reset_index(drop=True)
    n_dropped = n_before - len(df_valid)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with NaN in required columns "
              f"(slope NaN = insufficient GLCM coverage).")
    print(f"  OLS sample: {len(df_valid)} aza units.")

    # ---- Step 4: OLS residualisation -------------------------------------
    y = df_valid[phys_col].values
    X = df_valid[demo_cols].values
    X_names = list(demo_cols)
    n = len(y)
    print(f"\nStep 3: OLS: {phys_col} ~ {' + '.join(demo_cols)}  (n={n})")

    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.diagnostic import het_breuschpagan
    except ImportError as exc:
        raise RuntimeError("statsmodels required. pip install statsmodels") from exc

    result = _ols_statsmodels(
        y, X, X_names, n, vif_threshold,
        sm, variance_inflation_factor, het_breuschpagan,
    )
    if "error" in result:
        raise RuntimeError(f"OLS failed: {result['error']}")

    residuals = result["residuals"]
    _assert_residual_mean_zero(residuals)

    r2 = result["r_squared"]
    print(f"  Residualizing R2={r2:.3f}, adj_R2={result['adj_r_squared']:.3f}")
    if r2 > 0.50:
        print(f"  STOP FLAG: residualizing R2={r2:.3f} is above 0.50. "
              f"The trajectory is substantially explained by demography. "
              f"Inspect before proceeding.")

    # ---- Step 5: persist -------------------------------------------------
    out = df_valid[[aza_id_col]].copy()
    out[out_col] = residuals
    out[f"{phys_col}_fitted"] = result["fitted"]
    out[phys_col] = y

    out_path = Path(cfg["phase_b_root"]) / cfg["output"]["residuals_trajectory"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"\n  Trajectory target saved -> {out_path}")
    print(f"  Residuals: mean={float(np.mean(residuals)):.6f}, "
          f"std={float(np.std(residuals)):.4f}")
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


def _assert_not_demographic(physical_col: str) -> None:
    """Prevent a demographic ratio from being used as the physical target."""
    if physical_col in _DEMOGRAPHIC_LEAKAGE_COLS:
        raise ValueError(
            f"physical_indicator '{physical_col}' is a demographic ratio. "
            "The target must be an EO/remote-sensing feature."
        )


def _assert_no_leakage(physical_col: str, cfg: Dict[str, Any]) -> None:
    # Legacy entry point — delegate to the corrected check.
    _assert_not_demographic(physical_col)


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
    """Load config, build level target, return residual DataFrame."""
    if cfg_path is not None:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = _load_cfg()
    return build_target(cfg)


def run_trajectory_target_builder(cfg_path: Path | None = None) -> pd.DataFrame:
    """Load config, build GLCM trajectory target, return residual DataFrame."""
    if cfg_path is not None:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = _load_cfg()
    return build_trajectory_target(cfg)


# ---------------------------------------------------------------------------
# NDBI trajectory target (active gate fix — replaces GLCM contrast slope)
# ---------------------------------------------------------------------------

def build_ndbi_target(cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Option-3 trajectory target: OLS residual of NDBI_slope on 2015 demography.

    # Sign convention: more negative NDBI_slope = built-up loss = decline.
    NDBI_slope is already in classification_ready_aza.csv (computed by Phase A
    preprocessing). No recompute of the monthly panel is needed.

    Saves to target_ndbi_trajectory.parquet; does NOT overwrite prior targets.
    """
    if cfg is None:
        cfg = _load_cfg()

    phase_a_root = Path(cfg["phase_a_root"])
    aza_id_col = cfg["aza_id_col"]
    tgt_cfg = cfg["target"]
    demo_cols = list(tgt_cfg["demographic_regressors"])
    vif_threshold = float(tgt_cfg.get("vif_threshold", 10.0))
    out_col = tgt_cfg["output_col"]
    # more negative NDBI_slope = built-up loss = decline
    phys_col = tgt_cfg["physical_indicator_ndbi"]   # "NDBI_slope"

    _assert_not_demographic(phys_col)

    ready_path = phase_a_root / cfg["phase_a"]["classification_ready"]
    print(f"Loading Phase A classification-ready data: {ready_path}")
    df = pd.read_csv(ready_path, encoding="utf-8")
    _assert_columns(df, [aza_id_col, phys_col] + demo_cols, ready_path)

    required = [aza_id_col, phys_col] + demo_cols
    mask = df[required].notna().all(axis=1)
    n_before = len(df)
    df_valid = df.loc[mask].copy().reset_index(drop=True)
    n_dropped = n_before - len(df_valid)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with NaN in required columns.")
    print(f"  OLS sample: {len(df_valid)} aza units.")

    y = df_valid[phys_col].values
    X = df_valid[demo_cols].values
    X_names = list(demo_cols)
    n = len(y)
    print(f"  OLS: {phys_col} ~ {' + '.join(demo_cols)}  (n={n})")

    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.diagnostic import het_breuschpagan
    except ImportError as exc:
        raise RuntimeError("statsmodels required. pip install statsmodels") from exc

    result = _ols_statsmodels(
        y, X, X_names, n, vif_threshold,
        sm, variance_inflation_factor, het_breuschpagan,
    )
    if "error" in result:
        raise RuntimeError(f"OLS failed: {result['error']}")

    residuals = result["residuals"]
    _assert_residual_mean_zero(residuals)

    r2 = result["r_squared"]
    print(f"  Residualizing R2={r2:.3f}, adj_R2={result['adj_r_squared']:.3f}")
    if r2 > 0.50:
        print(f"  STOP FLAG: residualizing R2={r2:.3f} exceeds 0.50. "
              "Trajectory substantially explained by demography — inspect before proceeding.")

    out = df_valid[[aza_id_col]].copy()
    out[out_col] = residuals
    out[f"{phys_col}_fitted"] = result["fitted"]
    out[phys_col] = y  # kept for reference; more negative = built-up loss

    out_path = Path(cfg["phase_b_root"]) / cfg["output"]["residuals_ndbi"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"  Saved NDBI trajectory target -> {out_path}")
    print(f"  Residuals: mean={float(np.mean(residuals)):.6f}, "
          f"std={float(np.std(residuals)):.4f}")
    return out


def run_ndbi_target_builder(cfg_path: Path | None = None) -> pd.DataFrame:
    """Load config, build NDBI trajectory target, return residual DataFrame."""
    if cfg_path is not None:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = _load_cfg()
    return build_ndbi_target(cfg)


# ---------------------------------------------------------------------------
# DW bare-fraction slope target (active — passed pre-checks)
# ---------------------------------------------------------------------------

def build_dw_bare_target(cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Option-3 trajectory target: OLS residual of dw_bare_frac_slope on
    2015 demography (elderly_ratio + pop_total).

    # Sign convention: positive slope = bare area increasing over 2015-2024.
    dw_bare_frac_slope is in classification_ready_aza.csv — no recompute needed.

    Saves to target_dw_bare_trajectory.parquet; does NOT overwrite prior targets.
    """
    if cfg is None:
        cfg = _load_cfg()

    phase_a_root = Path(cfg["phase_a_root"])
    aza_id_col = cfg["aza_id_col"]
    tgt_cfg = cfg["target"]
    demo_cols = list(tgt_cfg["demographic_regressors"])
    vif_threshold = float(tgt_cfg.get("vif_threshold", 10.0))
    out_col = tgt_cfg["output_col"]
    phys_col = tgt_cfg["physical_indicator_dw_bare"]   # "dw_bare_frac_slope"

    _assert_not_demographic(phys_col)

    ready_path = phase_a_root / cfg["phase_a"]["classification_ready"]
    print(f"Loading Phase A classification-ready data: {ready_path}")
    df = pd.read_csv(ready_path, encoding="utf-8")
    _assert_columns(df, [aza_id_col, phys_col] + demo_cols, ready_path)

    required = [aza_id_col, phys_col] + demo_cols
    mask = df[required].notna().all(axis=1)
    n_before = len(df)
    df_valid = df.loc[mask].copy().reset_index(drop=True)
    n_dropped = n_before - len(df_valid)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with NaN in required columns.")
    print(f"  OLS sample: {len(df_valid)} aza units.")

    y = df_valid[phys_col].values
    X = df_valid[demo_cols].values
    X_names = list(demo_cols)
    n = len(y)
    print(f"  OLS: {phys_col} ~ {' + '.join(demo_cols)}  (n={n})")

    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.diagnostic import het_breuschpagan
    except ImportError as exc:
        raise RuntimeError("statsmodels required. pip install statsmodels") from exc

    result = _ols_statsmodels(
        y, X, X_names, n, vif_threshold,
        sm, variance_inflation_factor, het_breuschpagan,
    )
    if "error" in result:
        raise RuntimeError(f"OLS failed: {result['error']}")

    residuals = result["residuals"]
    _assert_residual_mean_zero(residuals)

    r2 = result["r_squared"]
    print(f"  Residualizing R2={r2:.3f}, adj_R2={result['adj_r_squared']:.3f}")
    if r2 > 0.50:
        print(f"  STOP FLAG: residualizing R2={r2:.3f} exceeds 0.50. "
              "Trajectory substantially explained by demography — inspect before proceeding.")

    out = df_valid[[aza_id_col]].copy()
    out[out_col] = residuals
    out[f"{phys_col}_fitted"] = result["fitted"]
    out[phys_col] = y

    out_path = Path(cfg["phase_b_root"]) / cfg["output"]["residuals_dw_bare"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"  Saved DW bare trajectory target -> {out_path}")
    print(f"  Residuals: mean={float(np.mean(residuals)):.6f}, "
          f"std={float(np.std(residuals)):.4f}")
    return out


def run_dw_bare_target_builder(cfg_path: Path | None = None) -> pd.DataFrame:
    """Load config, build DW bare-fraction slope target, return residual DataFrame."""
    if cfg_path is not None:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = _load_cfg()
    return build_dw_bare_target(cfg)


# ---------------------------------------------------------------------------
# DW bare-fraction Theil-Sen slope target (ACTIVE -- replaces plain-OLS dw_bare)
#
# Root cause of the plain-OLS dw_bare_frac_slope failure (see FINDINGS.md,
# "Target Validation: Theil-Sen + Genuine-Month-Coverage Fix" for the full
# diagnosis with numbers):
#
#   1. pipeline.py step 4e only overwrites the static gee_lulc.py dw_bare_frac
#      value with a genuine per-month value when GOOGLE/DYNAMICWORLD/V1
#      actually returned a non-null zonal mean for that (unit, month). Across
#      the whole aza panel, ~46% of (unit, month) cells never get a genuine
#      value (persistent cloud cover / early-archive sparsity) and silently
#      keep the old static, multi-year-average value instead.
#   2. A plain OLS slope fit on a series that is ~half a flat constant and
#      half noisy real observations is dominated by whichever genuine points
#      happen to land at the high-leverage ends of the time axis. A single
#      anomalous spike in the last genuine month can produce a slope many
#      times larger than the rest of the dataset's slopes.
#   3. Because most aza ARE mostly flat (raw slope IQR is tiny, ~0.0043),
#      RobustScaler then turns that one unstable estimate into an extreme
#      scaled value. 1.2% of aza (91 units) ended up responsible for >20% of
#      the target's total variance, and CV blocked by municipality collapses
#      whenever an outlier-loaded municipality lands in the held-out fold.
#
# The fix: (a) compute the slope using ONLY genuine (non-fallback) monthly
# observations, identified from the per-month cache CSVs that
# gee_dw_monthly.py writes to phase_a/outputs/gee/dw_monthly/ (a (unit,month)
# cell is genuine iff it appears there with a non-null dw_bare_frac); and
# (b) use Theil-Sen instead of OLS, which is the median of all pairwise
# slopes and is far less sensitive to one or two high-leverage points.
# ---------------------------------------------------------------------------

def _load_genuine_dw_bare_observations(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load every (unit_id, month) cell that holds a GENUINE Dynamic World
    monthly bare-fraction observation, as opposed to the static gee_lulc.py
    fallback value that pipeline.py step 4e leaves in place when extraction
    returned null for that cell.

    Source: the per-calendar-month cache CSVs written by
    data_preprocessing/gee_dw_monthly.py (phase_a/outputs/gee/dw_monthly/).
    A small number of aza have duplicate rows per month because some aza
    polygons are exported as multiple parts; these are collapsed by taking
    the mean before genuineness is determined.

    Returns
    -------
    pd.DataFrame  [unit_id, month, dw_bare_frac]  -- genuine observations only.
    """
    phase_a_root = Path(cfg["phase_a_root"])
    aza_id_col = cfg["aza_id_col"]
    cache_dir = phase_a_root / cfg["phase_a"]["dw_monthly_cache"]

    month_files = sorted(cache_dir.glob("dw_aza_*.csv"))
    if not month_files:
        raise FileNotFoundError(
            f"No dw_monthly cache files found in {cache_dir}. "
            "Run data_preprocessing/gee_dw_monthly.py (via pipeline.py step 4e) first."
        )

    print(f"  Loading {len(month_files)} dw_monthly cache files from {cache_dir} ...")
    frames = [pd.read_csv(f) for f in month_files]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={"unit_id": aza_id_col})

    n_raw = len(raw)
    n_dupe = int(raw.duplicated(subset=[aza_id_col, "month"]).sum())
    collapsed = (
        raw.groupby([aza_id_col, "month"], as_index=False)["dw_bare_frac"].mean()
    )
    genuine = collapsed.dropna(subset=["dw_bare_frac"]).reset_index(drop=True)

    print(f"  Cache rows: {n_raw:,} ({n_dupe:,} multi-part-polygon duplicates "
          f"collapsed by mean) -> {len(collapsed):,} unique (unit, month) cells")
    print(f"  Genuine (non-null) cells: {len(genuine):,} / {len(collapsed):,} "
          f"({len(genuine) / len(collapsed) * 100:.1f}%)")
    return genuine


def _compute_dw_bare_theilsen_slope(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute a robust dw_bare_frac trend per aza using Theil-Sen regression
    on genuine (non-fallback) monthly observations only.

    Time is normalised to [0, 1] over each unit's FULL panel span (same
    convention as compute_ols_slope / the retired dw_bare_frac_slope), so the
    slope remains comparable in units to the plain-OLS version; only the set
    of points used to FIT the slope differs (genuine-only, Theil-Sen instead
    of least squares).

    Returns
    -------
    pd.DataFrame  [aza_id_col, 'dw_bare_frac_slope_theilsen', 'n_genuine_months']
        Units with fewer genuine months than target.min_genuine_months get NaN.
    """
    from scipy.stats import theilslopes  # noqa: PLC0415

    phase_a_root = Path(cfg["phase_a_root"])
    aza_id_col = cfg["aza_id_col"]
    min_genuine = int(cfg["target"].get("min_genuine_months", 24))
    slope_col = cfg["target"]["physical_indicator_dw_bare_robust"]

    # Full monthly panel: defines the [0, 1] time-normalisation basis.
    ft_path = phase_a_root / cfg["phase_a"]["eo_trajectory"]
    print(f"  Loading full monthly panel: {ft_path}")
    ft = pd.read_parquet(ft_path, columns=[aza_id_col, "month"])
    n_aza = ft[aza_id_col].nunique()
    print(f"  Panel: {len(ft):,} rows, {n_aza} aza units")

    genuine = _load_genuine_dw_bare_observations(cfg)
    genuine_set = set(zip(genuine[aza_id_col], genuine["month"]))
    genuine_value = dict(zip(zip(genuine[aza_id_col], genuine["month"]),
                             genuine["dw_bare_frac"]))

    records = []
    n_below_min = 0
    for unit_id, grp in ft.groupby(aza_id_col):
        grp_sorted = grp.sort_values("month")
        months = grp_sorted["month"].values
        t = (pd.to_datetime(months) - pd.to_datetime(months[0])).days.values.astype(float)
        if len(t) < min_genuine or t[-1] == 0:
            records.append({aza_id_col: unit_id, slope_col: np.nan, "n_genuine_months": 0})
            continue
        t_norm = t / t[-1]

        is_genuine = np.array([(unit_id, m) in genuine_set for m in months])
        n_genuine = int(is_genuine.sum())
        if n_genuine < min_genuine:
            n_below_min += 1
            records.append({aza_id_col: unit_id, slope_col: np.nan,
                             "n_genuine_months": n_genuine})
            continue

        y_genuine = np.array([genuine_value[(unit_id, m)] for m in months[is_genuine]])
        t_genuine = t_norm[is_genuine]
        slope, _intercept, _lo, _hi = theilslopes(y_genuine, t_genuine)
        records.append({aza_id_col: unit_id, slope_col: round(float(slope), 6),
                         "n_genuine_months": n_genuine})

    df = pd.DataFrame(records)
    n_valid = df[slope_col].notna().sum()
    print(f"  {slope_col}: {n_valid}/{len(df)} aza with a robust slope "
          f"({n_below_min} below min_genuine_months={min_genuine} -> NaN, dropped from CV)")
    s = df[slope_col].dropna()
    if len(s):
        print(f"  Distribution: mean={s.mean():.4f}, std={s.std():.4f}, "
              f"min={s.min():.4f}, max={s.max():.4f}")
    print(f"  Genuine months per unit: min={df['n_genuine_months'].min()}, "
          f"median={df['n_genuine_months'].median():.0f}, "
          f"max={df['n_genuine_months'].max()}")
    return df


def build_dw_bare_robust_target(cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Option-3 trajectory target: OLS residual of the Theil-Sen,
    genuine-month-filtered dw_bare_frac slope on 2015 demography
    (elderly_ratio + pop_total).

    Saves to target_dw_bare_robust_trajectory.parquet; does not overwrite
    the superseded plain-OLS dw_bare target.
    """
    if cfg is None:
        cfg = _load_cfg()

    phase_a_root = Path(cfg["phase_a_root"])
    aza_id_col = cfg["aza_id_col"]
    tgt_cfg = cfg["target"]
    demo_cols = list(tgt_cfg["demographic_regressors"])
    vif_threshold = float(tgt_cfg.get("vif_threshold", 10.0))
    out_col = tgt_cfg["output_col"]
    phys_col = tgt_cfg["physical_indicator_dw_bare_robust"]  # "dw_bare_frac_slope_theilsen"

    _assert_not_demographic(phys_col)

    print("Step 1: computing genuine-month-filtered Theil-Sen dw_bare_frac slope...")
    slope_df = _compute_dw_bare_theilsen_slope(cfg)

    ready_path = phase_a_root / cfg["phase_a"]["classification_ready"]
    print(f"\nStep 2: loading 2015 demographic baseline: {ready_path.name}")
    base = pd.read_csv(ready_path, encoding="utf-8")
    _assert_columns(base, [aza_id_col] + demo_cols, ready_path)

    df = base[[aza_id_col] + demo_cols].merge(slope_df, on=aza_id_col, how="inner")
    required = [aza_id_col, phys_col] + demo_cols
    mask = df[required].notna().all(axis=1)
    n_before = len(df)
    df_valid = df.loc[mask].copy().reset_index(drop=True)
    n_dropped = n_before - len(df_valid)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with NaN in required columns "
              f"(insufficient genuine-month coverage).")
    print(f"  OLS sample: {len(df_valid)} aza units.")

    y = df_valid[phys_col].values
    X = df_valid[demo_cols].values
    X_names = list(demo_cols)
    n = len(y)
    print(f"\nStep 3: OLS: {phys_col} ~ {' + '.join(demo_cols)}  (n={n})")

    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.diagnostic import het_breuschpagan
    except ImportError as exc:
        raise RuntimeError("statsmodels required. pip install statsmodels") from exc

    result = _ols_statsmodels(
        y, X, X_names, n, vif_threshold,
        sm, variance_inflation_factor, het_breuschpagan,
    )
    if "error" in result:
        raise RuntimeError(f"OLS failed: {result['error']}")

    residuals = result["residuals"]
    _assert_residual_mean_zero(residuals)

    r2 = result["r_squared"]
    print(f"  Residualizing R2={r2:.4f}, adj_R2={result['adj_r_squared']:.4f}")
    if r2 > 0.50:
        print(f"  STOP FLAG: residualizing R2={r2:.3f} exceeds 0.50. "
              "Trajectory substantially explained by demography -- inspect before proceeding.")

    out = df_valid[[aza_id_col, "n_genuine_months"]].copy()
    out[out_col] = residuals
    out[f"{phys_col}_fitted"] = result["fitted"]
    out[phys_col] = y

    out_path = Path(cfg["phase_b_root"]) / cfg["output"]["residuals_dw_bare_robust"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"\n  Robust DW bare trajectory target saved -> {out_path}")
    print(f"  Residuals: mean={float(np.mean(residuals)):.6f}, "
          f"std={float(np.std(residuals)):.4f}")
    return out


def run_dw_bare_robust_target_builder(cfg_path: Path | None = None) -> pd.DataFrame:
    """Load config, build the Theil-Sen DW bare-fraction target, return residual DataFrame."""
    if cfg_path is not None:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = _load_cfg()
    return build_dw_bare_robust_target(cfg)


if __name__ == "__main__":
    import sys as _sys
    if "--dw_bare_robust" in _sys.argv:
        result = run_dw_bare_robust_target_builder()
    elif "--dw_bare" in _sys.argv:
        result = run_dw_bare_target_builder()
    elif "--ndbi" in _sys.argv:
        result = run_ndbi_target_builder()
    elif "--trajectory" in _sys.argv:
        result = run_trajectory_target_builder()
    else:
        result = run_target_builder()
    print(result.head())
