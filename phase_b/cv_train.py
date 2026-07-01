"""
Phase B — Grouped cross-validation and XGBoost training.

Groups are defined by municipality code (extracted from unit_id) so that
all aza within a municipality stay in the same fold.  This implements
spatial blocking that prevents spatial-autocorrelation leakage between
train and test sets.

Imports Phase A's cross_validation.py and leakage_analysis.py (read-only;
Phase A files are never modified).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

_HERE = Path(__file__).resolve().parent
_CFG_PATH = _HERE / "config" / "phase_b_config.yaml"


def _load_cfg() -> Dict[str, Any]:
    with open(_CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


_cfg = _load_cfg()
_PHASE_A_ROOT = Path(_cfg["phase_a_root"])
if str(_PHASE_A_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHASE_A_ROOT))

from classification.src.cross_validation import run_cross_validation  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_muni_code(unit_id: str) -> str:
    """First 5 chars of the numeric code = pref(2) + muni(3)."""
    return unit_id.split(":")[-1][:5]


def _build_xgboost_model(cfg: Dict[str, Any]) -> dict:
    """Wrap XGBoost regressor in the format expected by run_cross_validation."""
    from xgboost import XGBRegressor  # noqa: WPS433
    m_cfg = cfg["model"]["xgboost"]
    estimator = XGBRegressor(
        n_estimators=int(m_cfg["n_estimators"]),
        max_depth=int(m_cfg["max_depth"]),
        learning_rate=float(m_cfg["learning_rate"]),
        subsample=float(m_cfg["subsample"]),
        colsample_bytree=float(m_cfg["colsample_bytree"]),
        reg_alpha=float(m_cfg["reg_alpha"]),
        reg_lambda=float(m_cfg["reg_lambda"]),
        random_state=int(m_cfg["random_state"]),
        n_jobs=int(m_cfg["n_jobs"]),
        verbosity=0,
    )
    return {
        "xgboost": {
            "estimator": estimator,
            "display_name": "XGBoost (residual target)",
        }
    }


def _cv_cfg_for_phase_b(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the cross_validation sub-dict expected by run_cross_validation."""
    cv_cfg = cfg["cross_validation"]
    return {
        "n_splits": int(cv_cfg["n_splits"]),
        "n_repeats": int(cv_cfg["n_repeats"]),
        "random_state": int(cv_cfg["random_state"]),
        "grouping": cv_cfg["grouping"],
        "loo_check": cv_cfg["loo_check"],
    }


def _loo_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cross_validation": _cv_cfg_for_phase_b(cfg),
        "evaluation": {"primary_metric": "r2"},
    }


# ---------------------------------------------------------------------------
# Regression metrics (Phase A CV uses classification metrics; override here)
# ---------------------------------------------------------------------------

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_squared_error, r2_score  # noqa: WPS433
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "r2": float(r2_score(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(np.mean(np.abs(yt - yp))),
    }


def _run_grouped_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    models: dict,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run municipality-grouped CV via Phase A's StratifiedGroupKFold wrapper.

    Because the target is continuous we monkey-patch the metric computation
    inside each fold rather than using the classification metrics from Phase A.
    """
    from copy import deepcopy
    from sklearn.model_selection import StratifiedGroupKFold

    cv_cfg = _cv_cfg_for_phase_b(cfg)
    n_splits = cv_cfg["n_splits"]
    n_repeats = cv_cfg["n_repeats"]
    rs_base = cv_cfg["random_state"]

    X_arr = X.values
    y_arr = y.values

    # For StratifiedGroupKFold, we need a discrete stratum label.
    # Use the group (municipality) itself as the stratum so that the split
    # is purely group-blocked without requiring a class label.
    strata = groups

    all_results: Dict[str, Any] = {}

    for model_name, model_info in models.items():
        display = model_info["display_name"]
        print(f"  {display} ...", end=" ", flush=True)

        fold_metrics = []
        all_y_true, all_y_pred = [], []
        # Per-row out-of-fold prediction accumulators. Each row is tested
        # exactly once per repeat, so oof_count ends at n_repeats and the
        # mean is the repeat-averaged OOF prediction (positionally aligned
        # with X/y, hence with unit_ids returned by train()).
        oof_sum = np.zeros(len(y_arr), dtype=float)
        oof_count = np.zeros(len(y_arr), dtype=float)

        for r in range(n_repeats):
            sgkf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=rs_base + r,
            )
            for train_idx, test_idx in sgkf.split(X_arr, strata, groups=groups):
                est = deepcopy(model_info["estimator"])
                est.fit(X_arr[train_idx], y_arr[train_idx])
                y_pred = est.predict(X_arr[test_idx])
                fold_metrics.append(
                    _regression_metrics(y_arr[test_idx], y_pred)
                )
                all_y_true.append(y_arr[test_idx])
                all_y_pred.append(y_pred)
                oof_sum[test_idx] += y_pred
                oof_count[test_idx] += 1

        fold_df = pd.DataFrame(fold_metrics)
        mean_m = fold_df.mean().to_dict()
        std_m = fold_df.std().to_dict()
        print(f"R2={mean_m['r2']:.3f} +/- {std_m['r2']:.3f}, "
              f"RMSE={mean_m['rmse']:.4f}")

        oof_pred = np.divide(
            oof_sum, oof_count,
            out=np.full(len(y_arr), np.nan), where=oof_count > 0,
        )

        all_results[model_name] = {
            "fold_metrics": fold_df,
            "mean_metrics": mean_m,
            "std_metrics": std_m,
            "all_y_true": np.concatenate(all_y_true),
            "all_y_pred": np.concatenate(all_y_pred),
            "oof_pred": oof_pred,
        }

    return all_results


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    cfg: Dict[str, Any] | None = None,
    trajectory: bool = False,
    ndbi: bool = False,
    dw_bare: bool = False,
    dw_bare_robust: bool = False,
) -> Dict[str, Any]:
    """
    Load feature matrix + residual target, run grouped CV, save results.

    Parameters
    ----------
    trajectory : bool
        If True, use the GLCM contrast-slope target (RETIRED — kept for reference).
    ndbi : bool
        If True, use the NDBI_slope target (RETIRED — spatial gradient artifact).
    dw_bare : bool
        If True, use the plain-OLS dw_bare_frac_slope target (SUPERSEDED 2026-06-30
        — kept only for before/after comparison against dw_bare_robust).
    dw_bare_robust : bool
        If True, use the Theil-Sen, genuine-month-filtered dw_bare_frac slope
        target (ACTIVE). See target_builder.py:_compute_dw_bare_theilsen_slope
        and FINDINGS.md "Target Validation: Theil-Sen Fix" for the rationale.

    Returns
    -------
    dict with keys: cv_results, feature_names, groups, X, y, final_model.
    """
    if cfg is None:
        cfg = _load_cfg()

    phase_b_root = Path(cfg["phase_b_root"])
    aza_id_col = cfg["aza_id_col"]

    # Select target-variant paths
    if dw_bare_robust:
        res_key    = "residuals_dw_bare_robust"
        cv_key     = "cv_results_dw_bare_robust"
        model_name = "final_model_xgboost_dw_bare_robust.joblib"
        tag        = " [DW_BARE_ROBUST]"
    elif dw_bare:
        res_key    = "residuals_dw_bare"
        cv_key     = "cv_results_dw_bare"
        model_name = "final_model_xgboost_dw_bare.joblib"
        tag        = " [DW_BARE]"
    elif ndbi:
        res_key    = "residuals_ndbi"
        cv_key     = "cv_results_ndbi"
        model_name = "final_model_xgboost_ndbi.joblib"
        tag        = " [NDBI]"
    elif trajectory:
        res_key    = "residuals_trajectory"
        cv_key     = "cv_results_trajectory"
        model_name = "final_model_xgboost_trajectory.joblib"
        tag        = " [TRAJECTORY]"
    else:
        res_key    = "residuals"
        cv_key     = "cv_results"
        model_name = "final_model_xgboost.joblib"
        tag        = ""

    # Load feature matrix
    fm_path = phase_b_root / cfg["output"]["feature_matrix"]
    if not fm_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {fm_path}. "
            "Run feature_matrix.py first."
        )
    print(f"Loading feature matrix{tag}: {fm_path}")
    fm = pd.read_parquet(fm_path)

    # Load residual target
    res_path = phase_b_root / cfg["output"][res_key]
    if not res_path.exists():
        raise FileNotFoundError(
            f"Residual target not found: {res_path}. "
            "Run target_builder.py first."
        )
    print(f"Loading residual target: {res_path}")
    target_df = pd.read_parquet(res_path)

    target_col = cfg["target"]["output_col"]

    # Merge on unit_id
    data = fm.merge(
        target_df[[aza_id_col, target_col]],
        on=aza_id_col,
        how="inner",
    )
    print(f"  Combined: {len(data)} aza units after inner join.")

    # Derive municipality group for spatial blocking
    data["muni_code"] = data[aza_id_col].apply(_extract_muni_code)
    groups = data["muni_code"].values
    n_groups = len(set(groups))
    print(f"  Grouping: {n_groups} municipalities as spatial blocks.")

    # Feature columns: exclude identifiers, target, and physical-indicator
    # columns (both level and trajectory variants, whichever are present).
    meta_cols: set[str] = {
        aza_id_col, "pref_name", "muni_code", target_col,
        "city_name_ja", "unit_code",
        cfg["target"]["physical_indicator"],
        cfg["target"]["physical_indicator"] + "_fitted",
    }
    if trajectory:
        pi_traj = cfg["target"]["physical_indicator_trajectory"]
        meta_cols.add(pi_traj)
        meta_cols.add(pi_traj + "_fitted")
    if ndbi:
        pi_ndbi = cfg["target"]["physical_indicator_ndbi"]
        meta_cols.add(pi_ndbi)
        meta_cols.add(pi_ndbi + "_fitted")
    if dw_bare:
        pi_dw = cfg["target"]["physical_indicator_dw_bare"]
        meta_cols.add(pi_dw)
        meta_cols.add(pi_dw + "_fitted")
    if dw_bare_robust:
        pi_dwr = cfg["target"]["physical_indicator_dw_bare_robust"]
        meta_cols.add(pi_dwr)
        meta_cols.add(pi_dwr + "_fitted")
        meta_cols.add("n_genuine_months")

    feature_cols = [
        c for c in data.columns
        if c not in meta_cols
        and data[c].dtype in (np.float64, np.float32, np.int64, np.int32,
                              float, int)
        and c not in set(cfg["feature_matrix"]["banned_columns"])
    ]

    print(f"  Features: {len(feature_cols)} numeric columns.")

    X = data[feature_cols].astype(float)
    y = data[target_col].astype(float)

    # Drop only rows where y is NaN; XGBoost learns optimal split direction
    # for NaN in X natively — do not impute or drop those rows.
    valid = y.notna()
    n_dropped_y = int((~valid).sum())
    n_nan_x = int(X.loc[valid].isna().any(axis=1).sum())
    if n_dropped_y:
        print(f"  Dropped {n_dropped_y} rows with NaN in y.")
    if n_nan_x:
        print(f"  {n_nan_x} rows have NaN in X (kept; XGBoost handles natively).")
    X, y, groups = X.loc[valid], y.loc[valid], groups[valid]

    assert len(X) >= 10, (
        f"Only {len(X)} valid rows after dropping NaN — too few to cross-validate."
    )

    # Build XGBoost
    models = _build_xgboost_model(cfg)

    print(f"\nRunning municipality-grouped CV{tag} "
          f"({cfg['cross_validation']['n_splits']}-fold x "
          f"{cfg['cross_validation']['n_repeats']} repeats) ...")
    cv_results = _run_grouped_cv(X, y, groups, models, cfg)

    # Train final model on all data
    print("\nTraining final XGBoost model on full dataset...")
    from xgboost import XGBRegressor
    m_cfg = cfg["model"]["xgboost"]
    final_model = XGBRegressor(
        n_estimators=int(m_cfg["n_estimators"]),
        max_depth=int(m_cfg["max_depth"]),
        learning_rate=float(m_cfg["learning_rate"]),
        subsample=float(m_cfg["subsample"]),
        colsample_bytree=float(m_cfg["colsample_bytree"]),
        reg_alpha=float(m_cfg["reg_alpha"]),
        reg_lambda=float(m_cfg["reg_lambda"]),
        random_state=int(m_cfg["random_state"]),
        verbosity=0,
    )
    final_model.fit(X.values, y.values)
    print("  Final model trained.")

    # Persist CV results (metrics only — model saved via joblib)
    out_cv = phase_b_root / cfg["output"][cv_key]
    out_cv.parent.mkdir(parents=True, exist_ok=True)
    cv_summary = {}
    for mn, res in cv_results.items():
        cv_summary[mn] = {
            "mean_metrics": res["mean_metrics"],
            "std_metrics": res["std_metrics"],
            "n_folds": len(res["fold_metrics"]),
        }
    with open(out_cv, "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2, ensure_ascii=False)
    print(f"CV results saved -> {out_cv}")

    # Save final model
    import joblib
    model_path = phase_b_root / "outputs" / model_name
    joblib.dump(final_model, model_path)
    print(f"Final model saved -> {model_path}")

    return {
        "cv_results": cv_results,
        "feature_names": feature_cols,
        "groups": groups,
        "X": X,
        "y": y,
        "unit_ids": data.loc[valid, aza_id_col].to_numpy(),
        "final_model": final_model,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_cv_train(
    cfg_path: Path | None = None,
    trajectory: bool = False,
    ndbi: bool = False,
    dw_bare: bool = False,
    dw_bare_robust: bool = False,
) -> Dict[str, Any]:
    if cfg_path is not None:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = _load_cfg()
    return train(cfg, trajectory=trajectory, ndbi=ndbi, dw_bare=dw_bare,
                 dw_bare_robust=dw_bare_robust)


if __name__ == "__main__":
    import sys as _sys
    _dw_bare_robust = "--dw_bare_robust" in _sys.argv
    _dw_bare = "--dw_bare" in _sys.argv and not _dw_bare_robust
    _ndbi    = "--ndbi" in _sys.argv
    _traj    = "--trajectory" in _sys.argv
    results = run_cv_train(trajectory=_traj, ndbi=_ndbi, dw_bare=_dw_bare,
                            dw_bare_robust=_dw_bare_robust)
    print("\nCV Summary:")
    for model, res in results["cv_results"].items():
        mm = res["mean_metrics"]
        print(f"  {model}: R2={mm['r2']:.3f}, RMSE={mm['rmse']:.4f}")
