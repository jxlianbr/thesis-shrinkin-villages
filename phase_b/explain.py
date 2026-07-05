"""
Phase B — SHAP attributions and mechanism-cluster mapping.

Exports:
  1. Per-aza SHAP values (parquet)
  2. SHAP importance aggregated by Section 2.2 mechanism cluster (CSV)
  3. Case shortlist: aza units with the largest positive residuals (CSV).
     What a positive residual MEANS depends on the target variant: standing
     fabric above demographic expectation for the level target, bare ground
     expanding faster than demographic expectation for the dw_bare variants.

The mechanism taxonomy maps feature name prefixes to cluster labels
defined in the config under explain.mechanism_taxonomy.
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Mechanism taxonomy
# ---------------------------------------------------------------------------

def _assign_mechanism(feature_name: str, taxonomy: dict[str, str]) -> str:
    """Map a feature column name to its mechanism cluster via prefix lookup."""
    for prefix, cluster in taxonomy.items():
        if feature_name.startswith(prefix):
            return cluster
    return "other"


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap(
    model: Any,
    X: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute SHAP values using TreeExplainer (path-dependent method for trees).

    Returns DataFrame (n_samples x n_features) of SHAP values.
    """
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "shap is required for explain.py. Install with: pip install shap"
        ) from exc

    # TreeExplainer uses XGBoost's built-in path-dependent SHAP method;
    # no background dataset required (shap.kmeans returns legacy DenseData
    # that newer SHAP rejects as a masker).
    explainer = shap.TreeExplainer(model)

    print(f"Computing SHAP values for {len(X)} samples ...")
    shap_values = explainer.shap_values(X.values)

    shap_df = pd.DataFrame(
        shap_values,
        columns=X.columns,
        index=X.index,
    )
    return shap_df


# ---------------------------------------------------------------------------
# Mechanism aggregation
# ---------------------------------------------------------------------------

def aggregate_by_mechanism(
    shap_df: pd.DataFrame,
    taxonomy: dict[str, str],
) -> pd.DataFrame:
    """
    Aggregate mean absolute SHAP per mechanism cluster.

    Returns DataFrame: [mechanism, mean_abs_shap, n_features, features].
    """
    rows = []
    for feat in shap_df.columns:
        cluster = _assign_mechanism(feat, taxonomy)
        mean_abs = float(shap_df[feat].abs().mean())
        rows.append({"feature": feat, "mechanism": cluster,
                     "mean_abs_shap": mean_abs})
    feat_df = pd.DataFrame(rows)

    # Aggregate by mechanism
    agg = (
        feat_df.groupby("mechanism")
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            n_features=("feature", "count"),
            features=("feature", lambda x: ", ".join(sorted(x))),
        )
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
    )
    return agg


# ---------------------------------------------------------------------------
# Case shortlist
# ---------------------------------------------------------------------------

def build_case_shortlist(
    data: pd.DataFrame,
    shap_df: pd.DataFrame,
    target_col: str,
    aza_id_col: str,
    n: int,
    taxonomy: dict[str, str],
) -> pd.DataFrame:
    """
    Return the `n` aza units with the largest positive residuals.

    The interpretation of a positive residual is variant-dependent (no sign
    flip is applied anywhere in target_builder.py):
      - Level target (S2_NDBI_contrast_mean): higher contrast tracks
        physically stronger settlements, so positive = standing fabric ABOVE
        demographic expectation — resilience candidates for Chapter 6.
      - dw_bare variants (bare-ground slope): positive = bare ground expanding
        faster than demographic expectation — deterioration candidates.

    For each case, include the top-3 SHAP drivers and their mechanism cluster.
    """
    cases = data[[aza_id_col, target_col]].copy()
    cases = cases.loc[cases[target_col].notna()].copy()
    cases = cases.nlargest(n, target_col).reset_index(drop=True)

    # Add top SHAP drivers per case
    top_drivers = []
    for uid in cases[aza_id_col]:
        idx = data.index[data[aza_id_col] == uid]
        if len(idx) == 0 or idx[0] not in shap_df.index:
            top_drivers.append("")
            continue
        row_shap = shap_df.loc[idx[0]].abs().nlargest(3)
        drivers = "; ".join(
            f"{feat} [{_assign_mechanism(feat, taxonomy)}]"
            for feat in row_shap.index
        )
        top_drivers.append(drivers)

    cases["top_shap_drivers"] = top_drivers
    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_explain(
    cfg: Dict[str, Any] | None = None,
    trajectory: bool = False,
    ndbi: bool = False,
    dw_bare: bool = False,
    dw_bare_robust: bool = False,
) -> Dict[str, Any]:
    """
    Load trained model + feature matrix, compute SHAP, export all artefacts.

    Parameters
    ----------
    trajectory : bool
        If True, loads the GLCM contrast-slope model (RETIRED — kept for reference).
    ndbi : bool
        If True, loads the NDBI_slope model (RETIRED — spatial gradient artifact).
    dw_bare : bool
        If True, loads the plain-OLS dw_bare_frac_slope model (SUPERSEDED
        2026-06-30 — kept only for before/after comparison).
    dw_bare_robust : bool
        If True, loads the Theil-Sen, genuine-month-filtered dw_bare_frac
        slope model (ACTIVE) and writes outputs to the *_dw_bare_robust
        paths in config.

    Returns dict with: shap_df, mechanism_agg, case_shortlist.
    """
    if cfg is None:
        cfg = _load_cfg()

    phase_b_root = Path(cfg["phase_b_root"])
    aza_id_col = cfg["aza_id_col"]
    target_col = cfg["target"]["output_col"]
    taxonomy = cfg["explain"]["mechanism_taxonomy"]
    n_cases = int(cfg["explain"]["case_shortlist_n"])

    # Select variant paths
    if dw_bare_robust:
        model_filename = "final_model_xgboost_dw_bare_robust.joblib"
        res_key   = "residuals_dw_bare_robust"
        shap_key  = "shap_values_dw_bare_robust"
        case_key  = "case_shortlist_dw_bare_robust"
        mech_file = "mechanism_importance_dw_bare_robust.csv"
        tag = " [DW_BARE_ROBUST]"
    elif dw_bare:
        model_filename = "final_model_xgboost_dw_bare.joblib"
        res_key   = "residuals_dw_bare"
        shap_key  = "shap_values_dw_bare"
        case_key  = "case_shortlist_dw_bare"
        mech_file = "mechanism_importance_dw_bare.csv"
        tag = " [DW_BARE]"
    elif ndbi:
        model_filename = "final_model_xgboost_ndbi.joblib"
        res_key   = "residuals_ndbi"
        shap_key  = "shap_values_ndbi"
        case_key  = "case_shortlist_ndbi"
        mech_file = "mechanism_importance_ndbi.csv"
        tag = " [NDBI]"
    elif trajectory:
        model_filename = "final_model_xgboost_trajectory.joblib"
        res_key   = "residuals_trajectory"
        shap_key  = "shap_values_trajectory"
        case_key  = "case_shortlist_trajectory"
        mech_file = "mechanism_importance_trajectory.csv"
        tag = " [TRAJECTORY]"
    else:
        model_filename = "final_model_xgboost.joblib"
        res_key   = "residuals"
        shap_key  = "shap_values"
        case_key  = "case_shortlist"
        mech_file = "mechanism_importance.csv"
        tag = ""

    # Load final model
    import joblib
    model_path = phase_b_root / "outputs" / model_filename
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            f"Run cv_train.py{' --trajectory' if trajectory else ''} first."
        )
    model = joblib.load(model_path)
    print(f"Loaded model{tag} from {model_path}")

    # Load feature matrix
    fm_path = phase_b_root / cfg["output"]["feature_matrix"]
    fm = pd.read_parquet(fm_path)

    # Load residual target
    res_path = phase_b_root / cfg["output"][res_key]
    target_df = pd.read_parquet(res_path)

    data = fm.merge(
        target_df[[aza_id_col, target_col]],
        on=aza_id_col, how="inner",
    )

    # Determine feature columns — same logic as cv_train.py
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

    X = data[feature_cols].astype(float)
    # Keep all rows with a valid target; XGBoost TreeExplainer handles NaN in X
    # via the trained missing-value branches — no imputation or row-dropping needed.
    valid = data[target_col].notna()
    X, data_valid = X.loc[valid], data.loc[valid]

    n_nan_rows = int(X.isna().any(axis=1).sum())
    print(f"SHAP computation{tag} on {len(X)} aza units, {len(feature_cols)} features "
          f"({n_nan_rows} rows have NaN in X, handled natively).")

    # SHAP values
    shap_df = compute_shap(model, X, cfg)

    # Mechanism aggregation
    mech_agg = aggregate_by_mechanism(shap_df, taxonomy)
    print(f"\nMechanism cluster importance{tag}:")
    for _, row in mech_agg.iterrows():
        print(f"  {row['mechanism']:40s}  mean_abs_shap={row['mean_abs_shap']:.4f} "
              f"(n={row['n_features']} features)")

    # Case shortlist — positive-residual meaning depends on the target variant
    # (see build_case_shortlist docstring).
    if dw_bare or dw_bare_robust:
        case_desc = "bare ground expanding fastest vs demographic expectation"
    elif trajectory or ndbi:
        case_desc = "largest trend residual (retired variant, reference only)"
    else:
        case_desc = "standing fabric most above demographic expectation"
    case_shortlist = build_case_shortlist(
        data_valid, shap_df, target_col, aza_id_col, n_cases, taxonomy,
    )
    print(f"\nTop-{n_cases} positive residual cases ({case_desc}){tag}:")
    print(case_shortlist[[aza_id_col, target_col, "top_shap_drivers"]].to_string())

    # Persist
    shap_path = phase_b_root / cfg["output"][shap_key]
    shap_path.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_parquet(shap_path, index=True)
    print(f"\nSHAP values saved -> {shap_path}")

    mech_path = phase_b_root / "outputs" / mech_file
    mech_agg.to_csv(mech_path, index=False, encoding="utf-8")
    print(f"Mechanism importance saved -> {mech_path}")

    case_path = phase_b_root / cfg["output"][case_key]
    case_shortlist.to_csv(case_path, index=False, encoding="utf-8")
    print(f"Case shortlist saved -> {case_path}")

    return {
        "shap_df": shap_df,
        "mechanism_agg": mech_agg,
        "case_shortlist": case_shortlist,
    }


if __name__ == "__main__":
    import sys as _sys
    _dw_bare_robust = "--dw_bare_robust" in _sys.argv
    _dw_bare = "--dw_bare" in _sys.argv and not _dw_bare_robust
    _ndbi    = "--ndbi" in _sys.argv
    _traj    = "--trajectory" in _sys.argv
    results = run_explain(trajectory=_traj, ndbi=_ndbi, dw_bare=_dw_bare,
                           dw_bare_robust=_dw_bare_robust)
