"""
Phase B — SHAP attributions and mechanism-cluster mapping.

Exports:
  1. Per-aza SHAP values (parquet)
  2. SHAP importance aggregated by Section 2.2 mechanism cluster (CSV)
  3. Case shortlist: aza units with the largest positive residuals (CSV)

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

    Largest positive residual = physical condition worse than demography
    alone would predict — structurally deteriorating cases.

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

def run_explain(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Load trained model + feature matrix, compute SHAP, export all artefacts.

    Returns dict with: shap_df, mechanism_agg, case_shortlist.
    """
    if cfg is None:
        cfg = _load_cfg()

    phase_b_root = Path(cfg["phase_b_root"])
    aza_id_col = cfg["aza_id_col"]
    target_col = cfg["target"]["output_col"]
    taxonomy = cfg["explain"]["mechanism_taxonomy"]
    n_cases = int(cfg["explain"]["case_shortlist_n"])

    # Load final model
    import joblib
    model_path = phase_b_root / "outputs" / "final_model_xgboost.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run cv_train.py first."
        )
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    # Load feature matrix
    fm_path = phase_b_root / cfg["output"]["feature_matrix"]
    fm = pd.read_parquet(fm_path)

    # Load residual target
    res_path = phase_b_root / cfg["output"]["residuals"]
    target_df = pd.read_parquet(res_path)

    data = fm.merge(
        target_df[[aza_id_col, target_col]],
        on=aza_id_col, how="inner",
    )

    # Determine feature columns (same logic as cv_train.py)
    meta_cols = {aza_id_col, "pref_name", "muni_code", target_col,
                 cfg["target"]["physical_indicator"],
                 cfg["target"]["physical_indicator"] + "_fitted",
                 "city_name_ja", "unit_code"}
    feature_cols = [
        c for c in data.columns
        if c not in meta_cols
        and data[c].dtype in (np.float64, np.float32, np.int64, np.int32,
                              float, int)
        and c not in set(cfg["feature_matrix"]["banned_columns"])
    ]

    X = data[feature_cols].astype(float)
    valid = X.notna().all(axis=1) & data[target_col].notna()
    X, data_valid = X.loc[valid], data.loc[valid]

    print(f"SHAP computation on {len(X)} aza units, {len(feature_cols)} features.")

    # SHAP values
    shap_df = compute_shap(model, X, cfg)

    # Mechanism aggregation
    mech_agg = aggregate_by_mechanism(shap_df, taxonomy)
    print("\nMechanism cluster importance:")
    for _, row in mech_agg.iterrows():
        print(f"  {row['mechanism']:40s}  mean_abs_shap={row['mean_abs_shap']:.4f} "
              f"(n={row['n_features']} features)")

    # Case shortlist
    case_shortlist = build_case_shortlist(
        data_valid, shap_df, target_col, aza_id_col, n_cases, taxonomy,
    )
    print(f"\nTop-{n_cases} positive residual cases (structurally deteriorating):")
    print(case_shortlist[[aza_id_col, target_col, "top_shap_drivers"]].to_string())

    # Persist
    shap_path = phase_b_root / cfg["output"]["shap_values"]
    shap_path.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_parquet(shap_path, index=True)
    print(f"\nSHAP values saved -> {shap_path}")

    mech_path = phase_b_root / "outputs" / "mechanism_importance.csv"
    mech_agg.to_csv(mech_path, index=False, encoding="utf-8")
    print(f"Mechanism importance saved -> {mech_path}")

    case_path = phase_b_root / cfg["output"]["case_shortlist"]
    case_shortlist.to_csv(case_path, index=False, encoding="utf-8")
    print(f"Case shortlist saved -> {case_path}")

    return {
        "shap_df": shap_df,
        "mechanism_agg": mech_agg,
        "case_shortlist": case_shortlist,
    }


if __name__ == "__main__":
    results = run_explain()
