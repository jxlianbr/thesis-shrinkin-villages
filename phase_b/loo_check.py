"""
Phase B — Leave-one-municipality-out (LOO) robustness check.

Implements the check declared in phase_b_config.yaml under
cross_validation.loo_check (previously wired in config but never executed).
It backs the Chapter 5/6 claim that the cross-validated result is robust to
the exclusion of any single municipality, in two forms:

  1. LOO cross-validation: each of the 65 municipalities is held out once,
     the model is trained on the remaining 64, and the pooled out-of-fold
     predictions yield an overall R2 comparable to the grouped 5x5 CV.
  2. Jackknife on the metric: the pooled LOO R2 is recomputed 65 times,
     each time excluding one municipality's rows from the evaluation, so
     that no single municipality can be responsible for the headline number.

Persists outputs/loo_check.json (level target) or
outputs/loo_check_dw_bare_robust.json (--dw_bare_robust).

Data loading mirrors cv_train.py exactly (same feature-column selection,
same banned list, same NaN policy) so the LOO estimate is computed on the
identical matrix the grouped CV used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from cv_train import _load_cfg, _extract_muni_code

_HERE = Path(__file__).resolve().parent


def _pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_squared_error, r2_score
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _load_xy(cfg: Dict[str, Any], dw_bare_robust: bool) -> Dict[str, Any]:
    """Load feature matrix + residual target exactly as cv_train.train() does."""
    phase_b_root = Path(cfg["phase_b_root"])
    aza_id_col = cfg["aza_id_col"]

    res_key = "residuals_dw_bare_robust" if dw_bare_robust else "residuals"

    fm = pd.read_parquet(phase_b_root / cfg["output"]["feature_matrix"])
    target_df = pd.read_parquet(phase_b_root / cfg["output"][res_key])
    target_col = cfg["target"]["output_col"]

    data = fm.merge(target_df[[aza_id_col, target_col]], on=aza_id_col,
                    how="inner")
    data["muni_code"] = data[aza_id_col].apply(_extract_muni_code)
    groups = data["muni_code"].values

    meta_cols: set[str] = {
        aza_id_col, "pref_name", "muni_code", target_col,
        "city_name_ja", "unit_code",
        cfg["target"]["physical_indicator"],
        cfg["target"]["physical_indicator"] + "_fitted",
    }
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
    y = data[target_col].astype(float)
    valid = y.notna()
    return {
        "X": X.loc[valid],
        "y": y.loc[valid],
        "groups": groups[valid],
        "feature_cols": feature_cols,
    }


def run_loo_check(dw_bare_robust: bool = False) -> Dict[str, Any]:
    cfg = _load_cfg()
    loo_cfg = cfg["cross_validation"]["loo_check"]
    if not loo_cfg.get("enabled", False):
        print("loo_check disabled in config; nothing to do.")
        return {}

    tag = " [DW_BARE_ROBUST]" if dw_bare_robust else " [LEVEL]"
    print(f"Loading data for LOO check{tag} ...")
    d = _load_xy(cfg, dw_bare_robust)
    X, y, groups = d["X"].values, d["y"].values, d["groups"]
    munis = sorted(set(groups))
    print(f"  {len(y)} aza, {len(d['feature_cols'])} features, "
          f"{len(munis)} municipalities.")

    from xgboost import XGBRegressor
    m_cfg = cfg["model"]["xgboost"]

    def _fresh_model() -> "XGBRegressor":
        return XGBRegressor(
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

    oof_pred = np.full(len(y), np.nan)
    per_muni: Dict[str, Dict[str, Any]] = {}
    for i, muni in enumerate(munis):
        test_mask = groups == muni
        est = _fresh_model()
        est.fit(X[~test_mask], y[~test_mask])
        oof_pred[test_mask] = est.predict(X[test_mask])
        per_muni[muni] = {"n": int(test_mask.sum())}
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(munis)} municipalities done.")

    assert not np.isnan(oof_pred).any(), "Some rows never predicted."
    pooled = _pooled_metrics(y, oof_pred)
    print(f"  Pooled LOO: R2={pooled['r2']:.3f}, RMSE={pooled['rmse']:.4f}")

    # Jackknife: pooled R2 with each municipality's rows excluded from
    # evaluation. Small spread = no single municipality drives the result.
    jack = {}
    for muni in munis:
        keep = groups != muni
        jack[muni] = _pooled_metrics(y[keep], oof_pred[keep])["r2"]
    jack_vals = np.array(list(jack.values()))
    jack_summary = {
        "min": float(jack_vals.min()),
        "max": float(jack_vals.max()),
        "mean": float(jack_vals.mean()),
        "std": float(jack_vals.std()),
        "min_muni": min(jack, key=jack.get),
        "max_muni": max(jack, key=jack.get),
    }
    print(f"  Jackknife R2 excluding each municipality: "
          f"[{jack_summary['min']:.3f}, {jack_summary['max']:.3f}], "
          f"mean {jack_summary['mean']:.3f}")

    result = {
        "target_variant": "dw_bare_robust" if dw_bare_robust else "level",
        "n_rows": int(len(y)),
        "n_features": len(d["feature_cols"]),
        "n_municipalities": len(munis),
        "pooled_loo": pooled,
        "jackknife_r2_excluding_muni": jack,
        "jackknife_summary": jack_summary,
    }

    out_name = ("loo_check_dw_bare_robust.json" if dw_bare_robust
                else "loo_check.json")
    out_path = _HERE / "outputs" / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"LOO check saved -> {out_path}")
    return result


if __name__ == "__main__":
    import sys as _sys
    _dwr = "--dw_bare_robust" in _sys.argv
    run_loo_check(dw_bare_robust=_dwr)
