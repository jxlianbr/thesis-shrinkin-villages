"""
Baselines and capacity audit (2026-07-17 audit series).

Standalone, read-only. Reuses phase_b/cv_train.py machinery (identical
municipality-grouped stratified 5x5 CV, identical XGBoost config) for:

  1. One-feature baseline: level target from dist_did_m alone.
  2. Ridge baselines for both targets (alpha chosen inside training folds
     via municipality-grouped inner CV; median imputation + standardization
     fitted on training folds only).
  3. Capacity sweep: max_depth {3,4,6} x n_estimators {100,200,400} x
     learning_rate {0.03,0.05,0.1} for both targets.
  4. Negative controls on the level target: within-municipality and global
     target permutation under the frozen configuration.

Writes reports/audit_2026-07-17/baselines_capacity.json incrementally.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(r"D:\data_code_masterthesis\code\phase_a")
OUT = ROOT / "reports" / "audit_2026-07-17"
sys.path.insert(0, str(ROOT / "phase_b"))

from cv_train import (  # noqa: E402
    _extract_muni_code, _load_cfg, _run_grouped_cv,
)

cfg = _load_cfg()
JSON_PATH = OUT / "baselines_capacity.json"
results: Dict[str, Any] = {}


def save() -> None:
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data assembly — identical to cv_train.train()
# ---------------------------------------------------------------------------

def assemble(res_key: str, extra_meta: set[str]) -> tuple:
    fm = pd.read_parquet(ROOT / "phase_b" / cfg["output"]["feature_matrix"])
    tdf = pd.read_parquet(ROOT / "phase_b" / cfg["output"][res_key])
    target_col = cfg["target"]["output_col"]
    data = fm.merge(tdf[["unit_id", target_col]], on="unit_id", how="inner")
    data["muni_code"] = data["unit_id"].apply(_extract_muni_code)
    groups = data["muni_code"].values
    phys = cfg["target"]["physical_indicator"]
    meta = {
        "unit_id", "pref_name", "muni_code", target_col,
        "city_name_ja", "unit_code", phys, phys + "_fitted",
    } | extra_meta
    feats = [
        c for c in data.columns
        if c not in meta
        and data[c].dtype in (np.float64, np.float32, np.int64, np.int32,
                              float, int)
        and c not in set(cfg["feature_matrix"]["banned_columns"])
    ]
    X = data[feats].astype(float)
    y = data[target_col].astype(float)
    valid = y.notna()
    return (X.loc[valid].reset_index(drop=True),
            y.loc[valid].reset_index(drop=True),
            groups[valid.to_numpy()], feats)


def xgb_model(display: str, **overrides: Any) -> dict:
    from xgboost import XGBRegressor
    m = dict(cfg["model"]["xgboost"])
    m.update(overrides)
    est = XGBRegressor(
        n_estimators=int(m["n_estimators"]),
        max_depth=int(m["max_depth"]),
        learning_rate=float(m["learning_rate"]),
        subsample=float(m["subsample"]),
        colsample_bytree=float(m["colsample_bytree"]),
        reg_alpha=float(m["reg_alpha"]),
        reg_lambda=float(m["reg_lambda"]),
        random_state=int(m["random_state"]),
        n_jobs=-1,
        verbosity=0,
    )
    return {"model": {"estimator": est, "display_name": display}}


pi_dwr = cfg["target"]["physical_indicator_dw_bare_robust"]
level = assemble("residuals", set())
traj = assemble(
    "residuals_dw_bare_robust",
    {pi_dwr, pi_dwr + "_fitted", "n_genuine_months"},
)
log(f"level: n={len(level[1])}, {len(level[3])} features; "
    f"trajectory: n={len(traj[1])}, {len(traj[3])} features")


def cell(res: dict) -> dict:
    m = res["model"]
    return {"r2_mean": round(float(m["mean_metrics"]["r2"]), 4),
            "r2_sd": round(float(m["std_metrics"]["r2"]), 4),
            "rmse_mean": round(float(m["mean_metrics"]["rmse"]), 4)}


# ---------------------------------------------------------------------------
# 1. One-feature baseline (dist_did_m, level target)
# ---------------------------------------------------------------------------
log("=== Task 1: DID-only baseline (level) ===")
X, y, groups, feats = level
X_did = X[["dist_did_m"]]
res = _run_grouped_cv(X_did, y, groups, xgb_model("XGB dist_did_m only"), cfg)
results["task1_did_only_level"] = cell(res)
save()

# ---------------------------------------------------------------------------
# 2. Ridge baselines (both targets)
# ---------------------------------------------------------------------------
log("=== Task 2: ridge baselines ===")
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GroupKFold, StratifiedGroupKFold,
)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def ridge_cv(X: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> dict:
    """5x5 grouped CV; alpha picked per outer fold via inner GroupKFold(3)."""
    cv_cfg = cfg["cross_validation"]
    Xa, ya = X.values, y.values
    fold_r2, fold_rmse, picked = [], [], []
    for r in range(int(cv_cfg["n_repeats"])):
        sgkf = StratifiedGroupKFold(
            n_splits=int(cv_cfg["n_splits"]), shuffle=True,
            random_state=int(cv_cfg["random_state"]) + r,
        )
        for tr, te in sgkf.split(Xa, groups, groups=groups):
            best_alpha, best_score = None, -np.inf
            inner = GroupKFold(n_splits=3)
            for alpha in ALPHAS:
                scores = []
                for itr, ite in inner.split(
                        Xa[tr], ya[tr], groups=groups[tr]):
                    pipe = make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(), Ridge(alpha=alpha))
                    pipe.fit(Xa[tr][itr], ya[tr][itr])
                    scores.append(
                        r2_score(ya[tr][ite], pipe.predict(Xa[tr][ite])))
                if np.mean(scores) > best_score:
                    best_score, best_alpha = np.mean(scores), alpha
            pipe = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(), Ridge(alpha=best_alpha))
            pipe.fit(Xa[tr], ya[tr])
            pred = pipe.predict(Xa[te])
            fold_r2.append(r2_score(ya[te], pred))
            fold_rmse.append(np.sqrt(mean_squared_error(ya[te], pred)))
            picked.append(best_alpha)
    return {
        "r2_mean": round(float(np.mean(fold_r2)), 4),
        "r2_sd": round(float(np.std(fold_r2, ddof=1)), 4),
        "rmse_mean": round(float(np.mean(fold_rmse)), 4),
        "alphas_picked": {str(a): int(picked.count(a)) for a in ALPHAS
                          if picked.count(a)},
        "note": ("median imputation + standardization + Ridge, all fitted "
                 "inside training folds; alpha via inner GroupKFold(3) "
                 "on training municipalities"),
    }


results["task2_ridge_level"] = ridge_cv(*level[:3])
log(f"  ridge level: {results['task2_ridge_level']['r2_mean']}")
results["task2_ridge_trajectory"] = ridge_cv(*traj[:3])
log(f"  ridge trajectory: {results['task2_ridge_trajectory']['r2_mean']}")
save()

# ---------------------------------------------------------------------------
# 4. Negative controls (level target) — run before the long sweep
# ---------------------------------------------------------------------------
log("=== Task 4: negative controls (level) ===")
rng = np.random.RandomState(int(cfg["random_state"]))
y_within = y.copy().to_numpy()
for g in np.unique(groups):
    idx = np.where(groups == g)[0]
    y_within[idx] = y_within[rng.permutation(idx)]
res_w = _run_grouped_cv(
    X, pd.Series(y_within), groups,
    xgb_model("XGB within-muni permuted y"), cfg)
results["task4_negative_control_within_muni"] = cell(res_w)
save()

y_global = y.to_numpy()[rng.permutation(len(y))]
res_g = _run_grouped_cv(
    X, pd.Series(y_global), groups,
    xgb_model("XGB globally permuted y"), cfg)
results["task4_negative_control_global"] = cell(res_g)
save()

# ---------------------------------------------------------------------------
# 3. Capacity sweep (both targets)
# ---------------------------------------------------------------------------
log("=== Task 3: capacity sweep ===")
GRID = [(d, n, lr) for d in (3, 4, 6) for n in (100, 200, 400)
        for lr in (0.03, 0.05, 0.1)]
for tgt_name, (Xt, yt, gt, _) in [("level", level), ("trajectory", traj)]:
    sweep = []
    for d, n, lr in GRID:
        res = _run_grouped_cv(
            Xt, yt, gt,
            xgb_model(f"{tgt_name} d={d} n={n} lr={lr}",
                      max_depth=d, n_estimators=n, learning_rate=lr),
            cfg)
        c = cell(res)
        c.update({"max_depth": d, "n_estimators": n, "learning_rate": lr})
        sweep.append(c)
        results[f"task3_sweep_{tgt_name}"] = sweep
        save()
        log(f"  {tgt_name} d={d} n={n} lr={lr}: "
            f"{c['r2_mean']} +/- {c['r2_sd']}")

log("DONE")
save()
