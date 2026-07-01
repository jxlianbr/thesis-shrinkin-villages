"""
Task 1 audit: classify all feature-matrix columns and re-run CV with
sanctioned (institutional-only) features.

Run: python phase_b/task1_audit.py
Writes: phase_b/outputs/feature_audit.csv
Prints: old vs new R2/RMSE and top-10 SHAP
Does NOT modify any pipeline files.
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
sys.stdout.reconfigure(encoding="utf-8")

with open(_CFG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

phase_b_root = Path(cfg["phase_b_root"])

# ---------------------------------------------------------------------------
# 1. Load feature matrix
# ---------------------------------------------------------------------------
fm = pd.read_parquet(phase_b_root / cfg["output"]["feature_matrix"])
all_cols = list(fm.columns)
print(f"Feature matrix: {fm.shape[0]} rows x {fm.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. Column classifier
# ---------------------------------------------------------------------------
def classify_column(col: str) -> tuple[str, str]:
    """Returns (family, KEEP/DROP)."""
    if col in ("unit_id", "pref_name", "muni_code", "city_name_ja", "unit_code"):
        return "META", "DROP"

    # EO/RS — Sentinel-2 raw bands
    if col in ("B4_mean", "B4_std", "B8_mean", "B8_std", "B11_mean", "B11_std"):
        return "EO_RS_S2_BAND", "DROP"

    # EO/RS — NDVI / NDBI / MNDWI spectral trajectories
    if any(col.startswith(p) for p in ("NDVI_", "NDBI_", "MNDWI_")):
        return "EO_RS_TRAJECTORY", "DROP"

    # EO/RS — GLCM texture (IS the target component — most circular)
    if col.startswith("S2_NDBI_"):
        return "EO_RS_GLCM_TARGET_COMPONENT", "DROP"

    # EO/RS — Dynamic World LULC fractions
    if col.startswith("dw_"):
        return "EO_RS_LULC_DW", "DROP"

    # EO/RS — VIIRS nighttime lights
    if col.startswith("viirs_"):
        return "EO_RS_VIIRS", "DROP"

    # EO/RS — OSM built stock (no temporal constraint, from Phase A)
    if col.startswith("osm_built"):
        return "EO_RS_OSM_BUILT", "DROP"

    # EO_TERRAIN — satellite DEM derivatives (Copernicus/SRTM)
    if any(col.startswith(p) for p in ("elevation_", "slope_", "aspect_", "tri_")):
        return "EO_TERRAIN_DEM", "DROP"

    # SANCTIONED: demographic-threshold binary flags
    if col in ("elderly_flag_severely", "elderly_flag_shrinking",
               "household_size_small"):
        return "DEMOGRAPHIC_FLAG", "KEEP"

    # SANCTIONED: social/structural — census 2015 household size
    if col == "household_size":
        return "SOCIAL_STRUCTURAL", "KEEP"

    # SANCTIONED: institutional fiscal — MIC kessan-card PDFs (FY<=2014)
    if col.startswith("fin_"):
        return "INSTITUTIONAL_FISCAL", "KEEP"

    # SANCTIONED: institutional policy — kaso + Heisei merger
    if col in ("kaso_flag", "kaso_type", "merged_flag", "years_since_merger"):
        return "INSTITUTIONAL_POLICY", "KEEP"

    # SANCTIONED: durability — Housing and Land Survey 2013
    if col.startswith("hls_"):
        return "DURABILITY_HLS", "KEEP"

    # SANCTIONED: accessibility — NLNI distances and buffer counts
    if (col.startswith("dist_") or col == "in_did" or
            any(col.startswith(p) for p in
                ("n_bus_", "n_medical_", "n_school_", "n_community_"))):
        return "ACCESSIBILITY_NLNI", "KEEP"

    return "UNKNOWN", "FLAG"


rows = [
    {"column": col, **dict(zip(("family", "decision"), classify_column(col)))}
    for col in all_cols
]
audit_df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 3. Save feature_audit.csv
# ---------------------------------------------------------------------------
audit_path = phase_b_root / "outputs" / "feature_audit.csv"
audit_df.to_csv(audit_path, index=False, encoding="utf-8")
print(f"\nfeature_audit.csv -> {audit_path}")

# ---------------------------------------------------------------------------
# 4. Classification summary
# ---------------------------------------------------------------------------
summary = (
    audit_df.groupby(["family", "decision"])["column"]
    .count()
    .reset_index()
    .rename(columns={"column": "n_cols"})
    .sort_values(["decision", "family"])
)
print("\nColumn classification summary:")
print(summary.to_string(index=False))

unknown = audit_df[audit_df["decision"] == "FLAG"]
if len(unknown):
    print(f"\nWARNING: {len(unknown)} UNKNOWN/FLAG columns:")
    for _, r in unknown.iterrows():
        print(f"  {r['column']}")

# ---------------------------------------------------------------------------
# 5. Confirm demographic flags are binary (not continuous aging ratios)
# ---------------------------------------------------------------------------
flag_cols = ["elderly_flag_severely", "elderly_flag_shrinking", "household_size_small"]
print("\nDemographic flag unique values (must be binary 0/1):")
for col in flag_cols:
    if col in fm.columns:
        uniq = sorted(fm[col].dropna().unique().tolist())
        print(f"  {col}: {uniq}")
    else:
        print(f"  {col}: NOT IN MATRIX")

# ---------------------------------------------------------------------------
# 6. Check for any 2020-vintage demography columns
# ---------------------------------------------------------------------------
leaky_names = [c for c in all_cols if any(
    s in c for s in ("elderly_ratio", "aging_index", "youth_ratio",
                     "age_65", "age_u15", "pop_total", "pop_male", "pop_female")
)]
print(f"\n2020-vintage demography check: {leaky_names if leaky_names else 'none found (good)'}")

# ---------------------------------------------------------------------------
# 7. Old CV results (baseline with EO/RS leakage)
# ---------------------------------------------------------------------------
cv_path = phase_b_root / cfg["output"]["cv_results"]
with open(cv_path, encoding="utf-8") as f:
    old_cv = json.load(f)

print("\n=== OLD results (all 90 features, EO/RS leakage present) ===")
for model, res in old_cv.items():
    mm = res["mean_metrics"]
    sm = res["std_metrics"]
    print(f"  {model}: R2={mm['r2']:.3f} +/- {sm['r2']:.3f}, "
          f"RMSE={mm['rmse']:.4f} +/- {sm['rmse']:.4f}")

# ---------------------------------------------------------------------------
# 8. Sanctioned feature list
# ---------------------------------------------------------------------------
sanctioned_cols = [r["column"] for r in rows if r["decision"] == "KEEP"]
print(f"\nSanctioned (KEEP) features: {len(sanctioned_cols)}")
for fam in sorted({r["family"] for r in rows if r["decision"] == "KEEP"}):
    cols_in_fam = [r["column"] for r in rows if r["family"] == fam]
    print(f"  [{fam}] ({len(cols_in_fam)}): {', '.join(cols_in_fam)}")

# ---------------------------------------------------------------------------
# 9. Run CV with sanctioned-only features
# ---------------------------------------------------------------------------
target_df = pd.read_parquet(phase_b_root / cfg["output"]["residuals"])
target_col = cfg["target"]["output_col"]
aza_id_col = cfg["aza_id_col"]

data = fm.merge(
    target_df[[aza_id_col, target_col]],
    on=aza_id_col,
    how="inner",
)
print(f"\nMerged feature matrix + residual target: {len(data)} rows")

data["_muni_code"] = data[aza_id_col].apply(lambda x: x.split(":")[-1][:5])
groups_all = data["_muni_code"].values
print(f"Municipalities (spatial blocks): {len(set(groups_all))}")

# Subset to sanctioned columns that actually exist in data
present_sanctioned = [c for c in sanctioned_cols if c in data.columns]
missing_sanctioned = [c for c in sanctioned_cols if c not in data.columns]
if missing_sanctioned:
    print(f"WARNING: {len(missing_sanctioned)} sanctioned cols not in data: {missing_sanctioned}")

X_all = data[present_sanctioned].astype(float)
y_all = data[target_col].astype(float)

# Drop rows where y is NaN; XGBoost handles NaN in X natively
valid = y_all.notna()
n_dropped_y = int((~valid).sum())
n_nan_x = int(X_all.loc[valid].isna().any(axis=1).sum())
X_cv = X_all.loc[valid]
y_cv = y_all.loc[valid]
groups_cv = groups_all[valid]

print(f"  Rows with NaN in y (dropped): {n_dropped_y}")
print(f"  Rows with NaN in X (kept, XGBoost handles): {n_nan_x}")
print(f"  Rows entering CV: {len(X_cv)}")

# Show NaN distribution per sanctioned column
nan_counts = X_cv.isna().sum()
nan_cols = nan_counts[nan_counts > 0]
if len(nan_cols):
    print("\n  NaN counts per sanctioned column:")
    for col, cnt in nan_cols.items():
        print(f"    {col}: {cnt}")

# ---------------------------------------------------------------------------
# CV run (same hyperparams as cv_train.py, XGBoost handles NaN in X)
# ---------------------------------------------------------------------------
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

cv_cfg = cfg["cross_validation"]
n_splits = int(cv_cfg["n_splits"])
n_repeats = int(cv_cfg["n_repeats"])
rs_base = int(cv_cfg["random_state"])
m_cfg = cfg["model"]["xgboost"]

X_arr = X_cv.values
y_arr = y_cv.values

print(f"\nRunning {n_splits}-fold x {n_repeats} repeat municipality-grouped CV "
      f"on {len(present_sanctioned)} sanctioned features ...")

fold_metrics = []
for r in range(n_repeats):
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=rs_base + r
    )
    for train_idx, test_idx in sgkf.split(X_arr, groups_cv, groups=groups_cv):
        est = XGBRegressor(
            n_estimators=int(m_cfg["n_estimators"]),
            max_depth=int(m_cfg["max_depth"]),
            learning_rate=float(m_cfg["learning_rate"]),
            subsample=float(m_cfg["subsample"]),
            colsample_bytree=float(m_cfg["colsample_bytree"]),
            reg_alpha=float(m_cfg["reg_alpha"]),
            reg_lambda=float(m_cfg["reg_lambda"]),
            random_state=int(m_cfg["random_state"]),
            n_jobs=-1,
            verbosity=0,
        )
        est.fit(X_arr[train_idx], y_arr[train_idx])
        y_pred = est.predict(X_arr[test_idx])

        mask = ~(np.isnan(y_arr[test_idx]) | np.isnan(y_pred))
        yt, yp = y_arr[test_idx][mask], y_pred[mask]
        fold_metrics.append({
            "r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        })

fold_df = pd.DataFrame(fold_metrics)
new_r2_mean = fold_df["r2"].mean()
new_r2_std = fold_df["r2"].std()
new_rmse_mean = fold_df["rmse"].mean()
new_rmse_std = fold_df["rmse"].std()

print(f"\n=== NEW results (sanctioned features only, no EO/RS) ===")
print(f"  XGBoost: R2={new_r2_mean:.3f} +/- {new_r2_std:.3f}, "
      f"RMSE={new_rmse_mean:.4f} +/- {new_rmse_std:.4f}")

print(f"\n  R2 delta: {new_r2_mean - old_cv['xgboost']['mean_metrics']['r2']:+.3f}")
print(f"  RMSE delta: {new_rmse_mean - old_cv['xgboost']['mean_metrics']['rmse']:+.4f}")

# ---------------------------------------------------------------------------
# 10. SHAP on final model (institutional features only)
# ---------------------------------------------------------------------------
print("\nTraining final model for SHAP ...")
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
final_model.fit(X_arr, y_arr)

try:
    import shap
    explainer = shap.TreeExplainer(final_model)
    sv = explainer.shap_values(X_cv.values)
    shap_df = pd.DataFrame(sv, columns=X_cv.columns)
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    print("\nTop-10 SHAP features (institutional model only):")
    family_map = {r["column"]: r["family"] for r in rows}
    for feat, val in mean_abs.head(10).items():
        fam = family_map.get(feat, "?")
        print(f"  {feat:40s}  {val:.4f}  [{fam}]")
except Exception as exc:
    print(f"SHAP skipped: {exc}")

print("\n--- TASK 1 AUDIT COMPLETE. Stopping for review before Tasks 2-4. ---")
