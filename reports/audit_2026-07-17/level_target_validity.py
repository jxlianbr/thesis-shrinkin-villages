"""
Level-target validity audit (2026-07-17 audit series).

Standalone, read-only against Phase A and the Phase B pipeline outputs.
Adds nothing to the pipeline; reuses phase_b modules via import.

1. Area diagnostics: correlations of mean GLCM contrast (raw and scaled)
   and of the residual level target with polygon area / log area.
2. OSM diagnostics: presence of OSM footprint columns in the aza-scale
   tables; construct-validity correlations if present.
3. Sensitivity: residualize the level indicator on elderly_ratio +
   pop_total + log_area, re-run the identical grouped 5x5 XGBoost CV,
   and recompute the SHAP mechanism-cluster ranking.

Writes: reports/audit_2026-07-17/level_target_validity.json (all numbers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"D:\data_code_masterthesis\code\phase_a")
OUT = ROOT / "reports" / "audit_2026-07-17"
sys.path.insert(0, str(ROOT / "phase_b"))

from cv_train import (  # noqa: E402
    _build_xgboost_model, _extract_muni_code, _load_cfg, _run_grouped_cv,
)
from explain import aggregate_by_mechanism  # noqa: E402
from target_builder import _ols_statsmodels  # noqa: E402

cfg = _load_cfg()
results: dict = {}


def corr(x: np.ndarray, y: np.ndarray) -> dict:
    m = ~(np.isnan(x) | np.isnan(y))
    pr, pp = stats.pearsonr(x[m], y[m])
    sr, sp = stats.spearmanr(x[m], y[m])
    return {"pearson": round(float(pr), 4), "pearson_p": float(pp),
            "spearman": round(float(sr), 4), "spearman_p": float(sp),
            "n": int(m.sum())}


# ---------------------------------------------------------------------------
# Polygon area (dissolved multi-part, metric CRS — same handling as Phase B)
# ---------------------------------------------------------------------------
print("=== Areas ===")
import geopandas as gpd  # noqa: E402

gdf = gpd.read_file(ROOT / cfg["phase_a"]["aza_polygons"])
gdf = gdf.dissolve(by="unit_id", as_index=False)
gdf = gdf.to_crs(cfg["crs_project"])
gdf["area_m2"] = gdf.geometry.area
gdf["log_area"] = np.log(gdf["area_m2"])
areas = gdf[["unit_id", "area_m2", "log_area"]]
print(f"  {len(areas)} dissolved polygons")

# ---------------------------------------------------------------------------
# 1. Area diagnostics
# ---------------------------------------------------------------------------
print("=== Item 1: area diagnostics ===")
tgt = pd.read_parquet(ROOT / "phase_b/outputs/target_residuals.parquet")
raw_ind = pd.read_csv(
    ROOT / "typology/outputs_aza/tables/indicator_matrix_raw.csv")

d = tgt.merge(areas, on="unit_id").merge(
    raw_ind[["unit_id", "S2_NDBI_contrast_mean"]].rename(
        columns={"S2_NDBI_contrast_mean": "contrast_raw"}),
    on="unit_id", how="left")

item1 = {}
for var, col in [("contrast_scaled_parent", "S2_NDBI_contrast_mean"),
                 ("contrast_raw", "contrast_raw"),
                 ("residual_target", "physical_residual")]:
    for a_var in ["area_m2", "log_area"]:
        item1[f"{var}__vs__{a_var}"] = corr(
            d[col].to_numpy(dtype=float), d[a_var].to_numpy(dtype=float))
results["item1_area_diagnostics"] = item1
print(json.dumps(item1, indent=1))

# ---------------------------------------------------------------------------
# 2. OSM diagnostics
# ---------------------------------------------------------------------------
print("=== Item 2: OSM diagnostics ===")
panel_cols = pd.read_parquet(
    ROOT / "outputs/final/features_table_aza.parquet").columns.tolist()
ready = pd.read_csv(ROOT / "preprocessing/outputs/classification_ready_aza.csv")
osm_panel = [c for c in panel_cols if c.startswith("osm")]
osm_ready = [c for c in ready.columns if c.startswith("osm")]
item2 = {"osm_columns_in_monthly_panel": osm_panel,
         "osm_columns_in_classification_ready_aza": osm_ready}

if osm_panel:
    panel_osm = pd.read_parquet(
        ROOT / "outputs/final/features_table_aza.parquet",
        columns=["unit_id"] + osm_panel)
    per_unit = panel_osm.groupby("unit_id").first().reset_index()
    du = d.merge(per_unit, on="unit_id", how="left")
    for col in osm_panel:
        item2[f"contrast_raw__vs__{col}"] = corr(
            du["contrast_raw"].to_numpy(dtype=float),
            du[col].to_numpy(dtype=float))
        item2[f"contrast_scaled__vs__{col}"] = corr(
            du["S2_NDBI_contrast_mean"].to_numpy(dtype=float),
            du[col].to_numpy(dtype=float))
results["item2_osm_diagnostics"] = item2
print(json.dumps({k: v for k, v in item2.items()}, indent=1)[:2500])

# ---------------------------------------------------------------------------
# 3. Sensitivity: residualize with log_area added, identical CV + SHAP
# ---------------------------------------------------------------------------
print("=== Item 3: area-adjusted sensitivity run ===")
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: E402
from statsmodels.stats.outliers_influence import (  # noqa: E402
    variance_inflation_factor,
)

ready_a = ready.merge(areas, on="unit_id", how="inner")
phys_col = cfg["target"]["physical_indicator"]
demo_cols = list(cfg["target"]["demographic_regressors"]) + ["log_area"]
required = ["unit_id", phys_col] + demo_cols
ready_a = ready_a.loc[ready_a[required].notna().all(axis=1)].reset_index(
    drop=True)
y0 = ready_a[phys_col].values
X0 = ready_a[demo_cols].values
res = _ols_statsmodels(
    y0, X0, list(demo_cols), len(y0),
    float(cfg["target"].get("vif_threshold", 10.0)),
    sm, variance_inflation_factor, het_breuschpagan,
)
adj_target = pd.DataFrame({
    "unit_id": ready_a["unit_id"],
    "physical_residual_adj": res["residuals"],
})
results["item3_residualizing_ols"] = {
    "regressors_final": res["predictors_final"],
    "r_squared": res["r_squared"],
    "adj_r_squared": res["adj_r_squared"],
    "n": int(len(y0)),
    "residual_sd": round(float(np.std(res["residuals"])), 4),
}
print(json.dumps(results["item3_residualizing_ols"], indent=1))

# --- assemble X exactly as cv_train.train() does -------------------------
fm = pd.read_parquet(ROOT / "phase_b" / cfg["output"]["feature_matrix"])
data = fm.merge(adj_target, on="unit_id", how="inner")
data["muni_code"] = data["unit_id"].apply(_extract_muni_code)
groups = data["muni_code"].values
meta_cols = {
    "unit_id", "pref_name", "muni_code", "physical_residual_adj",
    "city_name_ja", "unit_code",
    phys_col, phys_col + "_fitted",
}
feature_cols = [
    c for c in data.columns
    if c not in meta_cols
    and data[c].dtype in (np.float64, np.float32, np.int64, np.int32,
                          float, int)
    and c not in set(cfg["feature_matrix"]["banned_columns"])
]
print(f"  Features: {len(feature_cols)} (main run uses 42)")
X = data[feature_cols].astype(float)
y = data["physical_residual_adj"].astype(float)
valid = y.notna()
X, y, groups = X.loc[valid], y.loc[valid], groups[valid]
print(f"  n = {len(X)} aza, {len(set(groups))} municipality blocks")

models = _build_xgboost_model(cfg)
cv_results = _run_grouped_cv(X, y, groups, models, cfg)
xgb = cv_results["xgboost"]
results["item3_cv_adjusted"] = {
    "mean_metrics": {k: round(float(v), 4)
                     for k, v in xgb["mean_metrics"].items()},
    "std_metrics": {k: round(float(v), 4)
                    for k, v in xgb["std_metrics"].items()},
    "n_folds": int(len(xgb["fold_metrics"])),
    "n_features": len(feature_cols),
}
print(json.dumps(results["item3_cv_adjusted"], indent=1))

# --- final model + SHAP cluster ranking ----------------------------------
print("  Fitting final model + SHAP ...")
from xgboost import XGBRegressor  # noqa: E402
import shap  # noqa: E402

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
explainer = shap.TreeExplainer(final_model)
shap_df = pd.DataFrame(
    explainer.shap_values(X.values), columns=X.columns, index=X.index)
mech_adj = aggregate_by_mechanism(
    shap_df, cfg["explain"]["mechanism_taxonomy"])
results["item3_mechanism_ranking_adjusted"] = [
    {"mechanism": r["mechanism"],
     "mean_abs_shap": round(float(r["mean_abs_shap"]), 5),
     "n_features": int(r["n_features"])}
    for _, r in mech_adj.iterrows()
]
mech_before = pd.read_csv(ROOT / "phase_b/outputs/mechanism_importance.csv")
results["item3_mechanism_ranking_before"] = [
    {"mechanism": r["mechanism"],
     "mean_abs_shap": round(float(r["mean_abs_shap"]), 5),
     "n_features": int(r["n_features"])}
    for _, r in mech_before.iterrows()
]
print(mech_adj.to_string(index=False))

with open(OUT / "level_target_validity.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT / 'level_target_validity.json'}")
