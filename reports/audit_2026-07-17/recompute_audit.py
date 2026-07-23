"""
Numbers audit 2026-07-17 — recomputation script.

Read-only against all pipeline outputs (Phase A frozen). Writes
recompute_results.json into reports/audit_2026-07-17/.

Covers:
  Item 1 — monthly panel shape / distinct units / exclusion stage evidence
  Item 2 — level-model out-of-fold RMSE vs R2 and target scaling
  Item 3 — Moran's I inventory (6 series, exact pipeline weight specs)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\data_code_masterthesis\code\phase_a")
OUT = ROOT / "reports" / "audit_2026-07-17"
sys.path.insert(0, str(ROOT / "phase_b"))

results: dict = {}

# ---------------------------------------------------------------------------
# Item 1 — monthly panel
# ---------------------------------------------------------------------------
print("=== Item 1: monthly panel ===")
panel = pd.read_parquet(ROOT / "outputs/final/features_table_aza.parquet")
month_col = "month" if "month" in panel.columns else [
    c for c in panel.columns if "month" in c.lower()][0]
n_units_panel = int(panel["unit_id"].nunique())
n_months = int(panel[month_col].nunique())
ready = pd.read_csv(ROOT / "preprocessing/outputs/classification_ready_aza.csv")
n_units_ready = int(ready["unit_id"].nunique())
panel_units = set(panel["unit_id"].unique())
ready_units = set(ready["unit_id"].unique())
results["item1_panel"] = {
    "panel_file": "outputs/final/features_table_aza.parquet",
    "shape_rows": int(panel.shape[0]),
    "shape_cols": int(panel.shape[1]),
    "n_distinct_unit_ids": n_units_panel,
    "n_distinct_months": n_months,
    "units_x_months": n_units_panel * n_months,
    "rows_equal_units_x_months": bool(
        panel.shape[0] == n_units_panel * n_months),
    "classification_ready_rows": int(len(ready)),
    "classification_ready_distinct_units": n_units_ready,
    "n_panel_units_not_in_ready": len(panel_units - ready_units),
    "ready_is_subset_of_panel": ready_units.issubset(panel_units),
}
print(json.dumps(results["item1_panel"], indent=2))

# ---------------------------------------------------------------------------
# Item 2 — level model RMSE vs R2, target scaling
# ---------------------------------------------------------------------------
print("\n=== Item 2: level-model RMSE vs R2 ===")
tgt = pd.read_parquet(ROOT / "phase_b/outputs/target_residuals.parquet")
sp = pd.read_parquet(ROOT / "phase_b/outputs/spatial_residuals.parquet")
cv = json.load(open(ROOT / "phase_b/outputs/cv_results.json"))
rep = json.load(open(
    ROOT / "preprocessing/outputs/preprocessing_report_aza.json"))

y = tgt["physical_residual"].to_numpy()
oof = sp["oof_residual"].to_numpy()
sd_pop = float(np.std(y))          # ddof=0
sd_sample = float(np.std(y, ddof=1))
oof_rmse = float(np.sqrt(np.mean(oof ** 2)))
pooled_r2 = float(1.0 - np.sum(oof ** 2) / np.sum((y - y.mean()) ** 2))
fold_r2 = cv["xgboost"]["mean_metrics"]["r2"]
fold_rmse = cv["xgboost"]["mean_metrics"]["rmse"]

# scaling of the target's parent variable
center = rep["transform_metadata"]["scaler_center"]["S2_NDBI_contrast_mean"]
scale = rep["transform_metadata"]["scaler_scale"]["S2_NDBI_contrast_mean"]
logged = rep["transform_metadata"].get("log_features") or rep[
    "transform_metadata"].get("log_transformed") or []
# is the target's parent column identical to the preprocessed one?
merged = tgt.merge(
    ready[["unit_id", "S2_NDBI_contrast_mean"]], on="unit_id",
    suffixes=("_tgt", "_ready"))
same = bool(np.allclose(
    merged["S2_NDBI_contrast_mean_tgt"], merged["S2_NDBI_contrast_mean_ready"]))
parent = ready["S2_NDBI_contrast_mean"].to_numpy()

results["item2_rmse"] = {
    "cv_results_fold_mean_r2": fold_r2,
    "cv_results_fold_mean_rmse": fold_rmse,
    "recomputed_pooled_oof_rmse": round(oof_rmse, 4),
    "recomputed_pooled_oof_r2": round(pooled_r2, 4),
    "target_sd_ddof0": round(sd_pop, 4),
    "target_sd_ddof1": round(sd_sample, 4),
    "parent_variable_sd": round(float(np.std(parent)), 4),
    "implied_rmse_from_fold_r2_and_sd": round(
        float(np.sqrt(1 - fold_r2) * sd_pop), 4),
    "rmse_if_target_had_unit_variance": round(float(np.sqrt(1 - fold_r2)), 4),
    "scaler_center_log1p_S2_NDBI_contrast_mean": center,
    "scaler_scale_iqr_log1p_S2_NDBI_contrast_mean": scale,
    "log1p_feature_list_recorded": logged if isinstance(logged, list) else str(logged),
    "target_parent_equals_classification_ready_column": same,
}
print(json.dumps(results["item2_rmse"], indent=2))

# ---------------------------------------------------------------------------
# Item 3 — Moran's I inventory
# ---------------------------------------------------------------------------
print("\n=== Item 3: Moran's I inventory ===")
from spatial_autocorrelation import (  # noqa: E402
    _align_to_units, _load_boundaries, _load_cfg, _morans_i,
)

cfg = _load_cfg()
perms = cfg["spatial"]["permutations"]
seed = cfg["spatial"]["seed"]
boundaries = _load_boundaries(cfg)  # dissolved multi-part, EPSG:6680

def moran_phaseb(df: pd.DataFrame, col: str) -> dict:
    d = df[df[col].notna()].reset_index(drop=True)
    ids = d["unit_id"].to_numpy()
    gdf, mask = _align_to_units(boundaries, ids)
    vals = d[col].to_numpy()[mask]
    return _morans_i(vals, gdf, perms, seed)

traj = pd.read_parquet(
    ROOT / "phase_b/outputs/target_dw_bare_robust_trajectory.parquet")
sp_traj = pd.read_parquet(
    ROOT / "phase_b/outputs/spatial_residuals_dw_bare_robust.parquet")

inventory = {}
print("level target residual ...")
inventory["level_target_residual"] = moran_phaseb(tgt, "physical_residual")
print("level OOF residual ...")
inventory["level_oof_residual"] = moran_phaseb(sp, "oof_residual")
print("trajectory target residual ...")
inventory["trajectory_target_residual"] = moran_phaseb(
    traj, "physical_residual")
print("trajectory OOF residual ...")
inventory["trajectory_oof_residual"] = moran_phaseb(sp_traj, "oof_residual")
print("raw Theil-Sen slope (before residualization) ...")
inventory["raw_theilsen_slope"] = moran_phaseb(
    traj, "dw_bare_frac_slope_theilsen")

# --- elderly-ratio regression residuals, exact typology step3 spec ---------
print("elderly-ratio OLS residuals (typology spec) ...")
import geopandas as gpd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from esda.moran import Moran  # noqa: E402
from libpysal.weights import Queen  # noqa: E402

raw_ind = pd.read_csv(
    ROOT / "typology/outputs_aza/tables/indicator_matrix_raw.csv")
physical = ["NDBI_slope", "viirs_mean_slope", "S2_NDBI_contrast_slope",
            "NDVI_cv", "NDVI_seasonal_amp", "NDBI_seasonal_amp",
            "S2_NDBI_contrast_mean", "S2_NDBI_entropy_mean"]
yv = raw_ind["elderly_ratio"].to_numpy()
X = raw_ind[physical]
valid = ~(np.isnan(yv) | X.isna().any(axis=1).to_numpy())
model = sm.OLS(yv[valid], sm.add_constant(X.loc[valid].to_numpy())).fit()
resid = np.asarray(model.resid)
unit_ids_valid = raw_ind.loc[valid, "unit_id"].to_numpy()

gdf_t = gpd.read_file(
    ROOT / "admin_demographics/boundaries/aza.gpkg")
gdf_t = gdf_t.drop_duplicates(subset="unit_id")  # typology keeps first part
gdf_m = gdf_t.set_index("unit_id").reindex(unit_ids_valid).reset_index()
gdf_m = gdf_m[gdf_m.geometry.notna()].reset_index(drop=True)
keep = np.isin(unit_ids_valid, gdf_m["unit_id"].values)
gdf_m = gdf_m.to_crs(epsg=6690)
w_t = Queen.from_dataframe(gdf_m, use_index=False, silence_warnings=True)
w_t.transform = "r"
np.random.seed(seed)
mi = Moran(resid[keep], w_t, permutations=999)
inventory["elderly_ratio_ols_residuals"] = {
    "morans_i": round(float(mi.I), 4),
    "z_sim": round(float(mi.z_sim), 4),
    "p_sim": round(float(mi.p_sim), 4),
    "n_units": int(keep.sum()),
    "ols_r_squared": round(float(model.rsquared), 4),
    "weights_spec": ("Queen contiguity, first-part-only polygons "
                     "(drop_duplicates), EPSG:6690, row-standardised, "
                     "999 permutations"),
}

results["item3_moran"] = inventory
print(json.dumps(inventory, indent=2))

with open(OUT / "recompute_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT / 'recompute_results.json'}")
