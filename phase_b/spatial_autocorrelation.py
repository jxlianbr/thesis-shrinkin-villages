"""
Phase B — Spatial autocorrelation diagnostics (Moran's I).

Implements FINDINGS.md improvement #5: quantify spatial autocorrelation in
(a) the residual target itself and (b) the municipality-grouped CV
out-of-fold residuals.  Mirrors the Phase A diagnostic in
typology/src/step3_relationships.py (Queen contiguity weights, row
standardisation, permutation-based pseudo p-value) at aza scale.

Interpretation guide:
  - Moran's I on the TARGET residual measures how spatially clustered the
    phenomenon itself is (context; high values are expected for any
    landscape-derived quantity).
  - Moran's I on the OOF residual measures the structure the model FAILS
    to capture.  If it stays high, the grouped-CV spatial blocking is not
    sufficient and a spatially explicit model (GWR, spatial error) would
    be warranted; if it drops toward zero, the sanctioned features absorb
    the spatial structure.

Outputs (per target variant, paths from phase_b_config.yaml):
  1. spatial_autocorrelation*.json — Moran's I, z, pseudo-p for both series
  2. spatial_residuals*.parquet    — per-unit target + OOF residuals, for
                                     later mapping / LISA follow-up

Usage:
    python phase_b/spatial_autocorrelation.py                    # level target
    python phase_b/spatial_autocorrelation.py --dw_bare_robust   # active trajectory target
    (flags --trajectory / --ndbi / --dw_bare also accepted, matching cv_train.py)
"""
from __future__ import annotations

import json
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
# Boundary handling
# ---------------------------------------------------------------------------

def _dissolve_multipart(gdf: Any) -> Any:
    """
    Collapse multi-part boundary rows sharing a unit_id into single
    (multi)polygons via geometric union.

    Dissolving (rather than drop_duplicates as in Phase A at mura scale)
    matters here: Queen contiguity computed on only the first part of a
    multi-part aza would miss the neighbours of the remaining parts.
    """
    n_dup = int(gdf["unit_id"].duplicated().sum())
    if n_dup:
        print(f"  Dissolved {n_dup} multi-part boundary rows on unit_id")
        gdf = gdf.dissolve(by="unit_id", as_index=False)
    return gdf


def _align_to_units(gdf: Any, unit_ids: np.ndarray) -> tuple[Any, np.ndarray]:
    """
    Subset and reorder boundaries to match unit_ids row-for-row.

    Returns (aligned_gdf, in_bounds_mask) where in_bounds_mask flags which
    of the requested unit_ids have a geometry.  aligned_gdf has exactly
    in_bounds_mask.sum() rows, in unit_ids order.
    """
    in_bounds = np.isin(unit_ids, gdf["unit_id"].values)
    if not in_bounds.all():
        print(f"  WARNING: {int((~in_bounds).sum())} units lack geometry")
    aligned = (
        gdf.set_index("unit_id")
        .reindex(unit_ids[in_bounds])
        .reset_index()
    )
    return aligned, in_bounds


def _load_boundaries(cfg: Dict[str, Any]) -> Any:
    """Load aza polygons, dissolve multi-part rows, project to metric CRS."""
    import geopandas as gpd

    gpkg = Path(cfg["phase_a_root"]) / cfg["phase_a"]["aza_polygons"]
    print(f"Loading boundaries: {gpkg}")
    gdf = gpd.read_file(gpkg)
    gdf = _dissolve_multipart(gdf)
    gdf = gdf.to_crs(cfg["crs_project"])
    print(f"  {len(gdf)} unit polygons (CRS {cfg['crs_project']})")
    return gdf


# ---------------------------------------------------------------------------
# Moran's I
# ---------------------------------------------------------------------------

def _morans_i(
    values: np.ndarray,
    gdf_aligned: Any,
    permutations: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Queen-contiguity Moran's I with permutation pseudo p-value.

    gdf_aligned rows must correspond positionally to values.
    """
    from esda.moran import Moran
    from libpysal.weights import Queen

    assert len(values) == len(gdf_aligned), (
        f"values ({len(values)}) and geometries ({len(gdf_aligned)}) misaligned"
    )

    w = Queen.from_dataframe(gdf_aligned, use_index=False, silence_warnings=True)
    n_islands = len(w.islands)
    if n_islands:
        print(f"    {n_islands} island units (no contiguous neighbour) "
              "contribute no spatial lag")
    w.transform = "r"

    np.random.seed(seed)  # esda permutations draw from the numpy global RNG
    mi = Moran(values, w, permutations=permutations)

    return {
        "morans_i": round(float(mi.I), 4),
        "expected_i": round(float(mi.EI), 4),
        "z_sim": round(float(mi.z_sim), 4),
        "p_sim": round(float(mi.p_sim), 4),
        "significant_005": bool(mi.p_sim < 0.05),
        "n_units": int(len(values)),
        "n_islands": n_islands,
        "permutations": int(permutations),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_spatial_autocorrelation(
    cfg: Dict[str, Any] | None = None,
    trajectory: bool = False,
    ndbi: bool = False,
    dw_bare: bool = False,
    dw_bare_robust: bool = False,
) -> Dict[str, Any]:
    """
    Compute Moran's I on the residual target and on CV OOF residuals.

    Variant flags select the target exactly as in cv_train.train().
    Re-runs the grouped CV (deterministic, random_state from config) to
    obtain out-of-fold predictions; cv_train persists its usual outputs
    with identical content as a side effect.
    """
    from cv_train import run_cv_train

    if cfg is None:
        cfg = _load_cfg()

    phase_b_root = Path(cfg["phase_b_root"])
    aza_id_col = cfg["aza_id_col"]
    target_col = cfg["target"]["output_col"]
    sp_cfg = cfg["spatial"]

    if dw_bare_robust:
        res_key, out_key, tag = ("residuals_dw_bare_robust",
                                 "dw_bare_robust", " [DW_BARE_ROBUST]")
    elif dw_bare:
        res_key, out_key, tag = "residuals_dw_bare", "dw_bare", " [DW_BARE]"
    elif ndbi:
        res_key, out_key, tag = "residuals_ndbi", "ndbi", " [NDBI]"
    elif trajectory:
        res_key, out_key, tag = ("residuals_trajectory",
                                 "trajectory", " [TRAJECTORY]")
    else:
        res_key, out_key, tag = "residuals", "", ""

    suffix = f"_{out_key}" if out_key else ""
    json_path = phase_b_root / cfg["output"][f"spatial_autocorrelation{suffix}"]
    parquet_path = phase_b_root / cfg["output"][f"spatial_residuals{suffix}"]

    print(f"=== Spatial autocorrelation diagnostics{tag} ===")

    # --- Residual target ---------------------------------------------------
    res_path = phase_b_root / cfg["output"][res_key]
    if not res_path.exists():
        raise FileNotFoundError(
            f"Residual target not found: {res_path}. Run target_builder.py first."
        )
    res_df = pd.read_parquet(res_path)
    res_df = res_df[res_df[target_col].notna()].reset_index(drop=True)
    print(f"Residual target: {res_path.name} ({len(res_df)} aza)")

    boundaries = _load_boundaries(cfg)

    unit_ids_target = res_df[aza_id_col].to_numpy()
    gdf_target, in_bounds = _align_to_units(boundaries, unit_ids_target)
    target_vals = res_df[target_col].to_numpy()[in_bounds]

    print("\nMoran's I on target residual ...")
    moran_target = _morans_i(
        target_vals, gdf_target, sp_cfg["permutations"], sp_cfg["seed"],
    )
    print(f"    I={moran_target['morans_i']:.4f}, "
          f"z={moran_target['z_sim']:.2f}, p={moran_target['p_sim']:.4f}")

    # --- CV out-of-fold residuals -------------------------------------------
    print("\nRe-running grouped CV to obtain out-of-fold predictions ...")
    cv_out = run_cv_train(
        trajectory=trajectory, ndbi=ndbi, dw_bare=dw_bare,
        dw_bare_robust=dw_bare_robust,
    )
    xgb_res = cv_out["cv_results"]["xgboost"]
    oof_resid_all = cv_out["y"].to_numpy() - xgb_res["oof_pred"]
    unit_ids_oof = cv_out["unit_ids"]

    gdf_oof, in_bounds_oof = _align_to_units(boundaries, unit_ids_oof)
    oof_vals = oof_resid_all[in_bounds_oof]

    print("\nMoran's I on CV out-of-fold residual ...")
    moran_oof = _morans_i(
        oof_vals, gdf_oof, sp_cfg["permutations"], sp_cfg["seed"],
    )
    print(f"    I={moran_oof['morans_i']:.4f}, "
          f"z={moran_oof['z_sim']:.2f}, p={moran_oof['p_sim']:.4f}")

    # --- Persist -------------------------------------------------------------
    results = {
        "variant": out_key or "level",
        "target_col": target_col,
        "residual_source": str(res_path.name),
        "weights": sp_cfg["weights"],
        "cv_mean_r2": round(
            float(xgb_res["mean_metrics"]["r2"]), 4,
        ),
        "moran_target_residual": moran_target,
        "moran_oof_residual": moran_oof,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved -> {json_path}")

    per_unit = pd.DataFrame({
        aza_id_col: unit_ids_oof[in_bounds_oof],
        "target_residual": (
            res_df.set_index(aza_id_col)[target_col]
            .reindex(unit_ids_oof[in_bounds_oof])
            .to_numpy()
        ),
        "oof_residual": oof_vals,
    })
    per_unit.to_parquet(parquet_path, index=False)
    print(f"Per-unit residuals saved -> {parquet_path}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    _dw_bare_robust = "--dw_bare_robust" in _sys.argv
    _dw_bare = "--dw_bare" in _sys.argv and not _dw_bare_robust
    _ndbi = "--ndbi" in _sys.argv
    _traj = "--trajectory" in _sys.argv
    out = run_spatial_autocorrelation(
        trajectory=_traj, ndbi=_ndbi, dw_bare=_dw_bare,
        dw_bare_robust=_dw_bare_robust,
    )
    print("\nSummary:")
    print(f"  Target residual:  I={out['moran_target_residual']['morans_i']:.4f} "
          f"(p={out['moran_target_residual']['p_sim']:.4f})")
    print(f"  OOF residual:     I={out['moran_oof_residual']['morans_i']:.4f} "
          f"(p={out['moran_oof_residual']['p_sim']:.4f})")
