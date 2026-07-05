# Phase B — Structural Trajectory Analysis

Phase B extends the Phase A aza pipeline to explain *why* villages show
divergent physical-space trajectories after controlling for their demographic
situation.  The predictor set covers accessibility, municipal finance,
housing durability, policy designations, and administrative history.

Phase A is **frozen** — no Phase A file is edited.  Phase B imports Phase A
helpers (OLS, CV) by adding `phase_a/` to `sys.path` at runtime.

---

## Run order

```
# 1. Build the decoupling target (OLS residual)
python phase_b/target_builder.py

# 2. Compute NLNI accessibility features
python phase_b/accessibility_features.py

# 3. Assemble the full feature matrix
python phase_b/feature_matrix.py

# 4. Municipality-grouped CV + final XGBoost model
python phase_b/cv_train.py

# 5. SHAP attributions + mechanism-cluster mapping + case shortlist
python phase_b/explain.py

# 6. Spatial autocorrelation diagnostics (Moran's I on target + OOF residuals)
python phase_b/spatial_autocorrelation.py
```

Each step persists its output under `phase_b/outputs/` before the next
step reads it.  Run from the repo root (`D:/data_code_masterthesis/code/`).

---

## Module summary

| Module | What it does |
|--------|-------------|
| `target_builder.py` | Builds the Phase B target as an OLS residual of a physical indicator on elderly_ratio + pop_total (VIF-pruned). Four indicator variants, selected by CLI flag: plain (level, `S2_NDBI_contrast_mean`), `--trajectory`/`--ndbi` (retired, spatial artifacts), `--dw_bare` (superseded plain-OLS slope), **`--dw_bare_robust` (active: Theil-Sen slope on genuine-month-only Dynamic World observations — see FINDINGS.md "Target Validation: Theil-Sen Fix")** |
| `accessibility_features.py` | Euclidean nearest-distance (m) from aza representative points to DID, medical, hospital-only, school, elementary, bus stop, bus route, community facility (P05); buffer counts |
| `feature_matrix.py` | Joins all predictor families on `unit_id`; enforces leakage guards before every merge; kaso flags from the official designation list with 一部過疎 resolved to aza accuracy via point-in-polygon against pre-merger N03 2000 boundaries; HLS durability broadcast muni→aza |
| `cv_train.py` | Municipality-blocked StratifiedGroupKFold CV; imports Phase A `cross_validation.py`; trains final XGBoost; same `--dw_bare_robust` etc. flags as `target_builder.py` |
| `loo_check.py` | Leave-one-municipality-out robustness check (executes the `cross_validation.loo_check` config key, which previously had no implementing code): each of the 65 municipalities held out once, pooled OOF R2 plus a jackknife recomputing that R2 without each municipality. Level target: pooled R2=0.524, jackknife [0.513, 0.543]; trajectory: 0.048 [0.018, 0.064]. Outputs `loo_check{,_dw_bare_robust}.json`; backs the thesis §5.2/§6.5 robustness claims |
| `explain.py` | SHAP TreeExplainer; aggregates by Section 2.2 mechanism cluster; exports case shortlist of largest-positive-residual units (meaning is variant-dependent: fabric above demographic expectation for the level target, fastest bare-ground expansion for the dw_bare variants); same target-variant flags |
| `spatial_autocorrelation.py` | Moran's I (Queen contiguity, permutation p) on the residual target and on the grouped-CV out-of-fold residuals; mirrors the Phase A diagnostic in `typology/src/step3_relationships.py`; same target-variant flags |

---

## Configuration

All paths, column names, and hyperparameters live in
`phase_b/config/phase_b_config.yaml`.  Key toggles:

| Key | Default | Purpose |
|-----|---------|---------|
| `target.physical_indicator` | `S2_NDBI_contrast_mean` | Physical indicator to decouple |
| `target.demographic_regressors` | `[elderly_ratio, pop_total]` | OLS RHS |
| `crs_project` | `EPSG:6680` | Metric CRS for all distance computation |
| `finance.max_fiscal_year` | `2014` | Hard leakage guard (raises if violated) |
| `cross_validation.grouping.enabled` | `true` | Municipality-blocked CV |
| `nlni.community_facilities.aomori/akita` | P05-10 shapefile paths | Community-facility layer (live data since the P05-10 download; no longer a stub) |

---

## Data blockers / known stubs

| Item | Status | Action needed |
|------|--------|---------------|
| P05 community facilities | **RESOLVED** — real shapefiles loaded (`P05-10_02_GML`/`P05-10_05_GML`); `accessibility_community` is a live, non-trivial SHAP mechanism | none |
| Finance pre-2015 | PDFs only; regex extractor implemented | Verify extraction quality on `1018-15-9_02.pdf`; may need manual QA |
| Kaso 一部過疎 aza assignment | **RESOLVED 2026-07-02** — point-in-polygon against pre-merger N03 2000 boundaries (`data/kaso_list/boundaries_2000/`); flags 2,604 → 180 aza | none |
| dw_bare_frac_slope trajectory target | **RESOLVED 2026-06-30** — see FINDINGS.md "Target Validation: Theil-Sen Fix"; `--dw_bare_robust` is now the active target | none (null result confirmed stable) |

---

## Tests

```
python -m pytest phase_b/tests/ -v
```

Test coverage:
- `target_builder`: residual mean-zero, residual ⊥ demography, leakage guard
- `accessibility_features`: inside-DID=0, nearest-point, dist-to-line,
  zero-match buffer, key alignment assertion
- `feature_matrix`: no leakage columns, key alignment fails loudly,
  HLS missing carried as NaN, muni_code extraction, kaso list completeness

---

## Key invariants

1. **AZA_ID = `unit_id`** (string, e.g. `aza:Aomori:022010010`).  
   Every merge asserts key alignment before joining.

2. **Leakage rule**: any column in `feature_matrix.banned_columns` raises
   `ValueError` if found in the feature matrix.  Finance years > 2014 raise
   `ValueError` at load time.

3. **HLS missing not silently imputed**: rows without HLS coverage carry
   `hls_pre1981_ratio = NaN` and `hls_missing = 1`.

4. **Municipality broadcast**: finance, HLS, merger, kaso are municipality-level
   predictors broadcast to aza via `city_name_ja` or `muni_code` (first 5 chars
   of `unit_code`).  The crosswalk is derived from the Phase A base frame and
   never hardcoded.

5. **P05 live**: `dist_community_facility_m` and `n_community_facility_*m`
   are computed from the real P05-10 shapefiles (formerly a NaN stub while
   the data was outstanding).  The columns are present in every output so
   downstream joins never break.
