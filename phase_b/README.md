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
```

Each step persists its output under `phase_b/outputs/` before the next
step reads it.  Run from the repo root (`D:/data_code_masterthesis/code/`).

---

## Module summary

| Module | What it does |
|--------|-------------|
| `target_builder.py` | Regresses GLCM-contrast on elderly_ratio + pop_total via OLS (VIF-pruned); per-aza residual is the Phase B target |
| `accessibility_features.py` | Euclidean nearest-distance (m) from aza representative points to DID, medical, hospital-only, school, elementary, bus stop, bus route; buffer counts; P05 stub |
| `feature_matrix.py` | Joins all predictor families on `unit_id`; enforces leakage guards before every merge; hard-coded kaso flags; HLS durability broadcast muni→aza |
| `cv_train.py` | Municipality-blocked StratifiedGroupKFold CV; imports Phase A `cross_validation.py`; trains final XGBoost |
| `explain.py` | SHAP TreeExplainer; aggregates by Section 2.2 mechanism cluster; exports case shortlist of structurally-deteriorating units |

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
| `nlni.community_facilities.aomori/akita` | `null` | Set to P05 paths when data arrives |

---

## Data blockers / known stubs

| Item | Status | Action needed |
|------|--------|---------------|
| P05 community facilities | **STUB** — all NaN | Drop P05 shapefiles into `nlni_data_japan/` and set paths in config |
| Finance pre-2015 | PDFs only; regex extractor implemented | Verify extraction quality on `1018-15-9_02.pdf`; may need manual QA |
| Kaso 一部過疎 roaza assignment | Municipality-level only | Requires 旧町村 boundary layer to assign individual aza |

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

5. **P05 stub**: `dist_community_facility_m` and `n_community_facility_*m`
   are `NaN` until P05 shapefiles are added and config paths are set.  The
   columns are present in every output so downstream joins never break.
