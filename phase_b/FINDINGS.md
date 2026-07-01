# Phase B — Findings, Implementation Notes, and Open Improvements

## Research Question

Phase B asks: **which socio-economic, institutional, and accessibility factors explain
physical deterioration in aza units that is *not* already explained by demographic ageing?**

The target is an OLS residual: `S2_NDBI_contrast_mean` regressed on `elderly_ratio + pop_total`.
R² of that regression = 0.092, confirming the residual is mostly orthogonal to demography.
Predicting this residual from institutional features — never from remote sensing — isolates the
structural mechanism from the demographic confound.

---

## Feature Matrix

**7 448 aza units × 42 predictor columns** (after banning EO/RS and target-leakage columns).

### Sanctioned predictor families

| Family | Features | Notes |
|--------|----------|-------|
| Demographic threshold flags | `elderly_flag_severely`, `elderly_flag_shrinking`, `household_size`, `household_size_small` | Binary flags from the same thresholds used to build the Phase A target; continuous ratios are banned |
| Durability — HLS 2013 | `hls_pre1981_ratio`, `hls_total_dwellings`, `hls_vacancy_rate`, `hls_vacancy_other_rate`, `hls_missing` | a047/a048 for construction period; a002 for vacancy. 33/65 municipalities covered; 32 carry NaN (XGBoost handles natively) |
| Institutional — fiscal | `fin_local_tax`, `fin_local_alloc_tax`, `fin_total_revenue`, `fin_total_expenditure`, `fin_std_fiscal_revenue`, `fin_std_fiscal_need`, `fin_real_balance`, `fin_fiscal_strength_index` | Averaged across FY2008–2014 from 決算カード PDFs; 65/65 municipalities covered |
| Institutional — policy | `kaso_flag`, `kaso_type` | Hard-coded from 過疎地域自立促進特別措置法 official list; 全部過疎=2, 一部過疎=1 |
| Institutional — merger | `merged_flag`, `years_since_merger` | Heisei-era mergers from MIC XLS; 32 events for Aomori+Akita |
| Accessibility — NLNI | 21 features: `dist_*_m`, `in_did`, `n_*_{1000,3000,5000}m` | Medical, hospital, school, elementary, bus stop, bus route, community facility, DID |

### Banned columns (50 total)

All EO/RS columns are banned. Key reasoning:

- **GLCM texture is the target component** — predicting it from EO features is circular by construction.
- **S2 bands, NDVI/NDBI/MNDWI trajectories, VIIRS, OSM** — all derived from post-2015 satellite observations; temporal leakage.
- **Dynamic World LULC, DEM terrain** — EO-derived; banned on both leakage and design grounds.
- **Continuous aging ratios** (`elderly_ratio`, `aging_index`, `youth_ratio`) — used to build the target; if kept, the model would predict the target from one of its own components.

---

## Data Pipeline — What Was Built

### `feature_matrix.py`
Assembles the feature matrix in 7 ordered steps:

1. **Demographic threshold flags** — binary from Phase A classification-ready CSV.
2. ~~GLCM proxy~~ — removed; all S2_NDBI_* columns are banned.
3. **Finance** — `_parse_kessancard_pdf()` parses 決算カード PDFs; averaged across FY2008–2014 per municipality then broadcast to aza via `city_name_ja`.
4. **HLS durability** — `_load_hls_durability()` parses a047/a048 XLS for pre-1981 housing ratio.
4b. **HLS vacancy** — `_load_hls_vacancy()` parses a002 XLS for `hls_vacancy_rate` (空き家 / 住宅総数) and `hls_vacancy_other_rate` (その他の住宅 / 住宅総数). Joined by 5-digit `muni_code` extracted from col 1 of a002.xls (first 5 chars of `0220146` → `02201`).
5. **Kaso flags** — hard-coded lookup against official designation list; 一部過疎 resolved to aza accuracy via point-in-polygon against pre-merger N03 2000 boundaries (see "Kaso 一部過疎 Aza-Accuracy Fix").
6. **Merger flags** — from MIC merger XLS; most-recent merger per municipality.
7. **Accessibility** — loaded from pre-computed `accessibility_features.parquet`.

Banned columns are stripped last; a final `_assert_no_leakage()` check raises if any survive.

### `cv_train.py`
- `StratifiedGroupKFold`, 5 splits × 5 repeats, grouped by 5-digit `muni_code`.
- Drops only NaN-in-y rows; NaN-in-X rows are kept (XGBoost learns optimal split directions for missing values natively).
- Trains a final model on the full dataset and saves it for SHAP.

### `explain.py`
- `shap.TreeExplainer` with path-dependent method (no background dataset needed).
- Runs on all `target_col`-valid rows (7 448); 1 206 have NaN in X — handled via trained missing-value branches.
- Aggregates mean absolute SHAP by mechanism cluster prefix.
- Outputs `shap_values.parquet`, `mechanism_importance.csv`, `case_shortlist.csv`.

---

## Bugs Fixed

### 1 — Finance PDF parser: all NaN (7 448 / 7 448)

Two independent causes:

**`_parse_toc`**: ToC page has two entries per line (e.g. `青森市 2　大間町 32`). The original code processed whole lines as a single name, so two-column lines produced wrong or missing municipality names.
Fix: rewrote using `re.findall` per line with a CJK/hiragana/katakana character class.

**`_extract_value`**: was called on fully-normalised text where inter-character spaces had been stripped, breaking the `\s+([\d,]+)` pattern.
Fix: introduced `_normalize_digits()` (digit-only normalisation; preserves all spacing) and used it instead of `_normalize_jp_text()` for value extraction.

### 2 — FY2014 `fin_local_alloc_tax` all NaN

pdfplumber merged the `地方交付税` row with the adjacent `地方特例交付金` column in FY2014 PDFs. The merged line had a non-digit character immediately after `税`, so the main pattern matched nothing.
Fix: added a fallback that sums `普通交付税 + 特別交付税 + 震災復興特別交付税` from the adjacent clean lines when the main extraction returns None.

### 3 — HLS and finance join failures for 42 town/village municipalities

Census `CITYNAME` for towns and villages includes a `郡` district prefix (e.g. `上北郡おいらせ町`). HLS XLS files and finance PDFs use plain names (`おいらせ町`).
Fix: `re.sub(r"^.+郡", "", name)` in `_build_muni_name_lookup()` strips the prefix. No-op for cities, which have no 郡 prefix.

### 4 — SHAP computed on only 5 747 / 7 448 aza

`explain.py` was filtering `valid = X.notna().all(axis=1) & target.notna()`, dropping every row with any NaN in X.
Fix: changed to `valid = target.notna()` only; XGBoost's `TreeExplainer` handles NaN via the trained missing-value branches.

### 5 — EO/RS leakage (R² was 0.945)

The original feature matrix included all Phase A columns, including GLCM texture (the target's physical indicator) and all EO-derived columns.
Fix: removed the GLCM proxy step from `build_feature_matrix()`; added 50 EO/RS + target-leakage columns to `banned_columns` in config; `_assert_no_leakage()` enforces this after assembly.

---

## Results

**CV strategy:** `StratifiedGroupKFold`, 5 splits × 5 repeats, grouped by 5-digit municipality code (65 spatial blocks).

| Metric | Value |
|--------|-------|
| R² | **0.505 ± 0.109** |
| RMSE | **0.484 ± 0.041** |
| MAE | **0.374 ± 0.030** |
| n folds | 25 |
| n aza | 7 448 |
| n features | 42 |

(Values reflect the aza-accurate kaso flags of 2026-07-02; the pre-fix run scored 0.515 ± 0.104 — see "Kaso 一部過疎 Aza-Accuracy Fix" below.)

### SHAP mechanism cluster importance

| Rank | Mechanism | mean \|SHAP\| | n features |
|------|-----------|--------------|------------|
| 1 | accessibility_medical | 0.0534 | 4 |
| 2 | accessibility_transit | 0.0467 | 5 |
| 3 | demographic_threshold | 0.0345 | 4 |
| 4 | accessibility_did | 0.0277 | 2 |
| 5 | accessibility_community | 0.0183 | 4 |
| 6 | accessibility_education | 0.0182 | 5 |
| 7 | accessibility_hospital | 0.0131 | 1 |
| 8 | institutional_merger | 0.0094 | 2 |
| 9 | institutional_fiscal | 0.0092 | 8 |
| 10 | durability_housing | 0.0082 | 5 |
| 11 | institutional_policy | 0.0030 | 2 |

**Interpretation:** Accessibility — especially to medical facilities and transit — accounts for the largest share of unexplained physical variation. Demographic flags remain relevant (households, severe ageing) even after the demographic trend is partialled out. Institutional and durability features contribute but are secondary.

---

## Incremental R² History

| Stage | Change | R² |
|-------|--------|----|
| All Phase A features (with EO leakage) | baseline | ~0.945 |
| EO/RS leakage removed | −0.44 | 0.507 |
| Finance parser fixed (was all NaN) | included in above | 0.507 |
| HLS vacancy added (a002.xls) | +0.008 | 0.515 |
| Kaso 一部過疎 aza-accuracy fix (2 604 → 180 flags) | −0.010 | **0.505** |

---

## What Could Be Improved

### High priority

**1 — HLS vacancy coverage for towns and villages**
The a002.xls files only cover municipalities that submitted returns to the prefectural government. 32/65 municipalities (all small towns and kaso-designated villages) have NaN for `hls_vacancy_rate`. These are the structurally most interesting units. Two possible fixes:
- Aggregate aza-level vacancy from the raw 2013 census housing tables (tblT000848 or similar), which have full coverage.
- Cross-check with the Ministry of Land Infrastructure Transport and Tourism (MLIT) Land Use Survey, which has municipality-level vacancy for all municipalities.

**2 — Finance features: FY coverage gaps**
Some municipalities are missing years within FY2008–2014 due to PDF extraction failures (garbled layout variants). Currently all FY are averaged together, which means a municipality missing FY2010–2012 will have a biased mean. Adding per-FY completeness logging would reveal how many municipalities have partial coverage.

**3 — Kaso partial designation (一部過疎) at aza level** ✅ DONE (2026-07-01)
`kaso_type=1` was assigned at municipality level. For partially-kaso municipalities (e.g. むつ市 where only 旧川内町/旧大畑町/旧脇野沢村 are designated), all aza received the same flag regardless of whether they actually fall within the designated 旧町村 area. A spatial join against the historical 旧町村 polygon boundaries would make this accurate.
→ Implemented via point-in-polygon against MLIT N03 2000-10-01 boundaries; see "Kaso 一部過疎 Aza-Accuracy Fix" section below.

**4 — Merger interaction with fiscal features**
Heisei mergers created fiscal consolidation effects (合併算定替 — special allocation-tax bonuses for 10 years post-merger). `years_since_merger` is in the model but there is no interaction term with `fin_local_alloc_tax`. Merged municipalities with the bonus still active vs. those past the 10-year cliff face structurally different fiscal constraints. Adding a `merger_bonus_active` binary flag (merger year + 10 >= 2015) would capture this.

**5 — Spatial autocorrelation in the residuals** ✅ DONE (2026-07-01)
The grouped CV blocks by municipality (65 groups), but aza within the same municipality are spatially autocorrelated and some cross-municipal spatial spillover likely exists. A Moran's I test on the target residuals and on the CV fold residuals would quantify this; a geographically-weighted version of the model could address it.
→ Implemented in `spatial_autocorrelation.py`; see "Spatial Autocorrelation Diagnostics" section below for results.

### Medium priority

**6 — Feature collinearity in the accessibility cluster**
The 21 accessibility features include both distance and count variants at 3 buffer radii each. Many of these are highly correlated (e.g. `dist_bus_stop_m` and `n_bus_stop_1000m`). A VIF screen or PCA reduction within each sub-cluster would improve interpretability and may marginally improve regularisation.

**7 — Temporal alignment of NLNI layers**
Bus stop and route data are from 2010/2011 NLNI vintages; medical facility data from 2014; school data from 2013. All pre-date 2015 which satisfies the leakage rule, but the different vintages introduce inconsistency. Using consistently 2013-vintage NLNI data for all layers would be cleaner.

**8 — Finance fiscal strength index cross-check**
`fin_fiscal_strength_index` is computed inline as `std_fiscal_revenue / std_fiscal_need`. The MIC publishes this ratio directly in the 財政力指数 table. A spot-check against published values for a sample of municipalities would validate the PDF extraction.

### Lower priority

**9 — Dilapidation (a057.xls)**
HLS 2013 includes a Table 57 (a057) with dwelling dilapidation counts. This would give a direct measure of structural deterioration stock, complementing the vacancy rate. The file was not explored in this session.

**10 — OSM building age proxy**
OSM building footprints contain `building:year` tags for a subset of buildings. Where available, this could provide an aza-level construction period estimate that covers the 32 municipalities lacking HLS a047/a048 data. Coverage is sparse in rural Aomori/Akita, so this is a stretch.

**11 — Hyperparameter tuning**
The XGBoost hyperparameters are fixed in config (`n_estimators=200, max_depth=4, lr=0.05`). A grouped CV grid search (Optuna or sklearn GridSearchCV with the same spatial folds) may recover a few R² points.

---

## Target Validation: Theil-Sen + Genuine-Month-Coverage Fix (2026-06-30)

**Context.** Everything above this section describes the `S2_NDBI_contrast_mean` level target (R²=0.515). That target was later superseded in `phase_b_config.yaml` by a trajectory target, `dw_bare_frac_slope` (plain OLS on Dynamic World's monthly bare-ground probability), because the thesis's conceptual framework (§2.5, see the Obsidian vault) explicitly defines the Phase B target as a *trajectory* residual, not a level residual. That trajectory target initially produced a CV R² of **-0.59 ± 0.78** — unstable and worse than predicting the mean. This section documents the root-cause diagnosis and the fix.

### Root cause

Two distinct problems were superimposed in the original `dw_bare_frac_slope`:

1. **Partial monthly coverage, silently back-filled.** `data_preprocessing/gee_dw_monthly.py` fetches a genuine per-(aza, month) Dynamic World bare-probability value from `GOOGLE/DYNAMICWORLD/V1`. `pipeline.py` (step 4e, ~line 215) only overwrites the static, multi-year-average `dw_bare_frac` value (originally broadcast to every month by `gee_lulc.py`) when the new monthly extraction returns non-null for that cell. Measured directly from the cache files in `outputs/gee/dw_monthly/dw_aza_*.csv`: **45.9% of all (unit, month) cells across the full 2015-06 to 2025-12 window are null** (persistent Tōhoku cloud cover and Sentinel-2 archive sparsity, consistent with the missingness already documented in the Phase A EDA), and the 2015-01 to 2015-05 months predate Dynamic World V1 entirely. Net effect: roughly half of every aza's 132-month series is a flat, repeated fallback constant, not a real observation.
2. **Plain OLS is dominated by high-leverage endpoints.** Fitting `np.polyfit` (the same routine as `preprocessing/src/temporal_aggregation.py:compute_ols_slope`) on a series that is ~50% a flat constant lets one or two genuine spike months — especially near the end of the time axis — dominate the fitted slope. Example: `aza:Aomori:022020350` sits at dw_bare_frac≈0.045 for ~130 months, then jumps to 0.36 (2025-09) and 0.24 (2025-11); those two points alone drive a large OLS slope. Because nearly all other aza are similarly flat (raw-slope IQR = 0.0043, from `preprocessing_report_aza.json`'s `scaler_center`/`scaler_scale`), `RobustScaler` then amplifies this handful of unstable estimates into extreme values: **91 aza (1.2% of rows) accounted for >20% of the target's total variance**, clustered in a small number of municipalities (8 of the top-20 outliers fall in one Aomori municipality code, 4 in another — consistent with neighbouring aza sharing the same anomalous satellite scene). Under municipality-grouped CV, whichever fold happened to hold one of those municipalities out collapsed (worst fold R²=-2.26); the average over 25 such folds was the reported -0.59 ± 0.78.

Diagnostic note: the demography→indicator OLS step was *not* the problem — it already returned R²≈0 (0.0022) before the fix, i.e. demography genuinely explains almost nothing of the bare-ground trajectory. The fix below targets the CV-collapse mechanism only.

### Fix

Implemented entirely in Phase B (Phase A remains frozen/unedited), in `phase_b/target_builder.py`:

| Function | Purpose |
|---|---|
| `_load_genuine_dw_bare_observations(cfg)` | Loads all 127 `outputs/gee/dw_monthly/dw_aza_*.csv` cache files, collapses 26,035 multi-part-polygon duplicate rows by mean, and returns only the (unit_id, month) cells with a non-null value — i.e. genuine satellite observations, not fallback. |
| `_compute_dw_bare_theilsen_slope(cfg)` | For each aza, normalises time to [0,1] over the *full* panel span (same convention as the original slope, for comparability), restricts the fitted points to genuine months only, requires `target.min_genuine_months` (config, default 24) genuine points, and fits `scipy.stats.theilslopes` (median-of-pairwise-slopes) instead of OLS. Theil-Sen has a breakdown point of ~29% versus OLS's 0%, so one or two leverage spikes can no longer dominate the estimate. |
| `build_dw_bare_robust_target(cfg)` / `run_dw_bare_robust_target_builder()` | Residualises the new slope on `elderly_ratio + pop_total` (same OLS/VIF machinery as every other variant) and persists to `target_dw_bare_robust_trajectory.parquet`. |

New physical indicator: `dw_bare_frac_slope_theilsen` (config key `target.physical_indicator_dw_bare_robust`). `cv_train.py` and `explain.py` gained a parallel `dw_bare_robust` branch (flag `--dw_bare_robust`) following the exact pattern already used for `--trajectory`/`--ndbi`/`--dw_bare`, writing to `*_dw_bare_robust` output paths so nothing already on disk was overwritten.

Coverage check (computed from the cache, see `_load_genuine_dw_bare_observations` output): genuine months per aza range from 49 to 110 (median 67) out of up to 127 possible — **every single aza clears the min_genuine_months=24 floor**, so the fix costs zero sample size.

Unit tests added in `phase_b/tests/test_phase_b.py::TestDwBareTheilSenFix`: Theil-Sen's robustness to a single high-leverage outlier (synthetic, mirrors the real `022020350` pattern), correct de-duplication/null-dropping in the genuine-month loader, and correct NaN-on-insufficient-coverage behaviour. Run via `python -m pytest phase_b/tests/ -v` (17/17 passing).

### Before / after

| | Raw-slope distribution | OLS R² (demography explains) | CV R² (XGBoost, grouped 5×5) | SHAP top mechanism |
|---|---|---|---|---|
| **Before** (`dw_bare_frac_slope`, plain OLS) | mean≈0.10\*, std≈0.92\* (scaled); 91/7448 rows are \|z\|>3 outliers, contributing >20% of variance | 0.0022 | **-0.590 ± 0.783** (worst fold -2.26) | durability_housing (0.072) — not trustworthy given negative R² |
| **After** (`dw_bare_frac_slope_theilsen`, genuine-month + robust) | mean=0.0009, std=0.0041, min=-0.087, max=0.066 (raw, no extreme tail) | 0.0033 | **-0.002 ± 0.105** (stable across all 25 folds) | none — all 11 mechanism clusters score 0.0000-0.0002, no leader |

\*scaled (RobustScaler) values, raw values are not directly comparable across the two rows; what matters is the elimination of the heavy tail and the resulting 7x reduction in CV standard deviation.

### Interpretation for the thesis

This is now a **clean, defensible null result**, not an artifact-contaminated one. Two things changed and both matter:
1. The *instability* is gone — fold-to-fold CV R² standard deviation dropped from 0.78 to 0.10, meaning the -0.002 mean is a trustworthy estimate, not an average dominated by a handful of catastrophic folds.
2. The *result itself* is still ~zero. Institutional, fiscal, durability, and accessibility features predict essentially none of the genuine bare-ground trajectory once demography is partialled out — performance is indistinguishable from predicting the mean, and SHAP confirms no mechanism cluster carries usable signal.

Combined with the two earlier retired trajectory variants (GLCM contrast slope R²=0.060, NDBI slope R²=-1.146, the latter itself diagnosed as a spatial-gradient artifact), **every trajectory-based operationalization of the Phase B target tried so far shows no predictable structure from socio-institutional features**, while the one target that does (`S2_NDBI_contrast_mean`, a static level, R²=0.515) does not match the trajectory-residual definition the conceptual framework (§2.5) commits to. This asymmetry — level decoupling is partially predictable, trajectory decoupling is not — is itself a substantive, citable finding for Chapter 5/6: it suggests structural/institutional features track *where* a settlement's physical fabric currently stands relative to its demography, but not *how fast* it is currently changing, at least at the temporal and spatial resolution available here (2018-2025 effective optical window, monthly aza-level Dynamic World coverage with ~46% nulls). Discuss in Chapter 6 alongside the temporal-lag caveat already registered in §2.5 ("physical condition trails functional decline by years").

---

## Spatial Autocorrelation Diagnostics: Moran's I on Phase B Residuals (2026-07-01)

**Context.** Improvement #5 (above) and the Phase A validity argument both lean on spatial-autocorrelation diagnostics; until now no such check had ever been run on Phase B. `spatial_autocorrelation.py` closes that gap. It mirrors the Phase A diagnostic (`typology/src/step3_relationships.py`): Queen contiguity weights on the aza polygons (multi-part rows dissolved by geometric union, projected to EPSG:6680), row-standardised, Moran's I with 999-permutation pseudo p-values. It tests two series per target variant:

1. **The residual target itself** — how spatially clustered the phenomenon is (context).
2. **The CV out-of-fold residual** — the structure the model *fails* to capture. Out-of-fold predictions are collected inside the existing municipality-grouped 5×5 CV (`cv_train.py` now accumulates per-row OOF predictions, averaged over the 5 repeats) so the residual is honest (never predicted by a model that saw the same municipality).

13 of 7,448 aza are contiguity islands (no touching neighbour); they carry zero spatial lag and are retained.

### Results

| Series | Level target (`S2_NDBI_contrast_mean` residual) | Trajectory target (`dw_bare_frac_slope_theilsen` residual) |
|---|---|---|
| Target residual | **I = 0.668** (z = 85.7, p = 0.001) | **I = 0.435** (z = 57.2, p = 0.001) |
| CV out-of-fold residual | **I = 0.443** (z = 57.6, p = 0.001) | **I = 0.406** (z = 53.1, p = 0.001) |
| CV R² (context) | 0.505 ± 0.109 | 0.003 ± 0.101 |

(OOF values reflect the aza-accurate kaso flags of 2026-07-02; the pre-fix run gave OOF I = 0.437 / 0.408 with R² 0.515 / −0.002 — same qualitative picture.)

Outputs: `outputs/spatial_autocorrelation{,_dw_bare_robust}.json` (statistics) and `outputs/spatial_residuals{,_dw_bare_robust}.parquet` (per-unit target + OOF residuals, for mapping / LISA follow-up).

### Interpretation

**Level target.** The target is strongly spatially clustered (I = 0.668), as expected for any landscape-derived quantity. The XGBoost model absorbs roughly a third of that structure (0.668 → 0.437) while explaining R² = 0.515 — but the OOF residual remains strongly autocorrelated. Two implications for the thesis:
- The municipality-grouped CV was the right call: with this much residual spatial structure, un-blocked CV would have leaked neighbour information and inflated R². The reported 0.515 is a fair estimate *because* of the blocking.
- The unexplained variation is not spatial white noise — it clusters. Whatever drives it (unmeasured local factors, scene-level EO artifacts, sub-municipal institutions) operates at a spatial scale the sanctioned predictor set does not capture. A spatially explicit model (GWR, spatial-error) or spatially interpolatable covariates could push beyond 0.515; cite as future work, not a flaw.

**Trajectory target (the null result).** The Theil-Sen trajectory residual is also significantly clustered (I = 0.435) — it is *not* spatially random noise — yet the OOF residual is essentially unchanged (0.408 ≈ 0.435, consistent with R² ≈ 0: the model predicts nearly a constant, so its residual ≈ the target). This sharpens the Chapter 5/6 null-result narrative: the bare-ground trajectory contains real, spatially organised signal (neighbouring aza change together — plausibly shared satellite scenes, local land-use dynamics, or municipal-scale processes), but none of it is predictable from the socio-institutional feature set. The null result is therefore "signal exists but is institutionally unexplained", which is stronger and more interesting than "the target is noise".

**Caveat for both.** Because aza within a municipality inherit identical municipality-level predictors (finance, kaso, merger, and HLS features are broadcast muni→aza), some OOF residual clustering is mechanical: the model cannot differentiate within a municipality using those features, so within-muni residual similarity is partly baked in by the feature construction, not only by missing covariates. Accessibility features (aza-specific) are the main within-muni discriminators.

---

## Kaso 一部過疎 Aza-Accuracy Fix (2026-07-02)

**Context.** `kaso_type=1` (partial designation) had been assigned at municipality level: every aza of the six partially-designated municipalities (弘前市, 八戸市, 十和田市, むつ市, 平川市, 秋田市) carried the flag, although the 過疎地域自立促進特別措置法 designation legally applies only to the pre-merger 旧町村 areas named in the official list (already hard-coded in `_KASO_ICHIBU`, incl. the 旧町村 names). Improvement #3 above.

**Implementation** (`feature_matrix.py`):

| Piece | Detail |
|---|---|
| Boundary layer | MLIT KSJ N03 administrative polygons, **2000-10-01 vintage** (`data_root/kaso_list/boundaries_2000/`), which predates all six relevant mergers (2005–2006). DBF is Shift_JIS; no .prj — CRS set from KS-META metadata (JGD2000, EPSG:4612). Config section `kaso:` in `phase_b_config.yaml`. |
| `_load_premerger_boundaries()` | Loads both prefectures, dissolves multi-part rows per pre-merger municipality code (N03_007), projects to EPSG:6680. 136 pre-merger municipalities. |
| `_assign_old_muni()` | Representative point of each (dissolved) aza polygon → point-in-polygon join against the pre-merger layer; 13/7,448 points miss every polygon (coastline mismatch) and fall back to nearest. |
| `_normalize_old_muni_name()` | Strips the 旧 prefix used in the designation list and unifies ヶ/ケ (designation PDF: 碇ヶ関村; N03: 碇ケ関村). |
| Fallback | If the shapefiles are unavailable the builder reverts to the old municipality-level behaviour with a warning (keeps tests/pipeline runnable without the download). |

**Validation.** Flagged aza names cross-checked against census NAME per municipality: 八戸市 = exactly the seven 南郷大字* aza; むつ市 = exactly the 川内町*/大畑町*/脇野沢* (+正津川) aza with central Mutsu unflagged; 弘前市 = exactly the 旧相馬村 大字 set (五所, 湯口, 黒滝, 藍内, 紙漉沢, 沢田, …); 平川市 = 碇ケ関/古懸/久吉; 十和田市 = 奥瀬/法量/沢田; 秋田市 = the 河辺* aza. One known edge case: 秋田市四ツ小屋末戸松本 straddles the old 河辺町 boundary and its representative point falls inside — accepted as point-in-polygon noise. Unit tests in `TestKasoAzaAccuracy` (24/24 passing).

**Effect.**

| | Before (municipality-level) | After (aza-accurate) |
|---|---|---|
| kaso_type=1 aza | 2 604 | **180** (−93%) |
| Total designated aza | 6 437 | 4 013 |
| Level-target CV R² | 0.515 ± 0.104 | 0.505 ± 0.109 |
| SHAP institutional_policy (level) | 0.0072 (rank 11) | **0.0030** (rank 11) |
| Trajectory CV R² | −0.002 ± 0.105 | 0.003 ± 0.101 |

**Interpretation.** Both shifts are informative, not regressions:

1. The R² drop (−0.010, well inside fold std) plus the halving of `institutional_policy` SHAP means the old flag's apparent signal was largely *miscoding*: a municipality-level kaso flag acts as a coarse municipality dummy, and the model was borrowing municipality identity through it. The corrected flag isolates the actual designated areas and shows the designation itself carries even less predictive signal for level decoupling than previously reported.
2. For the thesis this *strengthens* the institutional-features-are-secondary finding: it now rests on a legally accurate operationalization rather than a municipality-blurred proxy, and the designation-effects discussion (§2.2 / JentzschOvsiannikov2025) can cite an aza-accurate null rather than a confounded one.
3. The trajectory null result is unchanged, as expected.
