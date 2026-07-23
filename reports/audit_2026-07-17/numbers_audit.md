# Numbers Audit — Thesis Results Chapters vs. Pipeline Outputs

**Date:** 2026-07-17 (run 2026-07-18)
**Scope:** every statistic quoted in Chapters 5–7 (plus the two disputed sentences in 3.2 / 4.2), mapped to its source artifact and recomputed where the artifact permits. Phase A untouched; all recomputation read-only. Recomputation script: `reports/audit_2026-07-17/recompute_audit.py`, raw output `recompute_results.json`.
**Pre-registered rules applied:** a mismatch in items 1–3 corrects the thesis to the pipeline value, never the reverse. No other conclusion changes on the basis of this audit.

---

## Resolution 1 — Monthly panel row count: §4.2 is right, §3.2 is wrong

**Recomputed from `outputs/final/features_table_aza.parquet`:**

| Quantity | Value |
|---|---|
| Exact shape | **1,165,032 rows × 46 columns** |
| Distinct `unit_id` | **8,826** (Aomori 4,496 + Akita 4,330) |
| Distinct months | **132** |
| 8,826 × 132 | 1,165,032 — rows factorize exactly |
| Units also in `classification_ready_aza` | 7,448 (Aomori 3,723 + Akita 3,725) |
| Panel units absent downstream | **1,378** |

**Where the unbuilt parcels are actually dropped:** Stage 3 (preprocessing), step 5 target construction — `preprocessing/src/target_builder.py:50-55` (`build_target`). Units whose `elderly_ratio` is NaN/inf after the census join (no resident population) are dropped immediately before the shrinkage label is assigned. Stage 1 computes every indicator for all 8,826 units; Stage 2 EDA also sees all 8,826. The 1,378 parcels are therefore present in the monthly panel and in every indicator, and disappear only when the demographic target is built.

**Verdict:** the §4.2.1 sentence ("a monthly panel of 1,165,032 rows and 46 columns at the aza scale") is **correct**. The §3.2 sentence ("the 1,378 unbuilt parcels excluded in Section 3.1 are removed **before any indicator is computed**") is **wrong** and must be corrected to the pipeline behaviour.

**Suggested corrected wording for §3.2** (style rules respected — no em dashes, no semicolons, no colon setups):

> All four inputs are extracted for the full set of 8,826 *aza*, so the monthly panel still contains the 1,378 unbuilt parcels identified in Section~\ref{sec:data_studyarea}. Those parcels carry no census population and drop out at the preprocessing stage when the shrinkage target is constructed, and every analysis table downstream of the monthly panel therefore describes the 7,448 inhabited *aza* only.

---

## Resolution 2 — Level-model RMSE 0.48 vs R² 0.51: consistent, but "standardized target" is the wrong word

**Scaling actually applied to the target's parent variable** (from `preprocessing/outputs/preprocessing_report_aza.json`, `transform_metadata`): `S2_NDBI_contrast_mean` is in `log1p_features`, then scaled with **`scaler_type: "robust"`** (RobustScaler, median/IQR): center = 8.0379 (median of the log1p values), scale = 0.9781 (IQR). It is **not z-standardized**. The Phase B target is the OLS residual of this log1p+robust-scaled variable on `elderly_ratio + pop_total` (residualizing regression R² = 0.0923, matching the quoted 0.09).

**Recomputed numbers** (from `phase_b/outputs/target_residuals.parquet` and `spatial_residuals.parquet`; reference `cv_results.json`):

| Quantity | Value |
|---|---|
| SD of the residual target | **0.7023** (parent variable SD 0.7372) — not 1 |
| Fold-mean CV R² (cv_results.json) | 0.5080 ± 0.1057 |
| Fold-mean CV RMSE (cv_results.json) | 0.4826 ± 0.0399 |
| Pooled out-of-fold RMSE (recomputed) | **0.4822** |
| Pooled out-of-fold R² (recomputed) | 0.5286 |
| √(1 − 0.508) | 0.7014 |
| Implied RMSE = √(1 − R²) × SD = 0.7014 × 0.7023 | **0.4926** ≈ 0.48 |

**Reconciliation:** the apparent contradiction (R² = 0.51 should imply RMSE ≈ 0.70) assumes a unit-variance target. The target has SD ≈ 0.70 because RobustScaler divides by the IQR, not the SD, and the residualization further shrinks the spread. √(1 − 0.51) ≈ 0.70 multiplied by the actual SD of 0.70 gives ≈ 0.49, which matches the reported 0.48 (fold-mean RMSE and pooled RMSE differ from the algebraic identity only through fold weighting). Both thesis numbers are correct; only the word "standardized" is wrong.

**Suggested corrected wording for §5.2** (replacing "with an RMSE of $0.48$ on the standardized target"):

> with an RMSE of $0.48$ in the units of the robust-scaled target. The underlying texture indicator is log-transformed and scaled by median and interquartile range rather than z-standardized, which leaves the residual target with a standard deviation of $0.70$, so the reported $R^2$ and RMSE are mutually consistent.

---

## Resolution 3 — Moran's I inventory: all six thesis/pipeline values confirmed, **no transposition**; the project-note value 0.672 is wrong

**Weight matrix specifications actually used (two distinct specs, both recomputed exactly as coded):**

- **Phase B diagnostic** (`phase_b/spatial_autocorrelation.py`): Queen contiguity on the 8,826 aza polygons with multi-part rows **dissolved** by geometric union on `unit_id`, projected to EPSG:6680, row-standardised, 999 permutations, numpy seed 42; 13 island units retained with zero lag; n = 7,448.
- **Phase A typology diagnostic** (`typology/src/step3_relationships.py`): Queen contiguity on **first-part-only** polygons (`drop_duplicates` on `unit_id`), projected to EPSG:6690, row-standardised, 999 permutations; n = 7,282 (166 units lost to NaN in the OLS predictors).

| Series | Thesis / notes | Recomputed | Match |
|---|---|---|---|
| Level target residual | 0.67 | **0.6684** (z = 85.7, p = 0.001) | YES |
| Trajectory target residual | 0.43 | **0.4345** (z = 57.2, p = 0.001) | YES |
| Level model OOF residual | 0.44 | **0.4410** (z = 57.4, p = 0.001) | YES |
| Trajectory model OOF residual | 0.41 | **0.4058** (z = 53.1, p = 0.001) | YES |
| Raw Theil–Sen bare-fraction slope (before residualization) | notes: 0.672 | **0.4370** (z = 57.5, p = 0.001) | **NO — note is wrong** |
| Elderly-ratio OLS residuals | 0.22 | **0.2219** (z = 27.7, p = 0.001, n = 7,282) | YES |

**Transposition check:** none of the level/trajectory quantities are transposed. All four values quoted in §5.2 and reused in §6.5/§7.1 reproduce exactly from `spatial_autocorrelation{,_dw_bare_robust}.json` and from independent recomputation.

**The 0.672:** recomputing Moran's I on the raw `dw_bare_frac_slope_theilsen` column (identical Phase B weight spec) gives **0.437**, not 0.672. The 0.672 recorded in the project notes appears nowhere in any pipeline artifact and sits within rounding of the *level target's* 0.668, so it is almost certainly a mistranscription of the level value into the raw-slope slot. No thesis paragraph quotes 0.672, so **no chapter correction is required**; the project note should be corrected to 0.437. The recomputed 0.437 is also internally coherent, since residualization removes only 0.33% of the slope's variance, and the residual's Moran's I is accordingly nearly unchanged at 0.4345.

---

## Reproducibility caveat found in passing (no thesis number affected)

`preprocessing/outputs/classification_ready_aza.{parquet,csv}` was regenerated on 2026-07-05 23:55:56, an instant *after* `phase_b/outputs/target_residuals.parquet` (same second) — and its `S2_NDBI_contrast_mean` column now differs slightly from the copy the committed Phase B target build consumed: median |Δ| = 0.00016 scaled units, 54 units differ by > 0.01, 16 by > 0.1, max 1.19. The monthly panel itself is unchanged (2026-06-22). All thesis numbers trace to the committed run (`b15b5fc` artifacts), which is internally self-consistent, so nothing quoted changes, but a fresh end-to-end rerun of `target_builder.py` + `cv_train.py` from the current classification-ready table would shift the level results in roughly the third decimal. Worth a one-line regeneration before final submission if bitwise reproducibility from the current tree is wanted.

---

## General audit table — every further statistic in Chapters 5–7

Match = thesis value equals the artifact value at the thesis's rounding precision.

### Chapter 5.1 — Phase A results

| Statistic | Thesis | Recomputed / located | Source (file → key) | Match |
|---|---|---|---|---|
| n inhabited aza | 7,448 | 7,448 | `classification_ready_aza.parquet`; `typology_summary.json → n_units` | YES |
| Correlation coefficients tested | 64 | 8 physical × 4 demographic × 2 methods = 64 | `typology_summary.json → indicators` | YES |
| Pairs surviving Bonferroni | 49 | 49 | `typology_summary.json → relationships.n_significant_correlations` | YES |
| Largest association (Spearman, texture trend vs pop) | 0.50 | 0.4965 | `…top_correlations[0]` | YES |
| Strongest aging link (texture mean vs elderly ratio) | −0.34 | −0.3399 | `…top_correlations[2]` | YES |
| OLS elderly_ratio ~ 8 physical, R² | 0.15 | 0.154 (recomputed 0.1540) | `typology_summary.json → relationships.regression.r_squared`; recompute | YES |
| Balanced accuracy, RS-only, best model | 0.54, Logistic Regression | 0.5395, Logistic Regression | `classification/outputs_aza/tables/leakage_experiment_results.csv → rs_only` | YES |
| Balanced accuracy, no-demographic, best model | 0.54, Logistic Regression | 0.5444, Logistic Regression | same → `no_demographic` | YES |
| Non-best models "bunched between 0.47 and 0.54" | 0.47–0.54 | rs_only non-dummy range 0.4661–0.5386 | same | YES |
| Balanced accuracy incl. demographic counts | 0.92, SVM (linear) | 0.9150, SVM (Linear) | same → `no_leaky` | YES |
| Balanced accuracy leakage reference | 1.00, XGBoost | 1.0000, XGBoost (Gradient Boosting also 1.0000) | same → `all_features` | YES |
| Ungrouped (conventional stratified) CV accuracy | ≈ 0.58 | ~0.58 recorded in project notes only ("Phase A aza Results" note, Chapter 4 Codebase Reference); **no on-disk artifact** — the ungrouped run was superseded and its outputs overwritten | vault notes | FLAG — unverifiable from outputs; thesis hedge "approximately" is appropriate, or rerun ungrouped CV to pin it |
| Moran's I, elderly-ratio regression residuals | 0.22 (p = 0.001, 999 perms) | 0.2219 (p = 0.001; n = 7,282) | `typology_summary.json → relationships.spatial.morans_i`; recomputed | YES |
| LM lag and error both significant | stated | LM-lag 708.9 (p ≈ 0), LM-error 787.3 (p ≈ 0) | `…spatial.spatial_regression` | YES |
| Spatial error vs OLS AIC | −9,728 vs −9,077 | −9,728.4 vs −9,077.18 | `…spatial_regression.spatial_error.aic / ols.aic` | YES |
| Spatial lag pseudo-R² | 0.24 | 0.241 | `…spatial_regression.spatial_lag.pseudo_r_squared` | YES |
| Aspatial R² (same sentence) | 0.15 | 0.154 | `…ols.r_squared` | YES |
| K-means k = 3 primary; k = 4, 5 comparison | stated | primary_k = 3, report_k = [3, 4, 5] | `typology_summary.json → clustering` | YES |
| Bootstrap ARI (1,000 resamples) | 0.96 | 0.9601 | `…clustering.bootstrap.mean_ari` | YES |
| Type 3 mean elderly ratio "approaches one half" | qualitative | 0.482 | `typology/outputs_aza/tables/cluster_profiles_k3.csv` | YES |
| ARI vs demographic classification | 0.14 | 0.1354 | `…multi_k.3.supervised_ari` | YES |
| ARI physical-only respecification | 0.44 | 0.4420 | `…specification_robustness[physical_only]` | YES |
| ARI demographic-only respecification | 0.12 | 0.1174 | `…specification_robustness[demographic_only]` | YES |
| Crosstab row Type 1 | 1,902 / 1,039 / 529 (Σ 3,470) | 1,902 / 1,039 / 529 | `typology/outputs_aza/tables/cluster_crosstab_k3.csv` cluster 0 | YES |
| Crosstab row Type 2 | 258 / 606 / 431 (Σ 1,295) | 258 / 606 / 431 | same, cluster 1 | YES |
| Crosstab row Type 3 | 300 / 896 / 1,487 (Σ 2,683) | 300 / 896 / 1,487 | same, cluster 2 | YES |
| Class totals | 2,460 / 2,541 / 2,447 | stable 2,460, shrinking 2,541, severely 2,447 | `classification/outputs_aza/reports/classification_summary.json → class_distribution` | YES |

### Chapter 5.2 — Phase B results

| Statistic | Thesis | Recomputed / located | Source | Match |
|---|---|---|---|---|
| Residualizing regression R² (level) | 0.09 | 0.0923 (recomputed from target parquet) | `target_residuals.parquet`; FINDINGS.md 0.092 | YES |
| Level CV R² | 0.51 ± 0.11 | 0.5080 ± 0.1057 | `cv_results.json` | YES |
| Level CV RMSE | 0.48 | 0.4826 (pooled OOF recomputed 0.4822) | `cv_results.json`; `spatial_residuals.parquet` | YES (wording fix per Resolution 2) |
| n grouped folds | 25 | 25 | `cv_results.json → n_folds` | YES |
| LOO pooled R² | 0.52 | 0.5237 | `loo_check.json → pooled_loo.r2` | YES |
| Jackknife moves "no more than two points" | ≤ 0.02 | range 0.5128–0.5433, max |Δ| from pooled = 0.0110/0.0195 | `loo_check.json → jackknife_summary` | YES |
| Kaso correction changed R² "< 1 point" | stated | 0.515 → 0.508 (−0.007) | FINDINGS.md kaso section | YES |
| GLCM-trend variant | 0.06 ± 0.10 | 0.0603 ± 0.0985 | `cv_results_trajectory.json` | YES |
| NDBI-trend variant | −1.15 ± 1.98 | −1.1461 ± 1.9793 | `cv_results_ndbi.json` | YES |
| Bare-ground OLS variant | −0.59 ± 0.78 | −0.5898 ± 0.7834 | `cv_results_dw_bare.json` | YES |
| Bare-ground Theil–Sen variant | 0.00 ± 0.10 | 0.0030 ± 0.1018 | `cv_results_dw_bare_robust.json` | YES |
| Fallback share of monthly bare-ground panel | ≈ 46% | 45.9% (measured from `outputs/gee/dw_monthly/` cache) | FINDINGS.md target-validation section | YES |
| Moran's I trajectory target / OOF | 0.43 / 0.41 | 0.4345 / 0.4058 (recomputed) | `spatial_autocorrelation_dw_bare_robust.json` | YES |
| Moran's I level target / OOF | 0.67 / 0.44 | 0.6684 / 0.4410 (recomputed) | `spatial_autocorrelation.json` | YES |

### Chapter 5.3 — Mechanisms

| Statistic | Thesis | Located value | Source | Match |
|---|---|---|---|---|
| Accessibility, medical (level) | 0.054 | 0.05392 | `mechanism_importance.csv` | YES |
| Accessibility, transit | 0.047 | 0.04652 | same | YES |
| Demographic threshold | 0.034 | 0.03439 | same | YES |
| Accessibility, DID | 0.028 | 0.02833 | same | YES |
| Accessibility, community | 0.018 | 0.01814 | same | YES |
| Accessibility, education | 0.018 | 0.01796 | same | YES |
| Accessibility, hospital | 0.011 | 0.01114 | same | YES |
| Institutional, merger | 0.010 | 0.00958 | same | YES |
| Institutional, fiscal | 0.009 | 0.00919 | same | YES |
| Durability, housing | 0.008 | 0.00785 | same | YES |
| Institutional, designation | 0.005 | 0.00454 | same | YES |
| Trajectory model, all clusters | 0.0000–0.0002 | 0.0000329–0.0002063 | `mechanism_importance_dw_bare_robust.csv` | YES |
| dist_medical_m mean abs SHAP | 0.157 | 0.15740 | `shap_values.parquet` (recomputed mean abs) | YES |
| fin_fiscal_strength_index mean abs SHAP | 0.028 | 0.02763 | same | YES |
| Pre-fix policy attribution ("roughly 0.007", "reduced by about a third") | 0.007 → 0.005 | 0.0072 → 0.0045 (−37%) | FINDINGS.md kaso section | YES |

### Chapters 6–7 (numbers reused from Chapter 5)

| Statistic | Thesis | Located value | Source | Match |
|---|---|---|---|---|
| §6.2 ARI | 0.14 | 0.1354 | `typology_summary.json` | YES |
| §6.5 HLS vacancy unobservable municipalities | 32 of 65 | 32/65 | FINDINGS.md HLS verdict | YES |
| §6.5 fallback share | ≈ 46% | 45.9% | FINDINGS.md | YES |
| §6.5 level OOF Moran's I | 0.44 | 0.4410 | recomputed | YES |
| §7.1 R² physical→elderly | 0.15 | 0.154 | `typology_summary.json` | YES |
| §7.1 balanced accuracy / chance | 0.54 / 0.33 | 0.5395 / 0.3333 | leakage table | YES |
| §7.1 typology ARI | 0.14 | 0.1354 | `typology_summary.json` | YES |
| §7.1 residual Moran's I | 0.22 | 0.2219 | recomputed | YES |
| §7.1 level CV R² | 0.51 ± 0.11 | 0.5080 ± 0.1057 | `cv_results.json` | YES |
| §7.1 trajectory CV R² | 0.00 ± 0.10 | 0.0030 ± 0.1018 | `cv_results_dw_bare_robust.json` | YES |
| §7.1 trajectory target Moran's I | 0.43 | 0.4345 | recomputed | YES |
| §7.1 medical / transit clusters | 0.054 / 0.047 | 0.05392 / 0.04652 | `mechanism_importance.csv` | YES |

---

## Summary of required actions

1. **§3.2 (one sentence): correct** per Resolution 1. §3.1/§3.4/§4.2 need no change.
2. **§5.2 (one phrase): replace "on the standardized target"** per Resolution 2.
3. **Moran's I: no thesis change.** Correct the project note's raw-slope value 0.672 → 0.437. No transposition exists; §5.2's spatial-structure paragraphs are sign- and magnitude-consistent as written.
4. **Flag only:** the ≈0.58 ungrouped-CV figure has no surviving artifact (notes-only); the 2026-07-05 regeneration of `classification_ready_aza` differs microscopically from the committed target build's input (see caveat).

Every other quoted statistic in Chapters 5–7 (49 table rows above) reproduces from the committed pipeline artifacts at the thesis's stated precision.
