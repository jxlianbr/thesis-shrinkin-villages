# Level-Target Validity Audit — Area, OSM Construct Check, Area-Adjusted Sensitivity

**Date:** 2026-07-17 (run 2026-07-18)
**Script:** `reports/audit_2026-07-17/level_target_validity.py` (standalone, read-only; reuses `phase_b` modules by import). All numbers persisted in `level_target_validity.json`.
**Setup:** identical to the main Phase B run — same feature matrix (42 sanctioned features, n = 7,448), same XGBoost configuration, same municipality-grouped stratified 5×5 CV (65 blocks, seed 42). Areas from `aza.gpkg`, multi-part rows dissolved, EPSG:6680.

---

## VERDICT — the pre-registered failure trigger fires

**The area-adjusted CV R² is −0.14 ± 0.15, far below the 0.35 threshold, and the cluster ranking collapses.** Under the pre-registered interpretation rules this means: the level result is reframed as predominantly a settlement-size signal, the area-adjusted variant becomes the reported headline, and Section 6.1 is rewritten accordingly. This is not a marginal call — every diagnostic in this audit points the same way:

1. The residual level target correlates with log polygon area at **Pearson −0.86, Spearman −0.84**.
2. Adding log area to the residualizing regression raises its R² from **0.092 to 0.851**.
3. After area adjustment the identical model recovers **no** cross-validated signal (−0.14 ± 0.15).
4. The 42 sanctioned features predict **log polygon area itself** at CV R² = **0.41 ± 0.14** under the identical grouped CV, which accounts for the original 0.51 almost entirely (a target that is ≈ −0.86 × log area, predicted by features that recover log area at 0.41).
5. The OSM construct check **fails**: mean GLCM contrast is uncorrelated with building count (Spearman ≈ +0.04, Pearson ≈ −0.12 to −0.19), directly contradicting the claim that higher texture marks more standing fabric.

The mean GLCM contrast therefore fails the same audit that retired its slope. The trend was retired because its variance tracked unit area; the mean carries the same dependence, only stronger.

---

## 1. Area diagnostics

Correlations across the 7,448 inhabited aza (raw contrast from `typology/outputs_aza/tables/indicator_matrix_raw.csv`, n = 7,291 after its NaN; scaled parent and residual target from the pipeline's own `classification_ready_aza.csv` / `target_residuals.parquet`):

| Series | vs. area (Pearson) | vs. area (Spearman) | vs. log area (Pearson) | vs. log area (Spearman) |
|---|---|---|---|---|
| Mean GLCM contrast, raw | −0.25 | −0.93 | **−0.86** | **−0.93** |
| Mean GLCM contrast, scaled (target parent) | −0.45 | −0.90 | **−0.91** | −0.90 |
| Residual level target | −0.42 | −0.84 | **−0.86** | −0.84 |

All p < 10⁻¹⁰⁰. The demographic residualization (elderly ratio + population) barely touches the area dependence (−0.91 → −0.86), because population explains area poorly at aza scale. The level target is, to first order, inverse log unit area.

Mechanism (for the rewrite, not speculation-free): small aza are compact village cores whose 50 m zonal windows are dominated by heterogeneous built/unbuilt transitions (high NDBI texture), while large aza average the smooth texture of forest and farmland into the unit statistic. The indicator measures the built share of the polygon, i.e., unit geometry, more than it measures fabric.

## 2. OSM diagnostics

**Presence (settles the open project question):** the OSM footprint columns ARE in the aza re-run. The monthly panel (`features_table_aza.parquet`) carries `osm_built_area`, `osm_building_count`, `osm_built_ratio`; the analysis table (`classification_ready_aza.csv`) retains `osm_built_area` and `osm_built_ratio` (`osm_building_count` is removed by the Stage 3 multicollinearity step, as at mura scale). **No Chapter 3 / 4.2.1 correction is needed** — footprints are genuinely an input.

**Construct validity (fails for stock, holds only for density):**

| Pair | Pearson | Spearman | n |
|---|---|---|---|
| Contrast (raw) vs. building count | −0.12 | **+0.04** | 7,291 |
| Contrast (scaled) vs. building count | −0.19 | +0.04 | 7,448 |
| Contrast (raw) vs. built area | −0.15 | −0.01 | 7,291 |
| Contrast (scaled) vs. built area | −0.22 | −0.01 | 7,448 |
| Contrast (raw) vs. built ratio (density) | **+0.56** | **+0.55** | 7,291 |
| Contrast (scaled) vs. built ratio (density) | +0.50 | +0.54 | 7,448 |

The construct claim was that higher texture marks **more standing fabric**. The amount of fabric (count, area) is uncorrelated with texture — essentially zero rank correlation. What texture tracks is built **density** (built area ÷ unit area, r ≈ 0.5), i.e., a quantity whose denominator is polygon area. This independently corroborates the area diagnosis and, per the pre-registered rules, the sub-0.3 stock correlations are reported as a limitation — but in combination with items 1 and 3 they support the stronger reframing, since they refute the fabric-stock reading of the indicator outright.

## 3. Sensitivity run — area-adjusted level target

Residualizing regression (identical `_ols_statsmodels` machinery, VIF pruning retained all three regressors):

| | elderly + pop (pipeline) | elderly + pop + log area (adjusted) |
|---|---|---|
| OLS R² | 0.092 | **0.851** |
| Residual SD | 0.702 | 0.284 |
| n | 7,448 | 7,448 |

Identical XGBoost + grouped 5×5 CV on the two targets (same 42 features, 65 municipality blocks, 25 folds):

| | Pipeline target | Area-adjusted target |
|---|---|---|
| CV R² (mean ± SD) | **0.508 ± 0.106** | **−0.141 ± 0.154** |
| CV RMSE | 0.483 | 0.301 |

Cluster attribution ranking, before vs. after (mean absolute SHAP, mean-over-members convention):

| Rank | Before (pipeline) | | After (area-adjusted) | |
|---|---|---|---|---|
| 1 | accessibility_medical | 0.054 | accessibility_hospital | 0.010 |
| 2 | accessibility_transit | 0.047 | accessibility_medical | 0.007 |
| 3 | demographic_threshold | 0.034 | accessibility_community | 0.006 |
| 4 | accessibility_did | 0.028 | accessibility_transit | 0.006 |
| 5 | accessibility_community | 0.018 | institutional_merger | 0.005 |
| 6 | accessibility_education | 0.018 | durability_housing | 0.004 |
| 7 | accessibility_hospital | 0.011 | demographic_threshold | 0.004 |
| 8 | institutional_merger | 0.010 | accessibility_education | 0.004 |
| 9 | institutional_fiscal | 0.009 | institutional_fiscal | 0.004 |
| 10 | durability_housing | 0.008 | accessibility_did | 0.004 |
| 11 | institutional_policy | 0.005 | institutional_policy | 0.003 |

The after-ranking is the signature of a model with nothing to attribute: the spread between the strongest and weakest cluster shrinks from 12× to 3× and the ordering reshuffles arbitrarily (compare the trajectory null, where all clusters sit within a similarly narrow band). Given negative CV R², the after-attributions must not be interpreted — they are shown only to document that the accessibility dominance does not survive.

**Confirmatory check:** predicting log polygon area itself from the 42 sanctioned features under the identical grouped CV yields **R² = 0.41 ± 0.14**. Accessibility distances and counts are geometric functions of unit size and remoteness, so the feature set encodes area; a target that is −0.86 correlated with log area is then predictable at ≈ 0.5 without any institutional mechanism.

---

## Consequences under the pre-registered rules

- The level result (CV R² = 0.51) is **reframed as predominantly a settlement-size signal**. The area-adjusted variant (CV R² = −0.14 ± 0.15, a null) becomes the reported headline for the level target.
- **Section 6.1 is rewritten** around this. The audit note belongs in Section 4.3 next to the trend-variant retirements: the mean GLCM contrast fails the same area audit that retired the slope.
- No Chapter 3 correction (OSM columns are present); the near-zero fabric-stock correlations go to the limitations as pre-registered.

**Downstream ripple the rewrite must address** (flagged here, decisions belong to the author):
1. The **level-vs-trajectory asymmetry** (5.2's central finding, echoed in 6.x and 7.1) does not survive: with the level result reframed, every operationalization of the decoupling target is either an artifact or a null once its artifact is removed. The honest headline becomes a fully null Phase B: pre-2015 socio-institutional features predict neither the level nor the trajectory of demography-residualized physical condition at aza scale.
2. The **mechanism chapter (5.3)** and the medical/transit accessibility narrative rest on the unadjusted model and inherit its reframing — the attributions were substantially attributions of unit size.
3. The **spatial diagnostics** reframe consistently: the level target's Moran's I of 0.67 is to a large degree the spatial autocorrelation of polygon size, and the OOF residual clustering (0.44) partly reflects the size structure the features fail to fully absorb.
4. The **case shortlist** (largest positive residuals, carried to Chapter 6) is size-confounded and would need rebuilding from the area-adjusted residuals if retained at all.

**Caveat, stated for fairness:** adding log area removes not only the mechanical geometry component but also any genuine signal that scales with settlement size, so the adjustment is conservative and −0.14 is a lower bound on institutional signal. The OSM stock check is the reason this caveat does not rescue the original reading — if texture measured standing fabric, it would correlate with building count, and it does not (Spearman +0.04).
