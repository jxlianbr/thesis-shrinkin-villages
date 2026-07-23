# CV Stratification and Clustering Feature-Set Specification

**Date:** 2026-07-17 (run 2026-07-18)
**Method:** read-only code and artifact inspection (`phase_b/cv_train.py`, `typology/src/step1_indicators.py`, `typology/config/typology_config_aza.yaml`, `typology/outputs_aza/tables/indicator_matrix_{raw,scaled}.csv`). No pipeline runs.

---

## 1. What the Phase B regression CV actually stratifies: **nothing about the target**

`phase_b/cv_train.py`, `_run_grouped_cv` (lines 131–134 and 151–155):

```python
# For StratifiedGroupKFold, we need a discrete stratum label.
# Use the group (municipality) itself as the stratum so that the split
# is purely group-blocked without requiring a class label.
strata = groups
...
for r in range(n_repeats):
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=rs_base + r,
    )
    for train_idx, test_idx in sgkf.split(X_arr, strata, groups=groups):
```

The stratification label passed to `StratifiedGroupKFold` is the **municipality code itself** — not target bins (none are ever computed, the continuous residual is never discretized) and not a class label. Because every group consists of exactly one stratum, the stratification objective degenerates: each municipality must land whole in one fold, and the solver's only remaining effect is to assign the 65 municipalities to the 5 folds so that fold sizes stay approximately balanced in the number of aza. The mechanism is therefore **group-blocked five-fold CV with shuffled group-to-fold assignment, approximately size-balanced, repeated five times** (the code's own comment says as much: "purely group-blocked without requiring a class label"). Incidental corroboration: sklearn emits "the least populated class in y has only 2 members" warnings during these runs, which is the municipality label acting as the stratum.

For contrast and to scope the correction precisely: the Phase A **classification** runs (`classification/src/cross_validation.py:57`) use `StratifiedGroupKFold` with the three-class shrinkage label as `y`, where stratification is genuine. The word "stratified" is accurate there and only there.

**Per the pre-registered rule** (the code is what the thesis says), the word "stratified" is deleted from §4.4.2 and §4.2.3 wherever it refers to the regression CV. **Correct one-sentence description for §4.4.2:**

> Model validation uses municipality-grouped five-fold cross-validation with five repetitions, in which whole municipalities are assigned to folds with shuffled, approximately size-balanced assignment and the continuous target is never binned or stratified.

## 2. The twelve clustered indicators: statement **confirmed**

Verbatim column list of the clustered matrix (`typology/outputs_aza/tables/indicator_matrix_scaled.csv`, identifiers excluded):

**Physical (8):** `NDBI_slope`, `viirs_mean_slope`, `S2_NDBI_contrast_slope`, `NDVI_cv`, `NDVI_seasonal_amp`, `NDBI_seasonal_amp`, `S2_NDBI_contrast_mean`, `S2_NDBI_entropy_mean`
**Demographic (4):** `elderly_ratio`, `aging_index`, `household_size`, `pop_total`

The raw indicator matrix starts from 14 candidates; the Step 1 pruning (`typology_config_aza.yaml → indicators.pruning`, `drop_near_constant: true`) removes `NDVI_slope` and `youth_ratio` as near-constant (recorded in `indicator_report.json → near_constant_features`; the 0.85 correlation prune removed nothing, `high_correlation_pairs: []`). The surviving 12 are exactly the `physical_names` × `demo_names` consumed by the Step 3 correlation analysis — the same 8 × 4 sets behind the "64 coefficients" of §5.1.1.

**The §5.1.2 sentence is confirmed as written:** the twelve standardized indicators are the eight physical and four demographic indicators of the correlation analysis. One precision worth adding when the sentence is finalized: the four demographic indicators are elderly ratio, aging index, household size, and population — **youth ratio is not among them** (pruned as near-constant), so any enumeration in the text should not name it. The conservative-ARI remark **holds**: demographic indicators are genuinely inside the clustered feature set, so the ARI of 0.14 against the demographic label is computed on a typology that had access to demography, which makes the low agreement conservative.

## 3. K-means settings (optional check)

From `typology_config_aza.yaml → clustering` and `typology/src/step1_indicators.py` / `step2_clustering.py`:

| Setting | Value |
|---|---|
| Scaling before clustering | z-score standardization, `sklearn.preprocessing.StandardScaler` (`step1_indicators.py:92-93`) — distinct from the Stage 3 RobustScaler used for classification |
| k evaluated | 2–8 (`kmeans.k_range`) |
| k retained | 3 (primary; k = 4 and 5 reported for comparison) |
| `n_init` | 100 |
| `max_iter` | 300 |
| `random_state` | 42 (global `random_state`) |
| Clustered matrix | the 12 standardized indicators directly (the PCA-space refit is a robustness check only, ARI 0.976 vs. the raw-space solution) |
| Hierarchical comparison | Ward linkage on the same matrix |

Suggested methods sentence: k-means is run on the twelve z-standardized indicators with k from 2 to 8, 100 random initializations, 300 maximum iterations, and a fixed seed of 42, and the three-cluster solution is retained as primary.
