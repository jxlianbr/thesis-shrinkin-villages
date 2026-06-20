# Chapter 4 — Part 1: Phase A Results (aza / 小地域 scale)

> Source material for writing Chapter 4, Section 1. All figures referenced below
> are committed under each stage's `outputs_aza/` directory and render directly on
> GitHub. Numbers are taken from the committed run artifacts
> (`run_manifest_aza.json`, the stage `tables/` CSVs, and the stage report JSONs).

---

## 4.1.0  Study design: from municipality (mura) to small area (aza)

Phase A was re-run at the **aza** level (小地域 / 町丁・字等, "town-block / hamlet")
using the **identical** earth-observation logic developed for the 65 municipalities
(mura): the same Sentinel-2 / VIIRS / Dynamic World / Copernicus-DEM / OSM
indicators, the same `scale_m`, the same 2015–2025 monthly time window, and the
same zonal reducers. **Only the spatial unit of analysis changed.** This lets the
two scales be compared on equal methodological footing while testing whether a much
finer spatial grain reveals structure invisible at the municipal level.

| Property | mura run | aza run |
|---|---|---|
| Spatial units | 65 municipalities | **8,826** small areas (Aomori 4,496 + Akita 4,330) |
| Monthly panel | ~8,450 rows | **1,165,032 rows** (8,826 × 132 months) |
| Time window | 2015-01 → 2025-12 (132 months) | identical |
| Indicators | 30+ | identical set |

The aza boundary set contains 155 multi-part units (polygons split across 360 rows);
these were collapsed to one record per `unit_id` wherever geometry was needed.

---

## 4.1.1  Data acquisition (Stage 1)

The acquisition pipeline produced `features_table_aza.parquet`:
**1,165,032 rows × 46 columns**, every one of the 132 months populated for all
8,826 units (exactly 8,826 rows per month — no month is partially covered).

**Coverage / missingness (by design, not error):**

| Feature block | Missing | Reason |
|---|---|---|
| Sentinel-2 bands + NDVI/NDBI/MNDWI + GLCM | ~32% | The Sentinel-2 L2A surface-reflectance archive over Aomori/Akita effectively begins in spring 2018, so pre-2018 months carry no optical data (VIIRS, a separate collection, is unaffected). Remaining gaps are cloud-masked months. |
| Slope / aspect (terrain) | ~29% | Slope and aspect require a pixel neighbourhood; the many very small urban aza are smaller than the 30 m DEM kernel, so they return null. Elevation and TRI are complete (0% missing). |
| Demographics | 9–22% | Census-join misses / zero-population units. |
| LULC, OSM, elevation, VIIRS | ~0% | — |

> **For the chapter:** the ~32% optical gap is a *temporal* artefact (archive start),
> not a spatial one, and is shared identically with the mura run. The ~29%
> slope/aspect gap is the one genuinely new, scale-induced limitation of moving to
> aza — worth one sentence in the methods/limitations.

---

## 4.1.2  Exploratory data analysis (Stage 2)

Outputs: `eda/outputs_aza/` (18 figures, 11 tables, HTML report, 20 data-quality flags).

### Distributions and prefecture contrast
![Remote-sensing feature distributions](eda/outputs_aza/figures/distributions_rs.png)
![Distributions by prefecture](eda/outputs_aza/figures/boxplots_by_prefecture.png)

### Feature correlation structure
![Correlation heatmap](eda/outputs_aza/figures/correlation_heatmap.png)

### Temporal behaviour (2015–2025)
![Temporal trends](eda/outputs_aza/figures/temporal_trends.png)
![Seasonal patterns](eda/outputs_aza/figures/seasonal_patterns.png)

The temporal panel makes the 2018 optical-archive onset directly visible and shows
the strong NDVI seasonal cycle that motivates the seasonal-amplitude indicators.

### Spatial patterns (choropleths over 8,826 units)
![Population (spatial)](eda/outputs_aza/figures/spatial_pop_total.png)
![VIIRS night-lights (spatial)](eda/outputs_aza/figures/spatial_viirs_mean.png)
![NDVI (spatial)](eda/outputs_aza/figures/spatial_ndvi_mean.png)

### Remote-sensing ↔ demography (raw, pre-modelling)
![RS–demographic correlation](eda/outputs_aza/figures/rs_demo_correlation.png)
![VIIRS vs elderly ratio](eda/outputs_aza/figures/viirs_vs_elderly_ratio.png)
![NDBI vs population](eda/outputs_aza/figures/ndbi_vs_population.png)

### Missingness
![Missingness over time](eda/outputs_aza/figures/missing_by_time.png)

---

## 4.1.3  Preprocessing and target construction (Stage 3)

The 1.16 M-row monthly panel was collapsed to one cross-sectional row per unit
(temporal mean / std / OLS-trend / seasonal amplitude), demographic ratios were
engineered, and a 3-class shrinkage target was built.

**Output:** `classification_ready_aza.parquet` — **7,448 units × 53 columns**
(1,378 units with no valid `elderly_ratio` — missing/zero population — were dropped
so they could not be silently mislabelled).

**Target re-calibration.** The mura target used absolute `elderly_ratio` cut-points
of 0.37 / 0.42 (its 33rd/67th percentiles on 65 units). The aza `elderly_ratio`
distribution differs (mean 0.40), so applying the mura cut-points squeezed the
middle class to ~18%. The **same tercile *method*** was therefore re-fit to the aza
scale, giving cut-points **0.347 / 0.446** and three balanced classes:

| Class | `elderly_ratio` | Units | Share |
|---|---|---|---|
| stable | < 0.347 | 2,460 | 33.0% |
| shrinking | 0.347 – 0.446 | 2,541 | 34.1% |
| severely_shrinking | ≥ 0.446 | 2,447 | 32.9% |

> **For the chapter:** report aza class shares against the mura ones, and state
> explicitly that the threshold *rule* (terciles of elderly ratio) is held constant
> while the numeric cut-points are refit — this keeps the two scales comparable in
> construction without forcing an imbalanced aza target.

---

## 4.1.4  Supervised classification (Stage 4a)

Ten classifiers were evaluated with 5×5 repeated stratified cross-validation across
four leakage-controlled feature sets. Primary metric: **balanced accuracy**
(chance = 0.333). Leave-one-out was disabled at this scale (a small-n robustness
check; redundant and infeasible at n = 7,448).

### Balanced accuracy by experiment

| Model | all_features | no_leaky | no_demographic | **rs_only** |
|---|---|---|---|---|
| Random Forest | 0.999 | 0.673 | 0.583 | **0.578** |
| Gradient Boosting | 1.000 | 0.808 | 0.580 | **0.578** |
| XGBoost | 1.000 | 0.783 | 0.578 | **0.576** |
| SVM (RBF) | 0.908 | 0.695 | 0.571 | 0.565 |
| Logistic Regression | 0.983 | 0.917 | 0.568 | 0.564 |
| MLP | 0.965 | **0.925** | 0.566 | 0.563 |
| KNN | 0.699 | 0.577 | 0.542 | 0.536 |
| SVM (Linear) | 0.985 | 0.908 | 0.518 | 0.522 |
| Dummy | 0.333 | 0.333 | 0.333 | 0.333 |

![Leakage experiment comparison](classification/outputs_aza/figures/leakage_comparison.png)
![Model comparison (primary)](classification/outputs_aza/figures/model_comparison.png)

**Reading the four experiments:**

- **`all_features` ≈ 1.0** — contains the target-defining `elderly_ratio`; pure
  leakage, reference only (as designed).
- **`no_leaky` ≈ 0.91–0.93** — drops the deterministic ratios but *retains*
  `age_65_plus` and `pop_total`, whose quotient **is** the target. At n = 7,448 the
  smooth-function learners (MLP 0.925, Logistic 0.917, SVM-linear 0.908) reconstruct
  the ratio almost perfectly. **This is partial-leakage inflation amplified by
  sample size, not new predictive signal** — and must be framed as such (the mura
  `no_leaky` best was only 0.662).
- **`no_demographic` ≈ 0.58** and **`rs_only` ≈ 0.58** — the leakage-free results.

### The core thesis number: remote sensing alone

With **all demographic information removed**, satellite + terrain + land-cover
features classify shrinkage at **balanced accuracy ≈ 0.58 vs 0.33 chance** — a
modest but unambiguous signal that the physical landscape carries shrinkage
information. Tree ensembles (RF / GBM / XGB, ~0.578) lead here, **reversing the mura
finding** (where SVM-linear was best) — consistent with non-linear RS↔demography
relationships becoming learnable once thousands of units are available. Adding OSM
building footprints (`no_demographic`) over pure optical/terrain (`rs_only`) adds
only a hair (0.583 vs 0.578).

### Primary-experiment detail and diagnostics

Best primary (`no_leaky`) model — MLP: balanced acc 0.925, accuracy 0.925,
macro-F1 0.926, Cohen's κ 0.888.

![Cross-validation score dispersion](classification/outputs_aza/figures/cv_boxplots.png)
![Best-model ROC curves](classification/outputs_aza/figures/roc_curves_best.png)
![Best-model confusion matrix (MLP)](classification/outputs_aza/figures/confusion_matrix_mlp.png)

### What drives the classification
![Permutation importance](classification/outputs_aza/figures/permutation_importance.png)
![SHAP summary](classification/outputs_aza/figures/shap_summary.png)

Consensus ranking (lower = more important): `age_u15`, `age_65_plus`, `pop_total`
dominate (the partial-leakage demographic block), then the leading **physical**
features — `viirs_mean_std`, `tri_mean`, `viirs_mean_mean`, `dw_built_frac`,
`NDBI_slope`, `elevation_std`. Night-light variability and built-up trend are the
most informative purely-remote-sensing predictors.

![SelectKBest (ANOVA F)](classification/outputs_aza/figures/selectkbest_scores.png)
![PCA explained variance](classification/outputs_aza/figures/pca_variance.png)

---

## 4.1.5  Unsupervised typology (Stage 4b)

Indicators (physical RS trends/levels + demographic) were compiled for the 7,448
units, standardised, and clustered (K-means k = 2–8 and Ward hierarchical), with
gap statistic, 1,000-resample bootstrap stability, and specification-robustness
variants. Outputs: `typology/outputs_aza/`.

### Choosing k
![Optimal-k metrics](typology/outputs_aza/figures/optimal_k_metrics.png)
![Multi-k comparison](typology/outputs_aza/figures/multi_k_comparison.png)
![Dendrogram](typology/outputs_aza/figures/dendrogram.png)

A **3-cluster** solution is the primary choice: **bootstrap ARI = 0.96** (highly
stable). The data-driven optimum is k = 5, but every solution with k ≥ 4 merely
splits off a single-unit outlier cluster, so k = 3 is the substantive structure
(silhouette ≈ 0.18 — modest, as expected for noisy RS indicators at fine grain).

![Silhouette (k = 3)](typology/outputs_aza/figures/silhouette_diagram_k3.png)

### The three shrinkage types

| Cluster | n | elderly ratio | aging index | pop (mean) | NDBI slope | VIIRS slope | Interpretation |
|---|---|---|---|---|---|---|---|
| 0 | 3,470 | 0.338 | 4.1 | 329 | −0.03 | **−1.92** | **Younger, larger settlements with sharply dimming night-lights** |
| 1 | 1,295 | 0.417 | 6.1 | 186 | **+0.15** | −0.41 | **Small hamlets with rising built-up signal** |
| 2 | 2,683 | **0.482** | **9.3** | 300 | −0.02 | −0.05 | **Severely aged / shrinking** |

![Cluster profile heatmap](typology/outputs_aza/figures/cluster_profiles_heatmap.png)
![Cluster parallel coordinates](typology/outputs_aza/figures/cluster_parallel_coords.png)
![Cluster map](typology/outputs_aza/figures/cluster_map.png)

The clusters align meaningfully with the supervised shrinkage classes (crosstab):
Cluster 0 is dominated by *stable* units (1,902 / 3,470), Cluster 2 by
*severely_shrinking* (1,487 / 2,683).

![Cluster × shrinkage-class crosstab](typology/outputs_aza/figures/cluster_crosstab_heatmap.png)

### Remote-sensing ↔ demography relationships

**49 Bonferroni-significant** RS–demographic correlations. Strongest:

| Physical indicator | Demographic | ρ (Spearman) |
|---|---|---|
| `S2_NDBI_contrast_slope` | `pop_total` | **+0.50** |
| `S2_NDBI_contrast_mean` | `household_size` | −0.37 |
| `S2_NDBI_contrast_mean` | `elderly_ratio` | −0.34 |
| `S2_NDBI_contrast_mean` | `aging_index` | −0.30 |

![Spearman correlation heatmap](typology/outputs_aza/figures/correlation_heatmap_spearman.png)
![Subgroup correlations by cluster](typology/outputs_aza/figures/subgroup_correlations.png)

Texture (built-up GLCM contrast) is the recurring bridge between the physical and
demographic dimensions: where built-up texture is higher/more stable, populations
are larger and less aged.

### Regression and spatial dependence

OLS of `elderly_ratio` on the physical indicators: **R² = 0.15** (physical RS alone
explains ~15% of the cross-sectional variance in ageing).

![Regression coefficients](typology/outputs_aza/figures/regression_coefficients.png)

The OLS residuals are **strongly spatially autocorrelated** — **Moran's I = 0.222,
p = 0.001** — and both Lagrange-multiplier tests are highly significant
(LM-Lag 708.9, LM-Error 787.3). Fitting spatial models accordingly improves fit
substantially:

| Model | fit | AIC |
|---|---|---|
| OLS | R² = 0.154 | −9,077 |
| Spatial Lag (ML) | pseudo-R² = 0.241, ρ = 0.338 | −9,652 |
| **Spatial Error (ML)** | pseudo-R² = 0.152, λ = 0.380 | **−9,728 (best)** |

![Spatial model comparison](typology/outputs_aza/figures/spatial_model_comparison.png)

> **For the chapter — the headline aza result:** spatial dependence is invisible at
> municipal level but unmistakable across 7,282 small areas. Shrinkage signals
> cluster geographically (neighbouring aza resemble one another beyond what the
> covariates explain), and a spatial-error specification decisively out-performs
> OLS. This is the clearest payoff of moving the analysis to the finer aza grain.

---

## 4.1.6  Summary of key findings

1. **The pipeline ports cleanly to aza.** An identical EO workflow runs end-to-end
   over 8,826 units / 1.16 M monthly observations; data coverage matches the mura
   run except for a scale-induced slope/aspect gap on the smallest units.
2. **Remote sensing alone carries a real but modest shrinkage signal:** balanced
   accuracy ≈ 0.58 (vs 0.33 chance) with all demographics removed; night-light
   variability and built-up trend are the leading physical predictors.
3. **Beware sample-size-amplified leakage:** the `no_leaky` set still contains the
   target's numerator and denominator, inflating accuracy to ~0.92 at aza scale —
   this is not a genuine improvement over mura and must be reported as such.
4. **Tree ensembles overtake linear models** at aza scale, reversing the mura
   result — finer granularity exposes non-linear RS↔demography structure.
5. **A stable 3-type typology** (bootstrap ARI 0.96) separates younger/dimming,
   small/built-up-rising, and severely-aged areas, and aligns with the supervised
   labels.
6. **Strong spatial autocorrelation (Moran's I = 0.22)** is the distinctive aza
   finding: spatial-error regression beats OLS by a wide AIC margin, showing
   shrinkage is a geographically clustered process best modelled spatially.

---

*Artefacts:* `eda/outputs_aza/`, `classification/outputs_aza/`,
`typology/outputs_aza/`, `preprocessing/outputs/classification_ready_aza.*`.
Configs: `*/config/*_config_aza.yaml` and `config/config.yaml` (`run_mode.unit_level: aza`).
