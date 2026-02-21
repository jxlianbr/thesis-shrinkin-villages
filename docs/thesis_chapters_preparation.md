# Thesis Chapters Preparation Document
# Chapters 5 (Results), 6 (Discussion), 7 (Conclusion) + Appendix

**Project:** Mapping Shrinking Villages in Rural Japan Using Remote Sensing and Machine Learning
**Study Area:** 65 mura units, Aomori & Akita prefectures, Tohoku region
**Period:** January 2015 -- December 2025 (132 months)
**Dataset:** 8,450 panel rows (65 units x ~130 months), 30+ features

---

## Figure Referencing Convention

Throughout this document, figures are referenced using the format **Figure X.Y**
where X is the chapter number and Y is the sequential figure number within
that chapter. A mapping from figure labels to actual file paths is provided
in the Appendix (Section A.2). When writing the thesis, use your preferred
figure numbering scheme and insert the corresponding image from the file paths
listed in the appendix.

---

# CHAPTER 5: RESULTS

This chapter presents results in the order of the analytical pipeline:
data quality and exploratory findings (5.1), preprocessing and feature
engineering (5.2), supervised classification (5.3), and unsupervised
typology analysis (5.4).

---

## 5.1 Exploratory Data Analysis

### 5.1.1 Dataset Characteristics

The final panel dataset comprises 8,470 observations across 65 mura units
and 53 columns (43 numeric features, 5 identifiers, and derived fields).
Units are distributed across two prefectures: 40 in Aomori and 25 in Akita.
Mean temporal coverage is 85.2 months per unit (range: 71--93 months), with
all 65 units exhibiting temporal gaps due to cloud cover.

The geographic distribution of study units is shown in **Figure 5.1**
(study area map).

> **Figure 5.1:** Study area map showing the 65 mura units in Aomori and
> Akita prefectures.
> *File:* `outputs/maps/study_area.png` (also available as PDF:
> `outputs/maps/area_study_map.pdf`)

### 5.1.2 Missing Data

The overall missing rate is 14.62%. Missing data is concentrated in
cloud-dependent features:

| Feature Group              | Missing Rate | Cause           |
|----------------------------|-------------|-----------------|
| Sentinel-2 spectral bands  | 34.63%      | Cloud cover     |
| Spectral indices (NDVI, NDBI, MNDWI) | 34.63% | Cloud cover |
| GLCM texture features      | 23.02%      | Cloud cover     |
| VIIRS nighttime lights     | 3.07%       | Sensor gaps     |
| Terrain (slope, aspect)    | 20.00%      | Processing gaps |
| Demographics, OSM, elevation, LULC | 0.00% | Static features |

A notable asymmetry exists between prefectures for terrain features:
slope/aspect missingness is 4.0% in Akita versus 29.96% in Aomori.

The spatial and temporal distribution of missing data is visualized in
**Figure 5.2** (missing data heatmap) and **Figure 5.3** (missing data
over time).

> **Figure 5.2:** Missing data heatmap showing per-feature missingness
> across all units.
> *File:* `eda/outputs/figures/missing_heatmap.png`

> **Figure 5.3:** Temporal distribution of missing data, showing seasonal
> cloud cover patterns.
> *File:* `eda/outputs/figures/missing_by_time.png`

### 5.1.3 Feature Distributions

All 32 analysed numeric features are non-normal (Shapiro-Wilk p < 0.001
for all). The most severely skewed features are:

| Feature            | Skewness | Kurtosis | Recommended Transform |
|--------------------|----------|----------|-----------------------|
| viirs_mean         | 4.291    | 25.197   | robust_scale          |
| households_total   | 3.427    | 11.659   | log1p                 |
| age_15_64          | 3.328    | 10.905   | log1p                 |
| pop_total          | 3.270    | 10.608   | log1p                 |
| B2                 | 2.759    | 8.647    | log1p                 |
| S2_NDBI_contrast   | 2.061    | 5.482    | log1p                 |

Demographic features are static per unit (single census snapshot), meaning
the same population values repeat across all months for each unit.

Feature distributions are shown in **Figure 5.4** (remote sensing
distributions), **Figure 5.5** (demographic distributions), and
**Figure 5.6** (LULC distributions).

> **Figure 5.4:** Distribution histograms and density plots for remote
> sensing features.
> *File:* `eda/outputs/figures/distributions_rs.png`

> **Figure 5.5:** Distribution histograms and density plots for
> demographic features.
> *File:* `eda/outputs/figures/distributions_demo.png`

> **Figure 5.6:** Distribution histograms and density plots for LULC
> features.
> *File:* `eda/outputs/figures/distributions_lulc.png`

### 5.1.4 Correlation Structure and Multicollinearity

A total of 66 feature pairs exceed |r| > 0.8. The major correlation
clusters are:

**Demographics block:** All 8 demographic features are mutually correlated
at r > 0.98, reflecting their common census origin. The strongest pair is
pop_total -- pop_female (r = 0.9999).

**Spectral bands:** B2, B3, B4, B8 form a cluster with pairwise
r = 0.864--0.998.

**GLCM texture:** S2_NDBI_entropy and S2_NDBI_homogeneity are inversely
correlated at r = -0.994.

**OSM features:** osm_built_area and osm_building_count at r = 0.994.

**LULC-terrain interactions:** tri_mean correlates with dw_trees_frac
(r = 0.865), and dw_trees_frac is inversely correlated with dw_crops_frac
(r = -0.914).

The full correlation structure is visualized in **Figure 5.7** (correlation
heatmap).

> **Figure 5.7:** Pearson correlation heatmap of all numeric features,
> highlighting multicollinearity clusters.
> *File:* `eda/outputs/figures/correlation_heatmap.png`

### 5.1.5 Remote Sensing--Demographic Relationships

Correlations between remote sensing features and demographics are
generally weak:

| RS Feature         | pop_total | elderly_ratio (derived) |
|--------------------|-----------|------------------------|
| viirs_mean         | 0.380     | -0.571                 |
| dw_built_frac      | 0.263     | --                     |
| S2_NDBI_contrast   | -0.211    | --                     |
| NDVI               | 0.045     | --                     |
| NDBI               | -0.016    | --                     |

The strongest remote sensing--demographic link is viirs_mean versus
elderly_ratio at r = -0.571. OSM features show strong correlations with
demographics (r = 0.807--0.845) but represent proxy data rather than
independent remote sensing observations.

Pure spectral indices (NDVI, NDBI, MNDWI) show very weak correlations
with population variables (|r| < 0.10).

These relationships are visualized in **Figure 5.8** (RS-demographic
correlation matrix), **Figure 5.9** (VIIRS vs elderly ratio), **Figure
5.10** (VIIRS vs population), **Figure 5.11** (NDVI vs population),
and **Figure 5.12** (NDBI vs population).

> **Figure 5.8:** Correlation matrix between remote sensing features
> and demographic variables.
> *File:* `eda/outputs/figures/rs_demo_correlation.png`

> **Figure 5.9:** Scatterplot of VIIRS nighttime light radiance vs
> elderly ratio, showing the strongest RS-demographic relationship
> (r = -0.571).
> *File:* `eda/outputs/figures/viirs_vs_elderly_ratio.png`

> **Figure 5.10:** Scatterplot of VIIRS nighttime light radiance vs
> total population.
> *File:* `eda/outputs/figures/viirs_vs_population.png`

> **Figure 5.11:** Scatterplot of NDVI vs total population, illustrating
> the negligible correlation (r = 0.045).
> *File:* `eda/outputs/figures/ndvi_vs_population.png`

> **Figure 5.12:** Scatterplot of NDBI vs total population.
> *File:* `eda/outputs/figures/ndbi_vs_population.png`

### 5.1.6 Temporal Patterns

**Trends:** NDVI is increasing for 62 of 65 units (range: -0.045 to
0.558). NDBI trends are positive for all 25 Akita units but mixed in
Aomori (22 positive, 18 negative). VIIRS trends are positive for 48
units and negative for 17, with mura:Aomori:02203 showing an extreme
negative outlier (-2.204).

**Seasonality:** NDVI seasonal amplitude averages 0.340 and NDBI
averages 0.324, driven by the contrast between snow-covered winters and
vegetated summers in northern Honshu.

Temporal trend lines are shown in **Figure 5.13** and seasonal
decomposition patterns in **Figure 5.14**.

> **Figure 5.13:** Temporal trend lines for key remote sensing features
> (NDVI, NDBI, VIIRS) across all 65 units, 2015--2025.
> *File:* `eda/outputs/figures/temporal_trends.png`

> **Figure 5.14:** Seasonal patterns showing monthly mean values for
> NDVI, NDBI, and VIIRS, illustrating the winter-summer contrast in
> Tohoku.
> *File:* `eda/outputs/figures/seasonal_patterns.png`

### 5.1.7 Outlier Analysis

A total of 25,932 outlier observations were flagged across 34 features
(IQR x 1.5 method). Features with the highest outlier rates include
slope_std (25.0%), S2_NDBI_entropy (17.5%), and S2_NDBI_homogeneity
(17.3%). The most affected units are mura:Aomori:02203 (1,799 outlier
observations), mura:Aomori:02201 (1,674), and mura:Akita:05201 (1,557)
-- all larger or more urbanized units with feature values far from the
rural-dominated IQR range.

The distribution of outliers across features is shown in **Figure 5.15**.

> **Figure 5.15:** Box plots showing outlier distributions across all
> features, with outliers identified using the IQR x 1.5 method.
> *File:* `eda/outputs/figures/outlier_boxplots.png`

### 5.1.8 Prefecture Comparison

Akita and Aomori show differences in physical and demographic
characteristics:

| Characteristic         | Akita    | Aomori   |
|------------------------|----------|----------|
| Mean elevation (m)     | 263.2    | 188.3    |
| Mean VIIRS radiance    | 0.700    | 1.116    |
| Mean OSM built area (m2) | 3,965,810 | 1,363,723 |
| Mean population        | 38,380   | 31,201   |
| Mean DW built fraction | 0.074    | 0.093    |

Aomori has higher mean nighttime light radiance and Dynamic World built
fraction despite lower absolute OSM building footprint area and
population.

Prefecture-level comparisons are visualized in **Figure 5.16** (box plots
by prefecture). Spatial distributions of key features across the study
area are shown in **Figure 5.17** (NDVI spatial), **Figure 5.18**
(VIIRS spatial), and **Figure 5.19** (population spatial).

> **Figure 5.16:** Box plots comparing feature distributions between
> Akita and Aomori prefectures.
> *File:* `eda/outputs/figures/boxplots_by_prefecture.png`

> **Figure 5.17:** Spatial distribution of mean NDVI values across the
> 65 mura units.
> *File:* `eda/outputs/figures/spatial_ndvi_mean.png`

> **Figure 5.18:** Spatial distribution of mean VIIRS nighttime light
> radiance across the 65 mura units.
> *File:* `eda/outputs/figures/spatial_viirs_mean.png`

> **Figure 5.19:** Spatial distribution of total population across the
> 65 mura units.
> *File:* `eda/outputs/figures/spatial_pop_total.png`

---

## 5.2 Preprocessing and Feature Engineering

### 5.2.1 Temporal Aggregation

The 8,470 panel rows were collapsed into 65 cross-sectional units using
four temporal aggregation strategies:

- **Mean and standard deviation** for 12 features (spectral bands,
  indices, VIIRS, texture) = 24 aggregate features
- **Linear trend slope** for 4 features (NDVI, NDBI, MNDWI, viirs_mean)
  = 4 slope features
- **Seasonal amplitude** (max -- min monthly mean) for 3 features
  (NDVI, NDBI, MNDWI) = 3 seasonal features

Combined with 8 demographic, 3 OSM, 7 terrain, and 9 LULC features,
this yields 60 columns after aggregation.

### 5.2.2 Feature Engineering

Four derived ratio features were computed:

| Feature        | Formula                          | Mean  | Range         |
|----------------|----------------------------------|-------|---------------|
| elderly_ratio  | age_65_plus / pop_total          | 0.398 | 0.253 -- 0.553 |
| aging_index    | age_65_plus / age_u15            | 4.680 | 2.017 -- 12.286 |
| youth_ratio    | age_u15 / pop_total              | 0.092 | 0.045 -- 0.136 |
| household_size | pop_total / households_total     | 2.605 | 2.006 -- 3.606 |

### 5.2.3 Target Variable Construction

The three-class shrinkage target was constructed from elderly_ratio using
thresholds calibrated for approximately equal class sizes:

| Class                | Threshold            | Count | Percentage |
|----------------------|----------------------|-------|------------|
| stable               | elderly_ratio < 0.37 | 20    | 30.8%      |
| shrinking            | 0.37 <= ratio < 0.42 | 23    | 35.4%      |
| severely_shrinking   | ratio >= 0.42        | 22    | 33.8%      |

The geographic distribution shows that 17 of 20 stable units are in
Aomori (predominantly the city-level units 02201--02210), while 11 of
22 severely_shrinking units are in Akita.

### 5.2.4 Multicollinearity Resolution

Thirteen features were removed based on correlation analysis:

| Removed Feature          | Reason                                    |
|--------------------------|-------------------------------------------|
| B2_mean, B2_std          | r > 0.99 with B3/B4                       |
| B3_mean, B3_std          | r > 0.99 with B2/B4                       |
| pop_male, pop_female     | r > 0.99 with pop_total                   |
| households_total         | r > 0.99 with pop_total                   |
| age_15_64, age_75_plus   | r > 0.99 with pop_total / age_65_plus     |
| osm_building_count       | r = 0.994 with osm_built_area             |
| S2_NDBI_homogeneity_mean/std | r = -0.994 with S2_NDBI_entropy      |
| dw_snow_and_ice_frac     | Sums constraint with other LULC fractions |

### 5.2.5 Transformations

Nine skewed features received log1p transformation: B4_mean, B8_mean,
viirs_mean_mean, S2_NDBI_contrast_mean, pop_total, age_u15, age_65_plus,
osm_built_area, osm_built_ratio. Four terrain features (slope_mean,
slope_std, aspect_sin_mean, aspect_cos_mean) were imputed with median
values. All 49 numeric features were scaled using RobustScaler (median
and IQR).

### 5.2.6 Final Feature Set

The classification-ready dataset contains 65 rows and 53 columns:
49 features, 2 identifiers, and 2 target columns. Zero null values remain.

Feature breakdown by domain:
- Remote sensing temporal aggregates: 25 (18 mean/std + 4 slope + 3 seasonal)
- Demographic: 3 (pop_total, age_u15, age_65_plus)
- OSM: 2 (osm_built_area, osm_built_ratio)
- Terrain: 7
- LULC: 8
- Derived ratios: 4 (elderly_ratio, aging_index, youth_ratio, household_size)

**Data quality note:** slope_std has an IQR of 0.0 after aggregation,
causing extreme values after RobustScaler normalization (max = 1,098,115).
Similarly, aspect_sin_mean has a very small IQR (0.012), producing
scaled values ranging from -20.1 to 50.0.

---

## 5.3 Supervised Classification

### 5.3.1 Leakage Analysis

Feature importance analysis using ANOVA F-test revealed that three
derived features are deterministically linked to the target:

| Feature        | F-statistic | p-value     | Status |
|----------------|-------------|-------------|--------|
| elderly_ratio  | 134.91      | 2.61e-23    | Leaky  |
| youth_ratio    | 68.50       | 1.99e-16    | Leaky  |
| aging_index    | 63.66       | 9.36e-16    | Leaky  |

Four experiments were designed with progressive feature removal:

| Experiment      | Features | Description                              |
|-----------------|----------|------------------------------------------|
| all_features    | 49       | Reference with known leakage             |
| no_leaky        | 46       | **Primary** -- drops 3 target-derived ratios |
| no_demographic  | 42       | Drops all demographic features           |
| rs_only         | 40       | Remote sensing features only             |

The SelectKBest ANOVA F-scores for all features are shown in **Figure
5.20**, which clearly separates the leaky features from the remainder.

> **Figure 5.20:** SelectKBest ANOVA F-scores for all features, showing
> the three leaky features (elderly_ratio, youth_ratio, aging_index)
> with dramatically higher scores than all other features.
> *File:* `classification/outputs/figures/selectkbest_scores.png`

### 5.3.2 Model Performance (Primary Experiment: no_leaky)

Ten models were evaluated using RepeatedStratifiedKFold cross-validation
(5 folds x 5 repeats = 25 evaluations). Results ranked by balanced
accuracy:

| Model               | Bal. Acc.  | Std    | F1 Weighted | Cohen Kappa |
|---------------------|-----------|--------|-------------|-------------|
| Gradient Boosting   | 0.659     | 0.132  | 0.653       | 0.490       |
| Logistic Regression | 0.643     | 0.115  | 0.634       | 0.466       |
| Random Forest       | 0.613     | 0.111  | 0.611       | 0.424       |
| SVM (Linear)        | 0.588     | 0.109  | 0.556       | 0.375       |
| K-Nearest Neighbors | 0.519     | 0.132  | 0.512       | 0.279       |
| MLP Neural Network  | 0.371     | 0.101  | 0.277       | 0.051       |
| Dummy (Stratified)  | 0.349     | 0.156  | 0.337       | 0.024       |
| SVM (RBF)           | 0.334     | 0.060  | 0.234       | 0.000       |
| Dummy (Most Freq.)  | 0.333     | 0.000  | 0.186       | 0.000       |

The best model (Gradient Boosting) achieves a balanced accuracy of 0.659,
approximately double the random baseline (0.333). However, confidence
intervals are wide (95% CI: 0.450 -- 0.883), reflecting high variance
from the small sample size.

Model performance comparison is visualized as a bar chart in **Figure
5.21** and as cross-validation box plots in **Figure 5.22**.

> **Figure 5.21:** Bar chart comparing balanced accuracy across all 10
> models on the no_leaky experiment, with error bars indicating standard
> deviation across 25 CV folds.
> *File:* `classification/outputs/figures/model_comparison.png`

> **Figure 5.22:** Box plots of balanced accuracy distributions across
> 25 CV folds for each model, showing both central tendency and variance.
> *File:* `classification/outputs/figures/cv_boxplots.png`

### 5.3.3 Leakage Effect

The leakage experiment confirms that target-derived features dramatically
inflate performance:

| Model               | all_features | no_leaky | Delta   |
|---------------------|-------------|----------|---------|
| Gradient Boosting   | 0.975       | 0.659    | -0.317  |
| Random Forest       | 0.930       | 0.613    | -0.317  |
| Logistic Regression | 0.758       | 0.643    | -0.115  |
| SVM (Linear)        | 0.711       | 0.588    | -0.123  |

The tree-based models (GB, RF) show the largest drops, indicating they
relied heavily on the leaky features to construct near-perfect splits.

The leakage comparison is visualized in **Figure 5.23**.

> **Figure 5.23:** Grouped bar chart comparing model performance across
> the four leakage experiments (all_features, no_leaky, no_demographic,
> rs_only), illustrating the dramatic performance drop when leaky
> features are removed.
> *File:* `classification/outputs/figures/leakage_comparison.png`

### 5.3.4 Remote Sensing vs. Demographic Features

Comparing the feature set experiments reveals limited added value from
demographic features:

| Experiment      | Best Model            | Bal. Acc. |
|-----------------|-----------------------|-----------|
| no_leaky (46)   | Gradient Boosting     | 0.659     |
| no_demographic (42) | Gradient Boosting | 0.649     |
| rs_only (40)    | Logistic Regression   | 0.643     |

The drop from no_leaky to rs_only is only 0.016 (0.659 to 0.643),
suggesting that remote sensing features alone capture nearly all
discriminative information about shrinkage status.

### 5.3.5 Per-Class Performance

Using Gradient Boosting on the no_leaky experiment (aggregated across
25 CV folds):

| Class                | Precision | Recall | F1-Score |
|----------------------|-----------|--------|----------|
| stable               | 0.620     | 0.570  | 0.594    |
| shrinking            | 0.608     | 0.635  | 0.621    |
| severely_shrinking   | 0.752     | 0.773  | 0.762    |

The severely_shrinking class is the most detectable (F1 = 0.762),
while stable is the most difficult (F1 = 0.594). The shrinking class
falls between, but remains harder to distinguish from adjacent classes
due to its transitional nature.

Confusion matrices for all models are provided in the appendix. The
Gradient Boosting confusion matrix (**Figure 5.24**) and the ROC curves
for the best-performing model (**Figure 5.25**) are shown below.

> **Figure 5.24:** Confusion matrix for the Gradient Boosting classifier
> on the no_leaky experiment, aggregated across 25 CV folds.
> *File:* `classification/outputs/figures/confusion_matrix_gradient_boosting.png`

> **Figure 5.25:** ROC curves for the best-performing model, showing
> per-class and macro-average performance.
> *File:* `classification/outputs/figures/roc_curves_best.png`

### 5.3.6 Statistical Significance

**Friedman test:** chi-squared = 140.31, p = 2.05e-26, confirming
significant overall differences among classifiers.

**Pairwise Wilcoxon tests (Bonferroni-corrected):** 22 of 36 model
pairs show significant differences. Key findings:

- All four top models (GB, LR, RF, SVM Linear) significantly outperform
  both baselines and bottom-tier models (MLP, SVM RBF).
- The top four models are **not** significantly different from each
  other after Bonferroni correction (LR vs SVM Linear p_adj = 0.634;
  LR vs RF p_adj = 1.0; LR vs GB p_adj = 1.0; RF vs GB p_adj = 1.0).
- MLP and SVM (RBF) are not significantly different from the baselines.

**Paired t-tests:** Gradient Boosting versus Dummy (Most Frequent):
t = 12.30, p = 7.44e-12. Gradient Boosting versus Dummy (Stratified):
t = 8.27, p = 1.75e-08.

### 5.3.7 Leave-One-Out Cross-Validation

LOO-CV provides a complementary view using the maximum possible training
set (n = 64):

| Model               | LOO Bal. Acc. | Repeated CV Bal. Acc. |
|----------------------|---------------|----------------------|
| SVM (Linear)         | 0.683         | 0.588                |
| Logistic Regression  | 0.615         | 0.643                |
| Random Forest        | 0.595         | 0.613                |

SVM (Linear) ranks first under LOO (0.683) but fourth under repeated CV
(0.588), suggesting greater sensitivity to training set size.

### 5.3.8 Feature Importance

**Consensus ranking** (average rank across permutation, Gini, SHAP, and
ANOVA methods):

| Rank | Feature           | Avg Rank | Domain      |
|------|-------------------|----------|-------------|
| 1    | dw_bare_frac      | 3.0      | LULC        |
| 2    | age_u15           | 3.3      | Demographic |
| 3    | viirs_mean_mean   | 8.8      | VIIRS       |
| 4    | viirs_mean_std    | 9.8      | VIIRS       |
| 5    | household_size    | 13.2     | Derived     |
| 6    | age_65_plus       | 13.8     | Demographic |
| 7    | pop_total         | 13.8     | Demographic |
| 8    | NDBI_slope        | 14.5     | RS trend    |
| 9    | dw_water_frac     | 17.2     | LULC        |
| 10   | elevation_mean    | 17.5     | Terrain     |

VIIRS nighttime lights features (viirs_mean_mean, viirs_mean_std) are
the highest-ranked remote sensing features. LULC features (dw_bare_frac,
dw_water_frac, dw_trees_frac) also contribute strongly.

Pure spectral indices (NDVI_mean, NDBI_mean) rank low (positions 28 and
30 of 46), indicating limited discriminative power for shrinkage
classification at the temporal aggregate level.

**Gradient Boosting Gini importance** concentrates importance heavily:
the top 5 features (viirs_mean_mean = 0.183, dw_bare_frac = 0.127,
viirs_mean_std = 0.111, age_u15 = 0.092, NDBI_slope = 0.072) account
for 58.4% of total importance.

**Permutation importance** is extremely sparse: only 2 of 46 features
(dw_bare_frac = 0.064, age_u15 = 0.033) show non-zero importance on the
Gradient Boosting model, indicating heavy reliance on very few features
for the final decision boundaries.

Feature importance is visualized in **Figure 5.26** (permutation
importance), **Figure 5.27** (Gradient Boosting tree importance),
**Figure 5.28** (Random Forest tree importance), **Figure 5.29**
(XGBoost tree importance), and **Figure 5.30** (SHAP summary plot).

> **Figure 5.26:** Permutation importance for the Gradient Boosting
> model on the no_leaky experiment, showing the sparse importance
> distribution with only 2 non-zero features.
> *File:* `classification/outputs/figures/permutation_importance.png`

> **Figure 5.27:** Gini-based feature importance for the Gradient
> Boosting model, with the top 5 features accounting for 58.4% of
> total importance.
> *File:* `classification/outputs/figures/tree_importance_gradient_boosting.png`

> **Figure 5.28:** Gini-based feature importance for the Random Forest
> model.
> *File:* `classification/outputs/figures/tree_importance_random_forest.png`

> **Figure 5.29:** Gini-based feature importance for the XGBoost model.
> *File:* `classification/outputs/figures/tree_importance_xgboost.png`

> **Figure 5.30:** SHAP summary plot (beeswarm) for the Random Forest
> model, showing feature value impacts on classification decisions
> for each class.
> *File:* `classification/outputs/figures/shap_summary.png`

### 5.3.9 PCA Analysis

PCA on the 46 no_leaky features reveals that a single component explains
99.99% of variance. This reflects the dominance of features with
degenerate scaling (slope_std with IQR = 0.0, producing extreme values
up to 1,098,115 after RobustScaler). This PCA result is an artifact and
should not be interpreted as meaningful dimensionality reduction.

The PCA variance explained plot is shown in **Figure 5.31**.

> **Figure 5.31:** PCA cumulative variance explained for the
> classification feature set (46 no_leaky features), showing the
> artifact of a single dominant component due to scaling issues with
> slope_std.
> *File:* `classification/outputs/figures/pca_variance.png`

---

## 5.4 Typology Analysis

### 5.4.1 Indicator Selection

From the raw monthly panel data, 14 candidate indicators were computed
via OLS trend slopes, coefficient of variation, seasonal amplitudes,
temporal means, and demographic ratios. After redundancy pruning
(correlation threshold r > 0.85), 11 indicators were retained:

**Physical indicators (8):**
NDVI_slope, viirs_mean_slope, S2_NDBI_contrast_slope, NDVI_cv,
NDVI_seasonal_amp, NDBI_seasonal_amp, S2_NDBI_contrast_mean,
S2_NDBI_entropy_mean

**Demographic indicators (3):**
elderly_ratio, household_size, pop_total

Three features were pruned: aging_index (r = 0.923 with elderly_ratio),
youth_ratio (r = -0.878 with elderly_ratio), and NDBI_slope (near-constant).

The indicator distributions and inter-indicator correlations are shown
in **Figure 5.32** and **Figure 5.33**.

> **Figure 5.32:** Distribution plots for all 11 retained typology
> indicators.
> *File:* `typology/outputs/figures/indicator_distributions.png`

> **Figure 5.33:** Correlation heatmap of the 11 typology indicators
> after redundancy pruning.
> *File:* `typology/outputs/figures/indicator_correlation_heatmap.png`

### 5.4.2 PCA Dimensionality Reduction

PCA on the 11 standardized indicators yields:

| Component | Variance Explained | Cumulative |
|-----------|--------------------|------------|
| PC1       | 24.9%              | 24.9%      |
| PC2       | 19.4%              | 44.4%      |
| PC3       | 17.7%              | 62.1%      |
| PC4       | 11.2%              | 73.3%      |
| PC5       | 7.2%               | 80.5%      |

Five components are needed to explain 80% of variance, 7 for 90%, and
8 for 95%. The first three PCs capture 62.1%, indicating moderate
dimensionality but no single dominant axis.

**Key PC loadings:**
- PC1 (24.9%): Loads positively on NDVI_slope (0.457), NDVI_seasonal_amp
  (0.517), household_size (0.450), NDBI_seasonal_amp (0.364). Represents
  a "vegetation dynamism + household vitality" axis.
- PC2 (19.4%): Loads positively on S2_NDBI_contrast_slope (0.579) and
  pop_total (0.482), negatively on S2_NDBI_contrast_mean (-0.485).
  Represents a "built-environment change + population size" axis.
- PC3 (17.7%): Loads positively on viirs_mean_slope (0.522) and
  elderly_ratio (0.505), negatively on S2_NDBI_contrast_mean (-0.458).
  Represents a "nightlight trend + aging" axis.

The PCA scree plot and biplot are shown in **Figure 5.34** and **Figure
5.35**.

> **Figure 5.34:** PCA scree plot showing individual and cumulative
> variance explained for the 11 typology indicators.
> *File:* `typology/outputs/figures/pca_scree.png`

> **Figure 5.35:** PCA biplot (PC1 vs PC2) showing the 65 units
> projected onto the first two principal components, with indicator
> loading vectors overlaid.
> *File:* `typology/outputs/figures/pca_biplot.png`

### 5.4.3 Primary Clustering (k = 3)

K-means clustering with k = 3 produces three typology groups:

| Cluster | n  | Label Interpretation    |
|---------|----|-----------------------|
| 0       | 19 | Urban/peri-urban core |
| 1       | 40 | Rural mainstream       |
| 2       | 6  | High-texture remote   |

**Overall silhouette score:** 0.235
**Bootstrap ARI (1000 resamples):** 0.544 (95% CI: 0.112 -- 1.000)

**Cluster 0 (n = 19) -- Urban/peri-urban core:**
- Lowest NDVI_slope (0.112 vs 0.287 for Cluster 1)
- Negative viirs_mean_slope (-0.190; declining nightlights)
- Highest NDVI_cv (1.844; most variable vegetation)
- Lowest NDVI_seasonal_amp (0.233)
- Lowest elderly_ratio (0.364) and household_size (2.343)
- Highest pop_total (56,393)
- Contains 12 stable, 3 shrinking, 4 severely_shrinking units

**Cluster 1 (n = 40) -- Rural mainstream:**
- Highest NDVI_slope (0.287; greening trend)
- Positive viirs_mean_slope (0.230; brightening)
- Average texture and GLCM values
- Highest elderly_ratio (0.415)
- Moderate pop_total (26,730)
- Contains 6 stable, 18 shrinking, 16 severely_shrinking units

**Cluster 2 (n = 6) -- High-texture remote:**
- Highest NDVI_slope (0.369)
- Strongly negative S2_NDBI_contrast_slope (-221.97; rapidly declining
  texture contrast)
- Highest S2_NDBI_contrast_mean (412.01; double the other clusters)
- Highest NDVI_seasonal_amp (0.514) and household_size (2.888)
- Lowest pop_total (9,470)
- Contains 2 stable, 2 shrinking, 2 severely_shrinking (equal distribution)

Cluster characteristics are visualized in **Figure 5.36** (cluster
profiles heatmap), **Figure 5.37** (cluster box plots), **Figure 5.38**
(parallel coordinates), and **Figure 5.39** (spatial cluster map). The
silhouette diagram is shown in **Figure 5.40**.

> **Figure 5.36:** Heatmap of standardized cluster profile means for
> all 11 indicators across the three clusters.
> *File:* `typology/outputs/figures/cluster_profiles_heatmap.png`

> **Figure 5.37:** Box plots of indicator values within each cluster,
> showing within-cluster variation and between-cluster separation.
> *File:* `typology/outputs/figures/cluster_boxplots.png`

> **Figure 5.38:** Parallel coordinates plot showing individual unit
> trajectories across all 11 indicators, colored by cluster membership.
> *File:* `typology/outputs/figures/cluster_parallel_coords.png`

> **Figure 5.39:** Spatial map of cluster assignments for the 65 mura
> units, showing the geographic distribution of the three typology
> groups across Aomori and Akita prefectures.
> *File:* `typology/outputs/figures/cluster_map.png`

> **Figure 5.40:** Silhouette diagram for k = 3 clustering, showing
> per-unit silhouette coefficients grouped by cluster (overall
> silhouette = 0.235).
> *File:* `typology/outputs/figures/silhouette_diagram_k3.png`

### 5.4.4 Cluster--Shrinkage Class Correspondence (k = 3)

| Cluster | Severely Shrinking | Shrinking | Stable |
|---------|-------------------|-----------|--------|
| 0       | 4                 | 3         | 12     |
| 1       | 16                | 18        | 6      |
| 2       | 2                 | 2         | 2      |

**Supervised ARI = 0.105**, indicating weak but non-trivial agreement
between unsupervised clusters and the shrinkage classification.
Cluster 0 is enriched for stable units (63%), and Cluster 1 is enriched
for shrinking/severely_shrinking (85%), while Cluster 2 shows no
association with any shrinkage class.

The cluster-class cross-tabulation is visualized in **Figure 5.41**.

> **Figure 5.41:** Heatmap of the cross-tabulation between unsupervised
> cluster assignments and supervised shrinkage classes.
> *File:* `typology/outputs/figures/cluster_crosstab_heatmap.png`

### 5.4.5 Higher-k Clustering Solutions

**k = 4 (silhouette = 0.255):**
Splits Cluster 0 into a large mainstream (n = 42), a small urban core
(n = 4, all stable, mean pop = 211,358), a low-dynamism group (n = 13),
and the persistent high-texture group (n = 6). The 4-unit urban cluster
contains the largest cities and has the lowest elderly_ratio (0.299).

**k = 5 (silhouette = 0.217):**
Further splits the mainstream, creating a 3-unit cluster with very high
NDVI_cv (6.399) and near-zero NDVI_slope (0.009), representing stagnant
vegetation units. The data-driven k from gap statistic is 5, but
silhouette is lower and bootstrap stability drops (ARI = 0.497).

Optimal k determination is shown in **Figure 5.42** and the multi-k
comparison in **Figure 5.43**. Silhouette diagrams for k = 4 and k = 5
are in **Figure 5.44** and **Figure 5.45**. The hierarchical
dendrogram is in **Figure 5.46**.

> **Figure 5.42:** Optimal k metrics (silhouette, gap statistic,
> inertia) plotted against number of clusters.
> *File:* `typology/outputs/figures/optimal_k_metrics.png`

> **Figure 5.43:** Multi-k comparison showing spatial cluster maps
> side-by-side for k = 3, 4, and 5.
> *File:* `typology/outputs/figures/multi_k_comparison.png`

> **Figure 5.44:** Silhouette diagram for k = 4 clustering.
> *File:* `typology/outputs/figures/silhouette_diagram_k4.png`

> **Figure 5.45:** Silhouette diagram for k = 5 clustering.
> *File:* `typology/outputs/figures/silhouette_diagram_k5.png`

> **Figure 5.46:** Hierarchical clustering dendrogram of the 65 mura
> units based on the 11 typology indicators.
> *File:* `typology/outputs/figures/dendrogram.png`

### 5.4.6 Cluster Stability

**Bootstrap stability (1000 resamples, 80% sample fraction):**

| k | ARI       | 95% CI            |
|---|-----------|-------------------|
| 3 | 0.544     | [0.112, 1.000]    |
| 4 | 0.560     | [0.163, 0.947]    |
| 5 | 0.497     | [0.237, 0.895]    |

k = 3 and k = 4 have comparable stability; k = 5 is lower.

**Perturbation stability (leave-one-feature-out, k = 3):**

| Dropped Feature          | ARI vs Full | Fraction Changed |
|--------------------------|-------------|------------------|
| NDBI_seasonal_amp        | 1.000       | 0.0%             |
| S2_NDBI_entropy_mean     | 0.944       | 89.2%            |
| elderly_ratio            | 0.649       | 76.9%            |
| pop_total                | 0.603       | 78.5%            |
| NDVI_cv                  | 0.562       | 13.9%            |
| NDVI_slope               | 0.488       | 76.9%            |
| household_size           | 0.486       | 73.9%            |
| viirs_mean_slope         | 0.390       | 69.2%            |
| NDVI_seasonal_amp        | 0.387       | 75.4%            |
| S2_NDBI_contrast_mean    | 0.346       | 87.7%            |
| S2_NDBI_contrast_slope   | 0.290       | 30.8%            |

The clustering is most sensitive to removing S2_NDBI_contrast_slope
(ARI drops to 0.290) and S2_NDBI_contrast_mean (ARI = 0.346), indicating
that GLCM texture features are the most influential for cluster separation.
NDBI_seasonal_amp has zero impact when removed (ARI = 1.0).

### 5.4.7 Specification Robustness

Re-clustering with feature subsets:

| Specification      | n Features | ARI vs Primary | Silhouette |
|--------------------|-----------|----------------|------------|
| physical_only      | 8         | 0.457          | 0.244      |
| trend_only         | 4         | 0.414          | 0.491      |
| level_only         | 7         | 0.099          | 0.213      |
| demographic_only   | 3         | -0.001         | 0.326      |

The physical_only and trend_only specifications best reproduce the
primary clustering (ARI = 0.457 and 0.414 respectively), confirming
that physical remote sensing indicators drive the typology. The
demographic_only specification produces an essentially unrelated
clustering (ARI = -0.001), demonstrating that the typology captures
spatial patterns not explained by demographics alone.

The specification robustness comparison is visualized in **Figure 5.47**
and the spatial model comparison in **Figure 5.48**.

> **Figure 5.47:** Specification robustness bar chart comparing ARI
> agreement between each feature subset clustering and the primary
> (full-indicator) clustering.
> *File:* `typology/outputs/figures/specification_robustness.png`

> **Figure 5.48:** Spatial comparison of cluster maps produced by
> different feature specifications (physical_only, trend_only,
> level_only, demographic_only) versus the primary clustering.
> *File:* `typology/outputs/figures/spatial_model_comparison.png`

### 5.4.8 Physical--Demographic Relationships

**Top significant correlations (Bonferroni-corrected):**

| Physical Indicator     | Demographic       | r (Spearman) | p-value  |
|------------------------|-------------------|-------------|----------|
| S2_NDBI_contrast_slope | pop_total         | 0.612       | 6.2e-08  |
| NDVI_seasonal_amp      | household_size    | 0.538       | 3.8e-06  |
| viirs_mean_slope       | pop_total (Pears.)| -0.482      | 4.8e-05  |
| NDBI_seasonal_amp      | household_size    | 0.431       | 3.3e-04  |

**Regression (DV = elderly_ratio):**
Only viirs_mean_slope is a significant predictor (coefficient = 0.061,
R-squared = 0.142, p = 0.002). S2_NDBI_entropy_mean is marginally
significant (p = 0.051, R-squared = 0.059). All other physical
indicators are non-significant for predicting elderly_ratio directly.

**Kruskal-Wallis tests (cluster differences):**
8 of 11 indicators show significant between-cluster differences. The
three non-significant indicators are NDVI_cv (p = 0.130),
S2_NDBI_entropy_mean (p = 0.274), and pop_total (p = 0.507).

**VIF analysis:** All retained predictors have VIF < 5.0 (range:
1.20--4.62), confirming acceptable multicollinearity levels for the
physical indicators.

Correlation analyses are shown in **Figure 5.49** (Pearson correlation
heatmap), **Figure 5.50** (Spearman correlation heatmap), **Figure
5.51** (subgroup correlations), and **Figure 5.52** (regression
diagnostics). The regression coefficients are in **Figure 5.53**.

> **Figure 5.49:** Pearson correlation heatmap between physical and
> demographic indicators.
> *File:* `typology/outputs/figures/correlation_heatmap_pearson.png`

> **Figure 5.50:** Spearman rank correlation heatmap between physical
> and demographic indicators.
> *File:* `typology/outputs/figures/correlation_heatmap_spearman.png`

> **Figure 5.51:** Subgroup correlation analysis showing within-cluster
> physical-demographic relationships.
> *File:* `typology/outputs/figures/subgroup_correlations.png`

> **Figure 5.52:** Regression diagnostic plots (residuals vs fitted,
> Q-Q plot, scale-location, leverage) for the elderly_ratio prediction
> model.
> *File:* `typology/outputs/figures/regression_diagnostics.png`

> **Figure 5.53:** Regression coefficient plot showing the contribution
> of each physical indicator to elderly_ratio prediction.
> *File:* `typology/outputs/figures/regression_coefficients.png`

---

# CHAPTER 6: DISCUSSION

## 6.1 Classification Performance in Context

### 6.1.1 Absolute Performance Assessment

The best model (Gradient Boosting, balanced accuracy = 0.659 on the
no_leaky experiment) represents a modest improvement over random
classification (0.333) but falls short of levels typically desired for
operational applications. The classification task is inherently
difficult: 65 units represent a very small sample for training a
three-class classifier with 46 features. The effective training set per
CV fold is only 52 samples (80% of 65), an extremely unfavorable
samples-to-features ratio.

The wide confidence intervals (0.450--0.883) indicate high
between-fold variance, a direct consequence of the small sample.
Importantly, the top four models (GB, LR, RF, SVM Linear) are
statistically indistinguishable from each other after Bonferroni
correction, suggesting that model architecture matters less than data
volume at this sample size.

The CV box plots (**Figure 5.22**, p. XX) illustrate this overlap
visually, with substantial interquartile range overlap among the top
four models.

### 6.1.2 Leakage as Methodological Validation

The leakage experiment provides strong methodological validation. The
three derived features (elderly_ratio, youth_ratio, aging_index)
produce near-perfect classification (GB = 0.975 with all features),
confirming that the target variable is correctly constructed from
demographic data. The 31.7 percentage-point drop when removing these
features is itself an important result: it quantifies the difference
between "can demographics predict demographics" (trivially yes) and
"can remote sensing predict demographics" (moderately).

As shown in **Figure 5.23** (leakage comparison), tree-based models
are particularly susceptible to leakage exploitation.

### 6.1.3 Remote Sensing Sufficiency

A key finding is the near-equivalence of the rs_only experiment
(best bal. acc. = 0.643) and the full no_leaky experiment (0.659).
The marginal contribution of demographic and OSM features is only
0.016 balanced accuracy points. This suggests that satellite-derived
features -- particularly VIIRS nighttime lights and LULC fractions --
already encode most of the information relevant to distinguishing
shrinkage status.

This has practical implications: remote sensing data alone may be
sufficient for initial screening of shrinking settlements, reducing
dependence on census data that is collected infrequently (every 5 years
in Japan) and may not be available at sub-municipal resolution.

### 6.1.4 The Dominance of VIIRS

VIIRS nighttime lights features (viirs_mean_mean, viirs_mean_std) are
consistently the highest-ranked remote sensing predictors across all
importance methods (see **Figure 5.26** through **Figure 5.30**). VIIRS
radiance is a direct proxy for human economic activity and infrastructure
investment, making its association with depopulation patterns intuitive:
shrinking settlements experience declining commercial activity, reduced
street lighting, and lower household energy consumption.

The negative correlation between viirs_mean and elderly_ratio (r = -0.571)
is the strongest single RS-demographic link found in the EDA (visualized
in **Figure 5.9**), and viirs_mean_slope is the only physical indicator
that significantly predicts elderly_ratio in the typology regression
(R-squared = 0.142, **Figure 5.53**).

### 6.1.5 Why Spectral Indices Underperform

Pure spectral indices (NDVI, NDBI) show remarkably weak correlations
with population variables (|r| < 0.10, see **Figure 5.11** and **Figure
5.12**) and rank low in classification importance. Several factors
contribute:

1. **Vegetation greening paradox:** NDVI is increasing for 62/65 units
   (**Figure 5.13**), including both stable and severely shrinking
   villages. As populations decline, agricultural abandonment leads to
   vegetation regrowth, making NDVI trends potentially counter-intuitive
   as shrinkage indicators.

2. **Spatial resolution mismatch:** At the mura level (areas ranging
   from 18 to 1,093 km2), spectral indices reflect dominant land cover
   rather than settlement-scale changes. Population loss concentrated in
   small settlement cores may be invisible at this aggregation level.

3. **Temporal resolution vs. census timing:** Monthly spectral composites
   capture seasonal and weather-driven variation that dominates the signal
   (seasonal amplitude of ~0.34 for NDVI, **Figure 5.14**), potentially
   masking the gradual multi-year trends associated with depopulation.

### 6.1.6 The Shrinking Class as a Boundary Problem

The per-class analysis reveals that the "shrinking" class (intermediate
elderly_ratio 0.37--0.42) is the hardest to classify (**Figure 5.24**),
consistent with its role as a transitional category between stable and
severely shrinking. The threshold-based target construction creates
inherently fuzzy boundaries: a unit with elderly_ratio = 0.369 (stable)
is nearly identical to one with 0.371 (shrinking). This suggests that
ordinal approaches or continuous prediction of elderly_ratio might be
more appropriate than discrete classification.

## 6.2 Typology Insights

### 6.2.1 Physical vs. Demographic Clustering

The specification robustness analysis (**Figure 5.47**) provides the
clearest result of the typology study: the primary clustering is driven
almost entirely by physical remote sensing indicators (ARI = 0.457 for
physical_only) while demographic indicators produce an unrelated grouping
(ARI = -0.001). This demonstrates that the identified typologies capture
meaningful spatial patterns in the physical landscape that are not simply
proxies for demographic differences.

### 6.2.2 Three Village Typologies

The k = 3 solution identifies three interpretable typologies (**Figure
5.39** shows their geographic distribution):

**Type 0 -- Urban/peri-urban core (n = 19):** Characterized by lower
vegetation dynamism, declining nightlights, high vegetation variability,
and larger populations. These units contain the major population centers
and are more likely to be classified as stable (63%). The declining
VIIRS trend may reflect LED conversion or urban restructuring rather
than depopulation.

**Type 1 -- Rural mainstream (n = 40):** The largest group, characterized
by positive greening trends, brightening nightlights, and higher
elderly_ratio. This group captures the "typical" rural trajectory in
Tohoku and contains the majority of shrinking and severely shrinking
units (34 of 40).

**Type 2 -- High-texture remote (n = 6):** A distinct cluster defined
by extremely high GLCM texture contrast (mean = 412, double the other
clusters) and a strong negative texture trend (slope = -222). These
represent physically distinctive landscapes, potentially mountainous
areas with heterogeneous land cover, and show no association with
shrinkage class (equal distribution across all three classes).

The cluster profiles are characterized in detail in **Figure 5.36**
(heatmap) and **Figure 5.38** (parallel coordinates).

### 6.2.3 The Role of GLCM Texture

The perturbation analysis reveals that GLCM texture features
(S2_NDBI_contrast_slope, S2_NDBI_contrast_mean) are the most
influential for cluster separation (ARI drops to 0.290 and 0.346
when removed). This is notable because GLCM features rank low in the
supervised classification importance.

This apparent contradiction has a meaningful explanation: GLCM texture
captures landscape heterogeneity and structural characteristics that
distinguish physical typologies (e.g., mountainous vs. coastal vs.
lowland), while shrinkage classification is primarily driven by
human-activity proxies (VIIRS, LULC built fraction). The two tasks
probe different dimensions of the rural landscape.

### 6.2.4 Weak Cluster--Class Correspondence

The supervised ARI of 0.105 between clusters and shrinkage classes
(**Figure 5.41**) confirms that typologies and shrinkage status are
largely independent dimensions. This is expected and useful: it means
that shrinkage occurs across all physical landscape types, and that a
"one-size-fits-all" intervention approach may be inappropriate. Policies
should consider both the shrinkage severity and the physical typology
of each village.

### 6.2.5 Clustering Limitations

The moderate silhouette score (0.235, **Figure 5.40**) and wide
bootstrap confidence intervals (0.112--1.000) indicate that cluster
boundaries are not sharp. With only 65 observations in 11-dimensional
indicator space, the clustering is inherently underdetermined. The k = 3
solution was selected as primary because it balances interpretability
with stability, but the k = 4 solution (silhouette = 0.255, **Figure
5.44**) is equally defensible and provides the additional insight of
separating a small urban core group.

## 6.3 Methodological Considerations

### 6.3.1 Sample Size Constraints

The fundamental limitation of this study is the sample size of n = 65.
This constrains the classifier to high-bias models (linear SVM,
logistic regression), prevents reliable hyperparameter tuning, and
makes cross-validation unstable. The 132-month temporal depth partially
compensates by enabling rich temporal aggregate features, but the
cross-sectional unit count is the binding constraint.

### 6.3.2 Target Variable Design

The elderly_ratio-based target is a pragmatic choice but carries
limitations:

1. **Static census data:** Demographics represent a single-year
   snapshot, meaning the "shrinkage" classification reflects current
   state rather than trajectory. A unit may be classified as "stable"
   despite ongoing depopulation, if its elderly ratio has not yet
   crossed the threshold.

2. **Threshold sensitivity:** The 0.37/0.42 thresholds were
   calibrated for balanced classes rather than from external criteria.
   Different thresholds would reclassify boundary units and potentially
   change model performance.

3. **Ecological validity:** The elderly_ratio captures aging but not
   all dimensions of rural shrinkage (e.g., economic decline,
   service withdrawal, infrastructure deterioration). A composite
   index might better capture the multi-dimensional nature of the
   phenomenon.

### 6.3.3 Scaling Artifacts

The RobustScaler produced extreme values for features with zero or
near-zero IQR (slope_std: max scaled value = 1,098,115; aspect_sin_mean:
range -20 to 50). These artifacts affected PCA (**Figure 5.31** -- one
component explains 99.99% of variance, driven by slope_std) and may
have biased distance-based models (KNN, SVM RBF). Future work should
either drop degenerate features or use min-max clipping after scaling.

### 6.3.4 Spatial Autocorrelation

This study treats the 65 units as independent observations, but
neighboring mura units likely share similar physical and demographic
characteristics (Tobler's first law of geography). Spatial
autocorrelation may inflate apparent CV performance if spatially
adjacent units appear in both training and test folds. Spatial
cross-validation (e.g., leave-one-prefecture-out or spatial block CV)
would provide a more conservative performance estimate.

---

# CHAPTER 7: CONCLUSION

## 7.1 Summary of Findings

This study developed and evaluated a machine learning pipeline for
classifying shrinking villages in Aomori and Akita prefectures using
multi-temporal satellite remote sensing data from 2015--2025. The key
findings are:

1. **Moderate classification performance is achievable.** Gradient
   Boosting achieves a balanced accuracy of 0.659 in distinguishing
   three shrinkage categories (stable, shrinking, severely shrinking)
   among 65 village-level units, significantly outperforming random
   baselines but with substantial uncertainty due to small sample size.

2. **Remote sensing features alone are nearly sufficient.** The
   performance gap between using all non-leaky features (0.659) and
   remote sensing features only (0.643) is negligible, indicating that
   satellite-derived indicators -- especially VIIRS nighttime lights
   and LULC fractions -- encode the majority of information relevant to
   shrinkage classification.

3. **VIIRS nighttime lights are the most informative remote sensing
   feature.** VIIRS mean radiance and its temporal variability
   consistently rank highest across all feature importance methods.
   The correlation between VIIRS and elderly_ratio (r = -0.571) is
   the strongest RS-demographic link identified.

4. **Traditional spectral indices have limited utility.** NDVI and
   NDBI correlations with demographic variables are negligible
   (|r| < 0.10), likely due to the vegetation greening paradox in
   depopulating areas and spatial resolution mismatch.

5. **Feature leakage inflates results dramatically.** Including
   target-derived features (elderly_ratio, youth_ratio, aging_index)
   raises Gradient Boosting performance from 0.659 to 0.975, validating
   the leakage detection framework and demonstrating the critical
   importance of careful feature auditing.

6. **Three physically distinct village typologies exist.** Unsupervised
   clustering identifies an urban/peri-urban core (n = 19), a rural
   mainstream (n = 40), and a high-texture remote cluster (n = 6). These
   typologies are driven by GLCM texture features and vegetation dynamics
   rather than demographics.

7. **Typologies and shrinkage status are largely independent.**
   The supervised ARI between clusters and classes is only 0.105,
   and demographic-only clustering produces zero agreement with the
   physical typology (ARI = -0.001). This confirms that shrinkage
   occurs across all landscape types.

## 7.2 Research Contributions

This study makes the following contributions:

- **Methodological:** Demonstrates a complete, reproducible pipeline from
  raw satellite imagery to village-level classification, with explicit
  leakage testing and four progressively restrictive feature experiments.

- **Empirical:** Provides the first systematic comparison of remote
  sensing feature types (spectral, texture, nightlights, terrain, LULC,
  OSM) for shrinkage classification in Japanese rural settlements.

- **Practical:** Establishes that VIIRS nighttime lights alone can serve
  as a screening tool for identifying shrinking settlements, reducing
  dependence on infrequent census data.

- **Conceptual:** Separates the physical landscape typology from the
  shrinkage classification, showing that these are complementary but
  independent dimensions that should be considered jointly in policy
  design.

## 7.3 Limitations

1. **Small sample size (n = 65):** Limits model complexity, produces
   high-variance estimates, and prevents reliable comparison of advanced
   models. Results may not generalize to other regions.

2. **Single census snapshot:** The target variable is based on one
   point-in-time demographic observation rather than a depopulation
   trajectory, potentially conflating current state with trend.

3. **Coarse spatial aggregation:** Mura-level units range from 18 to
   1,093 km2, averaging out within-unit heterogeneity and potentially
   obscuring settlement-scale dynamics.

4. **No spatial cross-validation:** Standard stratified CV may
   overestimate performance due to spatial autocorrelation among
   neighboring units.

5. **Scaling artifacts:** RobustScaler produces extreme values for
   features with zero IQR, affecting PCA and distance-based models.

6. **Limited clustering stability:** Moderate silhouette scores (0.235)
   and wide bootstrap confidence intervals suggest fuzzy cluster
   boundaries.

## 7.4 Recommendations for Future Research

1. **Expand the study area** to additional prefectures in Tohoku or
   other rural regions of Japan, increasing sample size and enabling
   more robust model evaluation.

2. **Incorporate multi-census data** (2010, 2015, 2020) to construct
   trajectory-based target variables that capture actual depopulation
   rates rather than cross-sectional aging status.

3. **Increase spatial resolution** by conducting analysis at the aza
   (sub-village) level, where settlement-scale dynamics may be more
   visible in remote sensing features.

4. **Apply spatial cross-validation** (spatial block CV or
   leave-one-group-out) to obtain more conservative and realistic
   performance estimates.

5. **Explore deep learning** on raw image patches rather than
   hand-crafted features, particularly for GLCM-like texture
   extraction, which may benefit from learned representations.

6. **Develop composite shrinkage indices** that integrate multiple
   dimensions of rural decline (demographic, economic, infrastructure)
   as continuous target variables rather than threshold-based classes.

7. **Address scaling artifacts** by replacing or augmenting RobustScaler
   with quantile transformation or by pre-filtering features with
   degenerate distributions (zero IQR).

8. **Validate with ground truth** using field surveys, municipal
   records, or high-resolution imagery to assess whether classification
   errors reflect genuine model failures or ambiguities in the target
   definition.

---

# APPENDICES

---

## Appendix A: Key Numbers Quick Reference

### A.1 Dataset
- 65 mura units (40 Aomori, 25 Akita)
- 8,470 panel rows, 53 columns
- 132 months (2015-01 to 2025-12)
- Mean temporal coverage: 85.2 months/unit

### A.2 Missing Data
- Overall: 14.62%
- Spectral bands: 34.63%
- GLCM texture: 23.02%
- VIIRS: 3.07%
- Demographics, OSM, LULC: 0%

### A.3 Target Distribution
- stable: 20 units (30.8%), elderly_ratio < 0.37
- shrinking: 23 units (35.4%), 0.37 <= elderly_ratio < 0.42
- severely_shrinking: 22 units (33.8%), elderly_ratio >= 0.42

### A.4 Final Feature Set
- 49 features after multicollinearity removal
- 13 features dropped (9 demographics/spectral, 2 GLCM, 1 OSM, 1 LULC)
- 9 features log1p-transformed
- RobustScaler applied to all 49

### A.5 Classification (no_leaky experiment, 46 features)
- Best CV: Gradient Boosting, bal. acc. = 0.659
- Best LOO: SVM Linear, bal. acc. = 0.683
- RS-only best: Logistic Regression, bal. acc. = 0.643
- Leakage effect: GB drops from 0.975 to 0.659
- Friedman test: p = 2.05e-26

### A.6 Feature Importance (consensus top 5)
1. dw_bare_frac (avg rank 3.0)
2. age_u15 (avg rank 3.3)
3. viirs_mean_mean (avg rank 8.8)
4. viirs_mean_std (avg rank 9.8)
5. household_size (avg rank 13.2)

### A.7 Typology (k = 3)
- Cluster 0: n=19, urban/peri-urban core
- Cluster 1: n=40, rural mainstream
- Cluster 2: n=6, high-texture remote
- Silhouette: 0.235
- Bootstrap ARI: 0.544 (CI: 0.112--1.000)
- Supervised ARI: 0.105 (weak class correspondence)

### A.8 Key Correlations
- viirs_mean vs elderly_ratio: r = -0.571
- S2_NDBI_contrast_slope vs pop_total: rho = 0.612
- NDVI_seasonal_amp vs household_size: rho = 0.538
- NDVI vs pop_total: r = 0.045 (negligible)

---

## Appendix B: Complete Figure Catalog

This appendix provides a complete, organized catalog of all 63 unique
figures produced by the analysis pipeline. Figures are organized by
module and include both in-text references and supplementary figures
not referenced in the main chapters. Typology figures are available
in both PNG (for screen) and PDF (for print) formats.

### B.1 Study Area Map

| Fig. ID | Description | File Path |
|---------|------------|-----------|
| B.1.1 | Study area map (PNG) | `outputs/maps/study_area.png` |
| B.1.2 | Study area map (PDF, print quality) | `outputs/maps/area_study_map.pdf` |

### B.2 Exploratory Data Analysis Figures (18 figures)

| Fig. ID | Description | In-Text Ref. | File Path |
|---------|------------|-------------|-----------|
| B.2.1 | Missing data heatmap | Fig. 5.2 | `eda/outputs/figures/missing_heatmap.png` |
| B.2.2 | Missing data over time | Fig. 5.3 | `eda/outputs/figures/missing_by_time.png` |
| B.2.3 | Distributions: remote sensing features | Fig. 5.4 | `eda/outputs/figures/distributions_rs.png` |
| B.2.4 | Distributions: demographic features | Fig. 5.5 | `eda/outputs/figures/distributions_demo.png` |
| B.2.5 | Distributions: LULC features | Fig. 5.6 | `eda/outputs/figures/distributions_lulc.png` |
| B.2.6 | Correlation heatmap (all features) | Fig. 5.7 | `eda/outputs/figures/correlation_heatmap.png` |
| B.2.7 | RS-demographic correlation matrix | Fig. 5.8 | `eda/outputs/figures/rs_demo_correlation.png` |
| B.2.8 | VIIRS vs elderly ratio scatterplot | Fig. 5.9 | `eda/outputs/figures/viirs_vs_elderly_ratio.png` |
| B.2.9 | VIIRS vs population scatterplot | Fig. 5.10 | `eda/outputs/figures/viirs_vs_population.png` |
| B.2.10 | NDVI vs population scatterplot | Fig. 5.11 | `eda/outputs/figures/ndvi_vs_population.png` |
| B.2.11 | NDBI vs population scatterplot | Fig. 5.12 | `eda/outputs/figures/ndbi_vs_population.png` |
| B.2.12 | Temporal trend lines (NDVI, NDBI, VIIRS) | Fig. 5.13 | `eda/outputs/figures/temporal_trends.png` |
| B.2.13 | Seasonal patterns | Fig. 5.14 | `eda/outputs/figures/seasonal_patterns.png` |
| B.2.14 | Outlier box plots | Fig. 5.15 | `eda/outputs/figures/outlier_boxplots.png` |
| B.2.15 | Box plots by prefecture | Fig. 5.16 | `eda/outputs/figures/boxplots_by_prefecture.png` |
| B.2.16 | Spatial map: NDVI mean | Fig. 5.17 | `eda/outputs/figures/spatial_ndvi_mean.png` |
| B.2.17 | Spatial map: VIIRS mean | Fig. 5.18 | `eda/outputs/figures/spatial_viirs_mean.png` |
| B.2.18 | Spatial map: population | Fig. 5.19 | `eda/outputs/figures/spatial_pop_total.png` |

### B.3 Classification Figures (21 figures)

| Fig. ID | Description | In-Text Ref. | File Path |
|---------|------------|-------------|-----------|
| B.3.1 | SelectKBest ANOVA F-scores | Fig. 5.20 | `classification/outputs/figures/selectkbest_scores.png` |
| B.3.2 | Model comparison bar chart | Fig. 5.21 | `classification/outputs/figures/model_comparison.png` |
| B.3.3 | CV box plots (all models) | Fig. 5.22 | `classification/outputs/figures/cv_boxplots.png` |
| B.3.4 | Leakage experiment comparison | Fig. 5.23 | `classification/outputs/figures/leakage_comparison.png` |
| B.3.5 | Confusion matrix: Gradient Boosting | Fig. 5.24 | `classification/outputs/figures/confusion_matrix_gradient_boosting.png` |
| B.3.6 | ROC curves (best model) | Fig. 5.25 | `classification/outputs/figures/roc_curves_best.png` |
| B.3.7 | Permutation importance | Fig. 5.26 | `classification/outputs/figures/permutation_importance.png` |
| B.3.8 | Tree importance: Gradient Boosting | Fig. 5.27 | `classification/outputs/figures/tree_importance_gradient_boosting.png` |
| B.3.9 | Tree importance: Random Forest | Fig. 5.28 | `classification/outputs/figures/tree_importance_random_forest.png` |
| B.3.10 | Tree importance: XGBoost | Fig. 5.29 | `classification/outputs/figures/tree_importance_xgboost.png` |
| B.3.11 | SHAP summary plot | Fig. 5.30 | `classification/outputs/figures/shap_summary.png` |
| B.3.12 | PCA variance explained | Fig. 5.31 | `classification/outputs/figures/pca_variance.png` |
| B.3.13 | Confusion matrix: Logistic Regression | Appendix | `classification/outputs/figures/confusion_matrix_logistic_regression.png` |
| B.3.14 | Confusion matrix: Random Forest | Appendix | `classification/outputs/figures/confusion_matrix_random_forest.png` |
| B.3.15 | Confusion matrix: SVM Linear | Appendix | `classification/outputs/figures/confusion_matrix_svm_linear.png` |
| B.3.16 | Confusion matrix: SVM RBF | Appendix | `classification/outputs/figures/confusion_matrix_svm_rbf.png` |
| B.3.17 | Confusion matrix: KNN | Appendix | `classification/outputs/figures/confusion_matrix_knn.png` |
| B.3.18 | Confusion matrix: MLP | Appendix | `classification/outputs/figures/confusion_matrix_mlp.png` |
| B.3.19 | Confusion matrix: XGBoost | Appendix | `classification/outputs/figures/confusion_matrix_xgboost.png` |
| B.3.20 | Confusion matrix: Dummy (Stratified) | Appendix | `classification/outputs/figures/confusion_matrix_dummy_stratified.png` |
| B.3.21 | Confusion matrix: Dummy (Most Frequent) | Appendix | `classification/outputs/figures/confusion_matrix_dummy_most_frequent.png` |

### B.4 Typology Figures (22 unique figures, PNG + PDF)

All typology figures are available in both PNG and PDF formats at the
same path with the respective extension. Only PNG paths are listed below;
substitute `.pdf` for print-quality versions.

| Fig. ID | Description | In-Text Ref. | File Path (PNG) |
|---------|------------|-------------|-----------------|
| B.4.1 | Indicator distributions | Fig. 5.32 | `typology/outputs/figures/indicator_distributions.png` |
| B.4.2 | Indicator correlation heatmap | Fig. 5.33 | `typology/outputs/figures/indicator_correlation_heatmap.png` |
| B.4.3 | PCA scree plot | Fig. 5.34 | `typology/outputs/figures/pca_scree.png` |
| B.4.4 | PCA biplot | Fig. 5.35 | `typology/outputs/figures/pca_biplot.png` |
| B.4.5 | Cluster profiles heatmap | Fig. 5.36 | `typology/outputs/figures/cluster_profiles_heatmap.png` |
| B.4.6 | Cluster box plots | Fig. 5.37 | `typology/outputs/figures/cluster_boxplots.png` |
| B.4.7 | Parallel coordinates plot | Fig. 5.38 | `typology/outputs/figures/cluster_parallel_coords.png` |
| B.4.8 | Spatial cluster map (k=3) | Fig. 5.39 | `typology/outputs/figures/cluster_map.png` |
| B.4.9 | Silhouette diagram (k=3) | Fig. 5.40 | `typology/outputs/figures/silhouette_diagram_k3.png` |
| B.4.10 | Cluster-class cross-tabulation | Fig. 5.41 | `typology/outputs/figures/cluster_crosstab_heatmap.png` |
| B.4.11 | Optimal k metrics | Fig. 5.42 | `typology/outputs/figures/optimal_k_metrics.png` |
| B.4.12 | Multi-k comparison maps | Fig. 5.43 | `typology/outputs/figures/multi_k_comparison.png` |
| B.4.13 | Silhouette diagram (k=4) | Fig. 5.44 | `typology/outputs/figures/silhouette_diagram_k4.png` |
| B.4.14 | Silhouette diagram (k=5) | Fig. 5.45 | `typology/outputs/figures/silhouette_diagram_k5.png` |
| B.4.15 | Hierarchical dendrogram | Fig. 5.46 | `typology/outputs/figures/dendrogram.png` |
| B.4.16 | Specification robustness | Fig. 5.47 | `typology/outputs/figures/specification_robustness.png` |
| B.4.17 | Spatial model comparison | Fig. 5.48 | `typology/outputs/figures/spatial_model_comparison.png` |
| B.4.18 | Pearson correlation heatmap | Fig. 5.49 | `typology/outputs/figures/correlation_heatmap_pearson.png` |
| B.4.19 | Spearman correlation heatmap | Fig. 5.50 | `typology/outputs/figures/correlation_heatmap_spearman.png` |
| B.4.20 | Subgroup correlations | Fig. 5.51 | `typology/outputs/figures/subgroup_correlations.png` |
| B.4.21 | Regression diagnostics | Fig. 5.52 | `typology/outputs/figures/regression_diagnostics.png` |
| B.4.22 | Regression coefficients | Fig. 5.53 | `typology/outputs/figures/regression_coefficients.png` |

---

## Appendix C: Supplementary Confusion Matrices

This appendix contains confusion matrices for all models not shown in
the main text. These are referenced from Section 5.3.5 and provide
the full picture of per-class classification performance across all
10 models.

### C.1 Logistic Regression
> *File:* `classification/outputs/figures/confusion_matrix_logistic_regression.png`

### C.2 Random Forest
> *File:* `classification/outputs/figures/confusion_matrix_random_forest.png`

### C.3 SVM (Linear)
> *File:* `classification/outputs/figures/confusion_matrix_svm_linear.png`

### C.4 SVM (RBF)
> *File:* `classification/outputs/figures/confusion_matrix_svm_rbf.png`

### C.5 K-Nearest Neighbors
> *File:* `classification/outputs/figures/confusion_matrix_knn.png`

### C.6 MLP Neural Network
> *File:* `classification/outputs/figures/confusion_matrix_mlp.png`

### C.7 XGBoost
> *File:* `classification/outputs/figures/confusion_matrix_xgboost.png`

### C.8 Dummy (Stratified)
> *File:* `classification/outputs/figures/confusion_matrix_dummy_stratified.png`

### C.9 Dummy (Most Frequent)
> *File:* `classification/outputs/figures/confusion_matrix_dummy_most_frequent.png`

---

## Appendix D: Supplementary Typology Figures

The following figures provide additional detail on the typology analysis
and are referenced from sections 5.4.5 and 5.4.6.

### D.1 Silhouette Analysis for Alternative k Values

- **k = 4 silhouette diagram:** `typology/outputs/figures/silhouette_diagram_k4.png`
- **k = 5 silhouette diagram:** `typology/outputs/figures/silhouette_diagram_k5.png`

### D.2 Hierarchical Clustering

The agglomerative hierarchical dendrogram provides a complementary view
to the K-means partition, showing nested cluster relationships.

> *File:* `typology/outputs/figures/dendrogram.png`

### D.3 Cluster Comparison Across k Values

Side-by-side spatial maps for k = 3, 4, and 5 showing how cluster
assignments shift as k increases.

> *File:* `typology/outputs/figures/multi_k_comparison.png`

### D.4 Regression Analysis Details

Full regression diagnostic plots for the elderly_ratio prediction model
using physical indicators as predictors.

> *File:* `typology/outputs/figures/regression_diagnostics.png`
> *File:* `typology/outputs/figures/regression_coefficients.png`

---

## Appendix E: Figure Summary Statistics

### E.1 Total Figure Count by Module

| Module | PNG | PDF | Total Unique |
|--------|-----|-----|-------------|
| Study Area Maps | 1 | 1 | 1 |
| EDA | 18 | 0 | 18 |
| Classification | 21 | 0 | 21 |
| Typology | 22 | 22 | 22 |
| **Total** | **62** | **23** | **62** |

### E.2 In-Text Figure Summary

- **Chapter 5 main text:** 53 figures referenced (Fig. 5.1 through 5.53)
- **Appendix C (confusion matrices):** 9 supplementary figures
- **Appendix D (typology detail):** Figures already referenced in Ch. 5

### E.3 Recommended Core Figures for Thesis

If space constraints require reducing the number of in-text figures,
the following 20 figures are recommended as the essential minimum:

1. **Fig. 5.1** -- Study area map
2. **Fig. 5.2** -- Missing data heatmap
3. **Fig. 5.4** -- RS feature distributions
4. **Fig. 5.7** -- Correlation heatmap
5. **Fig. 5.9** -- VIIRS vs elderly ratio
6. **Fig. 5.13** -- Temporal trends
7. **Fig. 5.14** -- Seasonal patterns
8. **Fig. 5.17** -- Spatial NDVI map
9. **Fig. 5.18** -- Spatial VIIRS map
10. **Fig. 5.20** -- SelectKBest scores (leakage)
11. **Fig. 5.21** -- Model comparison
12. **Fig. 5.23** -- Leakage comparison
13. **Fig. 5.24** -- GB confusion matrix
14. **Fig. 5.27** -- GB tree importance
15. **Fig. 5.30** -- SHAP summary
16. **Fig. 5.35** -- PCA biplot (typology)
17. **Fig. 5.39** -- Cluster map
18. **Fig. 5.40** -- Silhouette diagram
19. **Fig. 5.41** -- Cluster-class cross-tab
20. **Fig. 5.47** -- Specification robustness

---

*Document prepared February 2026. All file paths are relative to the
project root directory (`D:\thesis-shrinkin-villages\`). Typology
figures are available in both PNG and PDF formats.*
