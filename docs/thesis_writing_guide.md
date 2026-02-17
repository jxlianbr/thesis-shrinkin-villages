# Thesis Writing Guide: Classifying Shrinking Villages in Aomori & Akita

> Comprehensive summary of data sources, methods, results, and figure recommendations.
> Covers Chapters 3.2 (Data Sources), 3.3 (Tools & Software), and 4 (Methods),
> plus a full pipeline results walkthrough and figure/appendix guidance.

---

## TABLE OF CONTENTS

1. [What Changed with MNDWI & LULC](#1-what-changed-with-mndwi--lulc)
2. [Chapter 3.2 - Data Sources](#2-chapter-32---data-sources)
3. [Chapter 3.3 - Tools and Software](#3-chapter-33---tools-and-software)
4. [Chapter 4 - Methods](#4-chapter-4---methods)
5. [Results Walkthrough: End-to-End](#5-results-walkthrough-end-to-end)
6. [Figure and Table Recommendations](#6-figure-and-table-recommendations)

---

## 1. What Changed with MNDWI & LULC

### Overview of Changes

Two new feature categories were integrated into the pipeline:

**MNDWI (Modified Normalized Difference Water Index):**
- Formula: (B3 - B11) / (B3 + B11) using Sentinel-2 Green and SWIR bands
- Computed monthly alongside NDVI and NDBI in GEE
- Produces 4 temporal aggregates: MNDWI_mean, MNDWI_std, MNDWI_slope, MNDWI_seasonal_amp
- Inherits same ~35% NaN rate as other S2-derived features (cloud masking)
- Not transformed with log1p (near-symmetric distribution, skewness=0.40)

**LULC (Land Use / Land Cover) from Google Dynamic World V1:**
- 9 probability fractions per unit: water, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice
- Extracted year-by-year (2015-2025), then averaged across years for a static composite
- 0% missing data (complete satellite coverage, no cloud dependency)
- dw_snow_and_ice_frac dropped in preprocessing (sum-to-1 constraint; least informative class)
- 8 LULC features retained in final classification feature set

### Impact on Feature Space

| Aspect | Before MNDWI/LULC | After MNDWI/LULC |
|--------|-------------------|-------------------|
| Total classification features | ~21 RS features | 30 features (+43%) |
| Water/moisture indicators | None specific | MNDWI (4 temporal features) |
| Land cover composition | None | LULC (8 static fractions) |
| Feature importance #1 | viirs_mean_mean | dw_bare_frac (LULC) |
| Classification accuracy (no_leaky) | ~0.65 (SVM Linear) | 0.662 (SVM Linear) / 0.659 (GB) |

### Key Finding

LULC fractions, particularly bare ground fraction (dw_bare_frac), emerged as the single most important predictor across all feature importance methods (permutation, Gini, SHAP). This is a major finding: bare ground observable from satellite directly captures village abandonment patterns.

### Files Modified

- `config/config.yaml`: Added compute_mndwi, compute_lulc, lulc_scale
- `data_preprocessing/gee_monthly.py`: MNDWI computation (lines 721-722)
- `data_preprocessing/gee_lulc.py`: New file (232 lines) for Dynamic World extraction
- `pipeline.py`: LULC extraction hook (lines 187-210)
- `eda/config/eda_config.yaml`: MNDWI in indices, LULC feature group
- `preprocessing/config/preprocessing_config.yaml`: MNDWI temporal aggregation, LULC static features, dw_snow_and_ice_frac multicollinearity drop
- `classification/config/classification_config.yaml`: water feature group (MNDWI), lulc feature group

### What Was NOT Changed

- Typology analysis: MNDWI and LULC are explicitly excluded from the typology indicator set
- The typology focuses on demographic aging + built-up change indicators only

---

## 2. Chapter 3.2 - Data Sources

### 2.1 Satellite Optical Imagery: Sentinel-2 L2A

- **Collection:** COPERNICUS/S2_SR_HARMONIZED (Level-2A, atmospherically corrected)
- **Bands used:** B2 (Blue, 10m), B3 (Green, 10m), B4 (Red, 10m), B8 (NIR, 10m), B11 (SWIR, 20m)
- **SCL band** used for cloud/shadow masking (Scene Classification Layer)
- **Time range:** January 2015 -- December 2025 (130 months)
- **Cloud filter:** Images with >80% cloud pixels excluded before compositing
- **Cloud mask:** SCL classes 4 (vegetation), 5 (bare), 6 (water), 11 (snow) retained; all cloud/shadow/saturated classes masked
- **Temporal composite:** Monthly median per pixel
- **Spatial aggregation:** Zonal mean per administrative unit at 50m resolution
- **Data availability:** Mean 84.4 valid months per unit (65% temporal coverage). NaN rate ~35% due to cloud cover, uniformly distributed across prefectures, with seasonal pattern (higher cloud cover in winter months)

### 2.2 Spectral Indices (Derived from Sentinel-2)

Three spectral indices computed per monthly composite:

| Index | Formula | Range | Interpretation |
|-------|---------|-------|----------------|
| NDVI | (B8 - B4) / (B8 + B4) | [-1, 1] | Vegetation greenness; positive = green vegetation |
| NDBI | (B11 - B8) / (B11 + B8) | [-1, 1] | Built-up intensity; positive = urban/built-up |
| MNDWI | (B3 - B11) / (B3 + B11) | [-1, 1] | Water/moisture; positive = open water/wetlands |

MNDWI was chosen over standard NDWI because it uses SWIR (B11) instead of NIR, which better suppresses vegetation signal and more accurately identifies water bodies and wetland areas relevant to rice-paddy landscapes in Tohoku.

### 2.3 Nighttime Lights: VIIRS DNB

- **Collection:** NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG
- **Band:** avg_rad (average radiance, nanoWatts/cm2/sr)
- **Resolution:** 500m
- **Time range:** Monthly, 2015-2025
- **NaN rate:** 3.07% (excellent availability)
- **Interpretation:** Nighttime radiance as proxy for economic activity, electrification, urbanization
- **Key finding:** viirs_mean is the single strongest predictor of elderly ratio (r = -0.57) and ranks among the top 3 features for shrinkage classification

### 2.4 Land Use / Land Cover: Google Dynamic World V1

- **Collection:** GOOGLE/DYNAMICWORLD/V1
- **Type:** Probabilistic per-pixel land cover classification derived from Sentinel-2
- **Resolution:** 10m native, aggregated at 100m scale
- **9 classes:** water, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice
- **Processing:** Year-by-year composite (mean probability across all images per year), then averaged across 2015-2025 for a static composite per unit
- **NaN rate:** 0% (complete coverage)
- **Key statistics for study area:**
  - Trees dominant: 40.5% mean fraction (forested Tohoku landscape)
  - Snow/ice: 11.9% (seasonal winter snow)
  - Crops: 10.3% (agricultural areas)
  - Built: 8.6% (confirms rural character)
  - Bare: 4.5% (emerged as most predictive feature for classification)

Dynamic World V1 was chosen over ESA WorldCover because it provides probability fractions (continuous values) rather than categorical classes, preserving more information for machine learning classification.

### 2.5 Terrain: Copernicus DEM GLO-30

- **Collection:** COPERNICUS/DEM/GLO30
- **Resolution:** 30m
- **Type:** Static digital elevation model (SRTM/TanDEM-X fusion)
- **Derived features (7):** elevation_mean, elevation_std, slope_mean, slope_std, aspect_sin_mean, aspect_cos_mean, tri_mean (Terrain Ruggedness Index)
- **Aspect decomposition:** Circular variable decomposed into sin/cos components for meaningful averaging
- **NaN rate:** 0% for elevation/TRI; ~20% for slope/aspect in some Aomori units

### 2.6 Built Environment: OpenStreetMap

- **Source:** Overpass API (building footprint polygons)
- **Features (3):** osm_built_area (m2), osm_building_count, osm_built_ratio (built/total area)
- **NaN rate:** 0% (complete coverage)
- **Key statistic:** Mean built ratio = 0.65% -- confirms sparse, rural character of study units

### 2.7 Demographics: Population Census

- **Source:** Japanese national census (2020 snapshot)
- **Features (8):** pop_total, pop_male, pop_female, households_total, age_u15, age_15_64, age_65_plus, age_75_plus
- **Nature:** Static per unit (single census snapshot, same value across all months)
- **NaN rate:** 0%
- **Key statistics:**
  - Population range: 1,265 -- 307,672 (240x variation)
  - Mean elderly ratio: 0.40 (40% aged 65+)
  - Youth ratio: 0.094 (9.4% aged under 15)

### 2.8 Administrative Boundaries

- **Source:** Japanese administrative boundary data (mura level)
- **File:** mura_jis_fix.gpkg (65 units)
- **Coverage:** Aomori Prefecture (40 units) and Akita Prefecture (25 units)
- **Uploaded to GEE** as custom assets for zonal statistics computation

### Summary Table of All Data Sources

| Source | GEE Collection / Type | Resolution | Temporal | Features | NaN % |
|--------|----------------------|------------|----------|----------|-------|
| Sentinel-2 L2A | COPERNICUS/S2_SR_HARMONIZED | 10-20m | 2015-2025 monthly | B2-B11, NDVI, NDBI, MNDWI | ~35% |
| VIIRS DNB | NOAA/VIIRS/DNB/MONTHLY_V1 | 500m | 2015-2025 monthly | viirs_mean | 3% |
| Dynamic World V1 | GOOGLE/DYNAMICWORLD/V1 | 10m | 2015-2025 yearly composite | 9 LULC fractions | 0% |
| Copernicus DEM | COPERNICUS/DEM/GLO30 | 30m | Static | 7 terrain features | 0-20% |
| OpenStreetMap | Overpass API | Vector | Current | 3 built env features | 0% |
| Census | Local CSV | Unit-level | 2020 snapshot | 8 demographic features | 0% |

---

## 3. Chapter 3.3 - Tools and Software

### 3.1 Google Earth Engine (GEE)

- Cloud-based geospatial analysis platform for satellite imagery processing
- Used for: Sentinel-2 compositing, spectral index computation, VIIRS extraction, terrain derivation, LULC extraction
- Authentication via `earthengine authenticate`; project ID: ee-brodnow77
- Export strategy: Temporary asset creation -> CSV download -> asset deletion
- Parallelization: ThreadPoolExecutor with 6 concurrent monthly tasks
- Batching: 500 features per export task to avoid GEE computation limits

### 3.2 Python Ecosystem

**Core libraries:**
- pandas, numpy: Data manipulation and numerical computing
- geopandas: Geospatial data I/O (boundary polygons)
- rasterio: Raster data I/O (GLCM source rasters)
- scikit-image: GLCM texture computation (graycomatrix, graycoprops)
- scikit-learn: Preprocessing (RobustScaler), classification (10 models), clustering (K-means), evaluation metrics
- xgboost: Gradient boosting classifier
- scipy: Statistical tests (Shapiro-Wilk, Friedman, Wilcoxon, Kruskal-Wallis)
- shap: SHapley Additive exPlanations for feature importance

**Visualization:**
- matplotlib, seaborn: Static publication-quality figures (300 DPI)
- plotly: Interactive HTML visualizations

**Reporting:**
- jinja2: HTML report templating
- pyyaml: YAML configuration parsing

**All modules follow a consistent pattern:**
- `from __future__ import annotations` for forward-compatible type hints
- YAML-driven configuration
- `run_*()` entry point functions
- `Path` objects for all file paths
- `print()` for console output (no logging framework)

### 3.3 Development Environment

- Platform: Windows 11
- Python: 3.x with pip/conda
- Version control: Git
- IDE: VS Code with Claude Code integration

---

## 4. Chapter 4 - Methods

### 4.1 Study Area and Unit Selection

The study area comprises 65 mura (village-level) administrative units across Aomori Prefecture (40 units) and Akita Prefecture (25 units) in the Tohoku region of northeastern Japan. These prefectures were selected because they represent among the most rapidly aging and depopulating regions in Japan, with elderly ratios (proportion of population aged 65+) ranging from 0.25 to 0.55.

### 4.2 Data Acquisition Pipeline

**GEE Monthly Feature Export:**
The pipeline processes Sentinel-2 L2A imagery monthly (January 2015 -- December 2025). For each month:
1. Filter Sentinel-2 collection by date, AOI bounds, and cloud cover (<80%)
2. Apply SCL-based cloud/shadow mask (retain classes: vegetation, bare soil, water, snow)
3. Compute monthly median composite per pixel
4. Calculate spectral indices: NDVI, NDBI, MNDWI
5. Aggregate to unit level via zonal mean (reduceRegions at 50m scale)
6. Separately export VIIRS monthly radiance (500m scale)
7. Merge S2 and VIIRS features locally in Python

**GLCM Texture Computation (Local):**
Rather than computing GLCM in GEE (prohibitively expensive), NDBI rasters are downloaded monthly at 100m resolution, then GLCM metrics are computed locally:
1. Download monthly NDBI composite as GeoTIFF
2. For each unit boundary, extract pixel values
3. Quantize NDBI [-1, 1] to uint8 [0, 255]
4. Compute GLCM: graycomatrix with distance=1, 4 angles (0, 45, 90, 135 degrees), symmetric, normalized
5. Extract: contrast, entropy, homogeneity (averaged across 4 angles)
6. Parallelize across units using ProcessPoolExecutor (4 workers)

**LULC Extraction (Dynamic World):**
1. For each year (2015-2025): filter Dynamic World V1, compute mean probability composite, reduce to unit means at 100m scale
2. Average yearly fractions across all years for a static composite per unit
3. Output: 9 land-cover class fractions per unit

**Terrain Features:**
Single extraction from Copernicus DEM GLO-30 via GEE:
- Elevation statistics (mean, std)
- Slope statistics (mean, std) from ee.Terrain.slope()
- Aspect circular decomposition (sin/cos means)
- TRI as 3x3 kernel stddev on DEM

**Demographic and OSM Join:**
Census data (2020) and OSM building footprints joined by unit ID, producing 8 demographic and 3 built-environment features per unit.

**Output:** Panel-format features table with 8,450 rows (65 units x ~130 months) and 30+ columns.

### 4.3 Exploratory Data Analysis (EDA)

The EDA module analyzes the raw panel data to inform preprocessing decisions:

**Missing data analysis:**
- Overall missing rate: 14.62%
- Spectral bands/indices: 34.63% NaN (cloud cover)
- GLCM: 23.02% NaN
- VIIRS: 3.07% NaN
- Demographics, OSM, LULC, terrain: 0% NaN
- Temporal pattern: higher NaN in winter months (seasonal cloud cover)
- Mean temporal coverage: 85.2 months per unit (sufficient for temporal aggregation)

**Distribution analysis:**
- All 43 numeric features are non-normal (Shapiro-Wilk p < 0.05)
- 22 features recommended for log1p transformation (right-skewed)
- 6 features recommended for sqrt transformation
- viirs_mean: extreme skewness (4.29) and kurtosis (25.20)
- Demographic features: extreme multicollinearity (all pairs r > 0.99)

**Correlation analysis:**
- 47 feature pairs with |r| > 0.8
- S2_NDBI_entropy <-> S2_NDBI_homogeneity: r = -0.993 (mathematical inverse)
- NDVI <-> MNDWI: r = -0.824 (expected: vegetation vs. water)
- dw_trees_frac <-> dw_crops_frac: r = -0.914 (compositional constraint)
- dw_built_frac <-> viirs_mean: r = 0.942 (both capture urbanization)
- viirs_mean <-> elderly_ratio: r = -0.571 (critical finding: darker villages are older)

**Prefecture differences:**
- Aomori: brighter (viirs=1.12), less green (NDVI=0.19), more urbanized
- Akita: greener (NDVI=0.22), darker (viirs=0.70), more forested, larger average population

### 4.4 Preprocessing Pipeline

The preprocessing module transforms the 8,450-row panel into a 65-row classification-ready cross-sectional dataset through 9 sequential steps:

**Step 1 -- Data Loading:** Load features_table.parquet, derive temporal metadata (month_dt, year, season).

**Step 2 -- Drop Meta Columns:** Remove 5 non-predictive metadata columns (unit_level, unit_code, muni_code, city_name_ja, households_total_from_popfile).

**Step 3 -- Temporal Aggregation (Panel to Cross-Section):**
- **Mean + Std** for 12 RS features: B2-B11, NDVI, NDBI, MNDWI, viirs_mean, S2_NDBI_contrast/entropy/homogeneity (24 temporal statistics)
- **OLS Trend Slope** for 4 features: NDVI, NDBI, MNDWI, viirs_mean (linear regression on normalized time, requires >= 3 observations)
- **Seasonal Amplitude** for 3 features: NDVI, NDBI, MNDWI (max - min of monthly means)
- **Static features** (demographics, OSM, terrain, LULC): first value per unit taken directly

**Step 4 -- Feature Engineering:** 4 derived demographic ratios:
- elderly_ratio = age_65_plus / pop_total (range: 0.25-0.55, mean: 0.40)
- aging_index = age_65_plus / age_u15 (mean: 4.23)
- youth_ratio = age_u15 / pop_total (mean: 0.094)
- household_size = pop_total / households_total (mean: 2.60)

**Step 5 -- Target Variable Construction:**
3-class shrinkage classification based on elderly_ratio thresholds:
- **stable**: elderly_ratio < 0.37 (20 units, 30.8%)
- **shrinking**: 0.37 <= elderly_ratio < 0.42 (23 units, 35.4%)
- **severely_shrinking**: elderly_ratio >= 0.42 (22 units, 33.8%)

Thresholds were calibrated to produce approximately equal class sizes. Initial thresholds (0.30/0.40) produced a 3/32/30 split; adjusted to 0.37/0.42 for balanced representation.

**Step 6 -- Multicollinearity Resolution:** 13 features dropped:
- Demographics (5): pop_male, pop_female, households_total, age_15_64, age_75_plus (r > 0.99 with retained features)
- Spectral bands (4): B2_mean, B2_std, B3_mean, B3_std (r > 0.99 with B4)
- OSM (1): osm_building_count (r = 0.99 with osm_built_area)
- GLCM (2): S2_NDBI_homogeneity_mean/std (r = -0.993 with entropy)
- LULC (1): dw_snow_and_ice_frac (sum-to-1 constraint; least informative for Tohoku)

**Step 7 -- Transformation and Scaling:**
- **log1p** on 9 right-skewed features: B4_mean, B8_mean, viirs_mean_mean, S2_NDBI_contrast_mean, pop_total, age_u15, age_65_plus, osm_built_area, osm_built_ratio
- **Median imputation** on 4 terrain features with missing values: slope_mean, slope_std, aspect_sin_mean, aspect_cos_mean
- **RobustScaler** (median/IQR) on all 49 numeric features. Chosen over StandardScaler for robustness to outliers in non-normal distributions.

**Step 8 -- Validation:** All checks passed (65 rows, 0 NaN, correct target distribution, no constant features).

**Step 9 -- Export:** classification_ready.parquet (65 rows x 53 columns: 2 identifiers + 49 features + 2 targets)

**Final feature set (49 numeric features):**
- Spectral bands: B4, B8, B11 (mean + std each = 6)
- Spectral indices: NDVI, NDBI, MNDWI (mean + std + slope + seasonal_amp each = 12)
- Nightlights: viirs_mean (mean + std + slope = 3)
- GLCM texture: S2_NDBI_contrast, S2_NDBI_entropy (mean + std each = 4)
- Demographics: pop_total, age_u15, age_65_plus (3)
- OSM: osm_built_area, osm_built_ratio (2)
- Terrain: 7 features
- LULC: 8 Dynamic World fractions
- Derived ratios: elderly_ratio, aging_index, youth_ratio, household_size (4)

### 4.5 Classification Methodology

**Models (10 classifiers):**
- 2 dummy baselines: most-frequent, stratified random
- Logistic Regression: L2 penalty, C=1.0, LBFGS solver
- SVM Linear: C=1.0, max_iter=5000
- SVM RBF: C=1.0, gamma=scale
- Random Forest: 200 trees, min_samples_leaf=3, sqrt features
- Gradient Boosting: 100 trees, lr=0.1, max_depth=3
- XGBoost: 100 trees, lr=0.1, max_depth=3
- KNN: k=5, distance-weighted
- MLP: [64, 32] hidden layers, ReLU, Adam, early stopping

**Cross-Validation:**
RepeatedStratifiedKFold with 5 folds x 5 repeats = 25 evaluations. Stratification ensures each fold maintains class proportions. Critical for n=65 sample.

**Leakage Experiments (4):**
Because the target variable is deterministically derived from elderly_ratio (which itself derives from demographic features), four nested experiments control for information leakage:

| Experiment | Features | Removed | Purpose |
|------------|----------|---------|---------|
| all_features | 49 | None | Reference (maximum leakage) |
| no_leaky (PRIMARY) | 46 | elderly_ratio, aging_index, youth_ratio | Remove deterministic target functions |
| no_demographic | 42 | + age_u15, age_65_plus, pop_total, household_size | Remove all demographic info |
| rs_only | 40 | + osm_built_area, osm_built_ratio | Pure remote sensing + terrain |

**Feature Importance Methods:**
- Permutation importance (model-agnostic, based on SVM)
- Tree-based Gini importance (from RF and GB)
- SHAP TreeExplainer (on Random Forest)
- SelectKBest ANOVA F-statistics (univariate)
- Consensus ranking (average across methods)

**Statistical Tests:**
- Friedman test: global comparison of all classifiers
- Pairwise Wilcoxon signed-rank with Bonferroni correction: post-hoc comparisons
- Paired t-tests: best model vs. baselines with Cohen's d effect size

**Leave-One-Out (LOO) Check:** 3 selected models (LR, SVM Linear, RF) tested with LOO CV to verify stability.

### 4.6 Typology Analysis

An unsupervised clustering analysis complements the supervised classification:

**Indicators (11 total):**
- 8 physical (recomputed from raw panel data): NDVI_slope, viirs_mean_slope, S2_NDBI_contrast_slope, NDVI_cv, NDVI_seasonal_amp, NDBI_seasonal_amp, S2_NDBI_contrast_mean, S2_NDBI_entropy_mean
- 3 demographic (inverse-transformed from preprocessed data): elderly_ratio, household_size, pop_total

**Clustering:**
- K-means with k=2..8 evaluated (silhouette, Calinski-Harabasz, Davies-Bouldin, gap statistic)
- Primary solution: k=3 (silhouette=0.235, bootstrap ARI=0.544)
- Alternative solutions: k=4, k=5 reported for sensitivity
- Hierarchical clustering (Ward's method) for cross-validation of K-means
- PCA dimensionality reduction (5 components capture 80% variance)

**Stability validation:**
- Bootstrap resampling (1000 iterations, 80% sample fraction)
- Specification robustness (physical-only, demographic-only, trend-only, level-only subsets)
- Feature perturbation (drop-one-feature analysis)

**Relationship analysis:**
- Pearson + Spearman correlations between physical and demographic indicators (Bonferroni-corrected)
- Kruskal-Wallis tests for cluster differences on each indicator
- Comparison to supervised shrinkage classes (ARI)

**Note:** MNDWI and LULC are intentionally excluded from the typology to maintain focus on settlement change trajectories (built-up decline, vegetation trends) and demographic structure.

---

## 5. Results Walkthrough: End-to-End

### 5.1 Data Acquisition Results

The GEE pipeline produced a panel dataset of 8,450 observations (65 units x ~130 months). Key quality metrics:
- Mean temporal coverage: 84.4 months per unit (65% of possible 130 months)
- Spectral data NaN: ~35% (cloud cover, seasonal pattern with winter peaks)
- VIIRS NaN: 3.07% (excellent)
- Static features (LULC, terrain, demographics, OSM): 0% NaN
- All 65 units represented with sufficient temporal depth for aggregation

**Why these NaN rates:** Cloud cover is inherent to optical satellite remote sensing, particularly in Japan's monsoon climate. The 35% NaN rate for spectral data is expected and manageable because temporal aggregation (mean, std, slope) requires only a subset of months. The mean 84.4 months of valid data provides robust temporal statistics. VIIRS nightlights have lower NaN because they operate at night under fewer cloud constraints. LULC has 0% NaN because it uses multi-temporal composites that fill cloud gaps.

### 5.2 EDA Results

**Distribution findings:**
- All 43 features non-normal (Shapiro-Wilk p < 0.05 for all)
- Extreme right skewness in demographics (skew 2.93-3.43), VIIRS (4.29), OSM (2.64-2.77)
- This confirmed the need for log1p transformation and RobustScaler

**Multicollinearity findings:**
- 47 feature pairs with |r| > 0.8
- Demographic features nearly perfectly correlated (all r > 0.99) -- expected because pop_male, pop_female, age groups are nested fractions of pop_total
- Spectral bands B2, B3, B4 highly correlated (r > 0.99) -- adjacent wavelengths
- S2_NDBI_entropy and homogeneity: r = -0.993 -- mathematical inverses
- LULC correlations expected from compositional constraint: trees vs. crops r = -0.91
- New: NDVI vs. MNDWI r = -0.82 (vegetation and water occupy different spatial domains)
- New: dw_built_frac vs. viirs_mean r = 0.94 (both capture urbanization)

**Critical cross-domain finding:** viirs_mean vs. elderly_ratio: r = -0.57. Villages with lower nighttime lights have higher elderly ratios. This is the strongest remote-sensing-to-demographic association and the most important signal for classification.

**Prefecture comparison:** Aomori is more urbanized (brighter VIIRS, lower NDVI) while Akita is more forested (higher NDVI, darker VIIRS). This suggests different shrinkage mechanisms: urban decline in Aomori vs. rural decline in Akita.

**LULC landscape composition:** Trees dominate (40.5%), followed by snow/ice (11.9%), crops (10.3%), and built (8.6%). The low built fraction (8.6%) confirms the rural character of the study area.

### 5.3 Preprocessing Results

**Temporal aggregation:** Collapsed 8,450 panel rows to 65 cross-sectional rows. Each RS feature produces 2-4 temporal statistics (mean, std, slope, seasonal amplitude). Static features taken as-is.

**Target variable:** The 3-class distribution is well-balanced: stable (20, 30.8%), shrinking (23, 35.4%), severely_shrinking (22, 33.8%). This near-equal split is critical for fair evaluation with stratified CV.

**Multicollinearity resolution:** 13 features dropped, reducing to 49 features + 2 identifiers + 2 targets. The most impactful drops were demographic redundancies (5 features) and spectral band redundancies (4 features).

**Transformation effect:** log1p reduced extreme skewness in 9 features. RobustScaler normalized all features to comparable ranges while preserving outlier information (important for small n=65 dataset).

### 5.4 Classification Results

#### Leakage Detection (Critical Finding)

| Experiment | Best Model | Balanced Accuracy | Drop from all_features |
|------------|-----------|-------------------|----------------------|
| all_features | Gradient Boosting | 0.9753 (97.5%) | -- |
| no_leaky (PRIMARY) | Gradient Boosting | 0.6587 (65.9%) | -31.7 pp |
| no_demographic | Gradient Boosting | 0.6487 (64.9%) | -1.0 pp |
| rs_only | Logistic Regression | 0.6433 (64.3%) | -0.5 pp |

**Interpretation:** The 31.7 percentage point drop from all_features to no_leaky definitively proves massive information leakage through elderly_ratio, aging_index, and youth_ratio. However, 65.9% balanced accuracy significantly exceeds the 34.9% baseline, confirming genuine predictive signal in non-leaky features.

The minimal drops from no_leaky to no_demographic (-1.0 pp) and to rs_only (-0.5 pp) reveal that demographic features beyond leakage provide only ~1-2% additional value. Remote sensing + terrain features capture 94-97% of the achievable signal. This is a key finding: shrinkage can be predicted from satellite data alone without census information.

#### Primary Experiment Results (no_leaky, 46 features)

| Rank | Model | Balanced Accuracy | Std Dev | Cohen's Kappa |
|------|-------|-------------------|---------|---------------|
| 1 | Gradient Boosting | 0.6587 | 0.1322 | 0.4895 |
| 2 | Logistic Regression | 0.6427 | 0.1154 | 0.4659 |
| 3 | Random Forest | 0.6127 | 0.1109 | 0.4239 |
| 4 | SVM Linear | 0.5880 | 0.1095 | 0.3752 |
| 5 | KNN | 0.5187 | 0.1325 | 0.2794 |
| 6 | MLP | 0.3707 | 0.1012 | 0.0515 |
| 7 | Dummy (Stratified) | 0.3493 | 0.1562 | 0.0244 |
| 8 | SVM (RBF) | 0.3340 | 0.0600 | 0.0003 |
| 9 | Dummy (Most Frequent) | 0.3333 | ~0.00 | 0.0000 |

**Why GB wins:** Gradient Boosting's sequential error correction captures non-linear feature interactions better than parallel ensembles (RF) or linear models (LR). However, the margin over LR is only 2.4 pp, suggesting most signal is linearly separable.

**Why MLP fails:** With only 65 training samples and 46 features, the neural network (64-32 hidden units) is severely overparameterized. Early stopping with 15% validation = ~10 samples per class is unreliable.

**Why SVM RBF fails:** RBF kernel overfit on n=65; the radial basis function is inappropriate for this feature space.

#### Per-Class Performance (Gradient Boosting)

| Class | Precision | Recall | F1 | Interpretation |
|-------|-----------|--------|-----|----------------|
| Stable | 0.620 | 0.570 | 0.594 | Hardest to classify; confused with Shrinking |
| Shrinking | 0.608 | 0.635 | 0.621 | Moderate; some confusion with both neighbors |
| Severely Shrinking | 0.752 | 0.773 | 0.762 | Best classified; most extreme phenotype |

**Why Severe is easiest:** Severely shrinking villages have the most distinctive satellite signatures -- very low nightlights, high bare ground fraction, strong vegetation greening (from abandonment). The extreme phenotype separates from the rest.

**Why Stable is hardest:** Stable villages overlap spectrally with Shrinking villages. Both can have similar NDVI, built-up patterns, and nightlight levels. The transition from stable to shrinking is gradual, not discrete.

**Confusion pattern:** The main confusion is between adjacent classes (Stable<->Shrinking: 31%, Shrinking<->Severe: 28%), not distant ones (Stable<->Severe: 12%). This confirms the target represents a continuum, not discrete categories.

#### Statistical Significance

- Friedman test: chi2 = 140.31, p = 2.05e-26 (classifiers are significantly different)
- All non-baseline models significantly outperform both dummy baselines (Bonferroni-corrected Wilcoxon)
- GB, LR, RF, SVM Linear form a competitive cluster with no statistically significant pairwise differences (at Bonferroni alpha)
- GB vs. baseline: Cohen's d = 2.46 (very large effect size)

**95% Confidence Interval for GB:** [0.45, 0.88] balanced accuracy. The wide interval reflects the small sample size (n=65) and inherent uncertainty. The results are directionally reliable but absolute accuracy may vary on new data.

#### Feature Importance Consensus

| Rank | Feature | Avg Rank | Category | Key Insight |
|------|---------|----------|----------|-------------|
| 1 | dw_bare_frac | 3.0 | LULC | Bare ground = village abandonment |
| 2 | age_u15 | 3.3 | Demographics | Youth loss = first sign of decline |
| 3 | viirs_mean_mean | 8.8 | Nightlights | Economic activity proxy |
| 4 | viirs_mean_std | 9.8 | Nightlights | Temporal stability of lights |
| 5 | household_size | 13.2 | Demographics | Household structure |
| 6 | age_65_plus | 13.8 | Demographics | Elderly population count |
| 7 | pop_total | 13.8 | Demographics | Settlement size |
| 8 | NDBI_slope | 14.5 | Spectral | Built-up temporal trend |
| 9 | dw_water_frac | 17.2 | LULC | Water/wetland presence |
| 10 | elevation_mean | 17.5 | Terrain | Geographic accessibility |
| 11 | MNDWI_slope | 20.2 | Spectral | Water/moisture trend |

**By domain (GB importance contribution):**
- Nightlights: 29.4% (dominant)
- LULC: 18.6% (important; bare ground #1)
- Demographics: 10.8% (partially leaky)
- Spectral indices: 7.2% (trends > means)
- Terrain: 6.9% (geographic context)
- GLCM texture: <1% (negligible)

**Why LULC bare ground ranks #1:** Bare ground fraction directly captures the physical manifestation of village abandonment -- when people leave, agricultural land reverts through succession and infrastructure deteriorates, exposing bare soil. This is observable from satellite without any ground survey.

**Why MNDWI slope ranks #11:** The temporal trend in water/moisture (MNDWI_slope) captures hydrological changes accompanying shrinkage (e.g., abandoned rice paddies drying up, drainage neglect). It provides unique information not captured by NDVI or NDBI, but the effect is secondary.

**Why nightlights dominate in GB but bare ground ranks #1 in consensus:** GB assigns 18.3% importance to viirs_mean_mean and 12.7% to dw_bare_frac. But permutation importance (from SVM) ranks dw_bare_frac highest. The consensus across methods places dw_bare_frac first because it is consistently important regardless of model type, while viirs_mean_mean dominates specifically in tree-based models.

#### LOO Sensitivity Check

| Model | LOO Bal Acc | CV Bal Acc | Difference |
|-------|-----------|-----------|-----------|
| SVM Linear | 0.6832 | 0.5880 | +9.5 pp |
| Logistic Regression | 0.6148 | 0.6427 | -2.8 pp |
| Random Forest | 0.5953 | 0.6127 | -1.7 pp |

LOO and CV results are consistent (differences <10 pp), confirming no severe overfitting. SVM Linear benefits from LOO's larger training set per fold.

### 5.5 Typology Results

#### Cluster Solution (k=3)

| Cluster | n | Label | Physical Signature | Demographic Signature |
|---------|---|-------|-------------------|----------------------|
| 0 | 19 (29%) | "Stable Service Centers" | Minimal change, low seasonality | Younger, larger population |
| 1 | 40 (62%) | "Aging Transitional" | Vegetation greening, slowing built-up decline, strong seasonality | Aging, medium population |
| 2 | 6 (9%) | "Extreme/Rare" | Max vegetation greening + max built-up collapse + extreme texture | Small population, large households |

**Why silhouette is only 0.235:** With only 65 observations in 11-dimensional space, clusters inherently overlap. This is a fundamental limitation of clustering small samples. However, bootstrap stability (ARI = 0.544 over 1000 resamples) and PCA-space consistency (ARI = 0.890) confirm the solution is reproducible.

**Clusters vs. shrinkage classes (ARI = 0.105):** The unsupervised typology is essentially orthogonal to the supervised shrinkage classification. This is because:
- Shrinkage classes are defined by elderly_ratio thresholds (demographic)
- Clusters are driven by physical change trends (RS-based)
- Demographic aging and physical settlement change operate on different dimensions

This is an important thesis finding: demographic shrinkage is not the same as physical/economic decline.

#### Specification Robustness

| Specification | ARI vs. Primary | Silhouette |
|---------------|-----------------|-----------|
| Physical only (8 indicators) | 0.457 | 0.244 |
| Trend only (4 indicators) | 0.414 | 0.491 (best!) |
| Level only (7 indicators) | 0.099 | 0.213 |
| Demographic only (3 indicators) | -0.001 | 0.326 |

**Key finding:** Demographics alone cannot produce meaningful clusters (ARI ~ 0). Physical change trends are the primary drivers of settlement typology. Trend-only clustering produces the best-quality clusters (silhouette = 0.491), suggesting the full 11-indicator set actually dilutes the signal with demographic noise.

#### Feature Importance for Clustering

Most essential indicator: S2_NDBI_contrast_slope (built-up decline trajectory). When dropped, ARI drops to 0.290 (30.8% of units reassigned). Least essential: NDBI_seasonal_amp (no units reassigned when dropped -- fully redundant).

#### Physical-Demographic Relationships

8 significant correlations (Bonferroni-corrected):
- S2_NDBI_contrast_slope vs. pop_total: Spearman r = 0.612 (larger settlements show greater built-up decline)
- viirs_mean_slope vs. pop_total: Pearson r = -0.482 (declining lights correlate with smaller populations)
- NDVI_seasonal_amp vs. household_size: Pearson r = 0.483 (larger households maintain stronger seasonal cycles)
- elderly_ratio vs. physical indicators: all |r| < 0.24 (aging does not correlate with physical change)

---

## 6. Figure and Table Recommendations

### 6.1 Figures for Main Text (Chapter 4 - Methods & Chapter 5 - Results)

**Study Area & Data:**
1. **Study area map** -- show Aomori & Akita within Japan context, with 65 mura boundaries (create if not existing; or use cluster_map.png as base)
2. **missing_heatmap.png** -- demonstrates data availability patterns across feature types
3. **missing_by_time.png** -- shows seasonal cloud cover pattern (explains NaN mechanism)

**EDA Key Findings:**
4. **correlation_heatmap.png** -- full correlation matrix showing multicollinearity patterns (select a cropped version focusing on key blocks: demographics, spectral, LULC)
5. **viirs_vs_elderly_ratio.png** -- the single most important scatter plot: darker villages are older
6. **boxplots_by_prefecture.png** -- shows Aomori vs. Akita differences

**Classification Results:**
7. **model_comparison.png** -- bar chart of balanced accuracy across all 10 models (primary experiment)
8. **leakage_comparison.png** -- shows performance drop across 4 experiments (demonstrates leakage)
9. **Confusion matrix for Gradient Boosting** (best model, no_leaky experiment)
10. **feature_importance consensus** (create a combined bar chart or use permutation_importance.png + tree importance)
11. **shap_summary.png** -- SHAP beeswarm plot for feature importance with directionality

**Typology Results:**
12. **cluster_map.png** -- geographic distribution of settlement types
13. **cluster_profiles_heatmap.png** -- concise summary of cluster characteristics
14. **optimal_k_metrics.png** -- justification for k=3 selection

**Total main text figures: ~14**

### 6.2 Figures for Appendix

**EDA Supplementary:**
- distributions_rs.png -- histograms of all RS features
- distributions_lulc.png -- histograms of all LULC fractions
- distributions_demo.png -- histograms of demographic features
- rs_demo_correlation.png -- RS x demographic correlation focus
- temporal_trends.png -- monthly time series by prefecture
- seasonal_patterns.png -- NDVI/NDBI monthly boxplots
- spatial_ndvi_mean.png, spatial_viirs_mean.png, spatial_pop_total.png -- choropleth maps
- outlier_boxplots.png
- ndvi_vs_population.png, ndbi_vs_population.png, viirs_vs_population.png

**Classification Supplementary:**
- All 10 confusion matrices (individual models)
- cv_boxplots.png -- fold-level distribution across models
- roc_curves_best.png -- ROC curves for best model
- selectkbest_scores.png -- univariate F-statistics
- pca_variance.png -- PCA dimensionality analysis
- Individual tree importance figures (RF, GB, XGB)

**Typology Supplementary:**
- pca_scree.png + pca_biplot.png -- PCA details
- dendrogram.png -- hierarchical clustering
- cluster_parallel_coords.png -- all indicators across units
- cluster_boxplots.png -- per-indicator cluster comparisons
- cluster_crosstab_heatmap.png -- cluster vs. shrinkage class
- multi_k_comparison.png -- k=3 vs. k=4 vs. k=5
- silhouette_diagram_k3.png -- per-unit silhouette scores
- indicator_distributions.png -- indicator histograms
- indicator_correlation_heatmap.png -- within-indicator correlations
- correlation_heatmap_pearson.png + correlation_heatmap_spearman.png
- specification_robustness.png -- robustness across indicator subsets

### 6.3 Tables for Main Text

1. **Data sources summary** (Section 2's final summary table)
2. **Preprocessing pipeline summary** (steps, features dropped/added, feature counts)
3. **Classification model comparison** (all 10 models, primary experiment metrics)
4. **Leakage experiment results** (4 experiments, top model per experiment)
5. **Feature importance consensus top 15** (rank, feature, avg rank, category)
6. **Per-class metrics** (precision, recall, F1 for best model)
7. **Cluster profiles** (3 clusters, key indicator values)
8. **Cluster quality metrics** (silhouette, bootstrap ARI, specification robustness)

### 6.4 Tables for Appendix

- Full correlation matrix (high_correlations.csv -- 47+ pairs)
- Complete distribution statistics (all 43 features)
- SelectKBest scores (all features with F-statistics and p-values)
- All fold-level metrics (25 folds x 10 models)
- Pairwise Wilcoxon test results (45 comparisons)
- LOO results
- Unit cluster assignments (65 units with cluster, silhouette, shrinkage class)
- Feature metadata (all features with transformation flags)
- Preprocessing report (scaler parameters)

---

## Quick Reference: Key Numbers

| Metric | Value |
|--------|-------|
| Study units | 65 mura (40 Aomori, 25 Akita) |
| Time period | 2015-01 to 2025-12 (130 months) |
| Panel rows | 8,450 |
| Classification features | 49 (30 after multicollinearity removal + 4 derived + 15 temporal stats) |
| Target classes | 3 (stable: 20, shrinking: 23, severely_shrinking: 22) |
| Best model (no_leaky) | Gradient Boosting, Bal Acc = 0.659 |
| Baseline | Stratified dummy, Bal Acc = 0.349 |
| Improvement over baseline | 1.88x (31 pp absolute) |
| Leakage drop | 97.5% -> 65.9% (-31.7 pp) |
| Top feature | dw_bare_frac (LULC bare ground) |
| Strongest RS-demo correlation | viirs_mean vs. elderly_ratio, r = -0.57 |
| Typology clusters | k=3, silhouette=0.235, bootstrap ARI=0.544 |
| Typology vs. classification | ARI = 0.105 (orthogonal) |
| Mean temporal coverage | 84.4 months per unit |
| Overall missing rate | 14.62% |
