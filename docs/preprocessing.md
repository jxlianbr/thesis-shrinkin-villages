# Preprocessing Module Documentation

## Overview

The preprocessing module transforms the panel-format feature table (4,680 rows: 65 units x 72 months) into a classification-ready cross-sectional dataset (65 rows). It performs temporal aggregation, feature engineering, target variable creation, multicollinearity resolution, and feature transformation, producing a clean dataset ready for supervised classification.

```
python preprocessing/run_preprocessing.py
python preprocessing/run_preprocessing.py --config path/to/config.yaml
```

---

## Directory Structure

```
preprocessing/
├── config/
│   └── preprocessing_config.yaml   # Central configuration
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Config loading, output helpers
│   ├── data_loader.py              # Parquet loading + temporal columns
│   ├── feature_dropper.py          # Drop meta-only and unusable columns
│   ├── temporal_aggregation.py     # Panel -> cross-section (mean/std/slope/seasonal)
│   ├── feature_engineering.py      # Derived ratios (elderly_ratio, aging_index, etc.)
│   ├── target_builder.py           # 3-class shrinkage label from elderly_ratio
│   ├── multicollinearity.py        # Drop correlated features per EDA findings
│   ├── transformer.py              # log1p transforms + RobustScaler
│   └── validation.py               # Final dataset checks
├── outputs/
│   ├── classification_ready.parquet  # Final output (Parquet)
│   ├── classification_ready.csv      # Final output (CSV)
│   ├── preprocessing_report.json     # Step-by-step processing report
│   └── feature_metadata.csv          # Per-column metadata (role, dtype, transforms)
└── run_preprocessing.py              # CLI entry point
```

---

## Pipeline Steps

The pipeline executes 9 sequential steps, each handled by a dedicated module:

```
Step 1: Load         Read Parquet, add month_dt/month_num/season
Step 2: Drop         Remove meta-only columns
Step 3: Aggregate    Collapse 72 months -> 1 row per unit (temporal stats)
Step 4: Engineer     Derive elderly_ratio, aging_index, youth_ratio, household_size
Step 5: Target       Create 3-class shrinkage label from elderly_ratio thresholds
Step 6: Multicoll    Drop highly correlated features per EDA findings
Step 7: Transform    log1p on skewed features + RobustScaler on all numeric
Step 8: Validate     Check shape, NaN, class balance, constant features
Step 9: Export       Save Parquet, CSV, JSON report, feature metadata
```

---

## Configuration

All pipeline parameters are controlled via `preprocessing/config/preprocessing_config.yaml`.

### `data`

| Key | Value | Description |
|-----|-------|-------------|
| `features_table` | `outputs/final/features_table.parquet` | Path to input panel-format Parquet |

### `output`

| Key | Value |
|-----|-------|
| `output_dir` | `preprocessing/outputs` |
| `parquet_path` | `preprocessing/outputs/classification_ready.parquet` |
| `csv_path` | `preprocessing/outputs/classification_ready.csv` |
| `report_path` | `preprocessing/outputs/preprocessing_report.json` |
| `metadata_path` | `preprocessing/outputs/feature_metadata.csv` |

### `drop`

Columns to remove before processing. Each key maps to a list of column names:

| Group | Columns | Rationale |
|-------|---------|-----------|
| `meta` | `unit_level`, `unit_code`, `muni_code`, `city_name_ja`, `households_total_from_popfile` | Administrative metadata with no predictive value |

### `temporal_aggregation`

Controls how panel data is collapsed to cross-section. Each feature list is processed differently:

| Key | Columns | Aggregation |
|-----|---------|-------------|
| `mean_std_features` | B2, B3, B4, B8, B11, NDVI, NDBI, viirs_mean, S2_NDBI_contrast, S2_NDBI_entropy, S2_NDBI_homogeneity | Mean + standard deviation over available (non-NaN) months |
| `trend_features` | NDVI, NDBI, viirs_mean | Additionally: OLS linear slope over normalised time [0,1] |
| `seasonal_features` | NDVI, NDBI | Additionally: seasonal amplitude (max - min of monthly means across months 1-12) |

### `demographic_features` and `osm_features`

Static features (identical across all 72 months per unit). The first value per unit is taken during aggregation.

- **Demographics:** pop_total, pop_male, pop_female, households_total, age_u15, age_15_64, age_65_plus, age_75_plus
- **OSM:** osm_built_area, osm_building_count, osm_built_ratio

### `derived_features`

Computed after temporal aggregation from static demographic columns:

| Feature | Formula | Notes |
|---------|---------|-------|
| `elderly_ratio` | `age_65_plus / pop_total` | Used both as feature AND to create target variable |
| `aging_index` | `age_65_plus / age_u15` | Division by zero prevented by clipping age_u15 to min 1 |
| `youth_ratio` | `age_u15 / pop_total` | Young population share |
| `household_size` | `pop_total / households_total` | Average persons per household |

### `target`

| Key | Value | Description |
|-----|-------|-------------|
| `name` | `shrinkage_class` | Target column name (string labels) |
| `method` | `elderly_ratio_thresholds` | Classification method |
| `thresholds.stable` | `0.37` | elderly_ratio < 0.37 -> "stable" |
| `thresholds.shrinking` | `0.42` | 0.37 <= elderly_ratio < 0.42 -> "shrinking" |
| `labels` | `["stable", "shrinking", "severely_shrinking"]` | Class labels (elderly_ratio >= 0.42 -> "severely_shrinking") |

These thresholds were calibrated to produce an approximately balanced 3-class split for the 65 units (~33rd and ~67th percentiles of the elderly ratio distribution).

### `multicollinearity`

Pre-defined features to drop, based on EDA `high_correlations.csv` (46 pairs with |r| > 0.8):

| Config Key | Dropped | Kept | Rationale |
|------------|---------|------|-----------|
| `drop_correlated` | pop_male, pop_female, households_total, age_15_64, age_75_plus | pop_total, age_65_plus, age_u15 | Demographics r > 0.99; keep pop_total (overall size), age_65_plus (target), age_u15 (aging_index) |
| `drop_spectral_correlated` | B2, B3 | B4, B8, B11 | B2/B3/B4 r > 0.99; keep B4 (red). B8 (NIR) distinct enough at r ~ 0.9 |
| `drop_osm_correlated` | osm_building_count | osm_built_area, osm_built_ratio | osm_built_area/building_count r = 0.99 |
| `drop_texture_correlated` | S2_NDBI_homogeneity | S2_NDBI_contrast, S2_NDBI_entropy | entropy/homogeneity r = -0.99; contrast is independent |

For temporal-aggregated features (`drop_spectral_correlated`, `drop_texture_correlated`), both `_mean` and `_std` variants are dropped.

### `transform`

| Key | Value | Description |
|-----|-------|-------------|
| `log1p_features` | B4, B8, viirs_mean, S2_NDBI_contrast, pop_total, age_u15, age_65_plus, osm_built_area, osm_built_ratio | Features with high skewness (|skew| > 1) to transform with `np.log1p` |
| `scaler` | `robust` | `"robust"` (RobustScaler, median/IQR) or `"standard"` (StandardScaler, mean/std) |

For temporal-aggregated features, log1p is applied to the `_mean` variant only (not `_std`).

### `identifiers`

Columns preserved in output but not treated as features or transformed: `unit_id`, `pref_name`.

---

## Module Details

### Step 1: Load (`data_loader.py`)

**`load_features_table(cfg) -> pd.DataFrame`**

Reads the Parquet file and adds derived temporal columns (`month_dt`, `year`, `month_num`, `season`). Uses the same `SEASON_MAP` as the EDA module.

**Input:** 4,680 rows x 30 columns
**Output:** 4,680 rows x 34 columns (+ 4 temporal columns)

---

### Step 2: Drop (`feature_dropper.py`)

**`drop_unusable_features(df, cfg) -> pd.DataFrame`**

Iterates over all groups in `cfg["drop"]` and removes matching columns from the DataFrame. The loop is dynamic -- it processes all keys in the `drop` config section, not just hardcoded group names.

**Input:** 4,680 rows x 30 columns (temporal columns are derived, not in original)
**Output:** 4,680 rows x 25 columns (5 meta columns dropped)

---

### Step 3: Temporal Aggregation (`temporal_aggregation.py`)

**`aggregate_to_cross_section(df, cfg) -> pd.DataFrame`**

This is the core transformation step. Collapses the panel data into one row per unit by computing different aggregations for different feature types:

**For `mean_std_features` (11 RS features):**
```
B4       -> B4_mean, B4_std
NDVI     -> NDVI_mean, NDVI_std
...
S2_NDBI_contrast -> S2_NDBI_contrast_mean, S2_NDBI_contrast_std
```
Mean and std are computed over available (non-NaN) months per unit. If only one valid month exists, std defaults to 0.0.

**For `trend_features` (NDVI, NDBI, viirs_mean):**
```
NDVI -> NDVI_slope
```
OLS linear slope is computed by:
1. Sorting observations by `month_dt`
2. Normalising time to [0, 1] over the unit's observation span
3. Fitting `np.polyfit(t, y, 1)` on valid (non-NaN) observations
4. The slope represents total feature change over the study period

Requires at least 3 valid observations; otherwise NaN.

**For `seasonal_features` (NDVI, NDBI):**
```
NDVI -> NDVI_seasonal_amp
```
Computed as max(monthly_mean) - min(monthly_mean) across months 1-12, capturing the annual seasonal swing.

**For demographic and OSM features:**
First value per unit is taken (these are static -- identical across all months).

**Identifier `pref_name`:** First value per unit is carried through.

**Input:** 4,680 rows
**Output:** 65 rows x 40 columns

---

### Step 4: Feature Engineering (`feature_engineering.py`)

**`engineer_features(df, cfg) -> pd.DataFrame`**

Computes four derived demographic ratios from the cross-sectional data:

| Feature | Formula | Safety |
|---------|---------|--------|
| `elderly_ratio` | `age_65_plus / pop_total` | `pop_total` clipped to min 1 |
| `aging_index` | `age_65_plus / age_u15` | `age_u15` clipped to min 1 |
| `youth_ratio` | `age_u15 / pop_total` | `pop_total` clipped to min 1 |
| `household_size` | `pop_total / households_total` | `households_total` clipped to min 1 |

Each derived feature is printed with its mean and range for verification.

**Output:** 65 rows x 44 columns (+ 4 derived)

---

### Step 5: Target Builder (`target_builder.py`)

**`build_target(df, cfg) -> pd.DataFrame`**

Creates the classification target from `elderly_ratio` using configurable thresholds:

| Class | Code | Condition | Current Count |
|-------|------|-----------|---------------|
| `stable` | 0 | `elderly_ratio < 0.37` | 20 units (30.8%) |
| `shrinking` | 1 | `0.37 <= elderly_ratio < 0.42` | 23 units (35.4%) |
| `severely_shrinking` | 2 | `elderly_ratio >= 0.42` | 22 units (33.8%) |

Two columns are added:
- `shrinkage_class` (str) -- human-readable label
- `shrinkage_code` (int) -- numeric code for model input

A warning is printed if any class has fewer than 10 samples. Thresholds can be adjusted in the YAML config without code changes.

**Output:** 65 rows x 46 columns (+ 2 target columns)

---

### Step 6: Multicollinearity Resolution (`multicollinearity.py`)

**`resolve_multicollinearity(df, cfg) -> pd.DataFrame`**

Drops pre-defined correlated features in four groups:

1. **Demographics** (`drop_correlated`): Direct column drop (pop_male, pop_female, households_total, age_15_64, age_75_plus)
2. **Spectral** (`drop_spectral_correlated`): Expands to `_mean` and `_std` suffixes (B2_mean, B2_std, B3_mean, B3_std)
3. **OSM** (`drop_osm_correlated`): Direct column drop (osm_building_count)
4. **Texture** (`drop_texture_correlated`): Expands to `_mean` and `_std` suffixes (S2_NDBI_homogeneity_mean, S2_NDBI_homogeneity_std)

Each dropped column is printed for transparency.

**Input:** 65 rows x 46 columns
**Output:** 65 rows x 34 columns (12 columns dropped)

---

### Step 7: Transform (`transformer.py`)

**`transform_features(df, cfg) -> tuple[pd.DataFrame, dict]`**

Applies two sequential transformations to numeric feature columns (excluding identifiers and target):

**Step 7a: log1p transform**

For each feature in `log1p_features`, the module resolves the actual column name:
1. Checks for `{feature}_mean` first (temporal-aggregated features)
2. Falls back to the raw name (static features)

If the column has negative values, it is shifted to make all values non-negative before applying `np.log1p`.

Current log1p features (9): B4_mean, B8_mean, viirs_mean_mean, S2_NDBI_contrast_mean, pop_total, age_u15, age_65_plus, osm_built_area, osm_built_ratio.

**Step 7b: RobustScaler**

`sklearn.preprocessing.RobustScaler` is applied to all numeric feature columns. This uses median and IQR (interquartile range) rather than mean and standard deviation, making it robust to outliers.

The scaler parameters (center, scale per feature) are stored in the metadata dict for reproducibility.

**Returns:** Tuple of (transformed DataFrame, metadata dict with scaler_type, log1p_features, scaled_features, n_features, scaler_center, scaler_scale).

---

### Step 8: Validate (`validation.py`)

**`validate_output(df, cfg) -> list[str]`**

Performs quality checks on the final dataset:

| Check | Criteria |
|-------|----------|
| Row count | Number of rows equals number of unique unit_ids |
| NaN check | No NaN values in any feature or target column |
| Target classes | Target column exists with exactly 3 unique classes |
| Constant features | No features with zero variance (std = 0) |
| Feature count | Reports total numeric feature columns |

Returns a list of warning messages (empty if all checks pass). Warnings are included in the preprocessing report.

---

### Step 9: Export (`run_preprocessing.py::_export`)

Saves four output files:

| File | Format | Description |
|------|--------|-------------|
| `classification_ready.parquet` | Apache Parquet | Primary output for downstream classification |
| `classification_ready.csv` | CSV | Human-readable backup |
| `preprocessing_report.json` | JSON | Step-by-step record of row/column counts, class distribution, transforms applied |
| `feature_metadata.csv` | CSV | Per-column metadata: role (identifier/feature/target), dtype, log1p_applied, scaled |

---

## Output Dataset

### Shape

**65 rows x 34 columns:** 30 numeric features + 2 identifiers + 2 target columns.

### Column Inventory

#### Identifiers (2)

| Column | Type | Description |
|--------|------|-------------|
| `unit_id` | object | Unique mura unit identifier |
| `pref_name` | object | Prefecture name (Aomori / Akita) |

#### Target (2)

| Column | Type | Description |
|--------|------|-------------|
| `shrinkage_class` | object | 3-class label: stable / shrinking / severely_shrinking |
| `shrinkage_code` | int64 | Numeric code: 0 / 1 / 2 |

#### Features (30)

**Spectral temporal aggregates (6):**

| Feature | Source | Transform |
|---------|--------|-----------|
| `B4_mean` | Sentinel-2 Red band, temporal mean | log1p + RobustScaler |
| `B4_std` | Sentinel-2 Red band, temporal std | RobustScaler |
| `B8_mean` | Sentinel-2 NIR band, temporal mean | log1p + RobustScaler |
| `B8_std` | Sentinel-2 NIR band, temporal std | RobustScaler |
| `B11_mean` | Sentinel-2 SWIR band, temporal mean | RobustScaler |
| `B11_std` | Sentinel-2 SWIR band, temporal std | RobustScaler |

**Index temporal aggregates (7):**

| Feature | Source | Transform |
|---------|--------|-----------|
| `NDVI_mean` | Vegetation index, temporal mean | RobustScaler |
| `NDVI_std` | Vegetation index, temporal std | RobustScaler |
| `NDVI_slope` | NDVI linear trend (OLS) | RobustScaler |
| `NDVI_seasonal_amp` | NDVI seasonal amplitude | RobustScaler |
| `NDBI_mean` | Built-up index, temporal mean | RobustScaler |
| `NDBI_std` | Built-up index, temporal std | RobustScaler |
| `NDBI_slope` | NDBI linear trend (OLS) | RobustScaler |
| `NDBI_seasonal_amp` | NDBI seasonal amplitude | RobustScaler |

**Nightlights temporal aggregates (3):**

| Feature | Source | Transform |
|---------|--------|-----------|
| `viirs_mean_mean` | VIIRS nightlights, temporal mean | log1p + RobustScaler |
| `viirs_mean_std` | VIIRS nightlights, temporal std | RobustScaler |
| `viirs_mean_slope` | VIIRS linear trend (OLS) | RobustScaler |

**GLCM texture temporal aggregates (4):**

| Feature | Source | Transform |
|---------|--------|-----------|
| `S2_NDBI_contrast_mean` | GLCM contrast, temporal mean | log1p + RobustScaler |
| `S2_NDBI_contrast_std` | GLCM contrast, temporal std | RobustScaler |
| `S2_NDBI_entropy_mean` | GLCM entropy, temporal mean | RobustScaler |
| `S2_NDBI_entropy_std` | GLCM entropy, temporal std | RobustScaler |

Note: S2_NDBI_homogeneity was dropped due to r = -0.99 with S2_NDBI_entropy.

**Static demographics (3):**

| Feature | Source | Transform |
|---------|--------|-----------|
| `pop_total` | Total population (census) | log1p + RobustScaler |
| `age_u15` | Population under 15 | log1p + RobustScaler |
| `age_65_plus` | Population 65 and over | log1p + RobustScaler |

Note: pop_male, pop_female, households_total, age_15_64, age_75_plus were dropped for multicollinearity.

**Static OSM (2):**

| Feature | Source | Transform |
|---------|--------|-----------|
| `osm_built_area` | Total built area from OSM | log1p + RobustScaler |
| `osm_built_ratio` | Built area / total area | log1p + RobustScaler |

Note: osm_building_count was dropped for multicollinearity with osm_built_area.

**Derived ratios (4):**

| Feature | Formula | Transform |
|---------|---------|-----------|
| `elderly_ratio` | age_65_plus / pop_total | RobustScaler |
| `aging_index` | age_65_plus / age_u15 | RobustScaler |
| `youth_ratio` | age_u15 / pop_total | RobustScaler |
| `household_size` | pop_total / households_total | RobustScaler |

---

## Data Flow Diagram

```
features_table.parquet (4,680 rows x 30 cols)
    │
    ├── Step 2: Drop meta columns (-5 cols)
    │   └── 4,680 rows x 25 cols
    │
    ├── Step 3: Temporal aggregation
    │   ├── 11 RS features -> 22 mean/std + 3 slopes + 2 seasonal = 27 RS cols
    │   ├── 8 demographic -> 8 static cols
    │   ├── 3 OSM -> 3 static cols
    │   └── 65 rows x 40 cols
    │
    ├── Step 4: Feature engineering (+4 derived)
    │   └── 65 rows x 44 cols
    │
    ├── Step 5: Target (+2 target cols)
    │   └── 65 rows x 46 cols
    │
    ├── Step 6: Multicollinearity (-12 correlated)
    │   └── 65 rows x 34 cols
    │
    ├── Step 7: Transform (log1p + RobustScaler)
    │   └── 65 rows x 34 cols (values transformed)
    │
    └── Step 8-9: Validate + Export
        └── classification_ready.parquet (65 rows x 34 cols)
```

---

## Preprocessing Report (`preprocessing_report.json`)

The JSON report captures metadata from each pipeline step:

```json
{
  "steps": [
    {"step": "load", "rows": 4680, "columns": 30},
    {"step": "drop_features", "columns_remaining": 25},
    {"step": "temporal_aggregation", "rows": 65, "columns": 40},
    {"step": "feature_engineering", "columns": 44},
    {"step": "build_target", "class_distribution": {...}},
    {"step": "multicollinearity", "columns_remaining": 34},
    {"step": "transform", "log1p_applied": [...], "scaler": "robust", "n_scaled_features": 30},
    {"step": "validation", "warnings": []}
  ],
  "final_shape": {"rows": 65, "columns": 34},
  "final_columns": [...],
  "transform_metadata": {
    "scaler_type": "robust",
    "log1p_features": [...],
    "scaled_features": [...],
    "scaler_center": {"B4_mean": ..., ...},
    "scaler_scale": {"B4_mean": ..., ...}
  }
}
```

The `scaler_center` (median) and `scaler_scale` (IQR) values are recorded for each feature, enabling exact inverse transformation if needed.

---

## Feature Metadata (`feature_metadata.csv`)

| Column | Description |
|--------|-------------|
| `column` | Column name in the output dataset |
| `role` | `identifier`, `feature`, or `target` |
| `dtype` | Pandas dtype (float64, object, int64) |
| `log1p_applied` | Whether log1p was applied to this column |
| `scaled` | Whether RobustScaler was applied (True for all numeric features) |

---

## Design Decisions

### Why elderly ratio for the target?

The elderly ratio (proportion of population aged 65+) is the standard demographic indicator for rural depopulation in Japan. The concept of *genkai-shuraku* (marginal settlement) is defined by communities where more than half the population is elderly. The thresholds 0.37 and 0.42 were chosen empirically to produce an approximately balanced 3-class split for classification, while remaining within the policy-relevant range.

### Why RobustScaler?

The EDA found all 22 numeric features are non-normal with significant outliers. RobustScaler uses median and IQR rather than mean and standard deviation, making it resilient to outliers. This is important because:
- Population counts span 3 orders of magnitude
- Remote sensing values contain cloud-contaminated outliers
- VIIRS nightlights have extreme positive skew (skewness = 4.4)

### Why log1p before scaling?

Features with skewness > 2 (B4, B8, viirs_mean, S2_NDBI_contrast, pop_total, etc.) benefit from log1p to reduce extreme positive skew. `log1p(x) = log(1 + x)` is used instead of `log(x)` to handle zero values safely. For columns with negative values, a shift is applied first.

### Why drop rather than PCA for multicollinearity?

With only 65 samples and features with clear domain-specific interpretations, dropping one feature per correlated pair preserves interpretability. The kept features were chosen based on domain relevance:
- `pop_total` over `pop_male`/`pop_female` (overall population size)
- `B4` over `B2`/`B3` (red band, most informative for vegetation)
- `S2_NDBI_entropy` over `S2_NDBI_homogeneity` (arbitrary choice; r = -0.99)

### Why only mean_std for GLCM (no slope/seasonal)?

GLCM texture features measure spatial pattern properties of NDBI. Unlike NDVI (which has a biological seasonal cycle) or nightlights (which can trend with urbanisation), texture metrics are not expected to show strong systematic temporal trends. Including mean and std captures the central tendency and variability, which is sufficient.

---

## Dependencies

```
pandas
numpy
scikit-learn (RobustScaler, StandardScaler)
pyyaml
pyarrow (for Parquet I/O)
```

---

## Shared Utilities (`utils.py`)

| Function | Description |
|----------|-------------|
| `load_config(path)` | Loads YAML config via `yaml.safe_load()` |
| `ensure_output_dirs(cfg)` | Creates the output directory |
| `write_json(path, obj)` | Writes dict to JSON with `default=str` serialisation |
