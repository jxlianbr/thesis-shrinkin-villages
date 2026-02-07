# EDA Module Documentation

## Overview

The Exploratory Data Analysis (EDA) module performs a comprehensive statistical and visual analysis of the `features_table.parquet` dataset -- a panel-format table containing 4,680 observations (65 mura administrative units x 72 months, January 2018 to December 2023) across 30 columns.

The module produces figures, tables, an interactive HTML report, a machine-readable JSON summary, and data quality flags that inform downstream preprocessing and classification decisions.

```
python eda/run_eda.py
python eda/run_eda.py --config path/to/eda_config.yaml
```

---

## Directory Structure

```
eda/
├── config/
│   └── eda_config.yaml           # Central configuration
├── src/
│   ├── __init__.py
│   ├── utils.py                  # Shared utilities (config, save, style)
│   ├── data_loader.py            # Parquet loading + schema validation
│   ├── summary_stats.py          # Descriptive statistics
│   ├── missing_data.py           # Missing data heatmap + summary
│   ├── distributions.py          # Histograms, boxplots, normality tests
│   ├── correlations.py           # Correlation matrices + high-pair extraction
│   ├── temporal_analysis.py      # Monthly trends, seasonality, coverage
│   ├── spatial_analysis.py       # Choropleth maps (geopandas)
│   ├── outlier_detection.py      # IQR-based outlier detection
│   ├── feature_relationships.py  # RS vs demographic scatter plots
│   └── report_generator.py       # HTML report, JSON summary, quality flags
├── outputs/
│   ├── figures/                  # 17 PNG visualisations
│   ├── tables/                   # 11 CSV analysis tables
│   └── reports/                  # HTML report, JSON summary, quality flags CSV
├── notebooks/
│   └── eda_interactive.ipynb     # Jupyter notebook for interactive exploration
└── run_eda.py                    # CLI entry point
```

---

## Configuration

All analysis parameters are controlled via `eda/config/eda_config.yaml`. The config is organised into the following sections:

### `data`

| Key | Description |
|-----|-------------|
| `features_table` | Path to the input Parquet file (`outputs/final/features_table.parquet`) |
| `boundaries_path` | Path to the GeoPackage for spatial analysis (`admin_demographics/boundaries/mura_jis.gpkg`) |

### `columns`

Declares feature groups used throughout the pipeline. Modules reference these groups to select which columns to analyse.

| Group | Columns | Description |
|-------|---------|-------------|
| `identifiers` | `unit_id`, `unit_level`, `unit_code`, `pref_name`, `month` | Non-numeric identifiers excluded from analysis |
| `spectral` | `B2`, `B3`, `B4`, `B8`, `B11` | Sentinel-2 spectral bands |
| `indices` | `NDVI`, `NDBI` | Derived vegetation and built-up indices |
| `texture` | `S2_NDBI_contrast`, `S2_NDBI_entropy`, `S2_NDBI_homogeneity` | GLCM texture features computed from NDBI |
| `nightlights` | `viirs_mean` | VIIRS nighttime lights |
| `osm` | `osm_built_area`, `osm_building_count`, `osm_built_ratio` | OpenStreetMap built environment |
| `demographic` | `pop_total`, `pop_male`, ... `age_75_plus` | Census demographics (8 columns) |
| `meta_only` | `muni_code`, `city_name_ja`, ... | Metadata excluded from numeric analysis |

### `analysis`

| Key | Default | Description |
|-----|---------|-------------|
| `outlier_method` | `iqr` | Outlier detection method |
| `outlier_iqr_factor` | `1.5` | IQR multiplier for outlier fences |
| `correlation_threshold` | `0.8` | Minimum |r| to flag as highly correlated |
| `missing_threshold` | `0.20` | Maximum acceptable missing fraction (20%) |
| `glcm_nan_note` | _(text)_ | Informational note about GLCM NaN provenance |

### `plot`

Controls all figure aesthetics: DPI (300), figsize presets, matplotlib style (`seaborn-v0_8-whitegrid`), font sizes, save formats (`png`), and per-prefecture colours (Aomori: blue, Akita: orange).

### `report`

| Key | Default | Description |
|-----|---------|-------------|
| `title` | `"EDA Report: Shrinking Villages Feature Table"` | HTML report title |
| `author` | `"Julian"` | Author name in report header |
| `generate_html` | `true` | Whether to produce the HTML report |
| `generate_pdf` | `false` | Whether to produce a PDF (requires `weasyprint`) |

---

## Pipeline Steps

`run_eda.py` orchestrates the pipeline in this order:

```
1. Load & validate       data_loader.py
2. Summary statistics    summary_stats.py
3. Missing data          missing_data.py
4. Distributions         distributions.py
5. Correlations          correlations.py
6. Temporal analysis     temporal_analysis.py
7. Spatial analysis      spatial_analysis.py
8. Outlier detection     outlier_detection.py
9. Feature relationships feature_relationships.py
10. Report generation    report_generator.py
```

Each analysis module receives the full DataFrame, the config dict, and the output directory. Each returns a summary dict that is collected and passed to the report generator.

---

## Module Details

### 1. Data Loader (`data_loader.py`)

**`load_features_table(cfg) -> pd.DataFrame`**

Reads the Parquet file and adds derived temporal columns:
- `month_dt` -- datetime parsed from the `month` column (format `"YYYY-MM"`)
- `year` -- integer year
- `month_num` -- integer month (1-12)
- `season` -- categorical (Winter, Spring, Summer, Autumn) using Northern Hemisphere mapping

**`validate_schema(df, cfg) -> list[str]`**

Cross-checks all column groups declared in config against the DataFrame. Returns warnings for missing columns, extra columns, and numeric type mismatches. This ensures the input data conforms to expectations before analysis begins.

---

### 2. Summary Statistics (`summary_stats.py`)

**`run_summary_stats(df, cfg, output_dir) -> dict`**

Computes and saves:

| Output | Description |
|--------|-------------|
| `tables/basic_statistics.csv` | `describe()` transposed + skewness, kurtosis, missing count/pct per feature |
| `tables/statistics_by_prefecture.csv` | Grouped mean/std/min/max/count per prefecture |

**Returns:** Row count, column count, unit count, time range, most variable features, most skewed features.

---

### 3. Missing Data (`missing_data.py`)

**`run_missing_analysis(df, cfg, output_dir) -> dict`**

| Output | Description |
|--------|-------------|
| `tables/missing_data_summary.csv` | Per-feature missing count, percentage, flag, and per-prefecture breakdown |
| `figures/missing_heatmap.png` | Heatmap of missing % by feature x prefecture (+ overall column) |
| `figures/missing_by_time.png` | Line plot of monthly missing counts for spectral/index features |

**Key logic:**
- All numeric features are checked against `missing_threshold` (default 20%)
- Features above threshold are flagged as `columns_above_threshold`
- The `glcm_nan_note` from config is carried through to the report as an informational note

**Returns:** Overall missing percentage, list of flagged columns, GLCM note, threshold used.

---

### 4. Distributions (`distributions.py`)

**`run_distribution_analysis(df, cfg, output_dir) -> dict`**

| Output | Description |
|--------|-------------|
| `figures/distributions_rs.png` | Histogram grid (with KDE overlay) for all remote sensing features including spectral, indices, texture, and nightlights |
| `figures/distributions_demo.png` | Histogram grid for demographic features |
| `figures/boxplots_by_prefecture.png` | Side-by-side boxplots for key features (NDVI, NDBI, viirs_mean, pop_total, age_65_plus) grouped by prefecture |
| `tables/distribution_statistics.csv` | Per-feature: n_valid, mean, std, skewness, kurtosis, Shapiro-Wilk p-value, normality flag, suggested transform |

**Transform suggestion logic:**
- |skewness| > 2 and min >= 0: `log1p`
- |skewness| > 2 and min < 0: `robust_scale`
- |skewness| > 1 and min >= 0: `sqrt`
- |skewness| > 1 and min < 0: `robust_scale`
- Otherwise: `none`

**Returns:** List of non-normal features, features needing transformation, count analysed.

---

### 5. Correlations (`correlations.py`)

**`run_correlation_analysis(df, cfg, output_dir) -> dict`**

| Output | Description |
|--------|-------------|
| `tables/correlation_matrix.csv` | Full Pearson correlation matrix of all numeric features |
| `figures/correlation_heatmap.png` | Upper-triangle-masked heatmap with annotated r values |
| `tables/high_correlations.csv` | Pairs with |r| >= threshold, sorted by absolute correlation |
| `tables/rs_demographic_correlations.csv` | Focused RS x demographic cross-correlation matrix |
| `figures/rs_demo_correlation.png` | Focused heatmap of RS (rows) vs demographic (columns) |

All numeric features (including GLCM texture) are included in the analysis.

**Returns:** Feature count, top 10 highly correlated pairs, top 10 RS-demographic associations, threshold used.

---

### 6. Temporal Analysis (`temporal_analysis.py`)

**`run_temporal_analysis(df, cfg, output_dir) -> dict`**

| Output | Description |
|--------|-------------|
| `figures/temporal_trends.png` | Monthly mean time series with +/-1 std shading for NDVI, NDBI, viirs_mean, faceted by prefecture |
| `figures/seasonal_patterns.png` | Monthly (1-12) boxplots aggregated across years for NDVI, NDBI |
| `tables/temporal_coverage.csv` | Per-unit: total months, valid months, valid %, first/last month, gap count |
| `tables/unit_trends.csv` | Per-unit OLS slope for key features (time normalised to [0,1]) |

**Temporal coverage:** Uses NDVI as the validity indicator. A gap is defined as >35 days between consecutive valid observations.

**OLS slope:** Time is normalised to [0,1] over each unit's observation span. `np.polyfit(t, y, 1)` extracts the linear slope, representing the total change in feature value over the study period.

**Returns:** Coverage summary (mean/min months, units with gaps), seasonal amplitude (max-min monthly mean for NDVI/NDBI), trend directions (increasing/decreasing).

---

### 7. Spatial Analysis (`spatial_analysis.py`)

**`run_spatial_analysis(df, cfg, output_dir) -> dict`**

Requires `geopandas` and the boundaries GeoPackage (`mura_jis.gpkg`). Falls back gracefully with a "skipped" status if either is unavailable.

| Output | Description |
|--------|-------------|
| `figures/spatial_ndvi_mean.png` | Choropleth of per-unit mean NDVI (YlGn colormap) |
| `figures/spatial_viirs_mean.png` | Choropleth of per-unit mean VIIRS (YlOrRd colormap) |
| `figures/spatial_pop_total.png` | Choropleth of per-unit mean population (Blues colormap) |

**Returns:** Number of units mapped, per-prefecture summary (unit count, mean NDVI, mean VIIRS, mean population).

---

### 8. Outlier Detection (`outlier_detection.py`)

**`run_outlier_detection(df, cfg, output_dir) -> dict`**

| Output | Description |
|--------|-------------|
| `tables/outlier_summary.csv` | Per-feature: n_valid, outlier count/pct, IQR bounds (Q1, Q3, lower, upper) |
| `tables/outlier_observations.csv` | Individual outlier observations with unit_id, month, value, bound exceeded |
| `figures/outlier_boxplots.png` | Grid of boxplots for all numeric features |

**Method:** IQR with factor 1.5 (configurable). Lower fence = Q1 - 1.5*IQR, upper fence = Q3 + 1.5*IQR.

All numeric features (including GLCM texture) are included.

**Returns:** Total outlier count, features with outliers, top 5 most affected features, top 5 most affected units.

---

### 9. Feature Relationships (`feature_relationships.py`)

**`run_feature_relationships(df, cfg, output_dir) -> dict`**

Generates scatter plots at the unit level (per-unit means averaged across months), coloured by prefecture with linear fit lines.

| Output | Description |
|--------|-------------|
| `figures/ndvi_vs_population.png` | Mean NDVI vs total population |
| `figures/viirs_vs_population.png` | Mean VIIRS nightlights vs total population |
| `figures/ndbi_vs_population.png` | Mean NDBI vs total population |
| `figures/viirs_vs_elderly_ratio.png` | Mean VIIRS vs elderly ratio (age_65_plus/pop_total) |

Each plot displays the Pearson r value in a text box.

**Returns:** List of key relationships with x, y column names and r values.

---

### 10. Report Generator (`report_generator.py`)

**`generate_report(summary, cfg, output_dir) -> None`**

Compiles all analysis results into three outputs:

| Output | Description |
|--------|-------------|
| `reports/eda_report.html` | Self-contained HTML report with embedded base64 figures, tables, validation checklist, and key findings |
| `reports/eda_summary.json` | Machine-readable JSON of all analysis summaries |
| `reports/data_quality_flags.csv` | Structured quality flags (type, feature, unit_id, month, detail) |

**HTML Report Sections:**
1. Data Overview -- row/column counts, units, prefectures, time range
2. Missing Data -- overall rate, threshold check, GLCM note
3. Feature Distributions -- normality counts, transform suggestions
4. Correlation Analysis -- highly correlated pairs table, RS-demographic associations
5. Temporal Patterns -- coverage summary, seasonal amplitude table
6. Spatial Patterns -- per-prefecture statistics, choropleth maps
7. Outlier Detection -- method, total count, most affected features/units
8. RS-Demographic Relationships -- scatter plot summaries
9. Data Quality Summary -- validation checklist with pass/warn status
10. Key Findings for Classification -- actionable findings with implications

**Quality Flag Types:**
- `high_missing` -- feature exceeds missing threshold
- `glcm_available` -- GLCM texture features available for all units
- `high_outlier_rate` -- feature has >5% outlier rate
- `temporal_gaps` -- units with temporal coverage gaps

---

## Output Inventory

### Figures (17 PNG files)

| Figure | Module |
|--------|--------|
| `missing_heatmap.png` | Missing data |
| `missing_by_time.png` | Missing data |
| `distributions_rs.png` | Distributions |
| `distributions_demo.png` | Distributions |
| `boxplots_by_prefecture.png` | Distributions |
| `correlation_heatmap.png` | Correlations |
| `rs_demo_correlation.png` | Correlations |
| `temporal_trends.png` | Temporal |
| `seasonal_patterns.png` | Temporal |
| `spatial_ndvi_mean.png` | Spatial |
| `spatial_viirs_mean.png` | Spatial |
| `spatial_pop_total.png` | Spatial |
| `outlier_boxplots.png` | Outliers |
| `ndvi_vs_population.png` | Feature relationships |
| `viirs_vs_population.png` | Feature relationships |
| `ndbi_vs_population.png` | Feature relationships |
| `viirs_vs_elderly_ratio.png` | Feature relationships |

### Tables (11 CSV files)

| Table | Module |
|-------|--------|
| `basic_statistics.csv` | Summary statistics |
| `statistics_by_prefecture.csv` | Summary statistics |
| `missing_data_summary.csv` | Missing data |
| `distribution_statistics.csv` | Distributions |
| `correlation_matrix.csv` | Correlations |
| `high_correlations.csv` | Correlations |
| `rs_demographic_correlations.csv` | Correlations |
| `temporal_coverage.csv` | Temporal |
| `unit_trends.csv` | Temporal |
| `outlier_summary.csv` | Outliers |
| `outlier_observations.csv` | Outliers |

### Reports (3 files)

| Report | Format |
|--------|--------|
| `eda_report.html` | Self-contained HTML with embedded figures |
| `eda_summary.json` | Machine-readable JSON summary |
| `data_quality_flags.csv` | Structured quality flags |

---

## Key Findings (current data)

These are the current findings from the most recent EDA run:

1. **Missing data: 4.9% overall.** Spectral bands and GLCM texture features have ~5-14% NaN from cloud cover. Demographics and OSM are 0% NaN. No features exceed the 20% threshold.

2. **All 22 numeric features are non-normal** (Shapiro-Wilk p < 0.05). Features with high skewness (|skew| > 2) that require log1p: B2, B3, B4, viirs_mean, S2_NDBI_contrast, pop_total, pop_male, pop_female, households_total, age_u15, age_15_64, age_65_plus, age_75_plus, osm_built_area, osm_building_count, osm_built_ratio.

3. **46 feature pairs with |r| > 0.8.** Major correlated clusters:
   - Demographics: pop_total/pop_male/pop_female/households_total (r > 0.99)
   - Spectral: B2/B3/B4 (r > 0.99), B8 correlated with B2-B4 (r ~ 0.9)
   - Texture: S2_NDBI_entropy/S2_NDBI_homogeneity (r = -0.99), S2_NDBI_contrast is independent
   - OSM: osm_built_area/osm_building_count (r = 0.99)

4. **Mean temporal coverage: 61.6 months per unit** out of 72. NDVI shows strong seasonality (amplitude ~0.2).

5. **Demographics are static** -- identical values across all 72 months per unit (single census snapshot).

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn (>= 0.14)
scipy
geopandas (for spatial analysis)
jinja2 (for HTML report)
pyyaml
pyarrow (for Parquet I/O)
```

Optional: `weasyprint` for PDF report generation.

---

## Shared Utilities (`utils.py`)

| Function | Description |
|----------|-------------|
| `load_eda_config(path)` | Loads YAML config via `yaml.safe_load()` |
| `ensure_output_dirs(cfg)` | Creates `figures_dir`, `tables_dir`, `reports_dir` |
| `save_figure(fig, name, cfg)` | Saves matplotlib figure to configured directory at configured DPI |
| `save_table(df, name, cfg)` | Saves DataFrame as CSV to configured directory |
| `write_json(path, obj)` | Writes dict to JSON with `default=str` serialisation |
| `get_numeric_columns(df, cfg)` | Returns all numeric columns excluding identifiers and meta-only |
| `get_feature_group(cfg, group)` | Returns column list for a named feature group |
| `get_existing_columns(df, columns)` | Filters a column list to those present in the DataFrame |
| `setup_plot_style(cfg)` | Applies matplotlib style and rcParams from config |
