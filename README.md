# Mapping Shrinking Villages in Rural Japan

Master thesis project: classifying shrinking villages in Aomori and Akita prefectures
using satellite remote sensing, demographic statistics, and machine learning.

## Project Summary

This repository implements a reproducible research pipeline that:

1. Acquires multi-temporal satellite imagery (Sentinel-2, VIIRS) via Google Earth Engine
2. Extracts spectral indices, texture features, terrain, land-use, and building footprints
3. Joins administrative demographics from the Japanese Census (e-Stat)
4. Performs exploratory data analysis on the resulting panel dataset
5. Preprocesses and aggregates monthly panel data into a cross-sectional feature table
6. Classifies 65 village-level units into three shrinkage categories
7. Identifies village typologies through unsupervised clustering

**Study area:** 65 mura (sub-municipal) units across Aomori and Akita prefectures, Tohoku region.
**Time range:** January 2015 -- December 2025 (132 months).
**Dataset:** 8,450 panel rows (65 units x ~130 months), 30+ features.

## Directory Structure

```
thesis-shrinkin-villages/
|
|-- pipeline.py                      Main GEE data acquisition orchestrator
|-- requirements.txt                 Python dependencies
|-- config/
|   |-- config.yaml                  Global pipeline configuration
|   +-- config_golden_sample.yaml    Validation subset config
|
|-- admin_demographics/              Boundary & census data preparation
|   |-- boundaries/                  GeoPackage boundary files (mura, aza)
|   |-- demographics/                Census CSV files
|   |-- aoi/                         Area-of-interest geometries
|   +-- *.py                         Boundary/demographics build scripts
|
|-- data_preprocessing/              Satellite feature extraction
|   |-- gee_monthly.py               Sentinel-2 + VIIRS monthly composites
|   |-- compute_glcm_local.py        GLCM texture features (local CPU)
|   |-- gee_terrain.py               Copernicus DEM terrain features
|   |-- gee_lulc.py                  Dynamic World land-use fractions
|   +-- osm_labels.py                OSM building footprint features
|
|-- eda/                             Exploratory Data Analysis
|   |-- run_eda.py                   Entry point
|   |-- config/eda_config.yaml       EDA configuration
|   |-- src/                         12 analysis modules
|   +-- outputs/                     Figures, tables, HTML report
|
|-- preprocessing/                   Feature engineering & scaling
|   |-- run_preprocessing.py         Entry point
|   |-- config/preprocessing_config.yaml
|   |-- src/                         8 processing modules
|   +-- outputs/                     classification_ready.parquet (65 rows)
|
|-- classification/                  Supervised classification
|   |-- run_classification.py        Entry point
|   |-- config/classification_config.yaml
|   |-- src/                         12 pipeline modules
|   +-- outputs/                     Models, figures, tables, HTML report
|
|-- typology/                        Unsupervised clustering & regression
|   |-- run_typology.py              Entry point
|   |-- config/typology_config.yaml
|   |-- src/                         5 analysis modules
|   +-- outputs/                     Cluster assignments, figures, tables
|
|-- outputs/                         Pipeline outputs (GEE exports, final tables)
|   |-- final/                       features_table.parquet (8,450 rows)
|   +-- gee/monthly/                 132 monthly CSV exports
|
|-- docs/                            Project documentation
|-- scripts/                         Utility scripts
+-- tests/                           Validation tests
```

## Pipeline Overview

The full pipeline runs in four sequential stages. Each stage has its own
entry point and configuration file.

### Stage 1: Data Acquisition (`pipeline.py`)

Orchestrates satellite data acquisition through Google Earth Engine and
local feature computation.

```
pipeline.py config/config.yaml
```

**Inputs:** GEE cloud assets, boundary GeoPackages, census CSV.
**Outputs:** `outputs/final/features_table.parquet` -- monthly panel table
(8,450 rows x 30+ columns).

Features extracted:
- Sentinel-2 bands (B2, B3, B4, B8, B11) and indices (NDVI, NDBI, MNDWI)
- GLCM texture on NDBI (contrast, entropy, homogeneity)
- VIIRS nighttime lights (monthly mean radiance)
- Copernicus DEM terrain (elevation, slope, aspect, TRI)
- Dynamic World land-use class fractions (9 classes)
- OSM building footprints (built area, count, density ratio)
- Census demographics (population, age groups, households)

### Stage 2: Exploratory Data Analysis (`eda/run_eda.py`)

Investigates feature distributions, missing data patterns, correlations,
temporal trends, and spatial patterns.

```
python eda/run_eda.py
```

**Input:** `outputs/final/features_table.parquet`
**Outputs:** 17 PNG figures, 11 CSV tables, HTML report, JSON summary.

Key findings: 25% NaN in GLCM features (cloud cover), 47 highly correlated
feature pairs, all features non-normal.

### Stage 3: Preprocessing (`preprocessing/run_preprocessing.py`)

Transforms the monthly panel into a cross-sectional classification-ready
dataset.

```
python preprocessing/run_preprocessing.py
```

**Input:** `outputs/final/features_table.parquet` (8,450 rows)
**Output:** `preprocessing/outputs/classification_ready.parquet` (65 rows x 34 columns)

Processing steps:
1. Temporal aggregation (mean, std, linear slope, seasonal amplitude)
2. Feature engineering (elderly_ratio, aging_index, youth_ratio, household_size)
3. Target variable construction: 3-class `shrinkage_class` from elderly_ratio
   thresholds (0.37 / 0.42) giving stable (20), shrinking (23), severely_shrinking (22)
4. Multicollinearity removal (9 redundant features dropped)
5. Skewness correction (log1p on 9 features) and RobustScaler normalization

### Stage 4a: Classification (`classification/run_classification.py`)

Trains and evaluates 10 classifiers under 4 leakage-controlled experiments.

```
python classification/run_classification.py
```

**Input:** `preprocessing/outputs/classification_ready.parquet`
**Outputs:** 10 saved models (.joblib), 21 figures, 14 CSV tables, HTML report.

Models: DummyClassifier (x2), Logistic Regression, SVM (linear), SVM (RBF),
Random Forest, Gradient Boosting, KNN, MLP, XGBoost.

Evaluation: RepeatedStratifiedKFold (5-fold x 5-repeat = 25 rounds).
Primary metric: balanced accuracy.

Leakage experiments:
- `all_features` (30 features) -- reference with known leakage
- `no_leaky` (27 features) -- **primary experiment**, drops target-derived ratios
- `no_demographic` (23 features) -- drops all demographic features
- `rs_only` (21 features) -- remote sensing features only

Best result: SVM Linear, balanced accuracy = 0.662 (no_leaky experiment).
Top features: viirs_mean_mean, viirs_mean_std, age_u15, household_size, age_65_plus.

### Stage 4b: Typology Analysis (`typology/run_typology.py`)

Unsupervised clustering to identify village shrinkage typologies and
analyze relationships between physical and demographic indicators.

```
python typology/run_typology.py
```

**Input:** `preprocessing/outputs/classification_ready.parquet`,
`outputs/final/features_table.parquet`
**Outputs:** 25+ figures, 30+ CSV tables, HTML report.

Analysis steps:
1. Indicator extraction from raw monthly data (trend slopes, seasonal amplitude,
   coefficient of variation, GLCM means, demographics)
2. Multi-k clustering (K-means k=2..8, hierarchical Ward) with gap statistic,
   bootstrap stability (1000 resamples), and perturbation robustness
3. Specification robustness: re-cluster with physical-only, demographic-only,
   trend-only, and level-only feature subsets
4. Relationship analysis: OLS regression, Pearson/Spearman correlations, VIF
   checks, subgroup regressions by cluster

## Setup

### Requirements

- Python 3.8+
- Google Earth Engine account (authenticated)
- ~5-10 GB disk space for full pipeline outputs

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt

# For GLCM texture computation:
pip install scikit-image

# Authenticate GEE:
earthengine authenticate
```

### Running

```bash
# Full data acquisition pipeline (2-6 hours)
python pipeline.py config/config.yaml

# Downstream analysis (run in order)
python eda/run_eda.py
python preprocessing/run_preprocessing.py
python classification/run_classification.py
python typology/run_typology.py
```

## Key Data Files

| File | Description | Shape |
|------|-------------|-------|
| `outputs/final/features_table.parquet` | Monthly panel dataset | 8,450 x 30+ |
| `preprocessing/outputs/classification_ready.parquet` | Cross-sectional, scaled | 65 x 34 |
| `admin_demographics/boundaries/mura_jis.gpkg` | Village unit boundaries | 65 polygons |
| `admin_demographics/demographics/village_demographics.csv` | Census snapshot | 65 rows |

## Configuration

All modules are YAML-configured:

| Config File | Controls |
|-------------|----------|
| `config/config.yaml` | GEE settings, data paths, feature flags, study area |
| `eda/config/eda_config.yaml` | Feature groups, plot styles, analysis thresholds |
| `preprocessing/config/preprocessing_config.yaml` | Aggregation, engineering, target, scaling |
| `classification/config/classification_config.yaml` | Models, experiments, CV strategy, metrics |
| `typology/config/typology_config.yaml` | Indicators, clustering k-range, relationships |

## Documentation

Detailed documentation is in the `docs/` directory:

- `codebase_documentation.md` -- Full API reference for the data pipeline
- `method.md` -- Methodological overview and reproducibility
- `preprocessing.md` -- Feature aggregation and engineering details
- `eda.md` -- EDA pipeline documentation
- `aoi_specification.md` -- Study area and golden sample definitions
- `label_strategy.md` -- Target variable design rationale
- `data_manifest.md` -- Data source inventory
- `provenance_licensing.md` -- Data licensing and attribution
- `compliance.md` -- Reproducibility checklist

## Codebase Conventions

- `from __future__ import annotations` in all modules
- Type hints on all functions (`Dict[str, Any]` for config)
- `print()` for output (no logging module)
- YAML config files loaded via `yaml.safe_load()`
- Snake_case functions, UPPER_CASE constants
- Module pattern: docstring, `run_*()` entry point, `if __name__ == "__main__"`
- Paths via `pathlib.Path`
- ASCII-only in `print()` statements (Windows cp1252 compatibility)

## Data Sources

| Source | Usage | License |
|--------|-------|---------|
| Sentinel-2 (Copernicus) | Spectral bands, indices, GLCM | Copernicus Open Access |
| VIIRS (NOAA) | Nighttime lights | NOAA Open Data |
| Copernicus DEM | Terrain features | Copernicus Open Access |
| Dynamic World (Google) | Land-use fractions | CC BY-4.0 |
| OpenStreetMap | Building footprints | ODbL |
| e-Stat (Statistics Bureau of Japan) | Demographics | Government Open Data |
