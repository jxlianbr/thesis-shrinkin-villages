# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Master's thesis pipeline classifying shrinking villages in rural Japan (Aomori and Akita prefectures, 65 mura units) using satellite remote sensing, demographic census data, and machine learning. Produces a 3-class shrinkage label (stable / shrinking / severely shrinking) per village unit.

## Commands

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install scikit-image        # optional: only needed for GLCM texture computation
earthengine authenticate        # GEE authentication required for Stage 1
```

### Running the Pipeline (sequential)
```bash
# Stage 1: Data acquisition from GEE (2-6 hours)
python pipeline.py config/config.yaml

# Stage 2-4: Downstream analysis
python eda/run_eda.py
python preprocessing/run_preprocessing.py
python classification/run_classification.py
python typology/run_typology.py
```

### Tests
```bash
python -m pytest tests/test_golden_sample.py -v
```
The test runs the full pipeline with `config/config_golden_sample.yaml` (10 units × 2 months = 20 rows) and validates schema and feature completeness. Timeout: 10 minutes.

## Architecture

The pipeline has four sequential stages, each self-contained with its own entry point, YAML config, `src/` modules, and `outputs/` directory.

### Stage 1 — Data Acquisition (`pipeline.py`)
Orchestrates 8 feature extraction steps, each implemented as a module in `data_preprocessing/`:
- `gee_monthly.py` — Sentinel-2 cloud-filtered monthly composites + VIIRS nighttime lights via GEE zone reduction
- `compute_glcm_local.py` — Downloads NDBI raster tiles from GEE and computes GLCM texture (contrast, entropy, homogeneity) locally using `scikit-image` with `ProcessPoolExecutor`; avoids per-pixel GEE computation cost
- `osm_labels.py` — Fetches building footprints from Overpass API, computes area/density per unit
- `gee_terrain.py` — Static Copernicus DEM terrain features (elevation, slope, aspect, TRI)
- `gee_lulc.py` — Static Dynamic World LULC fractions (9 classes)
- Demographics join is deterministic from local CSV (no API call)

**Output:** `outputs/final/features_table.parquet` — 8,450 rows (65 units × ~130 months), 30+ features

### Stage 2 — EDA (`eda/run_eda.py`)
12 analysis modules covering distributions, correlations, temporal trends, spatial patterns, and outlier detection. Outputs 17 PNG figures, 11 CSV tables, and an HTML report to `eda/outputs/`.

### Stage 3 — Preprocessing (`preprocessing/run_preprocessing.py`)
Collapses the 8,450-row monthly panel into 65 cross-sectional rows through 8 ordered steps implemented in `preprocessing/src/`:
1. Load + add temporal columns (`month_dt`, `month_num`, `season`)
2. Drop meta-only columns
3. **Temporal aggregation** — mean, std, OLS trend slope, and seasonal amplitude per unit
4. **Feature engineering** — `elderly_ratio`, `aging_index`, `youth_ratio`, `household_size`
5. **Target construction** — `shrinkage_class` from `elderly_ratio` thresholds (< 0.37 = stable, < 0.42 = shrinking, ≥ 0.42 = severely_shrinking)
6. **Multicollinearity removal** — 9 features dropped (e.g. `pop_male`, `pop_female`, `B2`, `B3`, `osm_building_count`)
7. **Transform + scale** — `log1p` on 9 skewed features, then `RobustScaler`
8. Validate + export

**Output:** `preprocessing/outputs/classification_ready.parquet` — 65 rows × 34 columns

### Stage 4a — Classification (`classification/run_classification.py`)
Trains 10 classifiers across 4 leakage-controlled feature-set experiments using `RepeatedStratifiedKFold` (5 splits × 5 repeats). Primary metric: balanced accuracy.

**Feature set experiments** (defined in `classification/src/leakage_analysis.py`):
- `all_features` — 30 features, includes known leakage (reference only)
- `no_leaky` — 27 features, **primary experiment**, drops target-derived ratios (`elderly_ratio`, `aging_index`, `youth_ratio`)
- `no_demographic` — 23 features
- `rs_only` — 21 features, remote sensing only (core thesis question)

Best result to date: SVM Linear, balanced accuracy = 0.662 (`no_leaky`).

### Stage 4b — Typology Analysis (`typology/run_typology.py`)
3-step unsupervised analysis:
1. Compile physical + demographic indicators from raw monthly data
2. Multi-k clustering (K-means k=2..8, hierarchical Ward) with gap statistic, bootstrap stability (1000 resamples), and 5 specification robustness variants
3. Relationship analysis: OLS regression, Pearson/Spearman correlations, VIF, spatial models, subgroup regressions

## Key Data Files

| File | Description | Shape |
|------|-------------|-------|
| `outputs/final/features_table.parquet` | Monthly panel (Stage 1 output) | 8,450 × 30+ |
| `preprocessing/outputs/classification_ready.parquet` | Cross-sectional, scaled (Stage 3 output) | 65 × 34 |
| `admin_demographics/boundaries/aza.gpkg` | Village unit boundaries | 65 polygons |
| `admin_demographics/demographics/` | Census CSV files (2015, 2020) | 65 rows |

## Configuration

All behaviour is YAML-driven — no hardcoded parameters. Key toggles in `config/config.yaml`:
- `features.*` — enable/disable NDVI, NDBI, MNDWI, terrain, LULC, GLCM, VIIRS, OSM
- `aoi.mode` — `full` (65 units) or `golden` (10 units for fast validation)
- `run.fast_mode` — reduces months processed for development
- `gee.export_mode` — controls GEE task export vs. direct download

Stage-specific configs live at `<stage>/config/<stage>_config.yaml`.

## Codebase Conventions

- `from __future__ import annotations` in every module
- Type hints on all functions; config dicts typed as `Dict[str, Any]`
- `print()` for all output — no `logging` module
- ASCII-only in `print()` statements (Windows cp1252 encoding compatibility)
- All file paths via `pathlib.Path`
- Module pattern: module docstring → helper functions → `run_*()` entry point → `if __name__ == "__main__"`
- `random_state=42` everywhere for reproducibility
- GEE operations use retry logic with exponential backoff (implemented in `gee_monthly.py`)
