# Codebase Documentation: Shrinking Villages Analysis Pipeline

**Version:** 1.0
**Date:** 2026-02-03
**Repository:** thesis-shrinking-villages
**Primary Language:** Python 3.8+

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [System Architecture](#3-system-architecture)
4. [Module Reference](#4-module-reference)
5. [Configuration Reference](#5-configuration-reference)
6. [Data Processing Pipeline](#6-data-processing-pipeline)
7. [Input Data Requirements](#7-input-data-requirements)
8. [Output Data Schema](#8-output-data-schema)
9. [Dependencies and Requirements](#9-dependencies-and-requirements)
10. [Execution Workflows](#10-execution-workflows)
11. [Testing and Validation](#11-testing-and-validation)
12. [Data Provenance and Compliance](#12-data-provenance-and-compliance)
13. [API Reference](#13-api-reference)

---

## 1. Project Overview

### 1.1 Purpose

This codebase implements a reproducible research pipeline for analyzing **shrinking villages** in rural Tohoku, Japan (Aomori and Akita prefectures). The pipeline integrates:

- **Satellite remote sensing** (Sentinel-2, Landsat 8, VIIRS night-lights)
- **Administrative boundary data** (village/sub-municipal levels)
- **Demographic statistics** (population, households, age structure)
- **OpenStreetMap building footprints** (built-up area indicators)

The system generates multi-temporal feature datasets at the municipal (mura) and sub-municipal (aza) administrative levels for subsequent analysis of rural depopulation patterns.

### 1.2 Scientific Context

Rural depopulation is a significant demographic challenge in Japan, particularly in the Tohoku region. This pipeline enables systematic monitoring of settlement dynamics by combining:

1. **Spectral indicators** (NDVI, NDBI) from optical satellite imagery
2. **Texture features** (GLCM metrics) characterizing built-up area heterogeneity
3. **Night-light radiance** (VIIRS) as a proxy for human activity
4. **Building density** (OSM footprints) for direct built-up measurement
5. **Demographic attributes** (population, age structure) from official statistics

### 1.3 Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Reproducibility** | Git versioning, pinned dependencies, configuration-driven |
| **Scalability** | Google Earth Engine for satellite preprocessing |
| **Modularity** | Separate modules for GEE, GLCM, OSM, demographics |
| **Privacy** | Aggregation at village level only, no person-level inference |
| **Provenance** | Machine-readable manifests, documented data sources |

---

## 2. Directory Structure

```
D:/thesis-shrinkin-villages/
├── pipeline.py                     # Main execution entrypoint
├── requirements.txt                # Core dependencies (pinned versions)
├── requirements-texture.txt        # Optional texture analysis dependencies
├── README.md                       # Project overview
├── .gitignore                      # Git exclusions
│
├── config/                         # Configuration files
│   ├── config.yaml                 # Production configuration
│   └── config_golden_sample.yaml   # Minimal test configuration
│
├── docs/                           # Documentation
│   ├── codebase_documentation.md   # This document
│   ├── method.md                   # Method implementation record
│   ├── aoi_specification.md        # AOI definitions
│   ├── data_manifest.md            # Dataset inventory template
│   ├── label_strategy.md           # OSM label source justification
│   ├── compliance.md               # Licensing and compliance
│   ├── provenance_licensing.md     # Data provenance details
│   └── task2_validation.md         # Validation run metadata
│
├── admin_demographics/             # Boundary and demographic preprocessing
│   ├── build_aoi.py                # Create AOI geometries
│   ├── build_mura_input_from_adm_gpkg.py    # Convert ADM to mura
│   ├── build_mura_boundaries_from_aza.py    # Aggregate aza to mura
│   ├── build_village_demographics_from_estat.py  # Process e-Stat data
│   ├── prepare_boundaries.py       # Standardize boundary schema
│   ├── export_for_gee.py           # Convert to Shapefile for GEE
│   ├── upload_aoi_to_gee.py        # Upload assets to GEE
│   ├── normalize_inputs.py         # Column renaming utility
│   └── check_gpkg.py               # GeoPackage diagnostic tool
│
├── data_preprocessing/             # Core data processing modules
│   ├── __init__.py                 # Package marker
│   ├── gee_monthly.py              # GEE satellite processing
│   ├── compute_glcm_local.py       # Local GLCM texture computation
│   └── osm_labels.py               # OSM building feature extraction
│
├── scripts/                        # Utility scripts
│   └── rerun_glcm_only.py          # Re-run GLCM without full pipeline
│
├── tests/                          # Test suite
│   └── test_golden_sample.py       # Reproducibility validation tests
│
├── outputs/                        # Generated outputs (gitignored)
│   ├── final/                      # Final feature tables
│   ├── logs/                       # Processing logs
│   ├── rasters/                    # Downloaded raster tiles
│   ├── osm/                        # OSM data cache
│   └── intermediate/               # Intermediate files
│
└── labels/                         # Reserved for label data
```

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE.PY                                     │
│                         (Main Orchestrator)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│  GEE_MONTHLY.PY   │   │ COMPUTE_GLCM_     │   │  OSM_LABELS.PY    │
│                   │   │    LOCAL.PY       │   │                   │
│ • Sentinel-2      │   │ • GLCM texture    │   │ • OSM download    │
│ • VIIRS           │   │ • Raster download │   │ • Building area   │
│ • NDVI/NDBI       │   │ • scikit-image    │   │ • Density ratio   │
│ • Zonal stats     │   │                   │   │                   │
└───────────────────┘   └───────────────────┘   └───────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │    FEATURES TABLE         │
                    │ (unit_id × month × features)│
                    └───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │   DEMOGRAPHICS JOIN       │
                    │ (village_demographics.csv) │
                    └───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │   OUTPUT FILES            │
                    │ • features_table.csv      │
                    │ • features_table.parquet  │
                    │ • run_manifest.json       │
                    └───────────────────────────┘
```

### 3.2 Data Flow

```
                     ┌──────────────────┐
                     │  CONFIGURATION   │
                     │  (config.yaml)   │
                     └────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Boundaries  │    │  Demographics │    │    AOI        │
│   (GeoPackage)│    │    (CSV)      │    │ (GeoPackage)  │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        │            ┌───────┴───────┐            │
        └────────────┤   PIPELINE    ├────────────┘
                     │   EXECUTION   │
                     └───────┬───────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     │                       │                       │
     ▼                       ▼                       ▼
┌──────────┐          ┌──────────┐          ┌──────────┐
│   GEE    │          │  Local   │          │   OSM    │
│ (Cloud)  │          │  GLCM    │          │ Overpass │
└────┬─────┘          └────┬─────┘          └────┬─────┘
     │                     │                     │
     └─────────────────────┼─────────────────────┘
                           ▼
                  ┌─────────────────┐
                  │  Merged Output  │
                  │  (CSV/Parquet)  │
                  └─────────────────┘
```

### 3.3 Processing Stages

| Stage | Description | Module |
|-------|-------------|--------|
| 1 | Data validation | `pipeline.py` |
| 2 | GEE satellite preprocessing | `gee_monthly.py` |
| 3 | VIIRS aggregation (optional) | `gee_monthly.py` |
| 4a | GLCM texture computation | `compute_glcm_local.py` |
| 4b | OSM building features | `osm_labels.py` |
| 5 | Aggregation to units | `gee_monthly.py` (reduceRegions) |
| 6 | Demographics join | `pipeline.py` |
| 7 | Export | `pipeline.py` |

---

## 4. Module Reference

### 4.1 Core Modules

#### 4.1.1 `pipeline.py`

**Purpose:** Main pipeline orchestrator. Single entry point for full pipeline execution.

**Location:** `D:/thesis-shrinkin-villages/pipeline.py`

**Key Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `main` | `main(config_path: str)` | Execute full pipeline |
| `load_config` | `load_config(path: str) -> Dict` | Load YAML configuration |
| `ensure_dirs` | `ensure_dirs(paths: List[str])` | Create output directories |
| `write_json` | `write_json(path: str, obj: Dict)` | Write JSON manifest |

**Pipeline Steps:**
1. Load configuration from YAML
2. Validate boundaries and demographics files exist
3. Validate AOI files (optional)
4. Execute GEE monthly feature export
5. Compute local GLCM (if enabled)
6. Compute OSM building features (if enabled)
7. Join demographics by unit_id
8. Export final features table (CSV + Parquet)
9. Write run manifest

---

#### 4.1.2 `data_preprocessing/gee_monthly.py`

**Purpose:** Scalable satellite preprocessing via Google Earth Engine.

**Location:** `D:/thesis-shrinkin-villages/data_preprocessing/gee_monthly.py`

**Key Functions:**

| Function | Description |
|----------|-------------|
| `run_gee_monthly_feature_export(cfg)` | Main orchestration - monthly composites + zonal stats |
| `_process_single_month(ym, ctx)` | Process one month (parallelizable) |
| `_retry_with_backoff(func, max_retries, base_delay)` | Resilient retry wrapper |
| `_wait_for_task(task, poll_s)` | Poll GEE task completion |
| `_export_fc_to_asset(fc, asset_id, description)` | Export FeatureCollection to GEE asset |
| `_download_table_asset_csv(cfg, asset_id, out_csv_path)` | Download table asset as CSV |
| `_export_viirs_for_month(...)` | Monthly VIIRS aggregation |
| `_batch_feature_collection(fc, batch_size)` | Split FC for batch processing |
| `_add_glcm_texture(img, src_band, ...)` | Add GLCM bands in GEE (expensive) |
| `export_map_rasters(cfg, output_dir, months)` | Export visualization rasters |

**Data Sources:**
- `COPERNICUS/S2_SR_HARMONIZED` - Sentinel-2 Surface Reflectance
- `LANDSAT/LC08/C02/T1_L2` - Landsat 8 Collection 2 (registered)
- `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` - VIIRS monthly night-light

**Processing Logic:**
1. Initialize GEE with cloud project
2. Load boundary FeatureCollection from GEE asset
3. Load AOI geometry for filterBounds
4. For each month in date range:
   - Filter Sentinel-2 by date, AOI, cloud percentage
   - Apply cloud masking (SCL band)
   - Create monthly median composite
   - Compute spectral indices (NDVI, NDBI)
   - Run reduceRegions for zonal statistics
   - Export to CSV via temporary asset
5. Optionally export VIIRS separately and merge locally
6. Concatenate all months into single DataFrame

---

#### 4.1.3 `data_preprocessing/compute_glcm_local.py`

**Purpose:** Local GLCM texture computation using scikit-image.

**Location:** `D:/thesis-shrinkin-villages/data_preprocessing/compute_glcm_local.py`

**Key Functions:**

| Function | Description |
|----------|-------------|
| `run_local_glcm(cfg, features_df, boundaries_path, rasters_dir)` | Main orchestration |
| `download_monthly_raster_from_gee(cfg, ym, band, out_path, scale)` | Download raster from GEE |
| `compute_glcm_metrics(arr, metrics, distances, angles)` | Compute GLCM for 2D array |
| `_process_unit_glcm(unit_id, geometry, raster_path, metrics)` | Process single unit |
| `_quantize_to_uint8(arr, vmin, vmax)` | Quantize array for GLCM |
| `_check_dependencies()` | Verify rasterio + scikit-image |

**Algorithm:**
1. Download NDBI raster tiles from GEE (clipped to study area)
2. For each administrative unit:
   - Extract pixels within boundary using rasterio mask
   - Quantize to uint8 (GLCM requires integer input)
   - Compute GLCM using scikit-image `graycomatrix`
   - Extract texture properties using `graycoprops`
3. Merge GLCM features with main features table

**Supported Metrics:**
- `contrast` - Local intensity variation
- `homogeneity` (IDM) - Closeness to diagonal
- `entropy` - Randomness/disorder measure
- `asm` - Angular second moment
- `correlation` - Linear dependency of gray levels

---

#### 4.1.4 `data_preprocessing/osm_labels.py`

**Purpose:** Download OSM building footprints and compute built-up area features.

**Location:** `D:/thesis-shrinkin-villages/data_preprocessing/osm_labels.py`

**Key Functions:**

| Function | Description |
|----------|-------------|
| `run_osm_labels(cfg, units_gdf, output_dir)` | Main entry point |
| `download_osm_buildings_overpass(prefectures, output_path, timeout)` | Download via Overpass API |
| `load_osm_buildings(path)` | Load cached OSM GeoPackage |
| `compute_osm_features(buildings_gdf, units_gdf, unit_id_field)` | Compute features per unit |
| `_parse_osm_json_to_gdf(osm_json, pref_name)` | Convert Overpass JSON to GeoDataFrame |
| `_get_prefecture_bbox(pref_name)` | Get bounding box for prefecture |

**Computed Features:**
- `osm_built_area` - Total building footprint area (m²)
- `osm_building_count` - Number of buildings
- `osm_built_ratio` - Building area / unit area (density)

**Processing Logic:**
1. Check for cached buildings or download via Overpass API
2. Spatial join buildings to administrative units
3. Clip building polygons to unit boundaries
4. Sum clipped areas per unit
5. Compute building count and density ratio

---

### 4.2 Administrative Preprocessing Modules

#### 4.2.1 `admin_demographics/build_aoi.py`

**Purpose:** Create Area of Interest geometries for GEE filtering.

**Functions:**
- `build_aoi_full(boundaries_path, prefectures)` - Dissolve all boundaries
- `build_aoi_golden(boundaries_path, golden_unit_ids)` - Union of 10 selected units
- `write_provenance(...)` - Write provenance JSON

**Outputs:**
- `admin_demographics/aoi/aoi_full.gpkg`
- `admin_demographics/aoi/aoi_golden.gpkg`
- `admin_demographics/aoi/aoi_provenance.json`

---

#### 4.2.2 `admin_demographics/upload_aoi_to_gee.py`

**Purpose:** Upload AOI and boundary assets to Google Earth Engine.

**Functions:**
- `upload_aoi_to_gee(local_path, asset_id, description, simplify_tolerance)` - Upload AOI
- `upload_boundaries_to_gee(local_path, asset_id, description, prefectures)` - Upload boundaries
- `gdf_to_ee_fc(gdf, simplify_tolerance)` - Convert GeoDataFrame to EE FeatureCollection
- `wait_for_task(task, poll_interval, timeout)` - Poll task completion

**Created Assets:**
- `projects/ee-brodnow77/assets/aoi_full`
- `projects/ee-brodnow77/assets/aoi_golden`
- `projects/ee-brodnow77/assets/mura_jis_shp`
- `projects/ee-brodnow77/assets/aza_shp`

---

#### 4.2.3 `admin_demographics/prepare_boundaries.py`

**Purpose:** Standardize raw boundary inputs to consistent schema.

**Schema Output:**
- `unit_id` (string) - Stable join key (format: `level:pref:code`)
- `unit_level` (string) - `mura` or `aza`
- `pref_name` (string) - Prefecture name (English)
- `unit_code` (string) - Administrative code
- `geometry` - Polygon

---

#### 4.2.4 `admin_demographics/build_village_demographics_from_estat.py`

**Purpose:** Process Japanese Statistics Bureau (e-Stat) data.

**Handles:**
- Multiple character encodings (UTF-8 BOM, CP932, Shift-JIS)
- 5-digit municipality code normalization
- Prefecture code to name mapping

**Output Schema:**
- `unit_id`, `pref_name`, `unit_code`
- `pop_total`, `pop_male`, `pop_female`
- `households_total`, `households_per_person`
- Age group columns

---

### 4.3 Utility Scripts

#### 4.3.1 `scripts/rerun_glcm_only.py`

**Purpose:** Re-compute GLCM features on existing feature table.

**Usage:**
```bash
python scripts/rerun_glcm_only.py config/config.yaml [features_parquet_path]
```

**Workflow:**
1. Load existing features table (Parquet)
2. Drop existing GLCM columns
3. Call `run_local_glcm()` to recompute
4. Save updated table

---

## 5. Configuration Reference

### 5.1 Configuration File Structure

Configuration is defined in YAML format. The main configuration file is `config/config.yaml`.

```yaml
project:
  name: shrinking-villages-japan
  run_id: "run_001"
  outputs_dir: "outputs"

time:
  start: "2018-01-01"
  end: "2023-12-31"
  composite: "monthly"

gee:
  enabled: true
  cloud_project: "ee-brodnow77"
  export_prefix: "shrinking_villages"
  scale_m: 50
  boundaries_asset_id_mura: "projects/ee-brodnow77/assets/mura_jis_shp"
  boundaries_asset_id_aza: "projects/ee-brodnow77/assets/aza_shp"
  export_target: "asset"
  export_asset_folder: "projects/ee-brodnow77/assets/thesis_exports"
  keep_export_assets: false
  cloudy_pixel_percentage_max: 80
  use_batching: true
  batch_size: 500
  max_retries: 3
  max_parallel_months: 6

data:
  boundaries_path: "admin_demographics/boundaries/mura_jis.gpkg"
  boundaries_layer: "mura"
  unit_id_field: "unit_id"
  demographics_path: "admin_demographics/demographics/village_demographics.csv"
  demographics_unit_id_field: "unit_id"

features:
  compute_ndvi: true
  compute_ndbi: true
  compute_glcm: false              # Disabled in GEE (too expensive)
  compute_glcm_local: true         # Compute locally instead
  glcm_restrict_to_golden: true    # Restrict GLCM rasters to golden AOI
  include_viirs: true
  glcm_source_s2: "NDBI"
  glcm_size: 3
  glcm_metrics: [contrast, entropy, homogeneity]
  glcm_scale: 100

labels:
  source: "osm"
  osm_buildings_path: null         # null = download on demand

outputs:
  features_table_csv: "outputs/final/features_table.csv"
  features_table_parquet: "outputs/final/features_table.parquet"
  run_manifest_json: "outputs/final/run_manifest.json"
  export_map_rasters: true
  map_rasters_dir: "outputs/rasters/map_figures"

study_area:
  prefectures:
    - Aomori
    - Akita

aoi:
  mode: "full"                     # "full" or "golden"
  aoi_full_path: "admin_demographics/aoi/aoi_full.gpkg"
  aoi_golden_path: "admin_demographics/aoi/aoi_golden.gpkg"
  aoi_full_asset_id: "projects/ee-brodnow77/assets/aoi_full"
  aoi_golden_asset_id: "projects/ee-brodnow77/assets/aoi_golden"
  golden_unit_ids:
    - "mura:Aomori:02423"
    - "mura:Aomori:02387"
    - "mura:Aomori:02201"
    - "mura:Aomori:02405"
    - "mura:Aomori:02443"
    - "mura:Akita:05303"
    - "mura:Akita:05202"
    - "mura:Akita:05363"
    - "mura:Akita:05215"
    - "mura:Akita:05207"

run_mode:
  unit_level: "mura"               # "mura" or "aza"
  fast_dev: false
  months_override: null            # null = use time range, or list
  unit_sample_n: 0                 # 0 = all units
  skip_existing_month_csv: true
  strict_no_empty_month: false
```

### 5.2 Configuration Parameters

#### `project` Section

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Project identifier |
| `run_id` | string | Unique run identifier |
| `outputs_dir` | string | Output directory path |

#### `time` Section

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | string | Start date (YYYY-MM-DD) |
| `end` | string | End date (YYYY-MM-DD) |
| `composite` | string | Aggregation interval ("monthly") |

#### `gee` Section

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled` | boolean | Enable GEE processing |
| `cloud_project` | string | GCP project ID for GEE |
| `scale_m` | integer | Export resolution in meters |
| `cloudy_pixel_percentage_max` | integer | Max cloud cover filter (%) |
| `use_batching` | boolean | Enable batch processing |
| `batch_size` | integer | Features per batch |
| `max_parallel_months` | integer | Concurrent month processing |

#### `features` Section

| Parameter | Type | Description |
|-----------|------|-------------|
| `compute_ndvi` | boolean | Compute NDVI index |
| `compute_ndbi` | boolean | Compute NDBI index |
| `compute_glcm` | boolean | Compute GLCM in GEE (expensive) |
| `compute_glcm_local` | boolean | Compute GLCM locally |
| `include_viirs` | boolean | Include VIIRS night-light |
| `glcm_metrics` | list | GLCM metrics to compute |
| `glcm_scale` | integer | GLCM raster resolution |

#### `run_mode` Section

| Parameter | Type | Description |
|-----------|------|-------------|
| `unit_level` | string | "mura" or "aza" |
| `months_override` | list/null | Override month list |
| `unit_sample_n` | integer | Limit to N units (0 = all) |
| `skip_existing_month_csv` | boolean | Skip existing exports |

---

## 6. Data Processing Pipeline

### 6.1 Stage 1: Data Validation

**Input:** Configuration file path

**Process:**
- Load YAML configuration
- Validate boundaries file exists
- Validate demographics file exists
- Validate AOI files (optional warning)
- Create output directories

**Output:** Initialized pipeline state

---

### 6.2 Stage 2: GEE Satellite Preprocessing

**Input:**
- Configuration with time range and study area
- Boundary assets in GEE
- AOI asset for filterBounds

**Process:**
1. Initialize Earth Engine with cloud project
2. Load boundary FeatureCollection from GEE
3. Filter by prefectures
4. Load Sentinel-2 collection:
   - Filter by date range
   - Filter by AOI (filterBounds)
   - Filter by cloud percentage (< 80%)
   - Select bands: B2, B3, B4, B8, B11, SCL
5. Apply cloud masking via SCL band:
   - Keep: vegetation (4), bare (5), water (6), snow (11)
6. For each month:
   - Create median composite
   - Compute NDVI: (B8 - B4) / (B8 + B4)
   - Compute NDBI: (B11 - B8) / (B11 + B8)
   - Run `reduceRegions` with mean reducer
   - Export to temporary asset
   - Download as CSV
7. Process in parallel with ThreadPoolExecutor

**Output:** Monthly CSV files with zonal statistics

---

### 6.3 Stage 3: VIIRS Aggregation (Optional)

**Input:** VIIRS monthly collection, boundary FC

**Process:**
1. Filter VIIRS to month
2. Compute mean image
3. Run `reduceRegions` with mean reducer at 500m scale
4. Export to temporary asset
5. Download and merge with S2 features locally

**Output:** `viirs_mean` column in features table

---

### 6.4 Stage 4a: Local GLCM Computation

**Input:**
- Configuration with GLCM settings
- Features DataFrame from GEE
- Boundaries GeoPackage

**Process:**
1. Download NDBI raster tiles from GEE (clipped to AOI)
2. For each administrative unit:
   - Mask raster to unit geometry using rasterio
   - Quantize to uint8 (scale -1,1 to 0,255)
   - Compute GLCM matrix (scikit-image)
   - Extract texture metrics (contrast, entropy, homogeneity)
3. Merge with features table on unit_id + month

**Output:** GLCM columns (S2_NDBI_contrast, S2_NDBI_entropy, etc.)

---

### 6.4b Stage 4b: OSM Building Features

**Input:**
- Configuration with OSM settings
- Unit boundaries GeoDataFrame

**Process:**
1. Download buildings via Overpass API (or load cache)
2. Reproject to JGD2011 (EPSG:6675) for area calculations
3. Spatial join buildings to units
4. Clip buildings to unit boundaries
5. Sum clipped areas per unit
6. Compute building count and density ratio

**Output:** OSM columns (osm_built_area, osm_building_count, osm_built_ratio)

---

### 6.5 Stage 5: Aggregation

Aggregation is performed in GEE via `reduceRegions`. The table is already at unit level.

---

### 6.6 Stage 6: Demographics Join

**Input:**
- Features DataFrame
- Demographics CSV

**Process:**
1. Load demographics with string dtypes for codes
2. Normalize join keys (strip whitespace)
3. For mura level: direct join on unit_id
4. For aza level: extract municipality code (first 5 digits), join on mura unit_id
5. Validate join success (< 95% NaN threshold)

**Output:** Features DataFrame with demographic columns

---

### 6.7 Stage 7: Export

**Process:**
1. Normalize string columns (unit_id, unit_code)
2. Zero-pad unit_code for mura level
3. Export to CSV
4. Export to Parquet
5. Write run manifest JSON

**Outputs:**
- `outputs/final/features_table.csv`
- `outputs/final/features_table.parquet`
- `outputs/final/run_manifest.json`

---

## 7. Input Data Requirements

### 7.1 Boundaries (GeoPackage)

**Path:** `admin_demographics/boundaries/mura_jis.gpkg`

**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `unit_id` | string | Primary key (format: `level:pref:code`) |
| `unit_level` | string | "mura" or "aza" |
| `pref_name` | string | Prefecture name (English) |
| `unit_code` | string | Administrative code |
| `geometry` | Polygon | Unit boundary |

**Example unit_id:** `mura:Aomori:02201`

---

### 7.2 Demographics (CSV)

**Path:** `admin_demographics/demographics/village_demographics.csv`

**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `unit_id` | string | Join key (matches boundaries) |
| `pop_total` | integer | Total population |
| `pop_male` | integer | Male population |
| `pop_female` | integer | Female population |
| `households_total` | integer | Number of households |

**Optional Columns:** Age group columns (e.g., `age_0_14`, `age_65_plus`)

---

### 7.3 AOI (GeoPackage)

**Paths:**
- `admin_demographics/aoi/aoi_full.gpkg` - Full study area
- `admin_demographics/aoi/aoi_golden.gpkg` - Golden sample subset

**Contents:** Single dissolved geometry for filterBounds operations.

---

## 8. Output Data Schema

### 8.1 Features Table Columns

#### Identifiers
| Column | Type | Description |
|--------|------|-------------|
| `unit_id` | string | Primary key |
| `unit_level` | string | "mura" or "aza" |
| `unit_code` | string | Administrative code |
| `pref_name` | string | Prefecture name |
| `month` | string | Year-month (YYYY-MM) |

#### Sentinel-2 Spectral Bands
| Column | Type | Description |
|--------|------|-------------|
| `B2` | float | Blue (490 nm) |
| `B3` | float | Green (560 nm) |
| `B4` | float | Red (665 nm) |
| `B8` | float | NIR (842 nm) |
| `B11` | float | SWIR (1610 nm) |

#### Computed Indices
| Column | Type | Description |
|--------|------|-------------|
| `NDVI` | float | Normalized Difference Vegetation Index |
| `NDBI` | float | Normalized Difference Built-up Index |

#### GLCM Texture (Optional)
| Column | Type | Description |
|--------|------|-------------|
| `S2_NDBI_contrast` | float | GLCM contrast |
| `S2_NDBI_entropy` | float | GLCM entropy |
| `S2_NDBI_homogeneity` | float | GLCM homogeneity |

#### VIIRS Night-Light (Optional)
| Column | Type | Description |
|--------|------|-------------|
| `viirs_mean` | float | Mean monthly radiance |

#### OSM Building Features (Optional)
| Column | Type | Description |
|--------|------|-------------|
| `osm_built_area` | float | Total building area (m²) |
| `osm_building_count` | integer | Number of buildings |
| `osm_built_ratio` | float | Built-up density (0-1) |

#### Demographics
| Column | Type | Description |
|--------|------|-------------|
| `pop_total` | integer | Total population |
| `pop_male` | integer | Male population |
| `pop_female` | integer | Female population |
| `households_total` | integer | Number of households |
| `age_*` | integer | Age group columns |

### 8.2 Run Manifest Schema

```json
{
  "project": {
    "name": "shrinking-villages-japan",
    "run_id": "run_001",
    "outputs_dir": "outputs"
  },
  "time": {
    "start": "2018-01-01",
    "end": "2023-12-31",
    "composite": "monthly"
  },
  "features": {
    "compute_ndvi": true,
    "compute_ndbi": true,
    "include_viirs": true
  },
  "aoi": {
    "mode": "full",
    "path": "admin_demographics/aoi/aoi_full.gpkg"
  },
  "started_utc": "2026-02-03T10:00:00Z",
  "finished_utc": "2026-02-03T12:30:00Z",
  "row_count": 4680,
  "steps": [
    {"step": "data_acquisition_hooks", "status": "ok", "ts_utc": "..."},
    {"step": "gee_optical_preprocessing_monthly_composites", "status": "ok", "ts_utc": "..."},
    {"step": "local_glcm_computation", "status": "ok", "ts_utc": "..."},
    {"step": "osm_label_computation", "status": "ok", "ts_utc": "..."},
    {"step": "feature_computation", "status": "ok", "ts_utc": "..."}
  ]
}
```

---

## 9. Dependencies and Requirements

### 9.1 Core Dependencies (requirements.txt)

```
rasterio==1.5.0              # Raster I/O
scikit-learn==1.8.0          # Machine learning utilities
tensorflow==2.20.0           # Deep learning framework
earthengine-api==1.7.10      # Google Earth Engine
numpy==2.1.3                 # Numerical computing
pandas==2.2.3                # Data manipulation
pyyaml==6.0.2                # YAML parsing
pyarrow==18.1.0              # Parquet I/O
geopandas==1.0.1             # Geospatial data frames
shapely==2.0.6               # Geometry operations
pyproj==3.7.0                # Coordinate transformations
google-cloud-storage==2.19.0 # GCS integration (optional)
```

### 9.2 Optional Dependencies

```
scikit-image                 # Required for local GLCM computation
```

### 9.3 System Requirements

| Requirement | Specification |
|-------------|---------------|
| Python | 3.8+ |
| GEE Account | Authenticated (`earthengine authenticate`) |
| Network | Access to GEE API, Overpass API |
| Disk | ~5-10 GB for full run outputs |

### 9.4 Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: texture analysis
pip install scikit-image

# Authenticate GEE
earthengine authenticate
```

---

## 10. Execution Workflows

### 10.1 Full Pipeline Execution

```bash
python pipeline.py config/config.yaml
```

**Expected Runtime:** 2-6 hours (depends on date range and number of units)

**Outputs:**
- `outputs/final/features_table.csv`
- `outputs/final/features_table.parquet`
- `outputs/final/run_manifest.json`

---

### 10.2 Golden Sample Validation

```bash
# Using pytest
python -m pytest tests/test_golden_sample.py -v

# Direct execution
python tests/test_golden_sample.py
```

**Configuration:** `config/config_golden_sample.yaml`
- 10 units × 2 months = 20 rows
- Runtime: ~5-10 minutes

---

### 10.3 Administrative Setup Workflow

```bash
# 1. Prepare boundaries from raw data
python admin_demographics/prepare_boundaries.py

# 2. Build AOI geometries
python admin_demographics/build_aoi.py config/config.yaml

# 3. Build demographics table
python admin_demographics/build_village_demographics_from_estat.py

# 4. Export for GEE upload
python admin_demographics/export_for_gee.py

# 5. Upload to GEE
python admin_demographics/upload_aoi_to_gee.py config/config.yaml
```

---

### 10.4 Re-compute GLCM Only

```bash
python scripts/rerun_glcm_only.py config/config.yaml outputs/final/features_table.parquet
```

---

## 11. Testing and Validation

### 11.1 Test Suite

**Location:** `tests/test_golden_sample.py`

**Test Cases:**

| Test | Description |
|------|-------------|
| `test_pipeline_runs_successfully` | Pipeline completes without errors |
| `test_output_files_exist` | CSV, Parquet, manifest created |
| `test_row_count` | Exactly 20 rows (10 units × 2 months) |
| `test_required_columns` | unit_id, month, NDVI, NDBI, viirs_mean present |
| `test_spectral_bands_present` | B2, B3, B4, B8, B11 present |
| `test_no_all_nan_features` | Feature columns not entirely NaN |
| `test_osm_features_present` | OSM columns present |
| `test_glcm_features_present` | GLCM columns present (if enabled) |
| `test_manifest_valid` | Valid JSON with expected fields |
| `test_unique_unit_month_pairs` | No duplicate unit-month rows |

### 11.2 Running Tests

```bash
# Full test suite
python -m pytest tests/ -v

# Single test
python -m pytest tests/test_golden_sample.py::test_row_count -v

# With coverage
python -m pytest tests/ --cov=data_preprocessing --cov-report=html
```

---

## 12. Data Provenance and Compliance

### 12.1 Data Sources

| Source | GEE Collection ID | License |
|--------|-------------------|---------|
| Sentinel-2 | `COPERNICUS/S2_SR_HARMONIZED` | Copernicus Open Access |
| Landsat 8 | `LANDSAT/LC08/C02/T1_L2` | USGS Public Domain |
| VIIRS | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | NOAA Open Data |
| OSM Buildings | Overpass API | ODbL |
| Demographics | e-Stat | Statistics Bureau of Japan |

### 12.2 Compliance Requirements

**Copernicus (Sentinel):**
- Free use under Copernicus regulation
- Maintain complete documentation of data sources
- Record collection ID, time range, processing in manifest

**USGS (Landsat):**
- Public domain
- Maintain proper attribution

**Privacy:**
- All outputs aggregated at village level
- No person-level inference
- No outputs below mura/aza aggregation

### 12.3 Provenance Recording

For each pipeline run, the following provenance is recorded:

1. **Run manifest** (`run_manifest.json`):
   - Time range
   - Enabled features
   - GEE dataset IDs
   - Processing steps and timestamps

2. **AOI provenance** (`aoi_provenance.json`):
   - Source boundaries
   - Unit counts
   - Creation timestamp

3. **Git versioning:**
   - All code changes tracked
   - Configuration changes versioned

---

## 13. API Reference

### 13.1 Main Entry Points

```python
# Full pipeline
from pipeline import main
main("config/config.yaml")

# GEE monthly export
from data_preprocessing.gee_monthly import run_gee_monthly_feature_export
df = run_gee_monthly_feature_export(cfg)

# Local GLCM
from data_preprocessing.compute_glcm_local import run_local_glcm
df = run_local_glcm(cfg, features_df, boundaries_path, rasters_dir)

# OSM features
from data_preprocessing.osm_labels import run_osm_labels
osm_df = run_osm_labels(cfg, units_gdf, output_dir)
```

### 13.2 Configuration Loading

```python
import yaml

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config("config/config.yaml")
```

### 13.3 Earth Engine Initialization

```python
import ee

# With cloud project
cloud_project = cfg["gee"]["cloud_project"]
ee.Initialize(project=cloud_project)

# Default project
ee.Initialize()
```

### 13.4 GeoDataFrame I/O

```python
import geopandas as gpd

# Read boundaries
gdf = gpd.read_file("admin_demographics/boundaries/mura_jis.gpkg")

# Filter by prefecture
gdf = gdf[gdf["pref_name"].isin(["Aomori", "Akita"])]

# Save
gdf.to_file("output.gpkg", driver="GPKG")
```

---

## Appendix A: Golden Sample Units

| Unit ID | Prefecture | Code | Area (km²) | Notes |
|---------|------------|------|------------|-------|
| mura:Aomori:02423 | Aomori | 02423 | 52 | Northernmost |
| mura:Aomori:02387 | Aomori | 02387 | 216 | Central |
| mura:Aomori:02201 | Aomori | 02201 | 887 | Largest (Aomori) |
| mura:Aomori:02405 | Aomori | 02405 | 84 | Small |
| mura:Aomori:02443 | Aomori | 02443 | 242 | Southernmost |
| mura:Akita:05303 | Akita | 05303 | 202 | Northernmost |
| mura:Akita:05202 | Akita | 05202 | 444 | Central |
| mura:Akita:05363 | Akita | 05363 | 18 | Smallest |
| mura:Akita:05215 | Akita | 05215 | 1093 | Largest |
| mura:Akita:05207 | Akita | 05207 | 790 | Southernmost |

---

## Appendix B: File Locations Summary

| Category | Path |
|----------|------|
| Main entrypoint | `pipeline.py` |
| Production config | `config/config.yaml` |
| Test config | `config/config_golden_sample.yaml` |
| GEE processing | `data_preprocessing/gee_monthly.py` |
| GLCM processing | `data_preprocessing/compute_glcm_local.py` |
| OSM processing | `data_preprocessing/osm_labels.py` |
| Boundary prep | `admin_demographics/prepare_boundaries.py` |
| AOI building | `admin_demographics/build_aoi.py` |
| GEE upload | `admin_demographics/upload_aoi_to_gee.py` |
| Tests | `tests/test_golden_sample.py` |
| Output CSV | `outputs/final/features_table.csv` |
| Output Parquet | `outputs/final/features_table.parquet` |
| Run manifest | `outputs/final/run_manifest.json` |

---

## Appendix C: Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| GEE authentication error | Token expired | Run `earthengine authenticate` |
| Missing boundaries file | Not created | Run `prepare_boundaries.py` |
| GLCM import error | Missing scikit-image | `pip install scikit-image` |
| GEE timeout | Large batch size | Reduce `batch_size` in config |
| Empty features table | No images in date range | Check cloud cover, date range |
| Demographics join fails | Mismatched unit_id | Verify unit_id format matches |

### Debugging Commands

```bash
# Check GEE authentication
earthengine authenticate --quiet

# List GEE assets
earthengine asset info projects/ee-brodnow77/assets/aoi_full

# Inspect GeoPackage
python admin_demographics/check_gpkg.py

# Run with verbose logging
python pipeline.py config/config.yaml 2>&1 | tee run.log
```

---

*Document generated: 2026-02-03*
*Repository: thesis-shrinking-villages*
*Version: 1.0*
