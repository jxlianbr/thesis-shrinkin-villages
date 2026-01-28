# Data sources, licensing, and provenance (Task 2)

## Purpose
This document freezes the data provenance and compliance rules for the thesis pipeline and provides a single place to audit:
- which datasets are used and how they are accessed,
- what usage/licensing constraints apply,
- how privacy is preserved through aggregation,
- what metadata must be recorded for reproducibility.

## Reproducibility backbone
- Primary implementation language: **Python** (libraries explicitly used in the implementation: `rasterio`, `scikit-learn`, `TensorFlow`).  
- Scalable satellite-archive access and preprocessing: **Google Earth Engine (GEE)**.  
- All processing steps and documents are versioned in **Git** to ensure traceability and reproducibility.  
- Each pipeline run writes a machine-readable provenance artifact:
  - `outputs/final/run_manifest.json` (run ID, time range, enabled features, and processing steps).

## Spatial scale and privacy constraints
- All analytical outputs remain aggregated at **village / sub-municipal** unit level (**mura / aza**).
- Administrative boundaries follow **Statistics Bureau of Japan** guidance to ensure privacy-compliant aggregation.
- The pipeline avoids finer-than-required spatial aggregation and follows a data-minimization principle to reduce re-identification risk when combining indicators.
- Demographic inputs are used at the same aggregation level as the spatial units used for analysis.

## Dataset inventory (authoritative list)

### Remote sensing (GEE)
The pipeline uses the following GEE collection IDs (as configured in `config/config.yaml`):

1. **Sentinel-2 Surface Reflectance (harmonized)**
- GEE collection ID: `COPERNICUS/S2_SR_HARMONIZED`
- Role: optical time series for monthly composites and derived indices (NDVI/NDBI).

2. **Landsat 8 Collection 2 Level-2 (registered/available)**
- GEE collection ID: `LANDSAT/LC08/C02/T1_L2`
- Role (Task 2): registered as a primary optical source and verified accessible in the workflow.
- Note: integration/fusion into the exported feature table is a later implementation step.

3. **VIIRS Day/Night Band monthly (optional)**
- GEE collection ID: `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`
- Role: optional monthly night-light indicator (`viirs_mean`) as a proxy for human activity dynamics.

### Administrative boundaries (local + EE assets)
- Analysis units: **mura** and **aza**.
- Local working copies are stored under `admin_demographics/boundaries/` (GeoPackage).
- Earth Engine assets are used for scalable zonal aggregation (FeatureCollection assets).
- Required properties on boundary features:
  - `unit_id` (stable join key used across the pipeline)
  - `unit_level` (`mura` or `aza`)
  - `pref_name`
  - `unit_code`

### Demographic statistics (local)
- Source category: official Japanese statistical authority (village-level administrative statistics).
- Storage:
  - raw downloads: `admin_demographics/demographics_raw/`
  - cleaned/standardized table used by pipeline: `admin_demographics/demographics/village_demographics.csv`
- Join key:
  - `unit_id` (same format as boundaries, e.g. `mura:Akita:05201`)

## Licensing and usage conditions (compliance rules)

### Copernicus / Sentinel
- The use of Copernicus remote-sensing data is treated as **free use** under the applicable Copernicus regulation, with a strict requirement to maintain **complete documentation** of data sources and usage conditions in the repository.
- Compliance action in this repo:
  - Keep dataset identifiers (GEE collection ID), time range, spatial scope, and processing configuration in `run_manifest.json`.
  - Keep this document under version control.

### USGS / Landsat
- Landsat data usage is governed by **USGS usage/licensing conditions**.
- Compliance action in this repo:
  - Record Landsat dataset identifiers (GEE collection ID), time range, spatial scope, and processing configuration in `run_manifest.json`.
  - Maintain a clear attribution statement in the thesis and/or repository documentation consistent with USGS requirements.

### Official demographic statistics
- Use only village-level/sub-municipal aggregated statistics and keep them aligned to the boundary authority used for aggregation.
- Compliance action in this repo:
  - Store raw inputs separately from cleaned outputs.
  - Preserve a record of the exact table(s) used via the Provenance Checklist below.

## Provenance checklist (fill for each dataset)
For reproducibility, complete these fields once per dataset and update if the source changes.

### Sentinel-2 SR Harmonized
- Dataset name:
- Provider:
- Access method: GEE
- GEE collection ID: `COPERNICUS/S2_SR_HARMONIZED`
- Temporal coverage used (start/end):
- Spatial scope (prefectures):
- Key preprocessing decisions (cloud mask, compositing):
- Derived variables exported:
- Retrieval date (local time):
- Notes:

### Landsat 8 C2 L2 (registered)
- Dataset name:
- Provider:
- Access method: GEE
- GEE collection ID: `LANDSAT/LC08/C02/T1_L2`
- Temporal coverage used (start/end):
- Spatial scope (prefectures):
- Retrieval date (local time):
- Notes (registered vs fused):

### VIIRS DNB Monthly (optional)
- Dataset name:
- Provider:
- Access method: GEE
- GEE collection ID: `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`
- Temporal coverage used (start/end):
- Spatial scope (prefectures):
- Aggregation variable(s):
- Retrieval date (local time):
- Notes:

### Administrative boundaries (mura / aza)
- Boundary authority alignment: Statistics Bureau of Japan guidance
- Local file(s):
- EE asset IDs:
- Required key fields present (`unit_id`, `unit_level`, `pref_name`, `unit_code`): yes/no
- Retrieval/source date:
- Notes:

### Demographic statistics (village-level)
- Statistical authority:
- Table IDs / dataset identifiers:
- Download format/encoding:
- Local raw files:
- Cleaned output file:
- Join key definition:
- Retrieval date:
- Notes:

## Where provenance is recorded in code outputs
- `outputs/final/run_manifest.json` records:
  - time range,
  - enabled features (NDVI/NDBI/VIIRS),
  - GEE dataset IDs used/registered,
  - run timestamps and step status.

## Non-negotiable rules
- No individual-level data is processed.
- No outputs are generated below the mura/aza aggregation level.
- Any change in dataset, collection ID, or preprocessing logic requires:
  - a Git commit,
  - an updated manifest (new run),
  - an update to this document if the source/provenance changed.


  # Data provenance and licensing (Task 2/3)

## Scope and processing layer
All large-scale satellite preprocessing and monthly compositing are executed in Google Earth Engine (GEE) to enable scalable processing of multi-year optical time series. The workflow is versioned with Git to ensure reproducibility. :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}

## Remote sensing datasets
### Sentinel-2 MSI (optical, primary)
- Role: optical time series for settlement-related indicators and monthly composites.
- Processing intent: atmospheric correction using Sen2Cor and cloud masking (FMask), followed by resampling to a uniform 10 m resolution and monthly compositing to reduce seasonal effects and outliers. :contentReference[oaicite:12]{index=12}

### Landsat 8 OLI/TIRS (optical, secondary)
- Role: complementary optical time series alongside Sentinel-2, aggregated into monthly composites and resampled to 10 m for harmonized feature extraction. :contentReference[oaicite:13]{index=13}

### VIIRS-DNB (night lights, optional)
- Role: monthly aggregation of night light intensity as an indicator of human activity. :contentReference[oaicite:14]{index=14}

## Administrative boundaries and demographics
- Analysis units: village (mura) and sub-municipal units (aza), aligned to the Statistics Bureau of Japan administrative boundary definitions. :contentReference[oaicite:15]{index=15}
- Demographic tables: village-level population and age-structure indicators are joined deterministically to the remote-sensing feature table using stable unit identifiers.

## Licensing and compliance
- Copernicus data usage is handled under the Copernicus data policy; the workflow maintains complete documentation of datasets and usage conditions. :contentReference[oaicite:16]{index=16}
- The project explicitly tracks compliance with the licensing conditions of Copernicus and USGS data sources. :contentReference[oaicite:17]{index=17} :contentReference[oaicite:18]{index=18}
- Outputs are aggregated at village level, preventing inference about individual persons. :contentReference[oaicite:19]{index=19}

## Provenance recording
For each pipeline run:
- configuration snapshot (YAML),
- produced monthly feature tables,
- merged final feature table (CSV/Parquet),
- run manifest JSON describing the processing steps and enabled sources
are written to the outputs directory.
s
