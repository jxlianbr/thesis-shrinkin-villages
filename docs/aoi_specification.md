# AOI Specification

This document describes the Area of Interest (AOI) geometries used in the shrinking villages analysis pipeline.

## Overview

The pipeline uses two AOI geometries for different purposes:

- **AOI_FULL**: The complete study area covering all Aomori and Akita prefectures
- **AOI_GOLDEN**: A subset of ~10 representative units for validation and testing

## AOI_FULL

**Definition**: Dissolved union of all municipal (mura) or sub-municipal (aza) boundaries in Aomori and Akita prefectures.

**Purpose**:
- Production runs: Filters satellite imagery to the full study area
- Reduces GEE computation by limiting spatial extent

**Source**: `admin_demographics/boundaries/mura_jis.gpkg` or `aza.gpkg`

## AOI_GOLDEN

**Definition**: Union of 10 selected administrative units spanning both prefectures.

**Purpose**:
- Fast validation runs (~5 minutes vs hours for full pipeline)
- Restricts GLCM raster downloads to a small area
- Generates visualization rasters for map figures
- End-to-end testing without full GEE computation costs

**Selected Units**:

### Aomori Prefecture (5 units)
| Unit ID | Unit Code | Area (km2) | Notes |
|---------|-----------|------------|-------|
| mura:Aomori:02423 | 02423 | 52 | Northernmost |
| mura:Aomori:02387 | 02387 | 216 | Central |
| mura:Aomori:02201 | 02201 | 887 | Largest |
| mura:Aomori:02405 | 02405 | 84 | Small |
| mura:Aomori:02443 | 02443 | 242 | Southernmost |

### Akita Prefecture (5 units)
| Unit ID | Unit Code | Area (km2) | Notes |
|---------|-----------|------------|-------|
| mura:Akita:05303 | 05303 | 202 | Northernmost |
| mura:Akita:05202 | 05202 | 444 | Central |
| mura:Akita:05363 | 05363 | 18 | Smallest |
| mura:Akita:05215 | 05215 | 1093 | Largest |
| mura:Akita:05207 | 05207 | 790 | Southernmost |

**Selection Criteria**:
- Geographic spread: North-to-south distribution within each prefecture
- Size variety: Range from 18 km2 to 1093 km2
- Both prefectures represented equally (5 + 5)

## File Locations

### Local Files
- `admin_demographics/aoi/aoi_full.gpkg` - Full study area geometry
- `admin_demographics/aoi/aoi_golden.gpkg` - Golden sample geometry
- `admin_demographics/aoi/aoi_provenance.json` - Metadata and provenance

### GEE Assets
- `projects/ee-brodnow77/assets/aoi_full` - Uploaded AOI_FULL
- `projects/ee-brodnow77/assets/aoi_golden` - Uploaded AOI_GOLDEN

## Usage

### Generating AOI Files

```bash
# Generate local AOI files from boundaries
python admin_demographics/build_aoi.py config/config.yaml
```

### Uploading to GEE

```bash
# Upload local files to GEE as FeatureCollection assets
python admin_demographics/upload_aoi_to_gee.py config/config.yaml
```

### Configuration

The pipeline is controlled via the `aoi` section in config:

```yaml
aoi:
  mode: "full"  # or "golden" for validation runs
  aoi_full_path: "admin_demographics/aoi/aoi_full.gpkg"
  aoi_golden_path: "admin_demographics/aoi/aoi_golden.gpkg"
  aoi_full_asset_id: "projects/ee-brodnow77/assets/aoi_full"
  aoi_golden_asset_id: "projects/ee-brodnow77/assets/aoi_golden"
  golden_unit_ids:
    - "mura:Aomori:02423"
    # ... (10 total)
```

### GLCM Restriction

When `features.glcm_restrict_to_golden: true`, GLCM raster downloads are limited to AOI_GOLDEN even when running in full mode. This prevents large raster downloads while still computing GLCM for golden sample units.

## Provenance

The `aoi_provenance.json` file contains:
- `created_utc`: Timestamp of AOI generation
- `source_boundaries`: Path to source boundary file
- `aoi_full.unit_count`: Number of units dissolved into AOI_FULL
- `aoi_golden.unit_ids`: Explicit list of golden sample unit IDs

## Regenerating AOIs

To regenerate AOIs (e.g., after boundary updates):

1. Update `golden_unit_ids` in config if needed
2. Run `build_aoi.py`
3. Run `upload_aoi_to_gee.py`
4. Verify assets exist: `earthengine asset info projects/ee-brodnow77/assets/aoi_full`
