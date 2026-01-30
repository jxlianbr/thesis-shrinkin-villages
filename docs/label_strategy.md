# Label Source Strategy: OSM Building Footprints

## Overview

We use OpenStreetMap (OSM) building footprints as the primary label source for built-up area detection in shrinking villages. This document describes the data source, features derived, and rationale for this choice.

## Data Source

- **Provider**: OpenStreetMap via Overpass API
- **Coverage**: Japan-wide (Tohoku region subset for Aomori/Akita prefectures)
- **Format**: GeoPackage (.gpkg) with building polygons
- **Update frequency**: Real-time (Overpass API) or weekly snapshots (Geofabrik)
- **License**: Open Database License (ODbL)

### Download Method

The pipeline automatically downloads building footprints using the Overpass API:

```python
from data_preprocessing.osm_labels import download_osm_buildings_overpass

buildings_gdf = download_osm_buildings_overpass(
    prefectures=["Aomori", "Akita"],
    output_path="outputs/osm/osm_buildings.gpkg"
)
```

For larger areas or repeated runs, pre-download from Geofabrik:
- https://download.geofabrik.de/asia/japan.html

## Features Derived

For each administrative unit (mura/aza), we compute:

| Feature | Description | Units |
|---------|-------------|-------|
| `osm_built_area` | Total building footprint area within unit | m² |
| `osm_building_count` | Number of buildings within unit | count |
| `osm_built_ratio` | Building area / unit area (built-up density) | ratio (0-1) |

### Computation Method

1. **Spatial Join**: Buildings are intersected with unit boundaries
2. **Clipping**: Building footprints are clipped to unit boundaries to avoid double-counting
3. **Aggregation**: Clipped areas are summed per unit

```python
from data_preprocessing.osm_labels import compute_osm_features

osm_features = compute_osm_features(
    buildings_gdf=buildings,
    units_gdf=units,
    unit_id_field="unit_id"
)
```

## Why OSM?

### Advantages

1. **Free and open data** with permissive license (ODbL)
2. **Good coverage** in Japan, especially for buildings
3. **Vector data** provides actual building shapes (not raster proxies)
4. **Historical extracts** available for temporal analysis
5. **Community maintained** with ongoing updates
6. **Standardized schema** across regions

### Comparison with Alternatives

| Source | Resolution | Coverage | Access | Notes |
|--------|-----------|----------|--------|-------|
| **OSM** | Vector | Good in Japan | Free | Our choice |
| GHS-BUILT | 30m raster | Global | Free | Less precise for rural areas |
| Japanese national data | High | Complete | Restricted | Better quality but limited access |
| Commercial (Google, etc.) | Varies | Complete | Paid | Expensive for research |

## Limitations

### Coverage Variability
- Urban areas have better coverage than rural
- Remote villages may have incomplete building data
- Coverage depends on local mapping activity

### Temporal Consistency
- OSM is continuously updated (no fixed timestamp)
- Historical analysis requires archived extracts
- Building additions/deletions may not match real-world timing

### Data Quality
- May undercount very small structures
- Some buildings may be missing in less-mapped areas
- Building types (residential/commercial) not always tagged

## Validation Approach

To assess OSM data quality for your study area:

1. **Visual inspection**: Compare with satellite imagery in QGIS/Google Earth
2. **Cross-validation**: Compare with census building counts if available
3. **Completeness check**: Look for systematic gaps in rural areas

## Configuration

Enable OSM labels in `config.yaml`:

```yaml
labels:
  source: "osm"
  osm_buildings_path: null  # null = download on demand
```

Or use a pre-downloaded file:

```yaml
labels:
  source: "osm"
  osm_buildings_path: "data/osm/japan_buildings.gpkg"
```

## Output Schema

The final features table includes these OSM-derived columns:

```
unit_id,month,NDVI,NDBI,...,osm_built_area,osm_building_count,osm_built_ratio
mura:Aomori:02201,2020-01,0.45,0.12,...,125000.5,342,0.0023
mura:Aomori:02202,2020-01,0.52,0.08,...,89500.2,215,0.0015
```

## References

- OpenStreetMap Wiki: https://wiki.openstreetmap.org/wiki/Key:building
- Overpass API: https://overpass-api.de/
- Geofabrik Downloads: https://download.geofabrik.de/
- ODbL License: https://opendatacommons.org/licenses/odbl/
