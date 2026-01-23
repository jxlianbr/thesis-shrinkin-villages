# Method Implementation Record (Task 1)

## Execution environments
- Primary implementation environment: Python.
- Core Python libraries used for the implementation: rasterio, scikit-learn, TensorFlow. :contentReference[oaicite:9]{index=9}
- Scalable preprocessing environment: Google Earth Engine (GEE) for large satellite datasets and preprocessing at scale. :contentReference[oaicite:10]{index=10}

## Reproducibility controls
- All processing steps, configuration, and code changes are versioned in Git. :contentReference[oaicite:11]{index=11}
- Dependencies are pinned in `requirements.txt` for deterministic re-installation. :contentReference[oaicite:12]{index=12}

## Pipeline entrypoint
- Single runnable entrypoint: `pipeline.py`
- Fixed execution order:
  1) data acquisition hooks
  2) optical preprocessing + monthly composites (GEE)
  3) optional VIIRS aggregation
  4) feature computation (NDVI/NDBI + optional GLCM)
  5) aggregation to village/sub-municipal units
  6) demographic join
  7) export final features table
