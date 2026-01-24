try:
    import geopandas as gpd
except ImportError:
    import sys
    print("Missing dependency: geopandas. Install with 'pip install geopandas' or see https://geopandas.org")
    sys.exit(1)
from pathlib import Path

paths = [ 
    Path("admin_demographics/boundaries_raw/mura_input.gpkg"),
    Path("admin_demographics/boundaries_raw/aza_input.gpkg")
]

for p in paths:
    if not p.exists():
        print(f"[SKIP] {p} not found")
        continue

    # If file has multiple layers, read the first layer. You can list layers if needed.
    try:
        gdf = gpd.read_file(p)
    except Exception as e:
        print(f"[FAIL] {p}: {e}")
        continue

    cols = set(gdf.columns)
    required = {"pref_name", "unit_code"}
    print(f"\n{p}")
    print("rows:", len(gdf))
    print("columns:", sorted(list(cols))[:40])
    print("has required:", required.issubset(cols))