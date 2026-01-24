from pathlib import Path
import geopandas as gpd

def export_layer(gpkg_path: str, layer: str, out_dir: Path):
    gdf = gpd.read_file(gpkg_path, layer=layer)

    # Reproject to EPSG:4326 (recommended for Earth Engine table uploads)
    if gdf.crs is None or str(gdf.crs).lower() != "epsg:4326":
        gdf = gdf.to_crs("EPSG:4326")

    out_dir.mkdir(parents=True, exist_ok=True)
    shp_path = out_dir / f"{layer}.shp"
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    print(f"Wrote {shp_path}")

def main():
    base = Path("admin_demographics/boundaries_export")
    export_layer("admin_demographics/boundaries/mura.gpkg", "mura", base / "mura_shp")
    export_layer("admin_demographics/boundaries/aza.gpkg", "aza", base / "aza_shp")

if __name__ == "__main__":
    main()
