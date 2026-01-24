from pathlib import Path
import geopandas as gpd

IN_PATH = Path("admin_demographics/boundaries_raw/mura_input_old.gpkg")
OUT_PATH = Path("admin_demographics/boundaries_raw/mura_input_fixed.gpkg")

def main():
    gdf = gpd.read_file(IN_PATH)

    # Required ADM columns present in your file
    needed = {"ADM1_EN", "ADM1_PCODE", "ADM2_JA", "ADM2_PCODE", "geometry"}
    missing = [c for c in needed if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    # Keep only Aomori (JP02) + Akita (JP05)
    gdf = gdf[gdf["ADM1_PCODE"].isin(["JP02", "JP05"])].copy()

    # STRICT mura only: municipality name ends with 村
    #gdf = gdf[gdf["ADM2_JA"].astype(str).str.endswith("村")].copy()

    # Create pipeline-required fields
    gdf["pref_name"] = gdf["ADM1_EN"].astype(str)
    gdf["unit_code"] = gdf["ADM2_PCODE"].astype(str)

    # Write minimal schema as GeoPackage
    out = gdf[["pref_name", "unit_code", "geometry"]].copy()
    out.to_file(OUT_PATH, layer="mura_input", driver="GPKG")

    print(f"Wrote {OUT_PATH} (rows={len(out)})")

if __name__ == "__main__":
    main()
