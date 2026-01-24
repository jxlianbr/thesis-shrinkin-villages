import geopandas as gpd
from pathlib import Path

RAW = Path("admin_demographics/boundaries_raw")

def normalize(in_path: Path, out_path: Path, pref_col: str, code_col: str):
    gdf = gpd.read_file(in_path)
    gdf["pref_name"] = gdf[pref_col].astype(str)
    gdf["unit_code"] = gdf[code_col].astype(str)
    keep = ["pref_name", "unit_code"] + [c for c in ["geometry"] if c in gdf.columns]
    gdf[keep].to_file(out_path, layer=out_path.stem, driver="GPKG")
    print(f"Wrote {out_path}")

# Adjust these mappings to match your actual column names in each file:
mura_in = RAW / "mura_input.gpkg"
aza_in  = RAW / "aza_input.gpkg"

if mura_in.exists():
    # Example mapping for your ADM-style file:
    # pref_col = "ADM1_EN", code_col = "ADM2_PCODE"
    normalize(mura_in, RAW / "mura_input.gpkg", pref_col="ADM1_EN", code_col="ADM2_PCODE")

if aza_in.exists():
    # Example mapping for your e-Stat small area file:
    # pref_col = "PREF_NAME", code_col = "KEY_CODE"
    normalize(aza_in, RAW / "aza_input.gpkg", pref_col="PREF_NAME", code_col="KEY_CODE")
