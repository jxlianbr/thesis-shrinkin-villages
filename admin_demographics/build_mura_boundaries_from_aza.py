from __future__ import annotations

from pathlib import Path
import geopandas as gpd

IN_GPKG = Path("admin_demographics/boundaries/aza.gpkg")
OUT_GPKG = Path("admin_demographics/boundaries/mura_jis.gpkg")

def main() -> None:
    aza = gpd.read_file(IN_GPKG, layer="aza")

    # unit_code in aza should be the small-area KEY_CODE; municipality code is first 5 digits. :contentReference[oaicite:2]{index=2}
    aza["muni_code"] = aza["unit_code"].astype(str).str.zfill(8).str[:5]

    muni = aza.dissolve(by=["pref_name", "muni_code"], as_index=False)

    muni["unit_level"] = "mura"
    muni["unit_code"] = muni["muni_code"]
    muni["unit_id"] = "mura:" + muni["pref_name"].astype(str) + ":" + muni["unit_code"].astype(str)

    out = muni[["unit_id", "unit_level", "pref_name", "unit_code", "geometry"]].copy()

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()
    out.to_file(OUT_GPKG, layer="mura", driver="GPKG")
    print(f"Wrote {OUT_GPKG} (rows={len(out)})")

if __name__ == "__main__":
    main()
