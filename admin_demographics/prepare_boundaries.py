from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

try:
    import geopandas as gpd
except ImportError as e:
    raise RuntimeError("geopandas is required for boundary preparation.") from e


def main():
    """
    Inputs (raw, local-only):
      - admin_demographics/boundaries_raw/mura_input.(gpkg|shp|geojson)
      - admin_demographics/boundaries_raw/aza_input.(gpkg|shp|geojson)

    Outputs (processed):
      - admin_demographics/boundaries/mura.gpkg
      - admin_demographics/boundaries/aza.gpkg
      - admin_demographics/boundaries/units_index.csv

    You MUST ensure both raw inputs originate from Statistics Bureau of Japan-aligned boundaries.
    The pipeline is keyed to: unit_id (string), unit_level in {mura, aza}.
    """
    raw_dir = Path("admin_demographics/boundaries_raw")
    out_dir = Path("admin_demographics/boundaries")
    out_dir.mkdir(parents=True, exist_ok=True)

    mura_in = next(raw_dir.glob("mura_input.*"), None)
    aza_in = next(raw_dir.glob("aza_input.*"), None)
    if mura_in is None or aza_in is None:
        raise FileNotFoundError(
            "Missing raw boundary inputs. Expect files named "
            "'mura_input.<ext>' and 'aza_input.<ext>' in admin_demographics/boundaries_raw/."
        )

    mura = gpd.read_file(mura_in, layer="mura_input")
    aza  = gpd.read_file(aza_in,  layer="aza_input")        

    # Expected minimal fields (configure/rename in-place as needed):
    # - pref_name: prefecture name (for filtering Aomori/Akita)
    # - unit_code: stable code for the unit (mura or aza)
    # If your raw files use different names, rename them here.
    REQUIRED = ["pref_name", "unit_code"]
    for df, name in [(mura, "mura"), (aza, "aza")]:
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{name} boundaries missing columns {missing}. Rename your raw fields to match.")

    # Filter to Aomori/Akita (pipeline constant)
    #target = {"Aomori", "Akita"}
    #mura = mura[mura["pref_name"].isin(target)].copy()
    #aza = aza[aza["pref_name"].isin(target)].copy()
    
    # normalize join/filter keys
    for df in (mura, aza):
        df["pref_name"] = df["pref_name"].astype(str).str.strip()
        df["unit_code"] = df["unit_code"].astype(str).str.strip()

    # allow Japanese prefecture names if present
    pref_map = {"青森県": "Aomori", "秋田県": "Akita"}
    mura["pref_name"] = mura["pref_name"].replace(pref_map)
    aza["pref_name"] = aza["pref_name"].replace(pref_map)

    # Create stable unit_id
    mura["unit_level"] = "mura"
    aza["unit_level"] = "aza"
    mura["unit_id"] = mura["unit_level"] + ":" + mura["pref_name"].astype(str) + ":" + mura["unit_code"].astype(str)
    aza["unit_id"] = aza["unit_level"] + ":" + aza["pref_name"].astype(str) + ":" + aza["unit_code"].astype(str)

    # Minimal schema written out
    mura_out = mura[["unit_id", "unit_level", "pref_name", "unit_code", "geometry"]]
    aza_out = aza[["unit_id", "unit_level", "pref_name", "unit_code", "geometry"]]

    mura_out.to_file(out_dir / "mura.gpkg", layer="mura", driver="GPKG")
    aza_out.to_file(out_dir / "aza.gpkg", layer="aza", driver="GPKG")

    # Index for joins/debugging
    idx = pd.concat(
        [
            mura_out.drop(columns="geometry").assign(n_geoms=len(mura_out)),
            aza_out.drop(columns="geometry").assign(n_geoms=len(aza_out)),
        ],
        ignore_index=True,
    )
    idx.to_csv(out_dir / "units_index.csv", index=False)

    print("Wrote admin_demographics/boundaries/mura.gpkg")
    print("Wrote admin_demographics/boundaries/aza.gpkg")
    print("Wrote admin_demographics/boundaries/units_index.csv")


if __name__ == "__main__":
    main()
