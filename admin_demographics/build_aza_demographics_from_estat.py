"""
Build aza (small-area / 小地域) demographics from raw e-Stat census extracts.

Reads the tblT*C*.txt files (comma-separated, CP932-encoded) for 2015 and 2020
census data from the per-year, subfolder-per-table layout under
admin_demographics/demographics/<year>/tbl<PREFIX>C<NN>/tbl<PREFIX>C<NN>.txt,
keeps small-area rows (HYOSYO != 1), and produces:
  1. aza_demographics.csv           -- backward-compatible single-year (2020)
  2. aza_demographics_multiyear.csv -- wide format with _2015 and _2020 cols

The unit_code is reconstructed to match admin_demographics/boundaries/aza.gpkg:
the e-Stat KEY_CODE drops the leading zero of the 2-digit prefecture code, so
unit_code = "0" + str(int(KEY_CODE)) (e.g. 22010010 -> 022010010, 2201002000 ->
02201002000). The same rule yields 5-digit municipality codes (2201 -> 02201).
unit_id = "aza:<PrefName>:<unit_code>".

Usage:
    python admin_demographics/build_aza_demographics_from_estat.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DEMO_DIR = Path("admin_demographics/demographics")
AZA_GPKG = Path("admin_demographics/boundaries/aza.gpkg")

# Table-ID mapping: census year -> {category: table_prefix}
CENSUS_TABLES: Dict[int, Dict[str, str]] = {
    2015: {"pop": "T000848", "age": "T000849", "hh": "T000850"},
    2020: {"pop": "T001081", "age": "T001082", "hh": "T001083"},
}

# Prefecture code (first 2 digits of unit_code) -> name
PREF_MAP: Dict[str, str] = {"02": "Aomori", "05": "Akita"}

# Column mapping within each table type (offset from prefix):
# Pop table: prefix001=total, prefix002=male, prefix003=female, prefix004=hh
# Age table: prefix017=u15, prefix018=15-64, prefix019=65+, prefix020=75+
# HH table:  prefix001=general_households


def read_estat_txt(path: Path) -> pd.DataFrame:
    """Read a raw e-Stat .txt file with auto-encoding detection."""
    for enc in ("cp932", "utf-8-sig", "shift_jis"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise ValueError(f"Failed to decode {path} with cp932/utf-8-sig/shift_jis")


def filter_small_area(df: pd.DataFrame) -> pd.DataFrame:
    """Keep small-area rows (HYOSYO != 1) with a valid KEY_CODE and build unit_code."""
    df = df[df["KEY_CODE"].notna()].copy()
    df = df[df["HYOSYO"] != 1].copy()
    key = pd.to_numeric(df["KEY_CODE"], errors="coerce")
    df = df[key.notna()].copy()
    df["unit_code"] = "0" + key[key.notna()].astype("int64").astype(str)
    df["pref_code"] = df["unit_code"].str[:2]
    df["pref_name"] = df["pref_code"].map(PREF_MAP)
    df = df[df["pref_name"].notna()].copy()
    return df


def find_files(year: int, table_prefix: str) -> List[Path]:
    """Find all tbl<prefix>C*/tbl<prefix>C*.txt files for a census year."""
    year_dir = DEMO_DIR / str(year)
    files = sorted(year_dir.glob(f"tbl{table_prefix}C*/tbl{table_prefix}C*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No files matching tbl{table_prefix}C*/tbl{table_prefix}C*.txt in {year_dir}"
        )
    return files


def extract_population(year: int, table_prefix: str) -> pd.DataFrame:
    """Extract pop_total, pop_male, pop_female, households_from_pop."""
    parts = []
    for fpath in find_files(year, table_prefix):
        df = filter_small_area(read_estat_txt(fpath))
        p = table_prefix
        parts.append(pd.DataFrame({
            "pref_name": df["pref_name"],
            "unit_code": df["unit_code"],
            "city_name_ja": df["CITYNAME"],
            "pop_total": pd.to_numeric(df[f"{p}001"], errors="coerce"),
            "pop_male": pd.to_numeric(df[f"{p}002"], errors="coerce"),
            "pop_female": pd.to_numeric(df[f"{p}003"], errors="coerce"),
            "households_total_from_popfile": pd.to_numeric(df[f"{p}004"], errors="coerce"),
        }))
    return pd.concat(parts, ignore_index=True)


def extract_age_groups(year: int, table_prefix: str) -> pd.DataFrame:
    """Extract age_u15, age_15_64, age_65_plus, age_75_plus."""
    parts = []
    for fpath in find_files(year, table_prefix):
        df = filter_small_area(read_estat_txt(fpath))
        p = table_prefix
        parts.append(pd.DataFrame({
            "pref_name": df["pref_name"],
            "unit_code": df["unit_code"],
            "age_u15": pd.to_numeric(df[f"{p}017"], errors="coerce"),
            "age_15_64": pd.to_numeric(df[f"{p}018"], errors="coerce"),
            "age_65_plus": pd.to_numeric(df[f"{p}019"], errors="coerce"),
            "age_75_plus": pd.to_numeric(df[f"{p}020"], errors="coerce"),
        }))
    return pd.concat(parts, ignore_index=True)


def extract_households(year: int, table_prefix: str) -> pd.DataFrame:
    """Extract households_total (general households count)."""
    parts = []
    for fpath in find_files(year, table_prefix):
        df = filter_small_area(read_estat_txt(fpath))
        p = table_prefix
        parts.append(pd.DataFrame({
            "pref_name": df["pref_name"],
            "unit_code": df["unit_code"],
            "households_total": pd.to_numeric(df[f"{p}001"], errors="coerce"),
        }))
    return pd.concat(parts, ignore_index=True)


def build_year(year: int) -> pd.DataFrame:
    """Build aza demographics DataFrame for a single census year."""
    tables = CENSUS_TABLES[year]

    pop = extract_population(year, tables["pop"])
    age = extract_age_groups(year, tables["age"])
    hh = extract_households(year, tables["hh"])

    demo = pop.merge(
        hh, on=["pref_name", "unit_code"], how="left"
    ).merge(
        age, on=["pref_name", "unit_code"], how="left"
    )

    demo = demo.drop_duplicates(subset=["pref_name", "unit_code"], keep="first")
    demo["unit_id"] = (
        "aza:" + demo["pref_name"].astype(str) + ":" + demo["unit_code"].astype(str)
    )

    print(f"  {year}: {len(demo)} aza units")
    return demo


def report_match_rate(demo: pd.DataFrame) -> None:
    """Report how many aza.gpkg unit_codes have a matching demographics row."""
    if not AZA_GPKG.exists():
        print(f"  (skip match check: {AZA_GPKG} not found)")
        return
    con = sqlite3.connect(str(AZA_GPKG))
    try:
        layer = con.execute("SELECT table_name FROM gpkg_contents").fetchone()[0]
        codes = pd.read_sql(f'SELECT unit_code FROM "{layer}"', con)["unit_code"].astype(str)
    finally:
        con.close()
    demo_codes = set(demo["unit_code"].astype(str))
    matched = codes.isin(demo_codes).sum()
    total = len(codes)
    print(f"  aza.gpkg match: {matched}/{total} ({matched / total:.1%}) unit_codes have demographics")


def main() -> None:
    """Build single-year and multi-year aza demographics CSVs."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    all_years: Dict[int, pd.DataFrame] = {}
    for year in sorted(CENSUS_TABLES.keys()):
        all_years[year] = build_year(year)

    # ------------------------------------------------------------------
    # Output 1: backward-compatible aza_demographics.csv (latest year)
    # ------------------------------------------------------------------
    latest_year = max(all_years.keys())
    demo_latest = all_years[latest_year].copy()

    single_year_cols = [
        "unit_id", "pref_name", "unit_code", "city_name_ja",
        "pop_total", "pop_male", "pop_female",
        "households_total", "households_total_from_popfile",
        "age_u15", "age_15_64", "age_65_plus", "age_75_plus",
    ]
    out_single = DEMO_DIR / "aza_demographics.csv"
    demo_latest[single_year_cols].sort_values(
        ["pref_name", "unit_code"]
    ).to_csv(out_single, index=False)
    print(f"Wrote {out_single} (rows={len(demo_latest)}, year={latest_year})")
    report_match_rate(demo_latest)

    # ------------------------------------------------------------------
    # Output 2: multi-year wide format
    # ------------------------------------------------------------------
    id_cols = ["unit_id", "pref_name", "unit_code", "city_name_ja"]
    demo_multi = demo_latest[id_cols].copy()

    numeric_cols = [
        "pop_total", "pop_male", "pop_female",
        "households_total", "households_total_from_popfile",
        "age_u15", "age_15_64", "age_65_plus", "age_75_plus",
    ]

    for year in sorted(all_years.keys()):
        year_df = all_years[year].copy()
        rename_map = {col: f"{col}_{year}" for col in numeric_cols}
        year_subset = year_df[["pref_name", "unit_code"] + numeric_cols].rename(
            columns=rename_map
        )
        demo_multi = demo_multi.merge(
            year_subset, on=["pref_name", "unit_code"], how="left"
        )

    out_multi = DEMO_DIR / "aza_demographics_multiyear.csv"
    demo_multi.sort_values(["pref_name", "unit_code"]).to_csv(out_multi, index=False)
    print(f"Wrote {out_multi} (rows={len(demo_multi)}, years={sorted(all_years.keys())})")


if __name__ == "__main__":
    main()
