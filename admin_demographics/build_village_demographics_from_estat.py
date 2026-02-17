"""
Build village demographics from raw e-Stat census extracts.

Reads the tblT*C*.txt files (comma-separated, CP932-encoded) for
2015 and 2020 census data, filters to municipality level (HYOSYO=1),
and produces:
  1. village_demographics.csv      -- backward-compatible single-year (2020)
  2. village_demographics_multiyear.csv -- wide format with _2015 and _2020 cols

Usage:
    python admin_demographics/build_village_demographics_from_estat.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

RAW_DIR = Path("admin_demographics/demographics_raw")
OUT_DIR = Path("admin_demographics/demographics")

# Table-ID mapping: census year -> {category: table_prefix}
CENSUS_TABLES: Dict[int, Dict[str, str]] = {
    2015: {"pop": "T000848", "age": "T000849", "hh": "T000850"},
    2020: {"pop": "T001081", "age": "T001082", "hh": "T001083"},
}

# File suffix -> prefecture code (first 2 digits of KEY_CODE)
PREF_SUFFIXES: Dict[str, str] = {"C02": "02", "C05": "05"}

# Prefecture code -> name
PREF_MAP: Dict[str, str] = {"02": "Aomori", "05": "Akita"}

# Column mapping within each table type (offset from prefix)
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


def filter_municipality(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only HYOSYO=1 rows (municipality level) with valid KEY_CODE."""
    df = df[df["KEY_CODE"].notna()].copy()
    df = df[df["HYOSYO"] == 1].copy()
    df["muni_code"] = (
        pd.to_numeric(df["KEY_CODE"], errors="coerce")
        .dropna()
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )
    df["pref_code"] = df["muni_code"].str[:2]
    df["pref_name"] = df["pref_code"].map(PREF_MAP)
    return df


def find_files(table_prefix: str) -> List[Path]:
    """Find all tblT{prefix}C*.txt files in RAW_DIR."""
    pattern = f"tbl{table_prefix}C*.txt"
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern} in {RAW_DIR}"
        )
    return files


def extract_population(table_prefix: str) -> pd.DataFrame:
    """Extract pop_total, pop_male, pop_female, households_from_pop."""
    parts = []
    for fpath in find_files(table_prefix):
        df = read_estat_txt(fpath)
        df = filter_municipality(df)
        p = table_prefix
        parts.append(pd.DataFrame({
            "pref_name": df["pref_name"],
            "muni_code": df["muni_code"],
            "city_name_ja": df["CITYNAME"],
            "pop_total": pd.to_numeric(df[f"{p}001"], errors="coerce"),
            "pop_male": pd.to_numeric(df[f"{p}002"], errors="coerce"),
            "pop_female": pd.to_numeric(df[f"{p}003"], errors="coerce"),
            "households_total_from_popfile": pd.to_numeric(
                df[f"{p}004"], errors="coerce"
            ),
        }))
    return pd.concat(parts, ignore_index=True)


def extract_age_groups(table_prefix: str) -> pd.DataFrame:
    """Extract age_u15, age_15_64, age_65_plus, age_75_plus."""
    parts = []
    for fpath in find_files(table_prefix):
        df = read_estat_txt(fpath)
        df = filter_municipality(df)
        p = table_prefix
        parts.append(pd.DataFrame({
            "pref_name": df["pref_name"],
            "muni_code": df["muni_code"],
            "age_u15": pd.to_numeric(df[f"{p}017"], errors="coerce"),
            "age_15_64": pd.to_numeric(df[f"{p}018"], errors="coerce"),
            "age_65_plus": pd.to_numeric(df[f"{p}019"], errors="coerce"),
            "age_75_plus": pd.to_numeric(df[f"{p}020"], errors="coerce"),
        }))
    return pd.concat(parts, ignore_index=True)


def extract_households(table_prefix: str) -> pd.DataFrame:
    """Extract households_total (general households count)."""
    parts = []
    for fpath in find_files(table_prefix):
        df = read_estat_txt(fpath)
        df = filter_municipality(df)
        p = table_prefix
        parts.append(pd.DataFrame({
            "pref_name": df["pref_name"],
            "muni_code": df["muni_code"],
            "households_total": pd.to_numeric(
                df[f"{p}001"], errors="coerce"
            ),
        }))
    return pd.concat(parts, ignore_index=True)


def build_year(year: int) -> pd.DataFrame:
    """Build demographics DataFrame for a single census year."""
    tables = CENSUS_TABLES[year]

    pop = extract_population(tables["pop"])
    age = extract_age_groups(tables["age"])
    hh = extract_households(tables["hh"])

    demo = pop.merge(
        hh, on=["pref_name", "muni_code"], how="left"
    ).merge(
        age, on=["pref_name", "muni_code"], how="left"
    )

    demo["unit_id"] = (
        "mura:" + demo["pref_name"].astype(str) + ":"
        + demo["muni_code"].astype(str)
    )

    print(f"  {year}: {len(demo)} units")
    return demo


def main() -> None:
    """Build single-year and multi-year demographics CSVs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build both years
    all_years: Dict[int, pd.DataFrame] = {}
    for year in sorted(CENSUS_TABLES.keys()):
        all_years[year] = build_year(year)

    # ------------------------------------------------------------------
    # Output 1: backward-compatible village_demographics.csv (2020 only)
    # ------------------------------------------------------------------
    latest_year = max(all_years.keys())
    demo_latest = all_years[latest_year].copy()

    single_year_cols = [
        "unit_id", "pref_name", "muni_code", "city_name_ja",
        "pop_total", "pop_male", "pop_female",
        "households_total", "households_total_from_popfile",
        "age_u15", "age_15_64", "age_65_plus", "age_75_plus",
    ]
    out_single = OUT_DIR / "village_demographics.csv"
    demo_latest[single_year_cols].sort_values(
        ["pref_name", "muni_code"]
    ).to_csv(out_single, index=False)
    print(f"Wrote {out_single} (rows={len(demo_latest)}, year={latest_year})")

    # ------------------------------------------------------------------
    # Output 2: multi-year wide format
    # ------------------------------------------------------------------
    # Start with identifiers from latest year
    id_cols = ["unit_id", "pref_name", "muni_code", "city_name_ja"]
    demo_multi = demo_latest[id_cols].copy()

    # Numeric columns to include per year
    numeric_cols = [
        "pop_total", "pop_male", "pop_female",
        "households_total", "households_total_from_popfile",
        "age_u15", "age_15_64", "age_65_plus", "age_75_plus",
    ]

    for year in sorted(all_years.keys()):
        year_df = all_years[year].copy()
        rename_map = {col: f"{col}_{year}" for col in numeric_cols}
        year_subset = year_df[["pref_name", "muni_code"] + numeric_cols].rename(
            columns=rename_map
        )
        demo_multi = demo_multi.merge(
            year_subset, on=["pref_name", "muni_code"], how="left"
        )

    out_multi = OUT_DIR / "village_demographics_multiyear.csv"
    demo_multi.sort_values(["pref_name", "muni_code"]).to_csv(
        out_multi, index=False
    )
    print(f"Wrote {out_multi} (rows={len(demo_multi)}, "
          f"years={sorted(all_years.keys())})")


if __name__ == "__main__":
    main()
