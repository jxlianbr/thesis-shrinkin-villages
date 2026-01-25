from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW_DIR = Path("admin_demographics/demographics_raw")
OUT_CSV = Path("admin_demographics/demographics/village_demographics.csv")

# e-Stat “standard area code” is 5 digits; first two are prefecture code. :contentReference[oaicite:0]{index=0}
PREF_MAP = {"02": "Aomori", "05": "Akita"}

def read_csv_auto(path: Path) -> pd.DataFrame:
    # Your downloads are typically UTF-8 with BOM; fallback to Japanese encodings if needed.
    for enc in ("utf-8-sig", "cp932", "shift_jis"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise UnicodeDecodeError("encoding", b"", 0, 1, f"Failed to decode {path} with utf-8-sig/cp932/shift_jis")

def norm_key_code(series: pd.Series) -> pd.Series:
    # KEY_CODE is numeric with NaN in header row; normalize to 5-digit string.
    s = pd.to_numeric(series, errors="coerce").dropna().astype(int).astype(str).str.zfill(5)
    return s

def main() -> None:
    pop_files = [
        RAW_DIR / "aomori_population_gender.csv",
        RAW_DIR / "akita_population_gender.csv",
    ]
    hh_files = [
        RAW_DIR / "aomori_households.csv",
        RAW_DIR / "akita_households.csv",
    ]
    age_files = [
        RAW_DIR / "aomori_age_group.csv",
        RAW_DIR / "akita_age_group.csv",
    ]

    for p in pop_files + hh_files + age_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    # ---- Population (total/male/female) ----
    pop_parts = []
    for p in pop_files:
        df = read_csv_auto(p)
        df = df[df["KEY_CODE"].notna()].copy()
        df["muni_code"] = norm_key_code(df["KEY_CODE"]).values
        df["pref_code"] = df["muni_code"].str[:2]
        df["pref_name"] = df["pref_code"].map(PREF_MAP)

        out = pd.DataFrame({
            "pref_name": df["pref_name"],
            "muni_code": df["muni_code"],
            "city_name_ja": df.get("CITYNAME"),
            "pop_total": pd.to_numeric(df["T001081001"], errors="coerce"),
            "pop_male": pd.to_numeric(df["T001081002"], errors="coerce"),
            "pop_female": pd.to_numeric(df["T001081003"], errors="coerce"),
            # This column is “household total” in the population-gender extract.
            "households_total_from_popfile": pd.to_numeric(df["T001081004"], errors="coerce"),
        })
        pop_parts.append(out)

    pop = pd.concat(pop_parts, ignore_index=True)

    # ---- Households (use “general households count” as households_total) ----
    hh_parts = []
    for p in hh_files:
        df = read_csv_auto(p)
        df = df[df["KEY_CODE"].notna()].copy()
        df["muni_code"] = norm_key_code(df["KEY_CODE"]).values
        df["pref_code"] = df["muni_code"].str[:2]
        df["pref_name"] = df["pref_code"].map(PREF_MAP)

        out = pd.DataFrame({
            "pref_name": df["pref_name"],
            "muni_code": df["muni_code"],
            # “一般世帯数…” is the total general households count in this extract.
            "households_total": pd.to_numeric(df["T001083001"], errors="coerce"),
        })
        hh_parts.append(out)

    hh = pd.concat(hh_parts, ignore_index=True)

    # ---- Age groups (use total columns) ----
    age_parts = []
    for p in age_files:
        df = read_csv_auto(p)
        df = df[df["KEY_CODE"].notna()].copy()
        df["muni_code"] = norm_key_code(df["KEY_CODE"]).values
        df["pref_code"] = df["muni_code"].str[:2]
        df["pref_name"] = df["pref_code"].map(PREF_MAP)

        out = pd.DataFrame({
            "pref_name": df["pref_name"],
            "muni_code": df["muni_code"],
            "age_u15": pd.to_numeric(df["T001082017"], errors="coerce"),     # 総数15歳未満
            "age_15_64": pd.to_numeric(df["T001082018"], errors="coerce"),   # 総数15～64歳
            "age_65_plus": pd.to_numeric(df["T001082019"], errors="coerce"), # 総数65歳以上
            "age_75_plus": pd.to_numeric(df["T001082020"], errors="coerce"), # 総数75歳以上 (optional)
        })
        age_parts.append(out)

    age = pd.concat(age_parts, ignore_index=True)

    # ---- Merge all ----
    demo = pop.merge(hh, on=["pref_name", "muni_code"], how="left").merge(
        age, on=["pref_name", "muni_code"], how="left"
    )

    # Pipeline join key (must match your municipality boundary unit_id):
    # unit_id = "mura:<pref_name>:<muni_code>"
    demo["unit_id"] = "mura:" + demo["pref_name"].astype(str) + ":" + demo["muni_code"].astype(str)

    # Write final demographics table
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "unit_id",
        "pref_name",
        "muni_code",
        "city_name_ja",
        "pop_total",
        "pop_male",
        "pop_female",
        "households_total",
        "households_total_from_popfile",
        "age_u15",
        "age_15_64",
        "age_65_plus",
        "age_75_plus",
    ]
    demo[cols].sort_values(["pref_name", "muni_code"]).to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} (rows={len(demo)})")

if __name__ == "__main__":
    main()
