"""
Compute migration proxy indicators from multi-year census data.

Derives inter-censal change metrics (2015-2020) that serve as proxies
for population dynamics including internal migration.

Usage:
    python admin_demographics/compute_migration_proxy.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MULTIYEAR_CSV = Path("admin_demographics/demographics/village_demographics_multiyear.csv")
OUT_CSV = Path("admin_demographics/demographics/migration_proxy.csv")


def main() -> None:
    """Compute change metrics between 2015 and 2020 census."""
    if not MULTIYEAR_CSV.exists():
        raise FileNotFoundError(
            f"Missing {MULTIYEAR_CSV}. "
            "Run build_village_demographics_from_estat.py first."
        )

    df = pd.read_csv(MULTIYEAR_CSV, dtype={"muni_code": "string"})
    print(f"Loaded {MULTIYEAR_CSV} ({len(df)} units)")

    result = df[["unit_id", "pref_name", "muni_code", "city_name_ja"]].copy()

    # Population change
    result["pop_change"] = df["pop_total_2020"] - df["pop_total_2015"]
    result["pop_change_pct"] = (
        result["pop_change"] / df["pop_total_2015"]
    ).round(4)

    # Working-age change (migration proxy)
    result["working_age_change"] = df["age_15_64_2020"] - df["age_15_64_2015"]
    result["working_age_change_pct"] = (
        result["working_age_change"] / df["age_15_64_2015"]
    ).round(4)

    # Youth change (family retention proxy)
    result["youth_change"] = df["age_u15_2020"] - df["age_u15_2015"]

    # Elderly change (natural aging component)
    result["elderly_change"] = df["age_65_plus_2020"] - df["age_65_plus_2015"]

    # Sort and save
    result = result.sort_values(["pref_name", "muni_code"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(result)} units)")

    # Summary statistics
    print(f"\n  Population change (2015-2020):")
    print(f"    Mean:   {result['pop_change'].mean():,.0f}")
    print(f"    Median: {result['pop_change'].median():,.0f}")
    print(f"    Min:    {result['pop_change'].min():,.0f}")
    print(f"    Max:    {result['pop_change'].max():,.0f}")
    n_decline = (result["pop_change"] < 0).sum()
    print(f"    Units declining: {n_decline}/{len(result)}")

    print(f"\n  Working-age change (migration proxy):")
    print(f"    Mean:   {result['working_age_change'].mean():,.0f}")
    print(f"    Median: {result['working_age_change'].median():,.0f}")
    print(f"    Mean pct: {result['working_age_change_pct'].mean():.1%}")


if __name__ == "__main__":
    main()
