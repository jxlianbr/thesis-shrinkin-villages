# Phase B Data Manifest

All paths relative to `D:/data_code_masterthesis/`.  
Generated 2026-06-21.

---

## Phase A outputs (code/phase_a/)

| File | Type | Shape / Notes |
|------|------|---------------|
| `outputs/final/features_table_aza.parquet` | Parquet | 1,165,032 × 46 — monthly EO panel for all aza units |
| `outputs/final/features_table_aza.csv` | CSV | same as above |
| `preprocessing/outputs/classification_ready_aza.parquet` | Parquet | 7,448 × 53 — cross-sectional Phase A features |
| `preprocessing/outputs/classification_ready_aza.csv` | CSV | same — includes `elderly_ratio`, `S2_NDBI_*`, `shrinkage_class` |
| `admin_demographics/boundaries/aza.gpkg` | GeoPackage | 9,031 polygons, EPSG:4612 — join key: `unit_id` (e.g. `aza:Aomori:022010010`) |

**Key columns in `features_table_aza` / `classification_ready_aza`:**  
`unit_id`, `unit_level`, `pref_name`, `unit_code`, `month` (panel only),  
`S2_NDBI_contrast_mean/std`, `S2_NDBI_entropy_mean/std` (GLCM texture),  
`NDVI_slope`, `NDBI_slope`, `MNDWI_slope` (trajectory),  
`elderly_ratio`, `aging_index`, `pop_total`, `age_65_plus`, `age_u15` (demographics),  
`shrinkage_class` / `shrinkage_code` (Phase A target — **BANNED** in Phase B features)

---

## NLNI accessibility layers (data/nlni_data_japan/)

| Folder | File | Layer | Vintage | CRS | Geometry | Features | Rows (Aomori) |
|--------|------|-------|---------|-----|----------|----------|---------------|
| `DID/A16-15_02_GML/` | `A16-15_02_DID.shp` | DID polygons | 2015 | None→EPSG:4612 | Polygon | A16_001–A16_010 (A16_011 datetime skipped: year-0) | 20 |
| `DID/A16-15_05_GML/` | `A16-15_05_DID.shp` | DID polygons | 2015 | None→EPSG:4612 | Polygon | same | — |
| `Hospitals/P04-14_02_GML/P04-14_02_GML/` | `P04-14_02-g_MedicalInstitution.shp` | Medical facilities | 2014 | EPSG:4612 | Point | P04_001 (type: 1=hospital), P04_002 (name), P04_003 (address), P04_004 (specialties), P04_005–P04_007 (beds) | 1,572 |
| `Hospitals/P04-14_05_GML/P04-14_05_GML/` | same pattern | Medical facilities | 2014 | EPSG:4612 | Point | same | — |
| `Schools/P29-13_02/P29-13_02/` | `P29-13_02.shp` | Schools | 2013 | EPSG:4612 | Point | P29_001 (muni code), P29_002 (category=16), P29_003 (type: 16001=middle, 16002=elementary…), P29_004–P29_007 | 609 |
| `Schools/P29-13_05/P29-13_05/` | same pattern | Schools | 2013 | EPSG:4612 | Point | same | — |
| `Bus_stops/P11-10_02_GML/P11-10_02_GML/` | `P11-10_02-jgd-g_BusStop.shp` | Bus stops | 2010 | None→EPSG:4612 | Point | P11_001 (stop code), P11_002 (stop name), P11_003_1..19 (route codes), P11_004_1..19 (operators) | 4,555 |
| `Bus_stops/P11-10_05_GML/P11-10_05_GML/` | same pattern | Bus stops | 2010 | None→EPSG:4612 | Point | same | — |
| `Bus_routes/N07-11_02_GML/N07-11_02_GML/` | `N07-11_02.shp` | Bus routes | 2011 | None→EPSG:4612 | LineString (M stripped) | N07_001–N07_007 | 343 |
| `Bus_routes/N07-11_05_GML/N07-11_05_GML/` | same pattern | Bus routes | 2011 | None→EPSG:4612 | LineString | same | — |
| `community_facilities/` | **ABSENT** | P05 community facilities | — | — | — | — | **STUB** |

**Known issues:**  
- DID `.shp` has no `.prj` file; CRS assigned as EPSG:4612 at load time.  
- DID `A16_011` is a `datetime64[D]` field with year-0 values (pyogrio error); skipped in loader.  
- Bus stops / routes have no `.prj`; CRS assigned as EPSG:4612.  
- Bus routes use Measured (M) geometry; pyogrio auto-converts to plain LineString.  
- P05 community facilities not yet available; `dist_community_facility_m` and `n_community_facility_*m` are NaN stubs.

---

## Census small-area age 2015 (data/census_data_japan_aza/2015/)

| File | Table | Encoding | Sep | Rows | Key column | Level col |
|------|-------|----------|-----|------|-----------|-----------|
| `tblT000848C02/tblT000848C02.txt` | T000848 pop+households, Aomori | CP932 | TAB | ~1,500 aza | `KEY_CODE` (float→int = unit_code int) | `HYOSYO` (2=aza) |
| `tblT000848C05/tblT000848C05.txt` | T000848, Akita | CP932 | TAB | — | same | same |
| `tblT000849C02/tblT000849C02.txt` | T000849 age breakdown, Aomori | CP932 | TAB | — | same | same |
| `tblT000849C05/tblT000849C05.txt` | T000849, Akita | CP932 | TAB | — | same | same |
| `tblT000850*/` | T000850, Aomori+Akita | CP932 | TAB | — | same | same |
| `tblT000851*/` | T000851, Aomori+Akita | CP932 | TAB | — | same | same |

**Key alignment:**  
`census KEY_CODE` (float, no leading zero) → `int(KEY_CODE)` == `int(unit_code)`  
e.g. `22010010.0` → `22010010` matches `unit_code = "022010010"` (leading zero added in gpkg).

---

## Municipal finance — 決算カード (data/finance_data_japan/)

**XLSX files (FY2015–FY2024 — ALL post-2014, violate leakage rule; not loaded)**

| File | Pref | FY |
|------|------|-----|
| `000473824.xlsx` | Aomori | 2015 (平成27) |
| `000473827.xlsx` | Akita | 2015 |
| … | … | 2016–2024 |
| `001063955.xlsx` | Akita | 2024 (令和6) |

**PDF files (pre-2015, usable under leakage rule)**

| File | Pref | FY (from PDF text) |
|------|------|--------------------|
| `1018-15-9_02.pdf` | Aomori | 2008 (平成20) |
| `1018-15-10_02.pdf` | Aomori | TBD |
| `1018-15-10_05.pdf` | Akita | TBD |
| `1018-15-11_02.pdf` | Aomori | TBD |
| `1018-15-11_05.pdf` | Akita | TBD |
| `1018-15-12_02.pdf` | Aomori | TBD |
| `1018-15-12_05.pdf` | Akita | TBD |
| `1018-15-13_02.pdf` | Aomori | TBD |
| `1018-15-13_05.pdf` | Akita | TBD |
| `1018-15-14_02.pdf` | Aomori | TBD |
| `1018-15-14_05.pdf` | Akita | TBD |
| `1018-15-15_02.pdf` | Aomori | TBD |
| `1018-15-15_05.pdf` | Akita | TBD |

Structure per PDF: page 1 = ToC (municipality → page number); pages 2+ = one municipality per page. Each page contains: 歳入総額, 歳出総額, 地方税, 地方交付税, 基準財政収入額/需要額 (from which fiscal strength index is computed). Text is multi-column and must be normalised before regex extraction.

---

## Housing and Land Survey 2013 (data/housing_data_japan/2013/)

| File | Coverage | Table | Key structure |
|------|----------|-------|---------------|
| `Aomori-ken/a002.xls` | Aomori cities | Table 1: dwellings by occupancy | muni code col 1; data rows: row_type col 2 |
| `Aomori-ken/a047.xls` | Aomori cities (市) | Table 21: dwellings by tenure × age × dilapidation | muni code col 1; total row: row_type=1; pre-1981 cols 6,7,8 |
| `Aomori-ken/a048.xls` | Aomori towns+villages (町村) | Table 22: same structure as a047 | same |
| `Akita-ken/a047.xls` | Akita cities | Table 21 | same |
| `Akita-ken/a048.xls` | Akita towns+villages | Table 22 | same |

Municipality code in col 1 (e.g. `220146`) maps to JISCD via `str(int(code))[:5]` → 5-digit muni code. Pre-1981 dwelling ratio (旧耐震率 proxy) = (cols 6+7+8) / col 5 (total), row_type=1 rows only.

---

## Heisei merger records (data/merger_data_japan/)

| File | Rows | Columns |
|------|------|---------|
| `000283317.xls` | ~500 | 都道府県, 合併期日, 名称, 合併の方式, 関係市町村 |
| `000283318.xls` | ~500 | same |

Filter to 都道府県 ∈ {青森県, 秋田県}; match `名称` to `city_name_ja` in aza base frame.

---

## Kaso designation list (data/kaso_list/)

| File | Format | Status |
|------|--------|--------|
| `000476767.pdf` | PDF | Machine-readable extraction not attempted; designations hard-coded from user-supplied verbatim in `feature_matrix.py` |

Hard-coded municipalities: 24 全部過疎 in Aomori, 22 in Akita; 5 一部過疎 in Aomori, 1 in Akita.

---

## Projection used for spatial operations

All distance computations are performed in **EPSG:6680** (JGD2011 Japan Zone 10, metric).  
Source geometries are in EPSG:4612 (JGD2000 geographic) and reprojected on load.
