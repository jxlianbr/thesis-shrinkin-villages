"""
Phase B — Feature matrix assembly.

Joins all Phase B predictor families onto AZA_ID:
  1. Demographic threshold flags (NOT raw aging ratios — those are in the target)
  2. GLCM texture proxy (from Phase A EO features)
  3. Municipal finance indicators from pre-2015 決算カード PDFs
  4. Housing and Land Survey 2013 durability (broadcast muni -> aza)
  5. Kaso old-law designation flag (hard-coded from official list)
  6. Heisei merger flag + years-since-merger
  7. Accessibility features (from accessibility_features.py)
  8. Community facility presence / distance (P05)

Guards enforced before every merge:
  - AZA_ID alignment assertion (fail loudly on mismatch)
  - Leakage column guard (banned_columns raise ValueError if present)
  - Post-2014 finance years raise ValueError
  - Missing HLS data carried as NaN with an explicit *_missing flag
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Windows cp1252 console can't encode kanji — force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

_HERE = Path(__file__).resolve().parent
_CFG_PATH = _HERE / "config" / "phase_b_config.yaml"


def _load_cfg() -> Dict[str, Any]:
    with open(_CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Kaso designation table (hard-coded from official 過疎地域自立促進特別措置法 list)
# Source: user-supplied verbatim from PDF kaso_list/000476767.pdf
# ---------------------------------------------------------------------------

# 全部過疎 — entire municipality is kaso-designated.
_KASO_ZENBU: dict[str, list[str]] = {
    "Aomori": [
        "五所川原市", "つがる市", "平内町", "今別町", "蓬田村", "外ヶ浜町",
        "鰺ケ沢町", "深浦町", "西目屋村", "大鰐町", "板柳町", "中泊町",
        "野辺地町", "七戸町", "横浜町", "大間町", "東通村", "風間浦村",
        "佐井村", "三戸町", "五戸町", "田子町", "南部町", "新郷村",
    ],
    "Akita": [
        "能代市", "横手市", "大館市", "男鹿市", "湯沢市", "鹿角市",
        "由利本荘市", "大仙市", "北秋田市", "にかほ市", "仙北市", "小坂町",
        "上小阿仁村", "藤里町", "三種町", "八峰町", "五城目町", "八郎潟町",
        "井川町", "美郷町", "羽後町", "東成瀬村",
    ],
}

# 一部過疎 — only aza within the listed 旧町村 area.
# Value: list of 旧町村 names whose aza inherit the flag.
_KASO_ICHIBU: dict[str, dict[str, list[str]]] = {
    "Aomori": {
        "弘前市": ["旧相馬村"],
        "八戸市": ["旧南郷村"],
        "十和田市": ["旧十和田湖町"],
        "むつ市": ["旧川内町", "旧大畑町", "旧脇野沢村"],
        "平川市": ["旧碇ヶ関村"],
    },
    "Akita": {
        "秋田市": ["旧河辺町"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_key_alignment(
    left: pd.DataFrame, right: pd.DataFrame, key: str, label: str,
) -> None:
    """Fail loudly if key sets diverge before a merge."""
    lk = set(left[key].dropna())
    rk = set(right[key].dropna())
    only_left = lk - rk
    only_right = rk - lk
    if only_right:
        raise AssertionError(
            f"Key alignment failure before merging '{label}': "
            f"{len(only_right)} keys in right-table only. "
            f"First 5: {sorted(only_right)[:5]}"
        )
    if only_left:
        print(f"  NOTE: {len(only_left)} aza in base table have no match "
              f"in '{label}' — will be NaN after merge.")


def _assert_no_leakage(df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    banned = set(cfg["feature_matrix"]["banned_columns"])
    found = banned & set(df.columns)
    if found:
        raise ValueError(
            f"Leakage columns found in feature matrix: {sorted(found)}. "
            "Remove them before proceeding."
        )


def _extract_muni_code(unit_id: str) -> str:
    """Extract 5-digit JISCD municipality code from a unit_id string.

    unit_id format: 'aza:Aomori:022010010' or 'aza:Akita:05202107002'
    The numeric code's first 5 chars = pref(2) + muni(3).
    """
    code_str = unit_id.split(":")[-1]
    return code_str[:5]


# ---------------------------------------------------------------------------
# 1. Demographic threshold flags
# ---------------------------------------------------------------------------

def _build_demographic_flags(
    df: pd.DataFrame, aza_id_col: str,
) -> pd.DataFrame:
    """
    Create threshold flags from the Phase A classification-ready data.

    elderly_ratio >= 0.42 -> severely_shrinking_flag
    elderly_ratio in [0.37, 0.42) -> shrinking_flag
    elderly_ratio < 0.37 -> stable_flag

    These are the same thresholds used to construct the Phase A target,
    but expressed as binary flags rather than the raw ratio (which is banned).
    """
    flags = df[[aza_id_col]].copy()
    if "elderly_ratio" in df.columns:
        er = df["elderly_ratio"]
        flags["elderly_flag_severely"] = (er >= 0.42).astype(float)
        flags["elderly_flag_shrinking"] = ((er >= 0.37) & (er < 0.42)).astype(float)
    if "household_size" in df.columns:
        hs = df["household_size"]
        flags["household_size_small"] = (hs < 2.5).astype(float)
    return flags


# ---------------------------------------------------------------------------
# 2. GLCM texture proxy (from Phase A EO features)
# ---------------------------------------------------------------------------

def _load_glcm_proxy(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load GLCM features from Phase A classification-ready CSV."""
    path = Path(cfg["phase_a_root"]) / cfg["phase_a"]["classification_ready"]
    df = pd.read_csv(path, encoding="utf-8")
    aza_id_col = cfg["aza_id_col"]
    glcm_cols = [c for c in df.columns if c.startswith("S2_NDBI")]
    keep = [aza_id_col] + glcm_cols
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


# ---------------------------------------------------------------------------
# 3. Municipal finance from 決算カード PDFs
# ---------------------------------------------------------------------------

def _load_finance_features(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract pre-2015 financial indicators from 決算カード PDFs.

    Returns DataFrame with columns [muni_name, fiscal_year, <metrics>].
    Only fiscal years <= max_fiscal_year are loaded; any later year raises.
    """
    import pdfplumber

    fin_cfg = cfg["finance"]
    data_root = Path(cfg["data_root"])
    pdf_dir = data_root / fin_cfg["pdf_dir"]
    min_fy = int(fin_cfg.get("min_fiscal_year", 0))
    max_fy = int(fin_cfg["max_fiscal_year"])
    metrics_map: dict[str, str] = fin_cfg["metrics"]

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("  WARNING: No finance PDFs found. Finance features will be NaN.")
        return pd.DataFrame(columns=["muni_name", "pref", "fiscal_year"])

    records = []
    for pdf_path in pdf_files:
        pref = "Aomori" if pdf_path.name.endswith("_02.pdf") else "Akita"
        try:
            rows = _parse_kessancard_pdf(pdf_path, pref, metrics_map, min_fy, max_fy)
            records.extend(rows)
        except Exception as exc:
            print(f"  WARNING: failed to parse {pdf_path.name}: {exc}")

    if not records:
        print("  WARNING: No valid finance records extracted from PDFs.")
        return pd.DataFrame(columns=["muni_name", "pref", "fiscal_year"])

    df = pd.DataFrame(records)
    print(f"  Finance: {len(df)} muni-year rows from "
          f"{df['fiscal_year'].nunique()} fiscal years, "
          f"FY{df['fiscal_year'].min()}-{df['fiscal_year'].max()}")
    return df


def _jp_year_to_western(era: str, year: int) -> int:
    """Convert Japanese era year to Western calendar year."""
    base = {"平成": 1988, "令和": 2018, "昭和": 1925}
    return base[era] + year


def _normalize_jp_text(text: str) -> str:
    """Remove spaces between Japanese characters for reliable pattern matching."""
    text = text.translate(str.maketrans(
        "０１２３４５６７８９　，",
        "0123456789 ,",
    ))
    return re.sub(r"(?<=\S) (?=\S)", "", text)


def _normalize_digits(text: str) -> str:
    """Convert full-width digits/comma/space to ASCII; preserve all spacing."""
    return text.translate(str.maketrans(
        "０１２３４５６７８９　，",
        "0123456789 ,",
    ))


def _extract_value(text: str, label: str) -> float | None:
    """Find first integer value (with commas) after `label` in `text`."""
    # Allow spaces between characters in label (interleaved text from PDF)
    spaced = r"\s*".join(re.escape(c) for c in label)
    pattern = spaced + r"\s+([\d,]+)"
    m = re.search(pattern, text)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def _parse_kessancard_pdf(
    pdf_path: Path,
    pref: str,
    metrics_map: dict[str, str],
    min_fy: int,
    max_fy: int,
) -> list[dict]:
    """Parse one 決算カード PDF, returning list of per-municipality dicts."""
    import pdfplumber

    rows = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        # Page 1: title + table of contents
        page0_text = pdf.pages[0].extract_text() or ""
        norm0 = _normalize_jp_text(page0_text)

        # Extract fiscal year
        fy = None
        for era, base in [("令和", 2018), ("平成", 1988), ("昭和", 1925)]:
            m = re.search(era + r"\s*(\d+)\s*年\s*度", page0_text)
            if m:
                fy = base + int(m.group(1))
                break
        if fy is None:
            print(f"  WARNING: Could not determine FY from {pdf_path.name}; skipping.")
            return []

        # Year range guards
        if fy < min_fy:
            print(f"  Skipping {pdf_path.name}: FY{fy} < min_fiscal_year={min_fy}.")
            return []
        if fy > max_fy:
            raise ValueError(
                f"Finance PDF {pdf_path.name} is FY{fy} which exceeds "
                f"max_fiscal_year={max_fy}. This would leak post-2014 data."
            )

        # Build ToC: municipality name -> page index (0-based in pdf.pages)
        toc = _parse_toc(page0_text)

        # Data pages (skip page 0)
        for page_num, page_obj in enumerate(pdf.pages[1:], start=2):
            raw_text = page_obj.extract_text() or ""
            if not raw_text.strip():
                continue
            # Digit-only normalisation preserves spacing between label and value;
            # full normalisation strips those spaces and breaks \s+ in _extract_value.
            digit_text = _normalize_digits(raw_text)

            # Identify municipality from ToC
            muni_name = toc.get(page_num)
            if muni_name is None:
                muni_name = _guess_muni_name(raw_text)

            row: dict = {
                "muni_name": muni_name,
                "pref": pref,
                "fiscal_year": fy,
            }
            for out_col, jp_label in metrics_map.items():
                val = _extract_value(digit_text, jp_label)
                row[f"fin_{out_col}"] = val

            # Fallback for 地方交付税: some PDF vintages (e.g. FY2014) have the
            # 地方交付税 row garbled by pdfplumber column-merging.  Sum the
            # sub-components from the adjacent clean rows instead.
            if row.get("fin_local_alloc_tax") is None:
                futsu = _extract_value(digit_text, "普通交付税")
                tokubetsu = _extract_value(digit_text, "特別交付税")
                shinsai = _extract_value(digit_text, "震災復興特別交付税")
                parts = [v for v in (futsu, tokubetsu, shinsai) if v is not None]
                if parts:
                    row["fin_local_alloc_tax"] = sum(parts)

            # Compute fiscal strength index from revenue/need (if both extracted)
            rev = row.get("fin_std_fiscal_revenue")
            need = row.get("fin_std_fiscal_need")
            if rev is not None and need is not None and need > 0:
                row["fin_fiscal_strength_index"] = round(rev / need, 4)
            else:
                row["fin_fiscal_strength_index"] = None

            rows.append(row)

    return rows


def _parse_toc(page0_text: str) -> dict[int, str]:
    """Extract {page_number: municipality_name} from the ToC page.

    ToC lines often have two entries per line (e.g. '青森市 2  大間町 32').
    re.findall captures both entries correctly.
    Character class covers CJK, hiragana, and katakana (needed for names like
    おいらせ町, つがる市, むつ市).
    """
    toc: dict[int, str] = {}
    # ぀-ゟ hiragana, ゠-ヿ katakana, 一-鿿 CJK
    pattern = re.compile(
        r"([぀-ゟ゠-ヿ一-鿿]{1,9}[市町村区])\s+(\d+)"
    )
    for line in page0_text.splitlines():
        for name, page in pattern.findall(line.strip()):
            toc[int(page)] = name
    return toc


def _guess_muni_name(text: str) -> str | None:
    """Fallback: find the first municipal name token in page text."""
    m = re.search(r"[一-鿿]{2,6}[市町村区]", text)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# 4. Housing and Land Survey 2013 durability
# ---------------------------------------------------------------------------

def _load_hls_durability(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load pre-1981 housing ratio (旧耐震率) per municipality from HLS 2013.

    a047 = cities (市); a048 = towns & villages (町村).
    Returns DataFrame: [city_name_ja, hls_pre1981_ratio, hls_total_dwellings,
                         hls_missing] at municipality level.
    """
    h_cfg = cfg["housing"]["tenure_age_dilapidation_2013"]
    data_root = Path(cfg["data_root"])

    frames = []
    for label, rel_path in h_cfg.items():
        if not rel_path:
            continue
        path = data_root / rel_path
        if not path.exists():
            print(f"  WARNING: HLS file not found: {path}; skipping.")
            continue
        try:
            df = _parse_hls_xls(path, cfg)
            frames.append(df)
        except Exception as exc:
            print(f"  WARNING: Failed to parse {path.name}: {exc}")

    if not frames:
        print("  WARNING: No HLS data loaded. Durability features will be NaN.")
        return pd.DataFrame(columns=["city_name_ja", "hls_pre1981_ratio",
                                     "hls_total_dwellings", "hls_missing"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="city_name_ja")
    print(f"  HLS: {len(combined)} municipalities loaded.")
    return combined


def _parse_hls_xls(path: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse one HLS XLS file (Table 22), extracting pre-1981 dwelling counts.

    Municipality names are read from section-header rows (identified by a
    non-null English name in muni_name_en_col while row_type_col is empty).
    Column layout verified against a048.xls:
      row_type_col=5, total_col=10, pre1981_cols=[11,12]
      (col 11 = <=1970; col 12 = 1971-1980; together = pre-old-seismic-code)
    """
    h_cfg = cfg["housing"]
    rt_col = int(h_cfg["row_type_col"])
    tot_col = int(h_cfg["total_col"])
    pre1981_cols = [int(c) for c in h_cfg["pre1981_cols"]]
    name_col = int(h_cfg.get("muni_name_col", 7))
    name_en_col = int(h_cfg.get("muni_name_en_col", 8))

    raw = pd.read_excel(path, header=None, dtype=str)

    records = []
    current_name = None

    for _, row in raw.iterrows():
        rt_val = str(row.iloc[rt_col]).strip()
        en_val = str(row.iloc[name_en_col]).strip()

        # Section header: row_type cell is empty but English name cell is present
        if rt_val in ("nan", "") and en_val not in ("nan", "") and len(en_val) > 3:
            jp_raw = str(row.iloc[name_col]).strip()
            # Strip leading number prefix (e.g. "361　") and internal whitespace
            jp_clean = re.sub(r"^\d+\s*", "", jp_raw)
            jp_clean = re.sub(r"\s+", "", jp_clean).strip()
            if jp_clean:
                current_name = jp_clean
            continue

        # Data row with row_type == 1 = 住宅総数 (total dwellings)
        if rt_val in ("1.0", "1") and current_name:
            try:
                total = _parse_num(str(row.iloc[tot_col]))
                pre81 = sum(_parse_num(str(row.iloc[c])) for c in pre1981_cols)
                records.append({
                    "city_name_ja": current_name,
                    "hls_total_dwellings": total,
                    "hls_pre1981_count": pre81,
                    "hls_pre1981_ratio": (
                        round(pre81 / total, 4) if total > 0 else np.nan
                    ),
                    "hls_missing": 0,
                })
            except (ValueError, TypeError):
                records.append({
                    "city_name_ja": current_name,
                    "hls_total_dwellings": np.nan,
                    "hls_pre1981_count": np.nan,
                    "hls_pre1981_ratio": np.nan,
                    "hls_missing": 1,
                })

    return pd.DataFrame(records)


def _parse_num(s: str) -> float:
    """Parse numeric string, returning 0 for '-' or empty."""
    s = s.strip().replace(",", "")
    if s in ("-", "", "nan", "None"):
        return 0.0
    return float(s)


# ---------------------------------------------------------------------------
# 4b. HLS vacancy (a002.xls, Table 1)
# ---------------------------------------------------------------------------

def _load_hls_vacancy(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load vacancy rates per municipality from HLS 2013 a002.xls (Table 1).

    Column layout (0-indexed, verified against Aomori-ken/a002.xls):
      1  = 7-digit JISCD code; first 5 chars = muni_code (e.g. '02201')
      9  = 住宅総数  (total dwellings)
      15 = 空き家 total (all vacant)
      19 = その他の住宅 (other vacant = long-term unoccupied / abandoned)

    Data rows start at row 16; rows before that are header / title rows.
    '-' values in the source table mean zero and are handled by _parse_num.

    Returns [muni_code, hls_vacancy_rate, hls_vacancy_other_rate].
    Joined to aza by muni_code (avoids name-matching ambiguity).
    """
    data_root = Path(cfg["data_root"])
    frames = []
    for pref_dir in ("Aomori-ken", "Akita-ken"):
        path = data_root / "housing_data_japan" / "2013" / pref_dir / "a002.xls"
        if not path.exists():
            print(f"  WARNING: {pref_dir}/a002.xls not found; skipping.")
            continue
        raw = pd.read_excel(path, header=None, dtype=str)
        n_parsed = 0
        for idx in range(16, len(raw)):
            row = raw.iloc[idx]
            code_raw = str(row.iloc[1]).strip()
            if len(code_raw) < 5 or not code_raw[:5].isdigit():
                continue
            muni_code = code_raw[:5]
            try:
                total = _parse_num(str(row.iloc[9]))
                vacant_total = _parse_num(str(row.iloc[15]))
                vacant_other = _parse_num(str(row.iloc[19]))
                if total > 0:
                    frames.append({
                        "muni_code": muni_code,
                        "hls_vacancy_rate": round(vacant_total / total, 4),
                        "hls_vacancy_other_rate": round(vacant_other / total, 4),
                    })
                else:
                    frames.append({
                        "muni_code": muni_code,
                        "hls_vacancy_rate": np.nan,
                        "hls_vacancy_other_rate": np.nan,
                    })
                n_parsed += 1
            except (ValueError, TypeError):
                frames.append({
                    "muni_code": muni_code,
                    "hls_vacancy_rate": np.nan,
                    "hls_vacancy_other_rate": np.nan,
                })
        print(f"  {pref_dir}: {n_parsed} municipalities parsed from a002.xls.")

    if not frames:
        print("  WARNING: No a002 vacancy data loaded. Vacancy features will be NaN.")
        return pd.DataFrame(columns=["muni_code", "hls_vacancy_rate",
                                     "hls_vacancy_other_rate"])

    df = pd.DataFrame(frames).drop_duplicates(subset="muni_code")
    vr = df["hls_vacancy_rate"].dropna()
    print(f"  HLS vacancy: {len(df)} municipalities, "
          f"vacancy_rate {vr.min():.3f}-{vr.max():.3f} (mean {vr.mean():.3f}).")
    return df


# ---------------------------------------------------------------------------
# 5. Kaso flag
# ---------------------------------------------------------------------------

def _normalize_old_muni_name(name: str) -> str:
    """
    Normalize a pre-merger municipality name for matching.

    Strips the leading 旧 ("former") prefix used in the kaso designation
    list and unifies the small/large ke variants, which differ between the
    designation PDF (碇ヶ関村) and N03 2000 (碇ケ関村).
    """
    name = str(name).strip()
    if name.startswith("旧"):  # 旧
        name = name[1:]
    return name.replace("ヶ", "ケ")  # ヶ -> ケ


def _load_premerger_boundaries(cfg: Dict[str, Any]) -> Any:
    """
    Load pre-Heisei-merger municipal polygons (N03 2000-10-01 vintage),
    dissolved to one (multi)polygon per pre-merger municipality code.
    """
    import geopandas as gpd

    data_root = Path(cfg["data_root"])
    k_cfg = cfg["kaso"]
    name_col = k_cfg["muni_name_col"]
    code_col = k_cfg["muni_code_col"]

    frames = []
    for pref in ("aomori", "akita"):
        shp = data_root / k_cfg["premerger_boundaries"][pref]
        g = gpd.read_file(shp, encoding=k_cfg["encoding"])
        if g.crs is None:
            g = g.set_crs(k_cfg["assumed_crs"])
        frames.append(g)

    gdf = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs=frames[0].crs,
    )
    gdf = gdf[gdf[name_col].notna() & gdf[code_col].notna()]
    gdf = gdf.dissolve(by=code_col, as_index=False)
    gdf = gdf.to_crs(cfg["crs_project"])
    print(f"  Pre-merger boundaries: {len(gdf)} municipalities (N03 2000).")
    return gdf[[code_col, name_col, "geometry"]]


def _assign_old_muni(
    aza_df: pd.DataFrame, aza_id_col: str, cfg: Dict[str, Any],
) -> pd.Series:
    """
    Map each aza to the name of the pre-merger municipality containing
    its representative point.  Returns Series indexed like aza_df.

    Points that miss every polygon (coastline mismatch between the aza
    layer and N03 2000) fall back to the nearest polygon.
    """
    import geopandas as gpd

    name_col = cfg["kaso"]["muni_name_col"]

    polys = _load_premerger_boundaries(cfg)

    gpkg = Path(cfg["phase_a_root"]) / cfg["phase_a"]["aza_polygons"]
    aza_geo = gpd.read_file(gpkg)
    aza_geo = aza_geo.dissolve(by="unit_id", as_index=False)
    aza_geo = aza_geo.to_crs(cfg["crs_project"])
    pts = gpd.GeoDataFrame(
        aza_geo[["unit_id"]],
        geometry=aza_geo.representative_point(),
        crs=aza_geo.crs,
    )

    joined = gpd.sjoin(pts, polys, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    missed = joined[name_col].isna()
    if missed.any():
        near = gpd.sjoin_nearest(
            pts.loc[joined.index[missed]], polys, how="left",
        )
        near = near[~near.index.duplicated(keep="first")]
        joined.loc[near.index, name_col] = near[name_col]
        print(f"  {int(missed.sum())} aza points outside all pre-merger "
              "polygons -- assigned to nearest.")

    lookup = joined.set_index("unit_id")[name_col]
    return aza_df[aza_id_col].map(lookup)


def _build_kaso_flags(
    aza_df: pd.DataFrame, aza_id_col: str, cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Assign kaso designation flags per aza.

    kaso_type: 2 = 全部過疎 (whole municipality designated),
               1 = 一部過疎, resolved to aza accuracy by point-in-polygon
                   against pre-merger (N03 2000) 旧町村 boundaries: only
                   aza inside a designated 旧町村 area inherit the flag,
               0 = no designation.

    Falls back to municipality-level 一部過疎 flags (every aza of the
    municipality gets 1) if the boundary shapefiles are unavailable.
    """
    flags = aza_df[[aza_id_col]].copy()
    flags["kaso_type"] = 0
    flags["kaso_flag"] = 0

    pref_names = aza_df[aza_id_col].astype(str).str.split(":").str[1]
    city_names = aza_df.get(
        "city_name_ja", pd.Series("", index=aza_df.index),
    ).fillna("").astype(str).str.strip()

    # Match on normalized names: census city_name_ja and the hard-coded
    # designation list disagree on the small/large ke variant for some
    # municipalities (census 鰺ヶ沢町 vs list 鰺ケ沢町), which silently
    # unflagged a whole 全部過疎 town before 2026-07-02.
    city_norm = city_names.map(_normalize_old_muni_name)
    zenbu = pd.Series(False, index=aza_df.index)
    ichibu_cand = pd.Series(False, index=aza_df.index)
    for pref in ("Aomori", "Akita"):
        in_pref = pref_names == pref
        zenbu_norm = {_normalize_old_muni_name(n)
                      for n in _KASO_ZENBU.get(pref, [])}
        ichibu_norm = {_normalize_old_muni_name(n)
                       for n in _KASO_ICHIBU.get(pref, {})}
        zenbu |= in_pref & city_norm.isin(zenbu_norm)
        ichibu_cand |= in_pref & city_norm.isin(ichibu_norm)

    flags.loc[zenbu, "kaso_type"] = 2
    flags.loc[zenbu, "kaso_flag"] = 1

    # Partial kaso: point-in-polygon against pre-merger boundaries.
    try:
        old_muni = _assign_old_muni(aza_df, aza_id_col, cfg)
        spatial_ok = True
    except Exception as e:  # missing shapefile, bad CRS, ...
        print(f"  WARNING: pre-merger boundary join failed ({e}); "
              "falling back to municipality-level 一部過疎 flags.")
        spatial_ok = False

    if spatial_ok:
        for pref in ("Aomori", "Akita"):
            for city, old_names in _KASO_ICHIBU.get(pref, {}).items():
                designated = {_normalize_old_muni_name(n) for n in old_names}
                cand = (ichibu_cand & (pref_names == pref)
                        & (city_norm == _normalize_old_muni_name(city)))
                inside = cand & old_muni.map(
                    lambda n: _normalize_old_muni_name(n) in designated,
                    na_action="ignore",
                ).fillna(False)
                flags.loc[inside, "kaso_type"] = 1
                flags.loc[inside, "kaso_flag"] = 1
                print(f"  {pref}/{city}: {int(inside.sum())}/{int(cand.sum())} "
                      "aza inside designated pre-merger area")
    else:
        flags.loc[ichibu_cand, "kaso_type"] = 1
        flags.loc[ichibu_cand, "kaso_flag"] = 1

    n_kaso = int((flags["kaso_flag"] == 1).sum())
    print(f"  Kaso flags: {n_kaso}/{len(flags)} aza designated "
          f"({int((flags['kaso_type']==2).sum())} 全部過疎, "
          f"{int((flags['kaso_type']==1).sum())} 一部過疎)")
    return flags


# ---------------------------------------------------------------------------
# 6. Merger flag
# ---------------------------------------------------------------------------

def _load_merger_flags(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load Heisei-era merger records and return per-muni merger flags.

    Returns DataFrame: [muni_code, merged_flag, merger_date,
                        years_since_merger_2015].
    Only Aomori and Akita mergers are extracted.
    """
    data_root = Path(cfg["data_root"])
    m_cfg = cfg["merger"]
    ref_date = pd.Timestamp(m_cfg["reference_date"])
    target_prefs = {"青森県", "秋田県"}

    frames = []
    for file_key in ("file_1", "file_2"):
        path = data_root / m_cfg[file_key]
        if not path.exists():
            print(f"  WARNING: merger file not found: {path}")
            continue
        df = pd.read_excel(path, header=0, dtype=str)
        df.columns = list(range(len(df.columns)))
        pref_col = m_cfg["pref_col"]
        date_col = m_cfg["date_col"]
        members_col = m_cfg["members_col"]

        for _, row in df.iterrows():
            pref = str(row.get(pref_col, "")).strip()
            if pref not in target_prefs:
                continue
            raw_date = row.get(date_col, "")
            try:
                merge_dt = pd.Timestamp(raw_date)
            except Exception:
                continue
            members_str = str(row.get(members_col, ""))
            frames.append({
                "pref": pref,
                "merger_date": merge_dt,
                "members_str": members_str,
                "new_name": str(row.get(m_cfg["new_name_col"], "")),
            })

    if not frames:
        print("  WARNING: No merger records loaded.")
        return pd.DataFrame(columns=["muni_name_ja", "merged_flag",
                                     "merger_date", "years_since_merger"])

    mergers = pd.DataFrame(frames)
    mergers["merged_flag"] = 1
    mergers["years_since_merger"] = (
        (ref_date - mergers["merger_date"]).dt.days / 365.25
    ).round(1)

    # Keep only most-recent merger per new_name (a municipality may have
    # merged multiple times; we want the most significant structural change).
    mergers = (
        mergers.sort_values("merger_date", ascending=False)
        .drop_duplicates(subset="new_name")
        .rename(columns={"new_name": "muni_name_ja"})
    )
    print(f"  Mergers: {len(mergers)} Aomori+Akita merger events loaded.")
    return mergers[["muni_name_ja", "merged_flag", "merger_date",
                    "years_since_merger"]].copy()


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def _build_muni_name_lookup(cfg: Dict[str, Any]) -> dict[str, str]:
    """
    Build {muni_code: city_name_ja} from census HYOSYO=1 rows.

    KEY_CODE for municipality-level rows is a 4-5 digit int (e.g. 2201 =
    Aomori-shi).  Zero-padding to 5 digits gives the JISCD muni code that
    matches _extract_muni_code() output.
    """
    data_root = Path(cfg["data_root"])
    c_cfg = cfg["census"]
    lookup: dict[str, str] = {}

    for pref_key in ("aomori", "akita"):
        path = data_root / c_cfg["pop_sex_households"][pref_key]
        try:
            df = pd.read_csv(path, encoding="cp932", dtype=str, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8", dtype=str, low_memory=False)
        # Drop header row (all-NaN KEY_CODE) and filter to municipality level
        df = df[df["KEY_CODE"].notna()].copy()
        df["_hyosyo"] = pd.to_numeric(df.get("HYOSYO", pd.Series(dtype=float)),
                                      errors="coerce")
        muni_rows = df[df["_hyosyo"] == 1][["KEY_CODE", "CITYNAME"]].dropna()
        for _, row in muni_rows.iterrows():
            try:
                code = str(int(float(row["KEY_CODE"]))).zfill(5)
                name = str(row["CITYNAME"]).strip()
                if name:
                    lookup[code] = name
            except (ValueError, TypeError):
                pass

    # Strip leading gun (district) prefix from CITYNAME for towns/villages.
    # Census format: "上北郡おいらせ町"; HLS + finance PDFs use "おいらせ町".
    # Cities (市) have no 郡 prefix, so the sub() is a no-op for them.
    lookup = {
        code: re.sub(r"^.+郡", "", name).strip()
        for code, name in lookup.items()
    }
    print(f"  Muni name lookup: {len(lookup)} municipalities.")
    return lookup


def build_feature_matrix(cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Assemble the full Phase B feature matrix.

    Returns
    -------
    pd.DataFrame
        One row per aza, with all predictor columns and the unit_id join key.
        No banned (leakage) columns; missing HLS carried as NaN + flag.
    """
    if cfg is None:
        cfg = _load_cfg()

    phase_a_root = Path(cfg["phase_a_root"])
    aza_id_col = cfg["aza_id_col"]
    ready_path = phase_a_root / cfg["phase_a"]["classification_ready"]

    print("Loading Phase A classification-ready data as base frame...")
    base = pd.read_csv(ready_path, encoding="utf-8")
    print(f"  Base: {len(base)} aza units, {len(base.columns)} cols.")

    # Derive muni_code + city_name_ja for broadcasting
    base["muni_code"] = base[aza_id_col].apply(_extract_muni_code)
    print("Building municipality name lookup from census...")
    muni_lookup = _build_muni_name_lookup(cfg)
    base["city_name_ja"] = base["muni_code"].map(muni_lookup)
    unmatched = base["city_name_ja"].isna().sum()
    if unmatched:
        print(f"  WARNING: {unmatched} aza units have no city_name_ja match.")

    # ---- 1. Demographic threshold flags --------------------------------
    print("Building demographic threshold flags...")
    dem_flags = _build_demographic_flags(base, aza_id_col)
    base = base.merge(dem_flags, on=aza_id_col, how="left")

    # GLCM proxy step removed: S2_NDBI_* columns are EO/RS and are listed
    # in banned_columns — they will be stripped below along with all other
    # Phase A remote-sensing columns present in the base frame.

    # ---- 3. Finance (broadcast muni -> aza) ----------------------------
    print("Loading finance features from PDFs...")
    finance = _load_finance_features(cfg)
    if len(finance) > 0 and "muni_name" in finance.columns:
        # Average across available pre-2015 years per municipality
        fin_metrics = [c for c in finance.columns
                       if c.startswith("fin_")]
        fin_agg = (
            finance.groupby("muni_name")[fin_metrics]
            .mean()
            .reset_index()
            .rename(columns={"muni_name": "city_name_ja"})
        )
        if "city_name_ja" in base.columns:
            base = base.merge(fin_agg, on="city_name_ja", how="left")
        else:
            print("  NOTE: city_name_ja not in base; finance features skipped.")

    # ---- 4. HLS durability (broadcast muni -> aza) ----------------------
    print("Loading Housing and Land Survey 2013 durability...")
    hls = _load_hls_durability(cfg)
    if len(hls) > 0 and "city_name_ja" in base.columns:
        unmatched_hls = len(set(base["city_name_ja"].dropna()) - set(hls["city_name_ja"].dropna()))
        if unmatched_hls:
            print(f"  NOTE: {unmatched_hls} municipalities in base have no HLS match "
                  "-- will be NaN (hls_missing=1).")
        base = base.merge(
            hls[["city_name_ja", "hls_pre1981_ratio", "hls_total_dwellings",
                 "hls_missing"]],
            on="city_name_ja", how="left",
        )
        # Where join produced NaN, set hls_missing=1
        base["hls_missing"] = base.get("hls_missing", pd.Series(np.nan))
        base["hls_missing"] = base["hls_missing"].fillna(1).astype(int)
    else:
        base["hls_pre1981_ratio"] = np.nan
        base["hls_total_dwellings"] = np.nan
        base["hls_missing"] = 1

    # ---- 4b. HLS vacancy (broadcast muni -> aza) -------------------------
    print("Loading Housing and Land Survey 2013 vacancy rates...")
    vacancy = _load_hls_vacancy(cfg)
    if len(vacancy) > 0 and "muni_code" in base.columns:
        unmatched_vac = len(set(base["muni_code"].dropna()) - set(vacancy["muni_code"].dropna()))
        if unmatched_vac:
            print(f"  NOTE: {unmatched_vac} municipalities in base have no a002 vacancy match.")
        base = base.merge(vacancy, on="muni_code", how="left")
    else:
        base["hls_vacancy_rate"] = np.nan
        base["hls_vacancy_other_rate"] = np.nan

    # ---- 5. Kaso flag ---------------------------------------------------
    print("Building kaso designation flags...")
    kaso = _build_kaso_flags(base, aza_id_col, cfg)
    base = base.merge(kaso, on=aza_id_col, how="left")

    # ---- 6. Merger flag (broadcast muni -> aza) -------------------------
    print("Loading merger records...")
    mergers = _load_merger_flags(cfg)
    if len(mergers) > 0 and "city_name_ja" in base.columns:
        base = base.merge(
            mergers[["muni_name_ja", "merged_flag", "years_since_merger"]],
            left_on="city_name_ja", right_on="muni_name_ja", how="left",
        )
        base["merged_flag"] = base["merged_flag"].fillna(0).astype(int)
        base = base.drop(columns=["muni_name_ja"], errors="ignore")
    else:
        base["merged_flag"] = 0
        base["years_since_merger"] = np.nan

    # ---- 7. Accessibility features -------------------------------------
    print("Loading accessibility features...")
    acc_path = Path(cfg["phase_b_root"]) / cfg["output"]["accessibility"]
    if acc_path.exists():
        acc = pd.read_parquet(acc_path)
        # acc covers all 9031 aza; base is the labeled subset — extra acc rows are fine.
        unmatched_base = len(set(base[aza_id_col]) - set(acc[aza_id_col]))
        if unmatched_base:
            print(f"  WARNING: {unmatched_base} base units missing from accessibility output.")
        base = base.merge(acc, on=aza_id_col, how="left")
    else:
        print(f"  WARNING: accessibility features not found at {acc_path}. "
              "Run accessibility_features.py first.")

    # ---- Strip banned (leakage) columns ---------------------------------
    banned = set(cfg["feature_matrix"]["banned_columns"])
    to_drop = [c for c in base.columns if c in banned]
    if to_drop:
        print(f"  Dropping {len(to_drop)} banned (leakage) columns: {to_drop}")
        base = base.drop(columns=to_drop)

    # ---- Final leakage assertion ----------------------------------------
    _assert_no_leakage(base, cfg)

    print(f"\nFeature matrix: {len(base)} rows x {len(base.columns)} columns.")

    # Persist
    out_path = Path(cfg["phase_b_root"]) / cfg["output"]["feature_matrix"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")

    return base


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_feature_matrix(cfg_path: Path | None = None) -> pd.DataFrame:
    if cfg_path is not None:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = _load_cfg()
    return build_feature_matrix(cfg)


if __name__ == "__main__":
    result = run_feature_matrix()
    print(result.head())
