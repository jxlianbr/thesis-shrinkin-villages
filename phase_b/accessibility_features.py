"""
Phase B — Per-aza accessibility features from NLNI layers.

Computes Euclidean nearest-distance (m) from each aza representative point
to: DID polygons, medical facilities, schools, bus stops, bus routes, and
community facilities (P05 LocalGovernmentOffice & PublicMeetingFacility).

All layers are reprojected to EPSG:6680 (JGD2011 Japan Zone 10, metres)
before distance computation.  Ties in sjoin_nearest are resolved by taking
the minimum distance per aza.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

_HERE = Path(__file__).resolve().parent
_CFG_PATH = _HERE / "config" / "phase_b_config.yaml"


def _load_cfg() -> Dict[str, Any]:
    with open(_CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Geometry loading helpers
# ---------------------------------------------------------------------------

def _load_nlni_points(path: Path, assumed_crs: str = "EPSG:4612") -> gpd.GeoDataFrame:
    """Load a point shapefile, fixing missing CRS and encoding issues."""
    try:
        gdf = gpd.read_file(path, encoding="shift_jis")
    except Exception:
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(assumed_crs)
    return gdf


def _load_nlni_lines(path: Path, assumed_crs: str = "EPSG:4612") -> gpd.GeoDataFrame:
    """Load a line shapefile, fixing missing CRS and stripping M-geometry."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            gdf = gpd.read_file(path, encoding="shift_jis")
        except Exception:
            gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(assumed_crs)
    # Measured (M) geometry is already stripped by pyogrio; just ensure CRS.
    return gdf


def _load_did(path: Path, skip_field: str, assumed_crs: str = "EPSG:4612") -> gpd.GeoDataFrame:
    """Load DID polygon shapefile, skipping the year-0 datetime field."""
    try:
        import pyogrio
        info = pyogrio.read_info(str(path))
        fields = [f for f in info["fields"] if f != skip_field]
        gdf = pyogrio.read_dataframe(str(path), columns=fields)
    except Exception:
        # Fallback: let geopandas load without datetime, ignore parse errors.
        gdf = gpd.read_file(path, ignore_geometry=False)

    if gdf.crs is None:
        gdf = gdf.set_crs(assumed_crs)
    return gdf


def _load_aza_polygons(phase_a_root: Path, rel_path: str) -> gpd.GeoDataFrame:
    path = phase_a_root / rel_path
    gdf = gpd.read_file(path)
    return gdf


# ---------------------------------------------------------------------------
# Distance helpers  (hook: swap these for network-routing equivalents)
# ---------------------------------------------------------------------------

def _euclidean_dist_point(
    aza_pts: gpd.GeoDataFrame,
    facility: gpd.GeoDataFrame,
    aza_id_col: str,
) -> pd.Series:
    """Return Series indexed by aza_id_col of min Euclidean distance (m)."""
    joined = gpd.sjoin_nearest(
        aza_pts[[aza_id_col, "geometry"]],
        facility[["geometry"]],
        how="left",
        distance_col="_dist",
    )
    return joined.groupby(aza_id_col)["_dist"].min()


def _euclidean_dist_line(
    aza_pts: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    aza_id_col: str,
) -> pd.Series:
    """Return Series of min Euclidean distance (m) from each aza point to any line."""
    joined = gpd.sjoin_nearest(
        aza_pts[[aza_id_col, "geometry"]],
        lines[["geometry"]],
        how="left",
        distance_col="_dist",
    )
    return joined.groupby(aza_id_col)["_dist"].min()


def _buffer_count(
    aza_pts: gpd.GeoDataFrame,
    facility: gpd.GeoDataFrame,
    radius_m: float,
    aza_id_col: str,
) -> pd.Series:
    """Count facilities within radius_m of each aza representative point."""
    aza_buf = aza_pts[[aza_id_col, "geometry"]].copy()
    aza_buf["geometry"] = aza_buf.geometry.buffer(radius_m)
    joined = gpd.sjoin(
        aza_buf,
        facility[["geometry"]],
        how="left",
        predicate="intersects",
    )
    counts = joined.groupby(aza_id_col)["index_right"].count()
    # Return a clean DataFrame so callers can merge on aza_id_col by column,
    # avoiding any index-dtype ambiguity that breaks right_index=True merges.
    ref = pd.DataFrame({aza_id_col: aza_pts[aza_id_col].unique()})
    result = ref.merge(counts.reset_index(name="_cnt"), on=aza_id_col, how="left")
    result["_cnt"] = result["_cnt"].fillna(0).astype(int)
    return result.set_index(aza_id_col)["_cnt"]


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_accessibility(cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Compute all accessibility features per aza.

    Returns
    -------
    pd.DataFrame
        One row per aza, columns:
          dist_did_m, in_did,
          dist_medical_m, dist_hospital_m,
          dist_school_m, dist_elementary_m,
          dist_bus_stop_m,
          dist_bus_route_m,
          dist_community_facility_m,
          n_medical_{r}m, n_school_{r}m, n_bus_stop_{r}m,
          n_community_facility_{r}m  (per buffer radius)
    """
    if cfg is None:
        cfg = _load_cfg()

    phase_a_root = Path(cfg["phase_a_root"])
    data_root = Path(cfg["data_root"])
    nlni_cfg = cfg["nlni"]
    aza_id_col = cfg["aza_id_col"]
    crs_proj = cfg["crs_project"]
    buffers = list(nlni_cfg["buffer_radii_m"])

    # --- Load aza polygons and compute representative points ---
    print("Loading aza polygons...")
    aza = _load_aza_polygons(phase_a_root, cfg["phase_a"]["aza_polygons"])
    aza_proj = aza.to_crs(crs_proj)
    aza_pts = aza_proj.copy()
    aza_pts["geometry"] = aza_proj.geometry.representative_point()
    print(f"  {len(aza_pts)} aza units loaded.")

    out = aza[[aza_id_col]].copy()

    # --- DID (Densely Inhabited District) ---
    print("Computing DID accessibility...")
    did_aomori = _load_did(
        data_root / nlni_cfg["did"]["aomori"],
        skip_field=nlni_cfg["did"]["skip_datetime_field"],
        assumed_crs=nlni_cfg["did"]["assumed_crs"],
    ).to_crs(crs_proj)
    did_akita = _load_did(
        data_root / nlni_cfg["did"]["akita"],
        skip_field=nlni_cfg["did"]["skip_datetime_field"],
        assumed_crs=nlni_cfg["did"]["assumed_crs"],
    ).to_crs(crs_proj)
    did_all = pd.concat([did_aomori, did_akita], ignore_index=True)
    did_all = gpd.GeoDataFrame(did_all, crs=crs_proj)

    # Nearest-DID distance
    joined_did = gpd.sjoin_nearest(
        aza_pts[[aza_id_col, "geometry"]],
        did_all[["geometry"]],
        how="left",
        distance_col="_dist_did",
    )
    dist_did = joined_did.groupby(aza_id_col)["_dist_did"].min()
    out = out.merge(dist_did.rename("dist_did_m"), on=aza_id_col, how="left")

    # in_did: distance == 0 means centroid is inside a DID polygon
    out["in_did"] = (out["dist_did_m"] == 0).astype(int)
    print(f"  DID: {out['in_did'].sum()} aza centroids inside DID.")

    # --- Medical facilities (P04) ---
    print("Computing medical accessibility...")
    med_aomori = _load_nlni_points(
        data_root / nlni_cfg["medical"]["aomori"],
    ).to_crs(crs_proj)
    med_akita = _load_nlni_points(
        data_root / nlni_cfg["medical"]["akita"],
    ).to_crs(crs_proj)
    med_all = pd.concat([med_aomori, med_akita], ignore_index=True)
    med_all = gpd.GeoDataFrame(med_all, crs=crs_proj)

    dist_med = _euclidean_dist_point(aza_pts, med_all, aza_id_col)
    out = out.merge(dist_med.rename("dist_medical_m"), on=aza_id_col, how="left")

    # Hospital-only filter (P04_001 == hospital_type_code)
    hosp_col = nlni_cfg["medical"]["hospital_type_col"]
    hosp_code = str(nlni_cfg["medical"]["hospital_type_code"])
    if hosp_col in med_all.columns:
        hosp_all = med_all[med_all[hosp_col].astype(str) == hosp_code].copy()
        if len(hosp_all) == 0:
            print(f"  WARNING: hospital filter on {hosp_col}={hosp_code} "
                  "returned zero rows; falling back to all medical.")
            hosp_all = med_all
    else:
        print(f"  WARNING: column {hosp_col} absent in P04; "
              "hospital-only distance equals all-medical distance.")
        hosp_all = med_all

    dist_hosp = _euclidean_dist_point(aza_pts, hosp_all, aza_id_col)
    out = out.merge(dist_hosp.rename("dist_hospital_m"), on=aza_id_col, how="left")

    # Medical buffer counts
    for r in buffers:
        cnt = _buffer_count(aza_pts, med_all, r, aza_id_col)
        out = out.merge(
            cnt.rename(f"n_medical_{r}m"),
            on=aza_id_col, how="left",
        )

    # --- Schools (P29) ---
    print("Computing school accessibility...")
    sch_aomori = _load_nlni_points(
        data_root / nlni_cfg["schools"]["aomori"],
    ).to_crs(crs_proj)
    sch_akita = _load_nlni_points(
        data_root / nlni_cfg["schools"]["akita"],
    ).to_crs(crs_proj)
    sch_all = pd.concat([sch_aomori, sch_akita], ignore_index=True)
    sch_all = gpd.GeoDataFrame(sch_all, crs=crs_proj)

    dist_sch = _euclidean_dist_point(aza_pts, sch_all, aza_id_col)
    out = out.merge(dist_sch.rename("dist_school_m"), on=aza_id_col, how="left")

    # Elementary-school-only filter (P29_003 == elementary_type_code)
    elem_col = nlni_cfg["schools"]["school_type_col"]
    elem_code = str(nlni_cfg["schools"]["elementary_type_code"])
    if elem_col in sch_all.columns:
        elem_all = sch_all[sch_all[elem_col].astype(str) == elem_code].copy()
        if len(elem_all) == 0:
            print(f"  WARNING: elementary filter on {elem_col}={elem_code} "
                  "returned zero rows; falling back to all schools.")
            elem_all = sch_all
    else:
        print(f"  WARNING: column {elem_col} absent in P29; "
              "elementary distance equals all-school distance.")
        elem_all = sch_all

    dist_elem = _euclidean_dist_point(aza_pts, elem_all, aza_id_col)
    out = out.merge(dist_elem.rename("dist_elementary_m"), on=aza_id_col, how="left")

    for r in buffers:
        cnt = _buffer_count(aza_pts, sch_all, r, aza_id_col)
        out = out.merge(
            cnt.rename(f"n_school_{r}m"),
            on=aza_id_col, how="left",
        )

    # --- Bus stops (P11) ---
    print("Computing bus-stop accessibility...")
    bs_aomori = _load_nlni_points(
        data_root / nlni_cfg["bus_stops"]["aomori"],
        assumed_crs=nlni_cfg["bus_stops"]["assumed_crs"],
    ).to_crs(crs_proj)
    bs_akita = _load_nlni_points(
        data_root / nlni_cfg["bus_stops"]["akita"],
        assumed_crs=nlni_cfg["bus_stops"]["assumed_crs"],
    ).to_crs(crs_proj)
    bs_all = pd.concat([bs_aomori, bs_akita], ignore_index=True)
    bs_all = gpd.GeoDataFrame(bs_all, crs=crs_proj)

    dist_bs = _euclidean_dist_point(aza_pts, bs_all, aza_id_col)
    out = out.merge(dist_bs.rename("dist_bus_stop_m"), on=aza_id_col, how="left")

    for r in buffers:
        cnt = _buffer_count(aza_pts, bs_all, r, aza_id_col)
        out = out.merge(
            cnt.rename(f"n_bus_stop_{r}m"),
            on=aza_id_col, how="left",
        )

    # --- Bus routes (N07) ---
    print("Computing bus-route accessibility...")
    br_aomori = _load_nlni_lines(
        data_root / nlni_cfg["bus_routes"]["aomori"],
        assumed_crs=nlni_cfg["bus_routes"]["assumed_crs"],
    ).to_crs(crs_proj)
    br_akita = _load_nlni_lines(
        data_root / nlni_cfg["bus_routes"]["akita"],
        assumed_crs=nlni_cfg["bus_routes"]["assumed_crs"],
    ).to_crs(crs_proj)
    br_all = pd.concat([br_aomori, br_akita], ignore_index=True)
    br_all = gpd.GeoDataFrame(br_all, crs=crs_proj)

    dist_br = _euclidean_dist_line(aza_pts, br_all, aza_id_col)
    out = out.merge(dist_br.rename("dist_bus_route_m"), on=aza_id_col, how="left")

    # --- Community facilities (P05) ---
    print("Computing community-facility accessibility...")
    p05_cfg = nlni_cfg["community_facilities"]
    p05_aomori_path = p05_cfg.get("aomori")
    p05_akita_path = p05_cfg.get("akita")
    if p05_aomori_path is None or p05_akita_path is None:
        print("  WARNING: P05 paths are null in config; columns will be NaN.")
        out["dist_community_facility_m"] = np.nan
        for r in buffers:
            out[f"n_community_facility_{r}m"] = np.nan
    else:
        p05_assumed = p05_cfg.get("assumed_crs", "EPSG:4612")
        p05_ao = _load_nlni_points(
            data_root / p05_aomori_path, assumed_crs=p05_assumed,
        ).to_crs(crs_proj)
        p05_ak = _load_nlni_points(
            data_root / p05_akita_path, assumed_crs=p05_assumed,
        ).to_crs(crs_proj)
        p05_all = gpd.GeoDataFrame(
            pd.concat([p05_ao, p05_ak], ignore_index=True), crs=crs_proj,
        )
        dist_p05 = _euclidean_dist_point(aza_pts, p05_all, aza_id_col)
        out = out.merge(dist_p05.rename("dist_community_facility_m"),
                        on=aza_id_col, how="left")
        for r in buffers:
            cnt = _buffer_count(aza_pts, p05_all, r, aza_id_col)
            out = out.merge(
                cnt.rename(f"n_community_facility_{r}m"),
                on=aza_id_col, how="left",
            )
        print(f"  P05: {len(p05_all)} facilities loaded "
              f"(Aomori {len(p05_ao)} + Akita {len(p05_ak)}).")

    # --- Final key alignment assertion ---
    _assert_key_alignment(out, aza, aza_id_col)

    # Persist
    out_path = Path(cfg["phase_b_root"]) / cfg["output"]["accessibility"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Accessibility features saved -> {out_path}  ({len(out)} rows, "
          f"{len(out.columns)} cols)")

    return out


def _assert_key_alignment(
    result: pd.DataFrame,
    aza_ref: gpd.GeoDataFrame,
    aza_id_col: str,
) -> None:
    ref_keys = set(aza_ref[aza_id_col])
    result_keys = set(result[aza_id_col])
    missing_from_result = ref_keys - result_keys
    if missing_from_result:
        raise AssertionError(
            f"{len(missing_from_result)} aza units from aza.gpkg are missing "
            f"from accessibility output. First 5: "
            f"{sorted(missing_from_result)[:5]}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_accessibility(cfg_path: Path | None = None) -> pd.DataFrame:
    if cfg_path is not None:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = _load_cfg()
    return compute_accessibility(cfg)


if __name__ == "__main__":
    result = run_accessibility()
    print(result.head())
    print("Columns:", list(result.columns))
