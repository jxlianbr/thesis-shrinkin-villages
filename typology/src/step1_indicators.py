"""
Step 1: Indicator compilation for the typology module.

Compiles a clean indicator matrix for clustering by deriving
settlement-level change indicators from the raw panel data and
inverse-transforming demographics from the preprocessed data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler


def compile_indicators(
    data: Dict[str, Any], cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the indicator matrix from raw panel data and preprocessed data.

    Args:
        data: Dict from load_typology_data().
        cfg: Typology configuration dict.

    Returns:
        Dict with raw_indicators, scaled_indicators, indicator_names,
        physical_names, demographic_names, slope_validation, quality.
    """
    ind_cfg = cfg["indicators"]

    # --- Compute physical indicators from raw panel ---
    print("  Computing physical indicators from raw panel data...")
    panel_indicators = _compute_panel_indicators(data["panel_df"], ind_cfg)

    # --- Inverse-transform demographics ---
    print("  Inverse-transforming demographic indicators...")
    demo_indicators = _inverse_transform_demographics(
        data["scaled_df"], data["preprocessing_report"],
        data["feature_metadata"], ind_cfg,
    )

    # --- Validate slopes against preprocessing ---
    print("  Validating slopes against preprocessing pipeline...")
    slope_validation = _validate_slopes(
        panel_indicators, data["scaled_df"],
        data["preprocessing_report"], data["feature_metadata"],
    )

    # --- Merge into indicator matrix ---
    identifiers = data["identifiers"].copy()
    raw_indicators = identifiers.copy()

    physical_cols = [c for c in panel_indicators.columns if c != "unit_id"]
    raw_indicators = raw_indicators.merge(panel_indicators, on="unit_id", how="left")

    demo_cols = [c for c in demo_indicators.columns if c != "unit_id"]
    raw_indicators = raw_indicators.merge(demo_indicators, on="unit_id", how="left")

    indicator_names = physical_cols + demo_cols

    # --- Quality checks ---
    print("  Running quality checks...")
    quality = _quality_checks(raw_indicators[indicator_names], ind_cfg)

    # --- Prune redundant indicators ---
    pruning_cfg = ind_cfg.get("pruning", {})
    pruning_report: Dict[str, Any] = {"enabled": False, "dropped": []}
    if pruning_cfg.get("enabled", False):
        print("  Pruning redundant indicators...")
        indicator_names, physical_cols, demo_cols, pruning_report = (
            _prune_redundant_indicators(
                raw_indicators, indicator_names, physical_cols, demo_cols,
                quality, ind_cfg,
            )
        )

    # --- Standardize for clustering ---
    print("  Standardizing indicators (z-score)...")
    numeric_df = raw_indicators[indicator_names].copy()

    # Handle any missing values before scaling
    n_missing = numeric_df.isna().sum().sum()
    if n_missing > 0:
        print(f"    {n_missing} missing values -- imputing with column median")
        for col in numeric_df.columns:
            if numeric_df[col].isna().any():
                numeric_df[col].fillna(numeric_df[col].median(), inplace=True)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(numeric_df.values)
    scaled_df = pd.DataFrame(
        scaled_values, columns=indicator_names, index=raw_indicators.index,
    )
    scaled_indicators = pd.concat(
        [identifiers.reset_index(drop=True), scaled_df.reset_index(drop=True)],
        axis=1,
    )

    # Print summary
    print(f"\n  Indicator matrix: {len(raw_indicators)} units x "
          f"{len(indicator_names)} indicators")
    print(f"    Physical: {len(physical_cols)} indicators")
    print(f"    Demographic: {len(demo_cols)} indicators")
    if quality["n_missing_total"] > 0:
        print(f"    Missing values (before imputation): {quality['n_missing_total']}")
    if quality["n_outlier_flags"] > 0:
        print(f"    Outlier flags (>3 SD): {quality['n_outlier_flags']}")
    print("  NOTE: n=65 units. Indicator statistics have limited precision.")

    return {
        "raw_indicators": raw_indicators,
        "scaled_indicators": scaled_indicators,
        "indicator_names": indicator_names,
        "physical_names": physical_cols,
        "demographic_names": demo_cols,
        "slope_validation": slope_validation,
        "quality": quality,
        "pruning": pruning_report,
    }


def _prune_redundant_indicators(
    raw_indicators: pd.DataFrame,
    indicator_names: List[str],
    physical_cols: List[str],
    demo_cols: List[str],
    quality: Dict[str, Any],
    ind_cfg: Dict[str, Any],
) -> tuple:
    """
    Remove near-constant and highly correlated indicators.

    Returns updated (indicator_names, physical_cols, demo_cols, pruning_report).
    """
    pruning_cfg = ind_cfg["pruning"]
    corr_threshold = pruning_cfg.get("correlation_threshold", 0.85)
    drop_near_constant = pruning_cfg.get("drop_near_constant", True)
    keep_list = set(pruning_cfg.get("keep_list", []))
    min_var = ind_cfg.get("quality", {}).get("min_variance", 0.01)

    dropped: List[Dict[str, Any]] = []
    current = list(indicator_names)

    # --- Step 1: Drop near-constant features ---
    if drop_near_constant:
        near_const = quality.get("near_constant_features", [])
        for feat in near_const:
            if feat in current and feat not in keep_list:
                current.remove(feat)
                dropped.append({
                    "feature": feat,
                    "reason": f"near-constant (var < {min_var})",
                })
                print(f"    Dropped {feat} (near-constant)")

    # --- Step 2: Iteratively drop high-correlation features ---
    while True:
        df = raw_indicators[current]
        corr = df.corr().abs()
        # Zero out diagonal
        for c in corr.columns:
            corr.loc[c, c] = 0.0

        max_r = corr.max().max()
        if max_r < corr_threshold:
            break

        # Find the worst pair
        for i in range(len(corr)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j] - max_r) < 1e-6:
                    feat_a = corr.columns[i]
                    feat_b = corr.columns[j]
                    break

        # Decide which to drop: prefer dropping the one with higher
        # mean |r| to all other features (most redundant),
        # unless one is in keep_list
        if feat_a in keep_list and feat_b not in keep_list:
            to_drop = feat_b
        elif feat_b in keep_list and feat_a not in keep_list:
            to_drop = feat_a
        else:
            mean_a = corr[feat_a].mean()
            mean_b = corr[feat_b].mean()
            to_drop = feat_a if mean_a >= mean_b else feat_b

        actual_r = raw_indicators[[feat_a, feat_b]].corr().iloc[0, 1]
        current.remove(to_drop)
        dropped.append({
            "feature": to_drop,
            "reason": f"|r|={abs(actual_r):.3f} with "
                      f"{feat_a if to_drop == feat_b else feat_b}",
        })
        print(f"    Dropped {to_drop} (corr {abs(actual_r):.3f} with "
              f"{feat_a if to_drop == feat_b else feat_b})")

    # Rebuild physical / demographic lists
    physical_out = [c for c in current if c in physical_cols]
    demo_out = [c for c in current if c in demo_cols]

    n_dropped = len(indicator_names) - len(current)
    print(f"    Pruning: {len(indicator_names)} -> {len(current)} indicators "
          f"({n_dropped} dropped)")

    pruning_report = {
        "enabled": True,
        "dropped": dropped,
        "n_before": len(indicator_names),
        "n_after": len(current),
        "correlation_threshold": corr_threshold,
    }

    return current, physical_out, demo_out, pruning_report


def _compute_panel_indicators(
    panel_df: pd.DataFrame, ind_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute per-unit physical indicators from the raw monthly panel.

    Replicates the temporal aggregation logic from
    preprocessing/src/temporal_aggregation.py for slope, seasonal
    amplitude, and additionally computes CV and GLCM means.
    """
    trend_features = ind_cfg["physical"]["trend_features"]
    cv_features = ind_cfg["physical"]["cv_features"]
    seasonal_features = ind_cfg["physical"]["seasonal_features"]
    glcm_mean_features = ind_cfg["physical"]["glcm_mean_features"]
    min_months = ind_cfg["physical"]["min_months_for_trend"]

    records: List[Dict[str, Any]] = []

    for unit_id, grp in panel_df.groupby("unit_id"):
        row: Dict[str, Any] = {"unit_id": unit_id}

        # --- Trend slopes ---
        if "month_dt" in grp.columns:
            grp_sorted = grp.sort_values("month_dt")
            t = (grp_sorted["month_dt"] - grp_sorted["month_dt"].min()).dt.days.values.astype(float)

            if len(t) >= min_months and t[-1] > 0:
                t_norm = t / t[-1]
                for feat in trend_features:
                    if feat in grp_sorted.columns:
                        valid = grp_sorted[feat].notna()
                        if valid.sum() >= min_months:
                            y = grp_sorted.loc[valid, feat].values.astype(float)
                            t_valid = t_norm[valid.values]
                            coeffs = np.polyfit(t_valid, y, 1)
                            row[f"{feat}_slope"] = round(float(coeffs[0]), 6)
                        else:
                            row[f"{feat}_slope"] = np.nan
                    else:
                        row[f"{feat}_slope"] = np.nan

        # --- Coefficient of variation ---
        for feat in cv_features:
            if feat in grp.columns:
                series = grp[feat].dropna()
                if len(series) > 1 and abs(series.mean()) > 1e-10:
                    row[f"{feat}_cv"] = float(series.std() / abs(series.mean()))
                else:
                    row[f"{feat}_cv"] = np.nan
            else:
                row[f"{feat}_cv"] = np.nan

        # --- Seasonal amplitude ---
        if "month_num" in grp.columns:
            for feat in seasonal_features:
                if feat in grp.columns:
                    monthly_means = grp.groupby("month_num")[feat].mean()
                    valid_months = monthly_means.dropna()
                    if len(valid_months) >= 2:
                        amp = float(valid_months.max() - valid_months.min())
                        row[f"{feat}_seasonal_amp"] = round(amp, 6)
                    else:
                        row[f"{feat}_seasonal_amp"] = np.nan
                else:
                    row[f"{feat}_seasonal_amp"] = np.nan

        # --- GLCM temporal means ---
        for feat in glcm_mean_features:
            if feat in grp.columns:
                series = grp[feat].dropna()
                if len(series) > 0:
                    row[f"{feat}_mean"] = float(series.mean())
                else:
                    row[f"{feat}_mean"] = np.nan
            else:
                row[f"{feat}_mean"] = np.nan

        records.append(row)

    result = pd.DataFrame(records)
    n_cols = len([c for c in result.columns if c != "unit_id"])
    print(f"    Computed {n_cols} physical indicators for {len(result)} units")
    return result


def _inverse_transform_demographics(
    scaled_df: pd.DataFrame,
    preproc_report: Dict[str, Any] | None,
    feature_metadata: pd.DataFrame | None,
    ind_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Recover raw demographic values by reversing RobustScaler (and log1p).

    Uses scaler_center and scaler_scale from preprocessing_report.json,
    and log1p flags from feature_metadata.csv.
    """
    demo_features = ind_cfg["demographic"]

    if preproc_report is None:
        print("    WARNING: No preprocessing report -- "
              "using scaled values as-is for demographics")
        result = scaled_df[["unit_id"] + demo_features].copy()
        return result

    transform_meta = preproc_report.get("transform_metadata", {})
    center = transform_meta.get("scaler_center", {})
    scale = transform_meta.get("scaler_scale", {})
    log1p_features = set(transform_meta.get("log1p_features", []))

    # Build log1p lookup from feature_metadata if available
    if feature_metadata is not None:
        log1p_from_meta = set(
            feature_metadata.loc[
                feature_metadata["log1p_applied"].astype(str).str.lower() == "true",
                "column",
            ].tolist()
        )
        log1p_features = log1p_features | log1p_from_meta

    records: List[Dict[str, Any]] = []
    for _, row in scaled_df.iterrows():
        rec = {"unit_id": row["unit_id"]}
        for feat in demo_features:
            val = row[feat]
            if pd.isna(val):
                rec[feat] = np.nan
                continue

            # Reverse RobustScaler: raw = scaled * scale + center
            feat_center = center.get(feat, 0.0)
            feat_scale = scale.get(feat, 1.0)
            raw_val = val * feat_scale + feat_center

            # Reverse log1p if it was applied
            if feat in log1p_features:
                raw_val = np.expm1(raw_val)

            rec[feat] = float(raw_val)
        records.append(rec)

    result = pd.DataFrame(records)
    print(f"    Inverse-transformed {len(demo_features)} demographic indicators")
    return result


def _validate_slopes(
    panel_indicators: pd.DataFrame,
    scaled_df: pd.DataFrame,
    preproc_report: Dict[str, Any] | None,
    feature_metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Cross-validate recomputed slopes against preprocessed slopes.

    Computes Pearson correlation between newly computed raw slopes
    and inverse-transformed preprocessed slopes.
    """
    if preproc_report is None:
        print("    WARNING: Cannot validate slopes -- no preprocessing report")
        return pd.DataFrame()

    transform_meta = preproc_report.get("transform_metadata", {})
    center = transform_meta.get("scaler_center", {})
    scale = transform_meta.get("scaler_scale", {})

    # Slopes to compare: NDVI_slope, NDBI_slope, viirs_mean_slope
    slope_pairs = [
        ("NDVI_slope", "NDVI_slope"),
        ("NDBI_slope", "NDBI_slope"),
        ("viirs_mean_slope", "viirs_mean_slope"),
    ]

    records = []
    for panel_col, preproc_col in slope_pairs:
        if panel_col not in panel_indicators.columns:
            continue
        if preproc_col not in scaled_df.columns:
            continue

        # Inverse-transform the preprocessed slope
        # Slopes were not log1p'd -- only RobustScaler
        s_center = center.get(preproc_col, 0.0)
        s_scale = scale.get(preproc_col, 1.0)
        preproc_raw = scaled_df[preproc_col] * s_scale + s_center

        # Merge on unit_id order
        merged = panel_indicators[["unit_id", panel_col]].merge(
            scaled_df[["unit_id"]].assign(**{f"{preproc_col}_raw": preproc_raw.values}),
            on="unit_id",
        )

        valid = merged[[panel_col, f"{preproc_col}_raw"]].dropna()
        if len(valid) >= 3:
            r, p = stats.pearsonr(valid[panel_col], valid[f"{preproc_col}_raw"])
        else:
            r, p = np.nan, np.nan

        records.append({
            "slope_feature": preproc_col,
            "pearson_r": round(r, 4) if not np.isnan(r) else np.nan,
            "p_value": round(p, 6) if not np.isnan(p) else np.nan,
            "n_valid": len(valid),
        })
        status = f"r={r:.3f}" if not np.isnan(r) else "insufficient data"
        print(f"    {preproc_col}: {status}")

    return pd.DataFrame(records)


def _quality_checks(
    indicators: pd.DataFrame, ind_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run quality checks on the indicator matrix.

    Reports missing values, near-constant features, and outliers.
    """
    quality_cfg = ind_cfg.get("quality", {})
    sd_thresh = quality_cfg.get("outlier_sd_threshold", 3.0)
    min_var = quality_cfg.get("min_variance", 0.01)

    # Missing values
    missing_per_col = indicators.isna().sum()
    missing_cols = missing_per_col[missing_per_col > 0]
    n_missing_total = int(missing_per_col.sum())

    # Near-constant features
    variances = indicators.var()
    near_constant = variances[variances < min_var].index.tolist()

    # Outliers (> sd_thresh standard deviations from mean)
    outlier_flags: Dict[str, List[int]] = {}
    n_outlier_flags = 0
    for col in indicators.columns:
        series = indicators[col].dropna()
        if len(series) < 3:
            continue
        z = np.abs((series - series.mean()) / series.std())
        outlier_idx = z[z > sd_thresh].index.tolist()
        if outlier_idx:
            outlier_flags[col] = outlier_idx
            n_outlier_flags += len(outlier_idx)

    # Correlation matrix
    corr_matrix = indicators.corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.8:
                high_corr_pairs.append({
                    "feature_1": corr_matrix.columns[i],
                    "feature_2": corr_matrix.columns[j],
                    "correlation": round(r, 3),
                })

    if missing_cols.any():
        print(f"    Missing values: {dict(missing_cols)}")
    if near_constant:
        print(f"    Near-constant features (var < {min_var}): {near_constant}")
    if high_corr_pairs:
        print(f"    High-correlation pairs (|r| > 0.8): {len(high_corr_pairs)}")

    return {
        "n_missing_total": n_missing_total,
        "missing_per_column": dict(missing_cols) if missing_cols.any() else {},
        "near_constant_features": near_constant,
        "n_outlier_flags": n_outlier_flags,
        "outlier_flags": {k: len(v) for k, v in outlier_flags.items()},
        "high_correlation_pairs": high_corr_pairs,
        "summary_stats": {
            col: {
                "mean": round(float(indicators[col].mean()), 4),
                "std": round(float(indicators[col].std()), 4),
                "min": round(float(indicators[col].min()), 4),
                "max": round(float(indicators[col].max()), 4),
                "median": round(float(indicators[col].median()), 4),
            }
            for col in indicators.columns
        },
    }
