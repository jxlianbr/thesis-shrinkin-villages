"""
Step 3: Physical-demographic relationship analysis.

Evaluates how RS-derived physical indicators relate to demographic
dynamics, and whether these relationships differ across cluster types.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def run_relationship_analysis(
    indicator_data: Dict[str, Any],
    cluster_data: Dict[str, Any],
    data: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run correlation analysis, OLS regression, spatial autocorrelation,
    and subgroup analysis.

    Args:
        indicator_data: Dict from compile_indicators().
        cluster_data: Dict from run_clustering().
        data: Dict from load_typology_data().
        cfg: Typology configuration dict.

    Returns:
        Dict with correlations, regression, spatial, subgroup results.
    """
    rel_cfg = cfg["relationships"]
    raw = indicator_data["raw_indicators"]
    physical_names = indicator_data["physical_names"]
    demo_names = indicator_data["demographic_names"]
    labels = cluster_data["cluster_labels"]

    print("  NOTE: n=65 units. Relationship analysis has limited "
          "statistical power.")

    # --- Correlation analysis ---
    print("\n  Computing correlations (Pearson + Spearman)...")
    corr_results = _correlation_analysis(
        raw, physical_names, demo_names, rel_cfg,
    )

    # --- OLS regression ---
    print("\n  Running OLS regression...")
    regression_results = _ols_regression(
        raw, physical_names, rel_cfg,
    )

    # --- Spatial autocorrelation ---
    print("\n  Testing spatial autocorrelation...")
    spatial_results = _spatial_analysis(
        regression_results.get("residuals"),
        data["boundaries"],
        data["identifiers"],
        cfg,
        raw=raw,
        physical_names=physical_names,
        regression_results=regression_results,
    )

    # --- Subgroup analysis ---
    print("\n  Running subgroup analysis by cluster...")
    subgroup_results = _subgroup_analysis(
        raw, physical_names, demo_names, labels,
        data["y_labels"], rel_cfg,
    )

    return {
        "correlations": corr_results,
        "regression": regression_results,
        "spatial": spatial_results,
        "subgroup": subgroup_results,
    }


# ------------------------------------------------------------------
# Correlation analysis
# ------------------------------------------------------------------

def _correlation_analysis(
    raw: pd.DataFrame,
    physical_names: List[str],
    demo_names: List[str],
    rel_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute Pearson and Spearman correlations between RS and demographic."""
    alpha = rel_cfg["correlation"]["alpha"]
    n_tests = len(physical_names) * len(demo_names)
    bonferroni_alpha = alpha / max(n_tests, 1)

    results: Dict[str, Any] = {}

    for method_name in rel_cfg["correlation"]["methods"]:
        corr_func = stats.pearsonr if method_name == "pearson" else stats.spearmanr

        corr_matrix = pd.DataFrame(
            index=physical_names, columns=demo_names, dtype=float,
        )
        pval_matrix = pd.DataFrame(
            index=physical_names, columns=demo_names, dtype=float,
        )

        for phys in physical_names:
            for demo in demo_names:
                x = raw[phys].values
                y = raw[demo].values
                valid = ~(np.isnan(x) | np.isnan(y))
                if valid.sum() >= 3:
                    r, p = corr_func(x[valid], y[valid])
                    corr_matrix.loc[phys, demo] = round(r, 4)
                    pval_matrix.loc[phys, demo] = p
                else:
                    corr_matrix.loc[phys, demo] = np.nan
                    pval_matrix.loc[phys, demo] = np.nan

        results[method_name] = corr_matrix
        results[f"{method_name}_pvalues"] = pval_matrix

    # Find significant pairs (Bonferroni-corrected)
    significant_pairs = []
    for method_name in rel_cfg["correlation"]["methods"]:
        pvals = results[f"{method_name}_pvalues"]
        corrs = results[method_name]
        for phys in physical_names:
            for demo in demo_names:
                p = pvals.loc[phys, demo]
                r = corrs.loc[phys, demo]
                if not np.isnan(p) and p < bonferroni_alpha:
                    significant_pairs.append({
                        "method": method_name,
                        "physical": phys,
                        "demographic": demo,
                        "correlation": round(float(r), 4),
                        "p_value": float(p),
                        "significant_bonferroni": True,
                    })

    # Sort by absolute correlation
    significant_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    results["significant_pairs"] = significant_pairs

    n_sig = len(significant_pairs)
    print(f"    {n_sig} significant pairs (Bonferroni alpha = {bonferroni_alpha:.5f})")
    if significant_pairs:
        print("    Top 5:")
        for pair in significant_pairs[:5]:
            print(f"      {pair['physical']} <-> {pair['demographic']}: "
                  f"r={pair['correlation']:.3f} ({pair['method']})")

    return results


# ------------------------------------------------------------------
# OLS regression
# ------------------------------------------------------------------

def _ols_regression(
    raw: pd.DataFrame,
    physical_names: List[str],
    rel_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    OLS regression: demographic DV ~ physical indicators.

    Uses statsmodels if available, falls back to scipy.
    """
    dependent_vars = rel_cfg["regression"]["dependent_vars"]
    vif_threshold = rel_cfg["regression"]["vif_threshold"]

    # Try statsmodels
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.diagnostic import het_breuschpagan
        has_statsmodels = True
    except ImportError:
        has_statsmodels = False
        print("    WARNING: statsmodels not installed -- using scipy fallback")

    all_results: Dict[str, Any] = {}

    for dv in dependent_vars:
        if dv not in raw.columns:
            print(f"    Skipping {dv} -- not in indicator matrix")
            continue

        print(f"\n    Dependent variable: {dv}")
        y = raw[dv].values
        X_vars = [c for c in physical_names if c in raw.columns]
        X_data = raw[X_vars].copy()

        # Drop rows with NaN
        valid = ~(np.isnan(y) | X_data.isna().any(axis=1).values)
        y_valid = y[valid]
        X_valid = X_data.loc[valid].values
        X_names = list(X_vars)
        n = len(y_valid)

        if has_statsmodels:
            result = _ols_statsmodels(
                y_valid, X_valid, X_names, n, vif_threshold,
                sm, variance_inflation_factor, het_breuschpagan,
            )
        else:
            result = _ols_scipy_fallback(y_valid, raw, physical_names, dv)

        result["dependent_var"] = dv
        result["n_obs"] = n
        result["n_predictors_initial"] = len(physical_names)

        all_results[dv] = result

    # Return the primary result (first DV)
    primary_dv = dependent_vars[0] if dependent_vars else None
    primary = all_results.get(primary_dv, {})
    primary["all_models"] = all_results
    return primary


def _ols_statsmodels(
    y: np.ndarray,
    X: np.ndarray,
    X_names: List[str],
    n: int,
    vif_threshold: float,
    sm: Any,
    vif_func: Any,
    het_bp_func: Any,
) -> Dict[str, Any]:
    """Run OLS with full diagnostics using statsmodels."""
    X_names = list(X_names)
    X_work = X.copy()

    # Iterative VIF removal
    vif_history = []
    while X_work.shape[1] > 1:
        X_with_const = sm.add_constant(X_work)
        vifs = []
        for i in range(1, X_with_const.shape[1]):
            vif = vif_func(X_with_const, i)
            vifs.append({"feature": X_names[i - 1], "vif": round(float(vif), 2)})

        vif_df = pd.DataFrame(vifs).sort_values("vif", ascending=False)
        max_vif = vif_df["vif"].max()

        if max_vif <= vif_threshold:
            break

        drop_feat = vif_df.iloc[0]["feature"]
        drop_idx = X_names.index(drop_feat)
        X_names.pop(drop_idx)
        X_work = np.delete(X_work, drop_idx, axis=1)
        vif_history.append({"dropped": drop_feat, "vif": float(max_vif)})
        print(f"      Dropped {drop_feat} (VIF={max_vif:.1f})")

    if len(X_names) == 0:
        print("      WARNING: All predictors dropped due to VIF")
        return {"error": "all_predictors_dropped"}

    # Final VIF
    X_with_const = sm.add_constant(X_work)
    final_vifs = []
    for i in range(1, X_with_const.shape[1]):
        vif = vif_func(X_with_const, i)
        final_vifs.append({"feature": X_names[i - 1], "vif": round(float(vif), 2)})
    vif_results = pd.DataFrame(final_vifs)

    # Fit OLS
    model = sm.OLS(y, X_with_const).fit()

    # Coefficients
    coefs = []
    param_names = ["const"] + X_names
    for i, name in enumerate(param_names):
        coefs.append({
            "feature": name,
            "coefficient": round(float(model.params[i]), 6),
            "std_error": round(float(model.bse[i]), 6),
            "t_stat": round(float(model.tvalues[i]), 4),
            "p_value": round(float(model.pvalues[i]), 6),
            "ci_lower": round(float(model.conf_int()[i, 0]), 6),
            "ci_upper": round(float(model.conf_int()[i, 1]), 6),
        })
    coefficients = pd.DataFrame(coefs)

    # Diagnostics
    residuals = model.resid
    fitted = model.fittedvalues

    # Shapiro-Wilk on residuals
    if len(residuals) >= 3:
        sw_stat, sw_p = stats.shapiro(residuals)
    else:
        sw_stat, sw_p = np.nan, np.nan

    # Breusch-Pagan
    try:
        bp_stat, bp_p, _, _ = het_bp_func(residuals, X_with_const)
    except Exception:
        bp_stat, bp_p = np.nan, np.nan

    ratio = n / len(X_names) if len(X_names) > 0 else float("inf")
    print(f"      R-squared: {model.rsquared:.3f}, "
          f"Adj R-squared: {model.rsquared_adj:.3f}")
    print(f"      F-stat: {model.fvalue:.2f}, p={model.f_pvalue:.4f}")
    print(f"      Sample/predictor ratio: {ratio:.1f}:1")
    if ratio < 10:
        print("      WARNING: Marginal sample-to-predictor ratio for OLS")

    return {
        "coefficients": coefficients,
        "vif_results": vif_results,
        "vif_history": vif_history,
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "f_stat": round(float(model.fvalue), 4),
        "f_pvalue": round(float(model.f_pvalue), 6),
        "residuals": np.asarray(residuals),
        "fitted": np.asarray(fitted),
        "predictors_final": X_names,
        "diagnostics": {
            "shapiro_wilk_stat": round(float(sw_stat), 4) if not np.isnan(sw_stat) else None,
            "shapiro_wilk_p": round(float(sw_p), 6) if not np.isnan(sw_p) else None,
            "breusch_pagan_stat": round(float(bp_stat), 4) if not np.isnan(bp_stat) else None,
            "breusch_pagan_p": round(float(bp_p), 6) if not np.isnan(bp_p) else None,
            "residual_normality": "normal" if sw_p > 0.05 else "non-normal" if not np.isnan(sw_p) else "unknown",
            "heteroscedasticity": "present" if bp_p < 0.05 else "absent" if not np.isnan(bp_p) else "unknown",
        },
        "model_summary_text": str(model.summary()),
    }


def _ols_scipy_fallback(
    y: np.ndarray,
    raw: pd.DataFrame,
    physical_names: List[str],
    dv: str,
) -> Dict[str, Any]:
    """Bivariate OLS fallback using scipy when statsmodels is unavailable."""
    results = []
    for feat in physical_names:
        x = raw[feat].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() >= 3:
            slope, intercept, r, p, se = stats.linregress(x[valid], y[valid])
            results.append({
                "feature": feat,
                "coefficient": round(slope, 6),
                "r_squared": round(r**2, 4),
                "p_value": round(p, 6),
            })

    coefficients = pd.DataFrame(results)
    print(f"      Bivariate regressions: {len(results)} computed")
    return {
        "coefficients": coefficients,
        "vif_results": pd.DataFrame(),
        "vif_history": [],
        "residuals": None,
        "fitted": None,
        "diagnostics": {"note": "statsmodels not available, bivariate only"},
    }


# ------------------------------------------------------------------
# Spatial analysis
# ------------------------------------------------------------------

def _spatial_analysis(
    residuals: np.ndarray | None,
    boundaries: Any,
    identifiers: pd.DataFrame,
    cfg: Dict[str, Any],
    raw: pd.DataFrame | None = None,
    physical_names: List[str] | None = None,
    regression_results: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Test spatial autocorrelation with Moran's I and fit spatial models."""
    if residuals is None:
        print("    Skipped -- no residuals available")
        return {"skipped": True, "reason": "no_residuals"}

    if boundaries is None:
        print("    Skipped -- no boundary geometries available")
        return {"skipped": True, "reason": "no_boundaries"}

    # Try libpysal / esda
    try:
        from libpysal.weights import Queen
        from esda.moran import Moran
    except ImportError:
        print("    Skipped -- libpysal/esda not installed. "
              "Install with: pip install libpysal esda spreg")
        return {"skipped": True, "reason": "libpysal_not_installed"}

    # Align boundaries with identifiers
    gdf = boundaries.copy()
    if "unit_id" in gdf.columns and "unit_id" in identifiers.columns:
        unit_order = identifiers["unit_id"].tolist()
        gdf = gdf.set_index("unit_id").loc[unit_order].reset_index()
    else:
        print("    WARNING: Cannot align boundaries with data")
        return {"skipped": True, "reason": "alignment_failed"}

    # Ensure projected CRS for queen contiguity
    if gdf.crs and gdf.crs.is_geographic:
        try:
            gdf_proj = gdf.to_crs(epsg=6690)  # JGD2011 Japan zone
        except Exception:
            gdf_proj = gdf  # proceed anyway
    else:
        gdf_proj = gdf

    # Build spatial weights
    try:
        w = Queen.from_dataframe(gdf_proj)
        w.transform = "r"  # Row-standardize
    except Exception as e:
        print(f"    WARNING: Could not build spatial weights: {e}")
        return {"skipped": True, "reason": f"weights_failed: {e}"}

    # Moran's I
    mi = Moran(residuals, w)
    print(f"    Moran's I: {mi.I:.4f}, p={mi.p_sim:.4f}")

    result: Dict[str, Any] = {
        "skipped": False,
        "morans_i": round(float(mi.I), 4),
        "morans_p": round(float(mi.p_sim), 4),
        "morans_z": round(float(mi.z_sim), 4),
        "significant": mi.p_sim < 0.05,
    }

    # Spatial regression if Moran's I is significant
    if mi.p_sim < 0.05 and cfg["relationships"]["spatial"]["enabled"]:
        print("    Spatial autocorrelation detected -- fitting spatial regression")
        spatial_reg = _spatial_regression(
            gdf_proj, w, cfg,
            raw=raw,
            physical_names=physical_names,
            regression_results=regression_results,
        )
        result["spatial_regression"] = spatial_reg
    else:
        if mi.p_sim >= 0.05:
            print("    No significant spatial autocorrelation (p >= 0.05)")
        result["spatial_regression"] = None

    return result


def _spatial_regression(
    gdf: Any,
    w: Any,
    cfg: Dict[str, Any],
    raw: pd.DataFrame | None = None,
    physical_names: List[str] | None = None,
    regression_results: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Fit OLS, spatial lag and spatial error models using spreg and
    compare AIC/pseudo-R2.

    Uses the same dependent variable and (VIF-pruned) predictors as
    the OLS regression from Step 3.
    """
    try:
        from spreg import OLS as SpregOLS, ML_Lag, ML_Error
    except ImportError:
        print("    spreg not installed -- spatial regression skipped. "
              "Install with: pip install spreg")
        return {"skipped": True, "reason": "spreg_not_installed"}

    if raw is None or physical_names is None or regression_results is None:
        print("    Spatial regression: missing original data -- skipped")
        return {"skipped": True, "reason": "missing_original_data"}

    # Use VIF-pruned predictor set if available, else physical_names
    final_predictors = regression_results.get("predictors_final", physical_names)
    dv = regression_results.get("dependent_var")
    if dv is None:
        dv = cfg["relationships"]["regression"]["dependent_vars"][0]

    if dv not in raw.columns:
        print(f"    Spatial regression: {dv} not in data -- skipped")
        return {"skipped": True, "reason": f"dv_{dv}_not_found"}

    X_vars = [c for c in final_predictors if c in raw.columns]
    if len(X_vars) == 0:
        print("    Spatial regression: no valid predictors -- skipped")
        return {"skipped": True, "reason": "no_predictors"}

    y = raw[dv].values.reshape(-1, 1)
    X = raw[X_vars].values

    # Drop rows with NaN
    valid = ~(np.isnan(y.ravel()) | np.isnan(X).any(axis=1))
    if valid.sum() < 10:
        print(f"    Spatial regression: only {valid.sum()} valid obs -- skipped")
        return {"skipped": True, "reason": "insufficient_obs"}

    y_clean = y[valid]
    X_clean = X[valid]

    # If NaN filtering removed rows, we need a matching weights matrix
    if valid.sum() < len(valid):
        print(f"    Spatial regression: {valid.sum()}/{len(valid)} valid obs")
        # Rebuild weights for valid subset if needed
        try:
            from libpysal.weights import Queen
            gdf_sub = gdf.iloc[np.where(valid)[0]].copy().reset_index(drop=True)
            w_sub = Queen.from_dataframe(gdf_sub)
            w_sub.transform = "r"
        except Exception as e:
            print(f"    Could not rebuild spatial weights for subset: {e}")
            return {"skipped": True, "reason": f"weights_subset_failed: {e}"}
    else:
        w_sub = w

    var_names = X_vars
    print(f"    DV: {dv}, predictors: {', '.join(var_names)} "
          f"(n={y_clean.shape[0]})")

    results: Dict[str, Any] = {"skipped": False, "dependent_var": dv}

    # --- OLS via spreg (for comparable AIC) ---
    try:
        ols = SpregOLS(
            y_clean, X_clean, w=w_sub,
            name_y=dv, name_x=var_names,
            spat_diag=True,
        )
        results["ols"] = {
            "r_squared": round(float(ols.r2), 4),
            "adj_r_squared": round(float(ols.ar2), 4),
            "aic": round(float(ols.aic), 2),
            "schwarz": round(float(ols.schwarz), 2),
            "log_likelihood": round(float(ols.logll), 4),
        }
        # LM diagnostics from spreg OLS with spat_diag=True
        # ols.lm_lag = (stat, p), ols.lm_error = (stat, p)
        if hasattr(ols, "lm_lag"):
            results["lm_lag"] = {
                "statistic": round(float(ols.lm_lag[0]), 4),
                "p_value": round(float(ols.lm_lag[1]), 6),
            }
            print(f"    LM-Lag:   stat={ols.lm_lag[0]:.3f}, "
                  f"p={ols.lm_lag[1]:.4f}")
        if hasattr(ols, "lm_error"):
            results["lm_error"] = {
                "statistic": round(float(ols.lm_error[0]), 4),
                "p_value": round(float(ols.lm_error[1]), 6),
            }
            print(f"    LM-Error: stat={ols.lm_error[0]:.3f}, "
                  f"p={ols.lm_error[1]:.4f}")
        if hasattr(ols, "rlm_lag"):
            results["rlm_lag"] = {
                "statistic": round(float(ols.rlm_lag[0]), 4),
                "p_value": round(float(ols.rlm_lag[1]), 6),
            }
        if hasattr(ols, "rlm_error"):
            results["rlm_error"] = {
                "statistic": round(float(ols.rlm_error[0]), 4),
                "p_value": round(float(ols.rlm_error[1]), 6),
            }

        print(f"    OLS:   R2={ols.r2:.3f}, AIC={ols.aic:.1f}")
    except Exception as e:
        print(f"    spreg OLS failed: {e}")
        results["ols"] = {"error": str(e)}

    # --- Spatial Lag Model (ML) ---
    try:
        lag = ML_Lag(
            y_clean, X_clean, w=w_sub,
            name_y=dv, name_x=var_names,
        )
        results["spatial_lag"] = {
            "pseudo_r_squared": round(float(lag.pr2), 4),
            "rho": round(float(lag.rho), 4),
            "aic": round(float(lag.aic), 2),
            "schwarz": round(float(lag.schwarz), 2),
            "log_likelihood": round(float(lag.logll), 4),
        }
        print(f"    Lag:   rho={lag.rho:.3f}, "
              f"pseudo-R2={lag.pr2:.3f}, AIC={lag.aic:.1f}")
    except Exception as e:
        print(f"    ML_Lag failed: {e}")
        results["spatial_lag"] = {"error": str(e)}

    # --- Spatial Error Model (ML) ---
    try:
        error = ML_Error(
            y_clean, X_clean, w=w_sub,
            name_y=dv, name_x=var_names,
        )
        results["spatial_error"] = {
            "pseudo_r_squared": round(float(error.pr2), 4),
            "lambda": round(float(error.lam), 4),
            "aic": round(float(error.aic), 2),
            "schwarz": round(float(error.schwarz), 2),
            "log_likelihood": round(float(error.logll), 4),
        }
        print(f"    Error: lambda={error.lam:.3f}, "
              f"pseudo-R2={error.pr2:.3f}, AIC={error.aic:.1f}")
    except Exception as e:
        print(f"    ML_Error failed: {e}")
        results["spatial_error"] = {"error": str(e)}

    # --- Model comparison summary ---
    models_ok = []
    if "aic" in results.get("ols", {}):
        models_ok.append(("OLS", results["ols"]["aic"]))
    if "aic" in results.get("spatial_lag", {}):
        models_ok.append(("Spatial Lag", results["spatial_lag"]["aic"]))
    if "aic" in results.get("spatial_error", {}):
        models_ok.append(("Spatial Error", results["spatial_error"]["aic"]))

    if models_ok:
        best_model = min(models_ok, key=lambda x: x[1])
        results["best_model"] = best_model[0]
        results["best_aic"] = best_model[1]
        print(f"    Best model by AIC: {best_model[0]} "
              f"(AIC={best_model[1]:.1f})")

    return results


# ------------------------------------------------------------------
# Subgroup analysis
# ------------------------------------------------------------------

def _subgroup_analysis(
    raw: pd.DataFrame,
    physical_names: List[str],
    demo_names: List[str],
    cluster_labels: np.ndarray,
    y_labels: pd.Series,
    rel_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Per-cluster and per-class correlation analysis."""
    indicator_names = physical_names + demo_names
    min_n = rel_cfg["subgroup"].get("min_group_size_for_regression", 15)

    # --- By cluster ---
    cluster_results = {}
    for cid in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cid
        n_cluster = int(mask.sum())
        subset = raw.loc[mask, indicator_names]

        cluster_info: Dict[str, Any] = {
            "n_units": n_cluster,
            "summary_stats": {},
        }

        # Summary stats
        for col in indicator_names:
            series = subset[col].dropna()
            if len(series) > 0:
                cluster_info["summary_stats"][col] = {
                    "mean": round(float(series.mean()), 4),
                    "std": round(float(series.std()), 4),
                    "median": round(float(series.median()), 4),
                }

        # Correlation between physical and demographic (if enough samples)
        if n_cluster >= 5:
            corr_mat = subset[physical_names + demo_names].corr()
            cluster_info["correlation_matrix"] = corr_mat
        else:
            cluster_info["correlation_matrix"] = None
            print(f"    Cluster {cid}: n={n_cluster} too small for correlation")

        if n_cluster < min_n:
            cluster_info["regression_note"] = (
                f"n={n_cluster} < {min_n}: too small for regression"
            )

        cluster_results[int(cid)] = cluster_info

    # --- Kruskal-Wallis across clusters ---
    kw_records = []
    for col in indicator_names:
        groups = [raw.loc[cluster_labels == cid, col].dropna().values
                  for cid in sorted(np.unique(cluster_labels))]
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) >= 2:
            try:
                h_stat, p_val = stats.kruskal(*groups)
                kw_records.append({
                    "indicator": col,
                    "h_statistic": round(float(h_stat), 4),
                    "p_value": round(float(p_val), 6),
                    "significant_005": p_val < 0.05,
                })
            except ValueError:
                pass

    kw_df = pd.DataFrame(kw_records).sort_values("p_value")

    n_sig_kw = kw_df["significant_005"].sum() if len(kw_df) > 0 else 0
    print(f"    Kruskal-Wallis: {n_sig_kw}/{len(kw_df)} indicators "
          f"differ significantly across clusters")

    return {
        "by_cluster": cluster_results,
        "kruskal_wallis": kw_df,
    }
