"""
Feature importance analysis for the classification module.

Computes permutation importance, tree-based (Gini/MDI) importance,
and optionally SHAP values for interpretability.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


TREE_MODELS = {"random_forest", "gradient_boosting", "xgboost"}


def run_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    cv_results: Dict[str, Any],
    models: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute feature importance via multiple methods.

    1. **Permutation importance**: on the best non-baseline model,
       using the full dataset with cross-validated scoring.
    2. **Tree-based (Gini/MDI)**: ``feature_importances_`` from
       Random Forest and Gradient Boosting, trained on the full data.
    3. **SHAP** (optional): TreeExplainer on the best tree model.

    Args:
        X:          Feature matrix (primary experiment).
        y:          Target vector.
        cv_results: Output of ``run_cross_validation``.
        models:     Model info dict from ``build_models``.
        cfg:        Configuration dict.

    Returns:
        Dict with keys:
            permutation       – pd.DataFrame (feature, importance_mean, importance_std)
            tree_based        – Dict[str, pd.DataFrame] per tree model
            shap_values       – np.ndarray | None
            shap_expected     – np.ndarray | None
            best_model_name   – str
            consensus_ranking – pd.DataFrame (feature, avg_rank)
    """
    fi_cfg = cfg["feature_importance"]
    feature_names = list(X.columns)

    # Identify best non-baseline model
    primary = cfg["evaluation"]["primary_metric"]
    best_name = _find_best_model(cv_results, models, primary)
    print(f"Feature importance (best model: {models[best_name]['display_name']})")

    results: Dict[str, Any] = {
        "best_model_name": best_name,
        "permutation": pd.DataFrame(),
        "tree_based": {},
        "shap_values": None,
        "shap_expected": None,
        "consensus_ranking": pd.DataFrame(),
    }

    rank_sources: list[pd.Series] = []

    # --- 1. Permutation importance ---
    if fi_cfg["permutation"]["enabled"]:
        print("  Computing permutation importance ...")
        perm_df = _compute_permutation_importance(
            models[best_name]["estimator"], X, y, cfg,
        )
        results["permutation"] = perm_df
        rank_sources.append(
            perm_df.set_index("feature")["importance_mean"].rank(ascending=False)
        )

    # --- 2. Tree-based importance ---
    if fi_cfg["tree_based"]["enabled"]:
        for tname in TREE_MODELS:
            if tname not in models:
                continue
            print(f"  Computing tree-based importance ({tname}) ...")
            est = _fit_full(models[tname]["estimator"], X, y)
            if hasattr(est, "feature_importances_"):
                imp = est.feature_importances_
                df_imp = pd.DataFrame({
                    "feature": feature_names,
                    "importance": imp,
                }).sort_values("importance", ascending=False).reset_index(drop=True)
                results["tree_based"][tname] = df_imp
                rank_sources.append(
                    df_imp.set_index("feature")["importance"].rank(ascending=False)
                )

    # --- 3. SHAP ---
    if fi_cfg["shap"]["enabled"]:
        shap_vals, shap_exp = _compute_shap_values(
            models, X, y, cfg,
        )
        results["shap_values"] = shap_vals
        results["shap_expected"] = shap_exp
        if shap_vals is not None:
            # Mean absolute SHAP as importance
            mean_abs = np.mean(np.abs(shap_vals), axis=0)
            shap_rank = pd.Series(mean_abs, index=feature_names).rank(ascending=False)
            rank_sources.append(shap_rank)

    # --- Consensus ranking ---
    if rank_sources:
        combined = pd.concat(rank_sources, axis=1)
        avg_rank = combined.mean(axis=1).sort_values()
        results["consensus_ranking"] = pd.DataFrame({
            "feature": avg_rank.index,
            "avg_rank": avg_rank.values,
        }).reset_index(drop=True)
        print("  Consensus top-5 features:")
        for _, row in results["consensus_ranking"].head(5).iterrows():
            print(f"    {row['feature']:30s}  avg_rank={row['avg_rank']:.1f}")

    return results


# ------------------------------------------------------------------ #
#  Helpers                                                           #
# ------------------------------------------------------------------ #

def _find_best_model(
    cv_results: Dict[str, Any],
    models: Dict[str, Dict[str, Any]],
    primary_metric: str,
) -> str:
    """Return the name of the best non-baseline model by primary metric."""
    best_name, best_score = "", -np.inf
    for name, res in cv_results.items():
        if models[name]["is_baseline"]:
            continue
        score = res["mean_metrics"].get(primary_metric, -np.inf)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def _fit_full(estimator: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    """Clone and fit an estimator on the full dataset."""
    from copy import deepcopy

    est = deepcopy(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X.values, y.values)
    return est


def _compute_permutation_importance(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Compute permutation importance using sklearn.inspection."""
    from copy import deepcopy
    from sklearn.model_selection import RepeatedStratifiedKFold

    fi_cfg = cfg["feature_importance"]["permutation"]
    est = deepcopy(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X.values, y.values)

    result = permutation_importance(
        est, X.values, y.values,
        n_repeats=fi_cfg["n_repeats"],
        random_state=fi_cfg["random_state"],
        scoring=fi_cfg["scoring"],
        n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def _compute_shap_values(
    models: Dict[str, Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute SHAP values for the best available tree model."""
    try:
        import shap
    except ImportError:
        print("  SHAP: skipped (shap not installed)")
        return None, None

    # Find best tree model
    for tname in ["random_forest", "gradient_boosting", "xgboost"]:
        if tname in models:
            print(f"  Computing SHAP values ({tname}) ...")
            est = _fit_full(models[tname]["estimator"], X, y)
            try:
                explainer = shap.TreeExplainer(est)
                shap_vals = explainer.shap_values(X.values)
                # For multiclass, shap_vals is list of arrays; stack → (n, p, c)
                if isinstance(shap_vals, list):
                    shap_vals = np.stack(shap_vals, axis=-1)
                # Average absolute across classes → (n, p)
                if shap_vals.ndim == 3:
                    shap_vals_2d = np.mean(np.abs(shap_vals), axis=-1)
                else:
                    shap_vals_2d = shap_vals
                return shap_vals_2d, explainer.expected_value
            except Exception as exc:
                print(f"  SHAP failed for {tname}: {exc}")
                continue

    print("  SHAP: no compatible tree model found")
    return None, None
