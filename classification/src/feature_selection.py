"""
Feature selection diagnostics for the classification module.

Provides SelectKBest rankings and PCA explained-variance analysis.
These are *diagnostic* — features are not removed automatically.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


SCORE_FUNCS = {
    "f_classif": f_classif,
}


def run_feature_selection(
    data: Dict[str, Any], cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform feature-selection diagnostics on the primary experiment set.

    Args:
        data: Must contain keys ``X`` (feature DataFrame) and ``y`` (target Series).
        cfg:  Configuration dict.

    Returns:
        Dict with keys:
            kbest_rankings  – pd.DataFrame (feature, score, p_value, rank)
            pca_variance    – pd.DataFrame (component, explained_variance,
                              cumulative)
            n_components_95 – int (components needed for 95 % variance)
    """
    X: pd.DataFrame = data["X"]
    y: pd.Series = data["y"]
    fs_cfg = cfg["feature_selection"]

    results: Dict[str, Any] = {}

    # --- SelectKBest ---
    if fs_cfg["select_k_best"]["enabled"]:
        print("Running SelectKBest diagnostics ...")
        k = min(fs_cfg["select_k_best"]["k"], X.shape[1])
        score_func = SCORE_FUNCS[fs_cfg["select_k_best"]["score_func"]]

        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)

        kbest = pd.DataFrame({
            "feature": X.columns,
            "score": selector.scores_,
            "p_value": selector.pvalues_,
        }).sort_values("score", ascending=False).reset_index(drop=True)
        kbest["rank"] = range(1, len(kbest) + 1)

        results["kbest_rankings"] = kbest
        print(f"  Top-{k} features by {fs_cfg['select_k_best']['score_func']}:")
        for _, row in kbest.head(k).iterrows():
            print(f"    {row['rank']:2d}. {row['feature']:30s}  "
                  f"F={row['score']:10.2f}  p={row['p_value']:.2e}")
    else:
        results["kbest_rankings"] = pd.DataFrame()

    # --- PCA ---
    if fs_cfg["pca"]["enabled"]:
        print("Running PCA explained-variance analysis ...")
        n_comp = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comp, random_state=cfg["random_state"])
        pca.fit(X)

        pca_df = pd.DataFrame({
            "component": range(1, n_comp + 1),
            "explained_variance": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        })

        threshold = fs_cfg["pca"]["variance_threshold"]
        n95 = int((pca_df["cumulative"] >= threshold).idxmax()) + 1
        results["pca_variance"] = pca_df
        results["n_components_95"] = n95
        print(f"  Total components: {n_comp}")
        print(f"  Components for {threshold*100:.0f}% variance: {n95}")
        print(f"  First 5 components explain "
              f"{pca_df['cumulative'].iloc[4]:.1%} of variance")
    else:
        results["pca_variance"] = pd.DataFrame()
        results["n_components_95"] = 0

    return results
