"""
Step 2: Cluster analysis for the typology module.

Identifies distinct shrinkage patterns through unsupervised clustering,
producing a defensible typology with stability assessment.  Reports
multiple k solutions (k=3, 4, 5 and data-driven optimum) so the
thesis author can compare and argue for the most interpretable one.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def run_clustering(
    indicator_data: Dict[str, Any],
    data: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the full clustering pipeline with multi-k reporting.

    Phase 1: PCA + optimal-k metrics for k=2..8.
    Phase 2: For each reported k, fit K-means, hierarchical,
             bootstrap stability, profiles, and cross-tab.

    Args:
        indicator_data: Dict from compile_indicators().
        data: Dict from load_typology_data().
        cfg: Typology configuration dict.

    Returns:
        Dict with pca, k_metrics, solutions (per-k dicts),
        primary_k, and backward-compatible top-level keys.
    """
    clust_cfg = cfg["clustering"]
    rng_seed = cfg["random_state"]

    scaled = indicator_data["scaled_indicators"]
    indicator_names = indicator_data["indicator_names"]
    X = scaled[indicator_names].values

    print(f"  Clustering input: {X.shape[0]} units x {X.shape[1]} indicators")
    print("  NOTE: n=65 with k clusters -> cluster sizes may be small. "
          "Bootstrap stability essential.")

    # ==================================================================
    # Phase 1: PCA + optimal k metrics
    # ==================================================================
    print("\n  Running PCA...")
    pca_results = _run_pca(X, indicator_names, clust_cfg)

    print("\n  Determining optimal k...")
    k_results = _optimal_k(X, clust_cfg, rng_seed)
    data_driven_k = k_results["optimal_k"]
    print(f"  Data-driven optimal k = {data_driven_k}")

    # ==================================================================
    # Phase 2: Multi-k solutions
    # ==================================================================
    report_k_values = list(clust_cfg.get("report_k_values", [3, 4, 5]))
    if clust_cfg.get("include_data_driven_k", True):
        if data_driven_k not in report_k_values:
            report_k_values.append(data_driven_k)
    report_k_values = sorted(set(report_k_values))

    # PCA-based clustering setup
    pca_cfg = clust_cfg.get("pca_clustering", {})
    pca_enabled = pca_cfg.get("enabled", False)
    X_pca_clust = None
    n_pca_components = 0
    if pca_enabled:
        pca_var_thresh = pca_cfg.get("variance_threshold", 0.80)
        thresh_key = f"{int(pca_var_thresh * 100)}%"
        n_pca_components = pca_results["n_for_threshold"].get(
            thresh_key, min(5, X.shape[1])
        )
        X_pca_clust = pca_results["X_pca"][:, :n_pca_components]
        print(f"\n  PCA clustering enabled: {n_pca_components} PCs "
              f"({thresh_key} variance)")

    solutions: Dict[int, Dict[str, Any]] = {}
    raw = indicator_data["raw_indicators"]

    for k in report_k_values:
        print(f"\n  --- k = {k} ---")

        # K-means (raw space)
        km_res = _fit_kmeans(X, k, rng_seed, clust_cfg)
        labels_km = km_res["labels"]

        sil = silhouette_score(X, labels_km)
        sil_samp = silhouette_samples(X, labels_km)
        cluster_sizes = pd.Series(labels_km).value_counts().sort_index()

        print(f"    K-means silhouette: {sil:.3f}")
        for cid, size in cluster_sizes.items():
            flag = " (small)" if size < 5 else ""
            print(f"      Cluster {cid}: {size} units{flag}")

        # Hierarchical
        hier_res = _fit_hierarchical(X, k, labels_km, clust_cfg)
        print(f"    Hierarchical ARI: {hier_res['ari']:.3f}")

        # Bootstrap stability
        boot_res = _bootstrap_stability(X, k, labels_km, clust_cfg, rng_seed)
        print(f"    Bootstrap ARI: {boot_res['mean_ari']:.3f} "
              f"(+/- {boot_res['std_ari']:.3f})")

        # PCA-based clustering
        pca_km_res = None
        if pca_enabled and X_pca_clust is not None:
            pca_km_res = _cluster_on_pca(
                X_pca_clust, k, labels_km, rng_seed, clust_cfg,
            )
            print(f"    PCA-space silhouette: {pca_km_res['silhouette']:.3f}, "
                  f"ARI vs raw: {pca_km_res['ari_vs_raw']:.3f}")

        # Cross-tab with supervised
        crosstab = _crosstab_with_supervised(labels_km, data["y_labels"])

        # Profiles
        profiles = _cluster_profiles(raw, indicator_names, labels_km)

        # Supervised ARI
        sup_ari = adjusted_rand_score(data["y_labels"].values, labels_km)

        # Unit assignments
        identifiers = data["identifiers"].copy()
        unit_clusters = identifiers.copy()
        unit_clusters["cluster"] = labels_km
        unit_clusters["silhouette"] = sil_samp
        unit_clusters["shrinkage_class"] = data["y_labels"].values

        solutions[k] = {
            "kmeans": km_res,
            "hierarchical": hier_res,
            "bootstrap": boot_res,
            "pca_kmeans": pca_km_res,
            "profiles": profiles,
            "crosstab": crosstab,
            "labels": labels_km,
            "silhouette_mean": round(sil, 4),
            "silhouette_samples": sil_samp,
            "cluster_sizes": dict(cluster_sizes),
            "supervised_ari": round(sup_ari, 4),
            "unit_clusters": unit_clusters,
        }

    # Perturbation stability (primary k only, to save time)
    primary_k = _select_primary_k(solutions, report_k_values)
    print(f"\n  Primary k selected: {primary_k}")

    perturbation_results = None
    if clust_cfg.get("perturbation", {}).get("enabled", True):
        print(f"\n  Running feature perturbation stability (k={primary_k})...")
        perturbation_results = _perturbation_stability(
            X, primary_k, solutions[primary_k]["labels"],
            indicator_names, rng_seed,
        )

    # Build backward-compatible top-level keys from primary solution
    primary = solutions[primary_k]
    return {
        "pca": pca_results,
        "k_metrics": k_results["metrics_df"],
        "solutions": solutions,
        "primary_k": primary_k,
        "report_k_values": report_k_values,
        "data_driven_k": data_driven_k,
        "perturbation": perturbation_results,
        # Backward-compatible keys (from primary solution)
        "optimal_k": primary_k,
        "cluster_labels": primary["labels"],
        "silhouette_mean": primary["silhouette_mean"],
        "silhouette_samples": primary["silhouette_samples"],
        "kmeans": primary["kmeans"],
        "hierarchical": primary["hierarchical"],
        "bootstrap": primary["bootstrap"],
        "profiles": primary["profiles"],
        "crosstab": primary["crosstab"],
        "unit_clusters": primary["unit_clusters"],
    }


def _select_primary_k(
    solutions: Dict[int, Dict[str, Any]],
    report_k_values: List[int],
) -> int:
    """Select primary k: best silhouette with parsimony tiebreak."""
    # Sort by k (ascending) for parsimony preference
    k_sil = [(k, solutions[k]["silhouette_mean"]) for k in sorted(report_k_values)]

    best_k, best_sil = k_sil[0]
    for k, sil in k_sil[1:]:
        # Only prefer larger k if silhouette improves by > 0.02
        if sil > best_sil + 0.02:
            best_k = k
            best_sil = sil

    return best_k


# ------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------

def _run_pca(
    X: np.ndarray, feature_names: List[str], clust_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run PCA and report variance explained."""
    n_components = min(X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    thresholds = clust_cfg["pca"]["variance_thresholds"]
    n_for_threshold = {}
    for thresh in thresholds:
        n_comp = int(np.searchsorted(cum_var, thresh) + 1)
        n_for_threshold[f"{int(thresh*100)}%"] = n_comp
        print(f"    Components for {int(thresh*100)}% variance: {n_comp}")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    variance_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(n_components)],
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": var_ratio,
        "cumulative_variance_ratio": cum_var,
    })

    return {
        "pca_model": pca,
        "X_pca": X_pca,
        "variance_df": variance_df,
        "loadings": loadings,
        "n_for_threshold": n_for_threshold,
    }


# ------------------------------------------------------------------
# Optimal k
# ------------------------------------------------------------------

def _optimal_k(
    X: np.ndarray, clust_cfg: Dict[str, Any], rng_seed: int,
) -> Dict[str, Any]:
    """Test k=2..8 and recommend optimal k (with cap)."""
    k_range = clust_cfg["kmeans"]["k_range"]
    n_init = clust_cfg["kmeans"]["n_init"]
    max_iter = clust_cfg["kmeans"]["max_iter"]
    max_k_cap = clust_cfg.get("max_k_for_selection", max(k_range))

    records = []
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter,
                     random_state=rng_seed)
        labels = km.fit_predict(X)
        inertia = km.inertia_
        inertias.append(inertia)

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        records.append({
            "k": k,
            "silhouette": round(sil, 4),
            "calinski_harabasz": round(ch, 2),
            "davies_bouldin": round(db, 4),
            "inertia": round(inertia, 2),
        })
        print(f"    k={k}: silhouette={sil:.3f}, CH={ch:.1f}, DB={db:.3f}")

    metrics_df = pd.DataFrame(records)

    # Gap statistic
    print("    Computing gap statistic...")
    gap_results = _gap_statistic(X, k_range, clust_cfg, rng_seed)
    metrics_df["gap"] = gap_results["gap_values"]
    metrics_df["gap_se"] = gap_results["gap_se"]

    # Votes (uncapped)
    votes_uncapped = {}
    votes_uncapped["silhouette"] = int(
        metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
    votes_uncapped["calinski_harabasz"] = int(
        metrics_df.loc[metrics_df["calinski_harabasz"].idxmax(), "k"])
    votes_uncapped["davies_bouldin"] = int(
        metrics_df.loc[metrics_df["davies_bouldin"].idxmin(), "k"])

    gap_vals = metrics_df["gap"].values
    gap_ses = metrics_df["gap_se"].values
    best_gap_k = k_range[-1]
    for i in range(len(k_range) - 1):
        if gap_vals[i] >= gap_vals[i + 1] - gap_ses[i + 1]:
            best_gap_k = k_range[i]
            break
    votes_uncapped["gap"] = best_gap_k

    inertia_arr = np.array(inertias)
    if len(inertia_arr) >= 3:
        d1 = np.diff(inertia_arr)
        d2 = np.diff(d1)
        elbow_idx = np.argmax(d2) + 1
        votes_uncapped["elbow"] = k_range[elbow_idx]
    else:
        votes_uncapped["elbow"] = k_range[0]

    print(f"\n    Votes (uncapped): {votes_uncapped}")

    # Apply cap for selection
    capped_votes = {m: min(k, max_k_cap) for m, k in votes_uncapped.items()}
    print(f"    Votes (capped at {max_k_cap}): {capped_votes}")

    vote_counts = Counter(capped_votes.values())
    max_count = vote_counts.most_common(1)[0][1]
    candidates = [k for k, c in vote_counts.items() if c == max_count]

    best_sil_capped = capped_votes["silhouette"]
    if best_sil_capped in candidates:
        optimal_k = best_sil_capped
    else:
        optimal_k = min(candidates)

    metrics_df["recommended"] = False
    metrics_df.loc[metrics_df["k"] == optimal_k, "recommended"] = True

    return {
        "metrics_df": metrics_df,
        "optimal_k": optimal_k,
        "votes_uncapped": votes_uncapped,
        "votes_capped": capped_votes,
        "gap_results": gap_results,
    }


def _gap_statistic(
    X: np.ndarray,
    k_range: List[int],
    clust_cfg: Dict[str, Any],
    rng_seed: int,
) -> Dict[str, Any]:
    """Compute gap statistic with reference uniform distributions."""
    n_refs = clust_cfg["gap_statistic"]["n_references"]
    n_init = clust_cfg["kmeans"]["n_init"]
    rng = np.random.RandomState(rng_seed)

    log_wk_obs = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=rng_seed)
        km.fit(X)
        log_wk_obs.append(np.log(km.inertia_))

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    log_wk_refs = np.zeros((n_refs, len(k_range)))
    for b in range(n_refs):
        X_ref = rng.uniform(mins, maxs, size=X.shape)
        for ki, k in enumerate(k_range):
            km = KMeans(n_clusters=k, n_init=min(10, n_init),
                         random_state=rng_seed)
            km.fit(X_ref)
            log_wk_refs[b, ki] = np.log(km.inertia_)

    gap_values = log_wk_refs.mean(axis=0) - np.array(log_wk_obs)
    gap_se = log_wk_refs.std(axis=0) * np.sqrt(1 + 1.0 / n_refs)

    print(f"    Gap statistic computed with {n_refs} references")

    return {
        "gap_values": [round(float(g), 4) for g in gap_values],
        "gap_se": [round(float(s), 4) for s in gap_se],
    }


# ------------------------------------------------------------------
# K-means, hierarchical, PCA clustering
# ------------------------------------------------------------------

def _fit_kmeans(
    X: np.ndarray, k: int, rng_seed: int, clust_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Fit final K-means at chosen k."""
    n_init = clust_cfg["kmeans"]["n_init"]
    max_iter = clust_cfg["kmeans"]["max_iter"]

    km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter,
                 random_state=rng_seed)
    labels = km.fit_predict(X)

    return {
        "model": km,
        "labels": labels,
        "inertia": km.inertia_,
        "centers": km.cluster_centers_,
        "n_iter": km.n_iter_,
    }


def _fit_hierarchical(
    X: np.ndarray,
    k: int,
    labels_kmeans: np.ndarray,
    clust_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Fit Ward's hierarchical clustering and compare with K-means."""
    method = clust_cfg["hierarchical"]["method"]
    Z = linkage(X, method=method)
    labels_hier = fcluster(Z, t=k, criterion="maxclust") - 1

    ari = adjusted_rand_score(labels_kmeans, labels_hier)
    nmi = normalized_mutual_info_score(labels_kmeans, labels_hier)

    return {
        "labels": labels_hier,
        "linkage_matrix": Z,
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
    }


def _cluster_on_pca(
    X_pca: np.ndarray,
    k: int,
    labels_raw: np.ndarray,
    rng_seed: int,
    clust_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run K-means in PCA space and compare with raw-space solution."""
    n_init = clust_cfg["kmeans"]["n_init"]
    max_iter = clust_cfg["kmeans"]["max_iter"]

    km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter,
                 random_state=rng_seed)
    labels_pca = km.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels_pca)
    ari = adjusted_rand_score(labels_raw, labels_pca)

    return {
        "labels": labels_pca,
        "silhouette": round(sil, 4),
        "ari_vs_raw": round(ari, 4),
        "centers": km.cluster_centers_,
        "n_pca_components": X_pca.shape[1],
    }


# ------------------------------------------------------------------
# Stability
# ------------------------------------------------------------------

def _bootstrap_stability(
    X: np.ndarray,
    k: int,
    labels_reference: np.ndarray,
    clust_cfg: Dict[str, Any],
    rng_seed: int,
) -> Dict[str, Any]:
    """Bootstrap resampling stability assessment."""
    n_resamples = clust_cfg["bootstrap"]["n_resamples"]
    fraction = clust_cfg["bootstrap"]["sample_fraction"]
    n = X.shape[0]
    n_sample = int(n * fraction)
    rng = np.random.RandomState(rng_seed)

    ari_scores = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n_sample, replace=False)
        X_boot = X[idx]
        labels_ref_boot = labels_reference[idx]

        km = KMeans(n_clusters=k, n_init=10, random_state=rng_seed)
        labels_boot = km.fit_predict(X_boot)

        ari = adjusted_rand_score(labels_ref_boot, labels_boot)
        ari_scores.append(ari)

    ari_arr = np.array(ari_scores)
    ci_lo = float(np.percentile(ari_arr, 2.5))
    ci_hi = float(np.percentile(ari_arr, 97.5))

    return {
        "mean_ari": round(float(ari_arr.mean()), 4),
        "std_ari": round(float(ari_arr.std()), 4),
        "ci_95": (round(ci_lo, 4), round(ci_hi, 4)),
        "n_resamples": n_resamples,
        "sample_fraction": fraction,
    }


def _perturbation_stability(
    X: np.ndarray,
    k: int,
    labels_reference: np.ndarray,
    feature_names: List[str],
    rng_seed: int,
) -> Dict[str, Any]:
    """Drop each feature one at a time and recluster."""
    records = []
    for i, feat_name in enumerate(feature_names):
        X_reduced = np.delete(X, i, axis=1)
        km = KMeans(n_clusters=k, n_init=50, random_state=rng_seed)
        labels_pert = km.fit_predict(X_reduced)

        ari = adjusted_rand_score(labels_reference, labels_pert)
        changed = float(np.mean(labels_reference != labels_pert))

        records.append({
            "dropped_feature": feat_name,
            "ari": round(ari, 4),
            "fraction_changed": round(changed, 4),
        })

    result_df = pd.DataFrame(records).sort_values("ari", ascending=True)

    unstable = result_df[result_df["fraction_changed"] > 0.20]
    if len(unstable) > 0:
        print(f"    Features whose removal changes >20% assignments:")
        for _, row in unstable.iterrows():
            print(f"      {row['dropped_feature']}: "
                  f"{row['fraction_changed']:.1%} changed "
                  f"(ARI={row['ari']:.3f})")
    else:
        print("    No single feature removal changes >20% of assignments")

    return {"results": result_df}


# ------------------------------------------------------------------
# Characterisation
# ------------------------------------------------------------------

def _cluster_profiles(
    raw_indicators: pd.DataFrame,
    indicator_names: List[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    """Compute mean and std of raw indicators per cluster."""
    df = raw_indicators[indicator_names].copy()
    df["cluster"] = labels

    means = df.groupby("cluster")[indicator_names].mean()
    means.columns = [f"{c}_mean" for c in means.columns]

    stds = df.groupby("cluster")[indicator_names].std()
    stds.columns = [f"{c}_std" for c in stds.columns]

    counts = df.groupby("cluster").size().rename("n_units")

    profiles = pd.concat([counts, means, stds], axis=1)
    return profiles


def _crosstab_with_supervised(
    labels_cluster: np.ndarray,
    labels_supervised: pd.Series,
) -> pd.DataFrame:
    """Cross-tabulate cluster labels vs supervised shrinkage_class."""
    ct = pd.crosstab(
        pd.Series(labels_cluster, name="cluster"),
        labels_supervised.reset_index(drop=True).rename("shrinkage_class"),
    )
    ari = adjusted_rand_score(labels_supervised.values, labels_cluster)
    nmi = normalized_mutual_info_score(labels_supervised.values, labels_cluster)
    print(f"    Cluster vs supervised: ARI={ari:.3f}, NMI={nmi:.3f}")
    return ct


# ------------------------------------------------------------------
# Specification robustness
# ------------------------------------------------------------------

def run_specification_robustness(
    indicator_data: Dict[str, Any],
    primary_k: int,
    primary_labels: np.ndarray,
    clust_cfg: Dict[str, Any],
    rng_seed: int,
) -> pd.DataFrame | None:
    """
    Re-cluster under alternative indicator subsets and compare with
    the primary solution via ARI and NMI.

    This tests whether the typology is robust to the choice of which
    indicators are included (specification sensitivity).

    Args:
        indicator_data: Dict from compile_indicators().
        primary_k: Number of clusters in the primary solution.
        primary_labels: Cluster labels from the primary solution.
        clust_cfg: clustering section of config.
        rng_seed: Random seed.

    Returns:
        DataFrame with one row per specification, or None if disabled.
    """
    spec_cfg = clust_cfg.get("specification_robustness", {})
    if not spec_cfg.get("enabled", False):
        print("  Specification robustness: disabled in config")
        return None

    specifications = spec_cfg.get("specifications", {})
    if not specifications:
        print("  Specification robustness: no specifications defined")
        return None

    scaled = indicator_data["scaled_indicators"]
    all_features = set(indicator_data["indicator_names"])

    n_init = clust_cfg["kmeans"].get("n_init", 100)
    max_iter = clust_cfg["kmeans"].get("max_iter", 300)

    records = []
    for spec_name, spec_features in specifications.items():
        # Filter to features that are actually available
        valid_features = [f for f in spec_features if f in all_features]
        if len(valid_features) < 2:
            print(f"    {spec_name}: skipped (only {len(valid_features)} "
                  f"valid features)")
            continue

        missing = [f for f in spec_features if f not in all_features]
        if missing:
            print(f"    {spec_name}: {len(missing)} features unavailable "
                  f"({', '.join(missing)})")

        X_spec = scaled[valid_features].values

        km = KMeans(
            n_clusters=primary_k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=rng_seed,
        )
        labels_spec = km.fit_predict(X_spec)

        ari = adjusted_rand_score(primary_labels, labels_spec)
        nmi = normalized_mutual_info_score(primary_labels, labels_spec)
        sil = silhouette_score(X_spec, labels_spec) if X_spec.shape[1] >= 2 else float("nan")

        records.append({
            "specification": spec_name,
            "n_features": len(valid_features),
            "features_used": ", ".join(valid_features),
            "ari_vs_primary": round(ari, 4),
            "nmi_vs_primary": round(nmi, 4),
            "silhouette": round(sil, 4),
        })

        print(f"    {spec_name} ({len(valid_features)} features): "
              f"ARI={ari:.3f}, NMI={nmi:.3f}, sil={sil:.3f}")

    if not records:
        return None

    return pd.DataFrame(records)
