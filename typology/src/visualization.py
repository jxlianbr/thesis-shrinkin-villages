"""
Visualization functions for the typology module.

All figures are publication-quality (300 dpi, PDF+PNG) with
colourblind-friendly palettes and font size >= 10pt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from sklearn.metrics import silhouette_samples

from typology.src.utils import save_figure


# ------------------------------------------------------------------
# Step 1: Indicator figures
# ------------------------------------------------------------------

def plot_indicator_distributions(
    raw_indicators: pd.DataFrame,
    indicator_names: List[str],
    cfg: Dict[str, Any],
) -> Path:
    """Histograms with KDE for each indicator."""
    n = len(indicator_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(indicator_names):
        ax = axes[i]
        data = raw_indicators[col].dropna()
        ax.hist(data, bins=15, color="#5B9BD5", edgecolor="white",
                alpha=0.7, density=True)
        if len(data) > 2:
            data.plot.kde(ax=ax, color="#2E4057", linewidth=1.5)
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Indicator Distributions (raw values)", fontsize=13, y=1.01)
    fig.tight_layout()
    return save_figure(fig, "indicator_distributions", cfg)


def plot_indicator_correlation_heatmap(
    raw_indicators: pd.DataFrame,
    indicator_names: List[str],
    cfg: Dict[str, Any],
) -> Path:
    """Annotated heatmap of indicator correlations."""
    corr = raw_indicators[indicator_names].corr()
    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_heatmap"])
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax,
        annot_kws={"size": 8}, linewidths=0.5,
    )
    ax.set_title("Indicator Correlation Matrix")
    fig.tight_layout()
    return save_figure(fig, "indicator_correlation_heatmap", cfg)


# ------------------------------------------------------------------
# Step 2: Clustering figures
# ------------------------------------------------------------------

def plot_pca_scree(
    pca_data: Dict[str, Any], cfg: Dict[str, Any],
) -> Path:
    """Scree plot with cumulative variance."""
    var_df = pca_data["variance_df"]
    fig, ax1 = plt.subplots(figsize=cfg["plot"]["figsize_single"])

    x = range(1, len(var_df) + 1)
    ax1.bar(x, var_df["explained_variance_ratio"], color="#5B9BD5",
            alpha=0.7, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")

    ax2 = ax1.twinx()
    ax2.plot(x, var_df["cumulative_variance_ratio"], "o-", color="#E74C3C",
             linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance Ratio")

    # Threshold lines
    for thresh, label in pca_data["n_for_threshold"].items():
        thresh_val = int(thresh.replace("%", "")) / 100
        ax2.axhline(thresh_val, color="gray", linestyle="--", alpha=0.5)
        ax2.text(len(var_df) * 0.85, thresh_val + 0.01, thresh,
                fontsize=9, color="gray")

    ax1.set_title("PCA Scree Plot")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.tight_layout()
    return save_figure(fig, "pca_scree", cfg)


def plot_pca_biplot(
    pca_data: Dict[str, Any],
    labels: np.ndarray,
    indicator_names: List[str],
    cfg: Dict[str, Any],
) -> Path:
    """PCA biplot: PC1 vs PC2 with feature loadings."""
    X_pca = pca_data["X_pca"]
    loadings = pca_data["loadings"]
    var_df = pca_data["variance_df"]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter points by cluster
    palette = sns.color_palette(cfg["plot"]["cluster_palette"], n_colors=len(np.unique(labels)))
    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[palette[cid]], label=f"Cluster {cid}",
                   s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

    # Loading arrows
    scale = max(abs(X_pca[:, 0]).max(), abs(X_pca[:, 1]).max()) * 0.8
    for feat in indicator_names:
        lx = loadings.loc[feat, "PC1"] * scale
        ly = loadings.loc[feat, "PC2"] * scale
        ax.annotate(
            "", xy=(lx, ly), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
        )
        ax.text(lx * 1.08, ly * 1.08, feat, fontsize=7, color="#333333",
                ha="center", va="center")

    var1 = var_df["explained_variance_ratio"].iloc[0] * 100
    var2 = var_df["explained_variance_ratio"].iloc[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")
    ax.set_title("PCA Biplot")
    ax.legend(loc="best")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return save_figure(fig, "pca_biplot", cfg)


def plot_optimal_k(
    k_metrics: pd.DataFrame,
    optimal_k: int,
    cfg: Dict[str, Any],
) -> Path:
    """2x2 panel: silhouette, CH, DB/elbow, gap statistic."""
    fig, axes = plt.subplots(2, 2, figsize=cfg["plot"]["figsize_heatmap"])

    k_vals = k_metrics["k"].values

    # Silhouette
    ax = axes[0, 0]
    ax.plot(k_vals, k_metrics["silhouette"], "o-", color="#2196F3")
    ax.axvline(optimal_k, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Analysis")

    # Calinski-Harabasz
    ax = axes[0, 1]
    ax.plot(k_vals, k_metrics["calinski_harabasz"], "o-", color="#4CAF50")
    ax.axvline(optimal_k, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("k")
    ax.set_ylabel("CH Index")
    ax.set_title("Calinski-Harabasz Index")

    # Elbow (inertia) + Davies-Bouldin
    ax = axes[1, 0]
    ax.plot(k_vals, k_metrics["inertia"], "o-", color="#FF9800", label="Inertia")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Method")
    ax.axvline(optimal_k, color="red", linestyle="--", alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(k_vals, k_metrics["davies_bouldin"], "s--", color="#9C27B0",
             alpha=0.7, label="DB Index")
    ax2.set_ylabel("Davies-Bouldin Index")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="best", fontsize=8)

    # Gap statistic
    ax = axes[1, 1]
    if "gap" in k_metrics.columns:
        gap_vals = k_metrics["gap"].values
        gap_se = k_metrics["gap_se"].values
        ax.errorbar(k_vals, gap_vals, yerr=gap_se, fmt="o-", color="#F44336",
                     capsize=3)
    ax.axvline(optimal_k, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("k")
    ax.set_ylabel("Gap Statistic")
    ax.set_title("Gap Statistic")

    fig.suptitle(f"Cluster Number Selection (optimal k={optimal_k})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "optimal_k_metrics", cfg)


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    unit_ids: List[str],
    optimal_k: int,
    cfg: Dict[str, Any],
) -> Path:
    """Ward's linkage dendrogram."""
    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_wide"])
    dendrogram(
        linkage_matrix,
        labels=unit_ids,
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=linkage_matrix[-(optimal_k - 1), 2],
        ax=ax,
    )
    ax.set_title(f"Hierarchical Clustering Dendrogram (Ward, cut at k={optimal_k})")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return save_figure(fig, "dendrogram", cfg)


def plot_cluster_profiles_heatmap(
    profiles: pd.DataFrame,
    indicator_names: List[str],
    cfg: Dict[str, Any],
) -> Path:
    """Heatmap of standardised cluster centroids."""
    # Extract mean columns
    mean_cols = [f"{c}_mean" for c in indicator_names]
    existing = [c for c in mean_cols if c in profiles.columns]
    data = profiles[existing].copy()
    data.columns = [c.replace("_mean", "") for c in existing]
    data.index = [f"Cluster {i}" for i in data.index]

    # Z-score within each column for display
    data_z = (data - data.mean()) / data.std().replace(0, 1)

    fig, ax = plt.subplots(figsize=(max(10, len(indicator_names) * 0.7), 4))
    sns.heatmap(
        data_z, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        ax=ax, linewidths=0.5, annot_kws={"size": 8},
    )
    ax.set_title("Cluster Profiles (z-scored indicator means)")
    fig.tight_layout()
    return save_figure(fig, "cluster_profiles_heatmap", cfg)


def plot_cluster_parallel_coords(
    raw_indicators: pd.DataFrame,
    indicator_names: List[str],
    labels: np.ndarray,
    cfg: Dict[str, Any],
) -> Path:
    """Parallel coordinates plot of indicators coloured by cluster."""
    df = raw_indicators[indicator_names].copy()

    # Normalize each column to [0, 1] for comparable axes
    for col in indicator_names:
        cmin = df[col].min()
        cmax = df[col].max()
        rng = cmax - cmin
        if rng > 0:
            df[col] = (df[col] - cmin) / rng
        else:
            df[col] = 0.5

    df["cluster"] = labels

    palette = sns.color_palette(cfg["plot"]["cluster_palette"],
                                n_colors=len(np.unique(labels)))

    fig, ax = plt.subplots(figsize=(max(12, len(indicator_names) * 0.9), 6))

    for cid in sorted(np.unique(labels)):
        subset = df[df["cluster"] == cid]
        for _, row in subset.iterrows():
            ax.plot(range(len(indicator_names)), row[indicator_names].values,
                    color=palette[cid], alpha=0.3, linewidth=0.8)
        # Plot mean
        mean_vals = subset[indicator_names].mean().values
        ax.plot(range(len(indicator_names)), mean_vals,
                color=palette[cid], linewidth=2.5, label=f"Cluster {cid}")

    ax.set_xticks(range(len(indicator_names)))
    ax.set_xticklabels(indicator_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalised Value [0-1]")
    ax.set_title("Parallel Coordinates (cluster profiles)")
    ax.legend(loc="best")
    fig.tight_layout()
    return save_figure(fig, "cluster_parallel_coords", cfg)


def plot_cluster_boxplots(
    raw_indicators: pd.DataFrame,
    indicator_names: List[str],
    labels: np.ndarray,
    cfg: Dict[str, Any],
) -> Path:
    """Box plot panel: one subplot per indicator grouped by cluster."""
    n = len(indicator_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    palette = sns.color_palette(cfg["plot"]["cluster_palette"],
                                n_colors=len(np.unique(labels)))
    df = raw_indicators[indicator_names].copy()
    df["cluster"] = labels

    for i, col in enumerate(indicator_names):
        ax = axes[i]
        sns.boxplot(
            data=df, x="cluster", y=col, hue="cluster",
            palette=palette, ax=ax, legend=False,
        )
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Cluster")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Indicator Distributions by Cluster", fontsize=13, y=1.01)
    fig.tight_layout()
    return save_figure(fig, "cluster_boxplots", cfg)


def plot_cluster_crosstab(
    crosstab: pd.DataFrame, cfg: Dict[str, Any],
) -> Path:
    """Confusion-style heatmap: cluster vs supervised labels."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        crosstab, annot=True, fmt="d", cmap="Blues", ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Cluster vs Supervised Shrinkage Class")
    ax.set_xlabel("Supervised Class")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    return save_figure(fig, "cluster_crosstab_heatmap", cfg)


def plot_cluster_map(
    gdf: Any,
    labels: np.ndarray,
    identifiers: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Path | None:
    """Choropleth map of cluster assignments."""
    try:
        import geopandas as gpd
    except ImportError:
        print("  geopandas not available -- skipping cluster map")
        return None

    gdf = gdf.copy()

    # Merge cluster labels
    id_df = identifiers.copy()
    id_df["cluster"] = labels
    gdf = gdf.merge(id_df[["unit_id", "cluster"]], on="unit_id", how="left")

    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_single"])
    gdf.plot(
        column="cluster",
        categorical=True,
        cmap=cfg["plot"]["cluster_palette"],
        legend=True,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        legend_kwds={"title": "Cluster", "loc": "lower right"},
    )
    ax.set_title("Cluster Assignments (spatial)")
    ax.set_axis_off()
    fig.tight_layout()
    return save_figure(fig, "cluster_map", cfg)


def plot_silhouette_diagram(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    cfg: Dict[str, Any],
    suffix: str = "",
) -> Path:
    """Per-sample silhouette diagram with horizontal bars grouped by cluster."""
    sil_vals = silhouette_samples(X, labels)
    sil_mean = float(np.mean(sil_vals))

    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_single"])
    palette = sns.color_palette(cfg["plot"]["cluster_palette"], n_colors=k)

    y_lower = 0
    for cid in range(k):
        mask = labels == cid
        cluster_sil = np.sort(sil_vals[mask])
        size_c = len(cluster_sil)

        y_upper = y_lower + size_c
        ax.barh(
            range(y_lower, y_upper), cluster_sil,
            height=1.0, color=palette[cid], edgecolor="none",
        )
        # Cluster label in the middle
        y_mid = y_lower + size_c / 2
        ax.text(-0.05, y_mid, f"C{cid}\n(n={size_c})",
                ha="right", va="center", fontsize=9, fontweight="bold")

        y_lower = y_upper + 2  # small gap between clusters

    # Mean silhouette line
    ax.axvline(sil_mean, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {sil_mean:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Samples (grouped by cluster)")
    ax.set_title(f"Silhouette Diagram (k={k})")
    ax.set_yticks([])
    ax.legend(loc="lower right")
    fig.tight_layout()

    fname = f"silhouette_diagram_k{k}" + (f"_{suffix}" if suffix else "")
    return save_figure(fig, fname, cfg)


def plot_multi_k_comparison(
    solutions: Dict[int, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Path:
    """2x2 panel comparing k values on key metrics."""
    k_vals = sorted(solutions.keys())
    sil_scores = [solutions[k]["silhouette_mean"] for k in k_vals]
    boot_aris = [solutions[k]["bootstrap"]["mean_ari"] for k in k_vals]
    boot_cis = [solutions[k]["bootstrap"]["ci_95"] for k in k_vals]
    sup_aris = [solutions[k]["supervised_ari"] for k in k_vals]

    fig, axes = plt.subplots(2, 2, figsize=cfg["plot"]["figsize_heatmap"])

    # 1) Silhouette
    ax = axes[0, 0]
    ax.bar(range(len(k_vals)), sil_scores, color="#2196F3", alpha=0.8)
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([f"k={k}" for k in k_vals])
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Mean Silhouette")
    for i, v in enumerate(sil_scores):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    # 2) Bootstrap ARI
    ax = axes[0, 1]
    ci_lo = [c[0] for c in boot_cis]
    ci_hi = [c[1] for c in boot_cis]
    yerr_low = [m - lo for m, lo in zip(boot_aris, ci_lo)]
    yerr_high = [hi - m for m, hi in zip(boot_aris, ci_hi)]
    ax.bar(range(len(k_vals)), boot_aris, color="#4CAF50", alpha=0.8,
           yerr=[yerr_low, yerr_high], capsize=5, ecolor="black")
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([f"k={k}" for k in k_vals])
    ax.set_ylabel("Bootstrap ARI")
    ax.set_title("Bootstrap Stability (95% CI)")

    # 3) Cluster size distribution
    ax = axes[1, 0]
    bar_width = 0.8 / max(1, max(k_vals))
    for i, k in enumerate(k_vals):
        sizes = sorted(solutions[k]["cluster_sizes"].values(), reverse=True)
        x_positions = [i + j * bar_width - (len(sizes) - 1) * bar_width / 2
                       for j in range(len(sizes))]
        palette = sns.color_palette(cfg["plot"]["cluster_palette"], n_colors=k)
        for j, (xp, s) in enumerate(zip(x_positions, sizes)):
            ax.bar(xp, s, width=bar_width * 0.9, color=palette[j % len(palette)])
            ax.text(xp, s + 0.5, str(s), ha="center", fontsize=7)
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([f"k={k}" for k in k_vals])
    ax.set_ylabel("Cluster Size")
    ax.set_title("Cluster Size Distribution")
    ax.axhline(15, color="red", linestyle="--", alpha=0.5, label="Min for regression")
    ax.legend(fontsize=8)

    # 4) Supervised ARI
    ax = axes[1, 1]
    ax.bar(range(len(k_vals)), sup_aris, color="#FF9800", alpha=0.8)
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([f"k={k}" for k in k_vals])
    ax.set_ylabel("ARI vs Supervised")
    ax.set_title("Agreement with Supervised Classes")
    for i, v in enumerate(sup_aris):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle("Multi-k Solution Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "multi_k_comparison", cfg)


# ------------------------------------------------------------------
# Step 3: Relationship figures
# ------------------------------------------------------------------

def plot_correlation_heatmaps(
    corr_results: Dict[str, Any],
    physical_names: List[str],
    demo_names: List[str],
    cfg: Dict[str, Any],
) -> List[Path]:
    """Annotated heatmaps for Pearson and Spearman correlations."""
    paths = []
    for method in ["pearson", "spearman"]:
        corr = corr_results.get(method)
        if corr is None:
            continue

        # Subset to physical (rows) x demographic (cols)
        corr_sub = corr.loc[
            [r for r in physical_names if r in corr.index],
            [c for c in demo_names if c in corr.columns],
        ]

        fig, ax = plt.subplots(figsize=(max(8, len(demo_names) * 1.2),
                                        max(6, len(physical_names) * 0.5)))
        sns.heatmap(
            corr_sub.astype(float), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 9}, linewidths=0.5,
        )
        ax.set_title(f"RS-Demographic Correlations ({method.title()})")
        fig.tight_layout()
        paths.append(save_figure(fig, f"correlation_heatmap_{method}", cfg))

    return paths


def plot_regression_diagnostics(
    regression_data: Dict[str, Any], cfg: Dict[str, Any],
) -> Path | None:
    """2x2 diagnostic panel: residuals vs fitted, Q-Q, histogram, VIF."""
    residuals = regression_data.get("residuals")
    fitted = regression_data.get("fitted")
    vif_results = regression_data.get("vif_results")

    if residuals is None or fitted is None:
        print("  No regression residuals -- skipping diagnostics plot")
        return None

    fig, axes = plt.subplots(2, 2, figsize=cfg["plot"]["figsize_heatmap"])

    # Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(fitted, residuals, alpha=0.6, s=30, color="#2196F3")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # Q-Q plot
    ax = axes[0, 1]
    from scipy import stats as sp_stats
    sorted_res = np.sort(residuals)
    theoretical = sp_stats.norm.ppf(
        (np.arange(1, len(sorted_res) + 1) - 0.5) / len(sorted_res)
    )
    ax.scatter(theoretical, sorted_res, alpha=0.6, s=30, color="#4CAF50")
    lims = [min(theoretical.min(), sorted_res.min()),
            max(theoretical.max(), sorted_res.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Q-Q Plot (residuals)")

    # Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=15, color="#FF9800", edgecolor="white", alpha=0.7)
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")

    # VIF bar chart
    ax = axes[1, 1]
    if vif_results is not None and len(vif_results) > 0:
        vif_sorted = vif_results.sort_values("vif", ascending=True)
        ax.barh(vif_sorted["feature"], vif_sorted["vif"], color="#9C27B0")
        ax.axvline(5, color="orange", linestyle="--", label="VIF=5")
        ax.axvline(10, color="red", linestyle="--", label="VIF=10")
        ax.set_xlabel("VIF")
        ax.set_title("Variance Inflation Factors")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "VIF data not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("VIF")

    fig.suptitle("OLS Regression Diagnostics", fontsize=13, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "regression_diagnostics", cfg)


def plot_regression_coefficients(
    coefficients: pd.DataFrame, cfg: Dict[str, Any],
) -> Path | None:
    """Horizontal bar chart of regression coefficients with CI."""
    if coefficients is None or len(coefficients) == 0:
        return None

    # Exclude constant
    coefs = coefficients[coefficients["feature"] != "const"].copy()
    if len(coefs) == 0:
        return None

    fig, ax = plt.subplots(figsize=cfg["plot"]["figsize_single"])
    coefs_sorted = coefs.sort_values("coefficient")

    y_pos = range(len(coefs_sorted))
    colors = ["#F44336" if c < 0 else "#4CAF50"
              for c in coefs_sorted["coefficient"]]

    ax.barh(y_pos, coefs_sorted["coefficient"], color=colors, alpha=0.7)

    # Error bars if CI available
    if "ci_lower" in coefs_sorted.columns and "ci_upper" in coefs_sorted.columns:
        xerr_low = coefs_sorted["coefficient"] - coefs_sorted["ci_lower"]
        xerr_high = coefs_sorted["ci_upper"] - coefs_sorted["coefficient"]
        ax.errorbar(
            coefs_sorted["coefficient"], y_pos,
            xerr=[xerr_low.values, xerr_high.values],
            fmt="none", color="black", capsize=3,
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(coefs_sorted["feature"].values)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Coefficient")
    ax.set_title("OLS Regression Coefficients (95% CI)")
    fig.tight_layout()
    return save_figure(fig, "regression_coefficients", cfg)


def plot_specification_robustness(
    spec_df: pd.DataFrame,
    primary_k: int,
    cfg: Dict[str, Any],
) -> Path:
    """
    Grouped bar chart showing ARI, NMI and silhouette per specification.

    Args:
        spec_df: DataFrame from run_specification_robustness().
        primary_k: Primary k for title.
        cfg: Config dict.

    Returns:
        Path to saved figure.
    """
    n_specs = len(spec_df)
    x = np.arange(n_specs)
    bar_width = 0.25

    fig, ax1 = plt.subplots(figsize=(max(10, n_specs * 2), 6))

    # ARI bars
    bars1 = ax1.bar(x - bar_width, spec_df["ari_vs_primary"],
                     bar_width, color="#2196F3", alpha=0.8, label="ARI")
    # NMI bars
    bars2 = ax1.bar(x, spec_df["nmi_vs_primary"],
                     bar_width, color="#4CAF50", alpha=0.8, label="NMI")
    # Silhouette bars
    bars3 = ax1.bar(x + bar_width, spec_df["silhouette"],
                     bar_width, color="#FF9800", alpha=0.8, label="Silhouette")

    ax1.set_xticks(x)
    labels = [f"{row['specification']}\n({row['n_features']} feat.)"
              for _, row in spec_df.iterrows()]
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Score")
    ax1.set_title(f"Specification Robustness (k={primary_k})")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 1.05)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                         f"{height:.2f}", ha="center", va="bottom", fontsize=7)

    # Reference line at ARI=0.5 (moderate agreement)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.text(n_specs - 0.5, 0.51, "moderate", fontsize=8, color="gray")

    fig.tight_layout()
    return save_figure(fig, "specification_robustness", cfg)


def plot_spatial_model_comparison(
    spatial_reg: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Path | None:
    """
    Bar chart comparing OLS, Spatial Lag, and Spatial Error model fit.

    Shows AIC (lower is better) and pseudo-R2/R2 side by side.
    """
    ols_res = spatial_reg.get("ols", {})
    lag_res = spatial_reg.get("spatial_lag", {})
    err_res = spatial_reg.get("spatial_error", {})

    # Collect models that succeeded
    models = []
    if "aic" in ols_res:
        models.append({
            "name": "OLS",
            "aic": ols_res["aic"],
            "r2": ols_res.get("r_squared", 0),
        })
    if "aic" in lag_res:
        models.append({
            "name": "Spatial Lag",
            "aic": lag_res["aic"],
            "r2": lag_res.get("pseudo_r_squared", 0),
        })
    if "aic" in err_res:
        models.append({
            "name": "Spatial Error",
            "aic": err_res["aic"],
            "r2": err_res.get("pseudo_r_squared", 0),
        })

    if len(models) < 2:
        print("  Not enough spatial models for comparison plot")
        return None

    names = [m["name"] for m in models]
    aics = [m["aic"] for m in models]
    r2s = [m["r2"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # AIC comparison
    ax = axes[0]
    colors = ["#2196F3", "#4CAF50", "#FF9800"][:len(models)]
    bars = ax.bar(names, aics, color=colors, alpha=0.8)
    ax.set_ylabel("AIC (lower is better)")
    ax.set_title("Model Comparison: AIC")
    for bar, val in zip(bars, aics):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    # Highlight best
    best_idx = np.argmin(aics)
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(2)

    # R2 comparison
    ax = axes[1]
    bars = ax.bar(names, r2s, color=colors, alpha=0.8)
    ax.set_ylabel("R-squared / Pseudo-R2")
    ax.set_title("Model Comparison: Fit")
    ax.set_ylim(0, max(max(r2s) * 1.3, 0.5))
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    dv = spatial_reg.get("dependent_var", "")
    fig.suptitle(f"Spatial Regression Model Comparison (DV: {dv})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "spatial_model_comparison", cfg)


def plot_subgroup_correlations(
    subgroup_data: Dict[str, Any],
    physical_names: List[str],
    demo_names: List[str],
    cfg: Dict[str, Any],
) -> Path | None:
    """Correlation heatmaps faceted by cluster."""
    by_cluster = subgroup_data.get("by_cluster", {})
    clusters_with_corr = {
        k: v for k, v in by_cluster.items()
        if v.get("correlation_matrix") is not None
    }

    if not clusters_with_corr:
        print("  No cluster subgroups with enough data for correlation")
        return None

    n_clusters = len(clusters_with_corr)
    fig, axes = plt.subplots(1, n_clusters,
                              figsize=(6 * n_clusters, max(5, len(physical_names) * 0.4)))
    if n_clusters == 1:
        axes = [axes]

    for idx, (cid, info) in enumerate(sorted(clusters_with_corr.items())):
        ax = axes[idx]
        corr = info["correlation_matrix"]

        # Subset
        phys_avail = [c for c in physical_names if c in corr.columns]
        demo_avail = [c for c in demo_names if c in corr.columns]
        if phys_avail and demo_avail:
            corr_sub = corr.loc[phys_avail, demo_avail].astype(float)
            sns.heatmap(
                corr_sub, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 7}, linewidths=0.5,
            )
        ax.set_title(f"Cluster {cid} (n={info['n_units']})", fontsize=10)

    fig.suptitle("RS-Demographic Correlations by Cluster", fontsize=13, y=1.02)
    fig.tight_layout()
    return save_figure(fig, "subgroup_correlations", cfg)
