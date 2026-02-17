"""
Main entry point for the shrinkage typology pipeline.

Compiles physical and demographic indicators, performs unsupervised
clustering to identify shrinkage types, and analyses relationships
between remote sensing and demographic dimensions.

Usage:
    python typology/run_typology.py
    python typology/run_typology.py --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from typology.src.utils import (
    load_config,
    ensure_output_dirs,
    save_table,
    setup_plot_style,
    write_json,
)
from typology.src.data_loader import load_typology_data
from typology.src.step1_indicators import compile_indicators
from typology.src.step2_clustering import run_clustering, run_specification_robustness
from typology.src.step3_relationships import run_relationship_analysis
from typology.src.visualization import (
    plot_indicator_distributions,
    plot_indicator_correlation_heatmap,
    plot_pca_scree,
    plot_pca_biplot,
    plot_optimal_k,
    plot_dendrogram,
    plot_cluster_profiles_heatmap,
    plot_cluster_parallel_coords,
    plot_cluster_boxplots,
    plot_cluster_crosstab,
    plot_cluster_map,
    plot_silhouette_diagram,
    plot_multi_k_comparison,
    plot_correlation_heatmaps,
    plot_regression_diagnostics,
    plot_regression_coefficients,
    plot_subgroup_correlations,
    plot_specification_robustness,
    plot_spatial_model_comparison,
)
from typology.src.report_generator import generate_report


def main(config_path: str = "typology/config/typology_config.yaml") -> None:
    """Run the full typology pipeline."""
    t0 = time.time()

    print("=" * 60)
    print("  Typology Pipeline - Shrinking Villages")
    print("=" * 60)

    cfg = load_config(config_path)
    ensure_output_dirs(cfg)
    setup_plot_style(cfg)

    summary: Dict[str, Any] = {}

    # ----------------------------------------------------------------
    # Step 1: Load Data + Indicator Compilation
    # ----------------------------------------------------------------
    print("\n--- Step 1: Load Data + Indicator Compilation ---")
    data = load_typology_data(cfg)
    summary["n_units"] = data["n_units"]

    indicator_data = compile_indicators(data, cfg)
    indicator_names = indicator_data["indicator_names"]

    # Save tables
    save_table(indicator_data["raw_indicators"], "indicator_matrix_raw", cfg)
    save_table(indicator_data["scaled_indicators"], "indicator_matrix_scaled", cfg)
    if len(indicator_data["slope_validation"]) > 0:
        save_table(indicator_data["slope_validation"], "slope_validation", cfg)

    # Save indicator report JSON
    indicator_report_path = str(
        Path(cfg["output"]["reports_dir"]) / "indicator_report.json"
    )
    write_json(indicator_report_path, {
        "n_indicators": len(indicator_names),
        "physical_indicators": indicator_data["physical_names"],
        "demographic_indicators": indicator_data["demographic_names"],
        "quality": indicator_data["quality"],
    })

    # Figures
    plot_indicator_distributions(
        indicator_data["raw_indicators"], indicator_names, cfg,
    )
    plot_indicator_correlation_heatmap(
        indicator_data["raw_indicators"], indicator_names, cfg,
    )

    summary["indicators"] = {
        "n_indicators": len(indicator_names),
        "n_physical": len(indicator_data["physical_names"]),
        "n_demographic": len(indicator_data["demographic_names"]),
        "quality": indicator_data["quality"],
    }

    # ----------------------------------------------------------------
    # Step 2: Cluster Analysis
    # ----------------------------------------------------------------
    print("\n--- Step 2: Cluster Analysis ---")
    cluster_data = run_clustering(indicator_data, data, cfg)

    # Save shared tables (PCA, k metrics)
    save_table(cluster_data["pca"]["variance_df"], "pca_variance", cfg)
    save_table(cluster_data["pca"]["loadings"], "pca_loadings", cfg)
    save_table(cluster_data["k_metrics"], "optimal_k_metrics", cfg)

    if cluster_data["perturbation"] is not None:
        save_table(
            cluster_data["perturbation"]["results"],
            "perturbation_stability", cfg,
        )

    # Save primary-k backward-compatible tables
    save_table(cluster_data["profiles"], "cluster_profiles", cfg)
    save_table(cluster_data["crosstab"], "cluster_crosstab", cfg)
    save_table(cluster_data["unit_clusters"], "unit_cluster_assignments", cfg)

    # Save per-k tables and figures
    solutions = cluster_data["solutions"]
    scaled_X = indicator_data["scaled_indicators"][indicator_names].values

    for k, sol in solutions.items():
        save_table(sol["unit_clusters"], f"unit_cluster_assignments_k{k}", cfg)
        save_table(sol["profiles"], f"cluster_profiles_k{k}", cfg)
        save_table(sol["crosstab"], f"cluster_crosstab_k{k}", cfg)

        # Per-k silhouette diagram
        plot_silhouette_diagram(scaled_X, sol["labels"], k, cfg)

        # Per-k profiles heatmap (use suffix in filename)
        # Re-use the existing function but the primary k gets default name
        # Additional k values get suffixed names handled by saving manually

    # Shared figures (not per-k)
    plot_pca_scree(cluster_data["pca"], cfg)
    plot_pca_biplot(
        cluster_data["pca"], cluster_data["cluster_labels"],
        indicator_names, cfg,
    )
    plot_optimal_k(cluster_data["k_metrics"], cluster_data["optimal_k"], cfg)
    plot_dendrogram(
        cluster_data["hierarchical"]["linkage_matrix"],
        data["identifiers"]["unit_id"].tolist(),
        cluster_data["optimal_k"], cfg,
    )

    # Primary k figures (default names, backward-compatible)
    plot_cluster_profiles_heatmap(
        cluster_data["profiles"], indicator_names, cfg,
    )
    plot_cluster_parallel_coords(
        indicator_data["raw_indicators"], indicator_names,
        cluster_data["cluster_labels"], cfg,
    )
    plot_cluster_boxplots(
        indicator_data["raw_indicators"], indicator_names,
        cluster_data["cluster_labels"], cfg,
    )
    plot_cluster_crosstab(cluster_data["crosstab"], cfg)

    if data["boundaries"] is not None:
        plot_cluster_map(
            data["boundaries"], cluster_data["cluster_labels"],
            data["identifiers"], cfg,
        )

    # Multi-k comparison figure
    plot_multi_k_comparison(solutions, cfg)

    # Specification robustness
    spec_robustness = None
    if cfg["clustering"].get("specification_robustness", {}).get("enabled", False):
        print("\n  Running specification robustness tests...")
        spec_robustness = run_specification_robustness(
            indicator_data,
            cluster_data["primary_k"],
            cluster_data["cluster_labels"],
            cfg["clustering"],
            cfg["random_state"],
        )
        if spec_robustness is not None:
            save_table(spec_robustness, "specification_robustness", cfg)
            plot_specification_robustness(
                spec_robustness, cluster_data["primary_k"], cfg,
            )
        else:
            print("  Specification robustness: no results produced")

    # Build multi-k summary for report
    multi_k_summary = {}
    for k, sol in solutions.items():
        multi_k_summary[k] = {
            "silhouette": sol["silhouette_mean"],
            "bootstrap_ari": sol["bootstrap"]["mean_ari"],
            "bootstrap_ci": sol["bootstrap"]["ci_95"],
            "supervised_ari": sol["supervised_ari"],
            "cluster_sizes": sol["cluster_sizes"],
            "hier_ari": sol["hierarchical"]["ari"],
        }
        if sol.get("pca_kmeans") is not None:
            multi_k_summary[k]["pca_silhouette"] = sol["pca_kmeans"]["silhouette"]
            multi_k_summary[k]["pca_ari_vs_raw"] = sol["pca_kmeans"]["ari_vs_raw"]

    summary["clustering"] = {
        "primary_k": cluster_data["primary_k"],
        "optimal_k": cluster_data["optimal_k"],
        "data_driven_k": cluster_data["data_driven_k"],
        "report_k_values": cluster_data["report_k_values"],
        "pca_thresholds": cluster_data["pca"]["n_for_threshold"],
        "silhouette_mean": cluster_data["silhouette_mean"],
        "kmeans_hier_ari": cluster_data["hierarchical"]["ari"],
        "kmeans_hier_nmi": cluster_data["hierarchical"]["nmi"],
        "bootstrap": cluster_data["bootstrap"],
        "multi_k": multi_k_summary,
        "specification_robustness": spec_robustness,
    }

    # ----------------------------------------------------------------
    # Step 3: Relationship Analysis
    # ----------------------------------------------------------------
    print("\n--- Step 3: Relationship Analysis ---")
    rel_data = run_relationship_analysis(
        indicator_data, cluster_data, data, cfg,
    )

    # Save tables
    for method in ["pearson", "spearman"]:
        if method in rel_data["correlations"]:
            save_table(
                rel_data["correlations"][method],
                f"correlations_{method}", cfg,
            )

    regression = rel_data["regression"]
    if regression.get("coefficients") is not None:
        save_table(regression["coefficients"], "regression_coefficients", cfg)
    if regression.get("vif_results") is not None and len(regression["vif_results"]) > 0:
        save_table(regression["vif_results"], "regression_vif", cfg)

    subgroup = rel_data["subgroup"]
    if subgroup.get("kruskal_wallis") is not None and len(subgroup["kruskal_wallis"]) > 0:
        save_table(subgroup["kruskal_wallis"], "kruskal_wallis_tests", cfg)

    # Figures
    plot_correlation_heatmaps(
        rel_data["correlations"],
        indicator_data["physical_names"],
        indicator_data["demographic_names"], cfg,
    )
    if regression.get("residuals") is not None:
        plot_regression_diagnostics(regression, cfg)
    if regression.get("coefficients") is not None:
        plot_regression_coefficients(regression["coefficients"], cfg)
    plot_subgroup_correlations(
        subgroup, indicator_data["physical_names"],
        indicator_data["demographic_names"], cfg,
    )

    # Spatial model comparison plot
    spatial_reg = rel_data["spatial"].get("spatial_regression")
    if spatial_reg and not spatial_reg.get("skipped", True):
        plot_spatial_model_comparison(spatial_reg, cfg)

    summary["relationships"] = {
        "n_significant_correlations": len(
            rel_data["correlations"].get("significant_pairs", [])
        ),
        "top_correlations": rel_data["correlations"].get(
            "significant_pairs", []
        )[:5],
        "regression": {
            "r_squared": regression.get("r_squared"),
            "adj_r_squared": regression.get("adj_r_squared"),
            "f_stat": regression.get("f_stat"),
            "f_pvalue": regression.get("f_pvalue"),
            "dependent_var": regression.get("dependent_var"),
        },
        "spatial": {
            "morans_i": rel_data["spatial"].get("morans_i"),
            "morans_p": rel_data["spatial"].get("morans_p"),
            "significant": rel_data["spatial"].get("significant"),
            "spatial_regression": rel_data["spatial"].get("spatial_regression"),
        },
    }

    # ----------------------------------------------------------------
    # Step 4: Generate Report
    # ----------------------------------------------------------------
    print("\n--- Step 4: Generate Report ---")
    generate_report(summary, cfg)

    # Done
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  Typology pipeline complete.")
    print(f"  Primary k: {cluster_data['primary_k']}  "
          f"(data-driven: {cluster_data['data_driven_k']})")
    print(f"  Reported k values: {cluster_data['report_k_values']}")
    print(f"  Silhouette (primary): {cluster_data['silhouette_mean']:.3f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Outputs: {cfg['output']['base_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Typology pipeline for shrinking villages",
    )
    parser.add_argument(
        "--config",
        default="typology/config/typology_config.yaml",
        help="Path to typology config YAML",
    )
    args = parser.parse_args()
    main(args.config)
