"""
Report generation for the typology module.

Compiles all results into an HTML report with embedded figures
and a JSON summary document.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from jinja2 import Template

from typology.src.utils import write_json


# ------------------------------------------------------------------ #
#  HTML report                                                       #
# ------------------------------------------------------------------ #

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ title }}</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; line-height: 1.6; }
  h1 { color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 10px; }
  h2 { color: #1976D2; margin-top: 40px; }
  h3 { color: #1E88E5; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; }
  th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
  th { background-color: #E3F2FD; font-weight: 600; }
  tr:nth-child(even) { background-color: #f9f9f9; }
  .note { background: #FFF3E0; padding: 12px; border-left: 4px solid #FF9800; margin: 15px 0; }
  .warning { background: #FFEBEE; padding: 12px; border-left: 4px solid #F44336; margin: 15px 0; }
  .figure { text-align: center; margin: 20px 0; }
  .figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
  .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #777; font-size: 0.9em; }
</style>
</head>
<body>

<h1>{{ title }}</h1>
<p><strong>Author:</strong> {{ author }}</p>

<h2>1. Data Overview</h2>
<ul>
  <li>Units: {{ n_units }}</li>
  <li>Indicators: {{ n_indicators }} ({{ n_physical }} physical + {{ n_demographic }} demographic)</li>
  <li>Supervised classes: {{ class_labels | join(', ') }}</li>
</ul>

<div class="note">
  <strong>Note:</strong> n={{ n_units }} is a small sample. All statistical
  results should be interpreted with appropriate caution regarding
  precision and generalisability.
</div>

<h2>2. Indicator Matrix</h2>
{% if quality_notes %}
<ul>
  {% for note in quality_notes %}
  <li>{{ note }}</li>
  {% endfor %}
</ul>
{% endif %}

{% if figures.indicator_distributions %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.indicator_distributions }}" alt="Indicator distributions">
</div>
{% endif %}

{% if figures.indicator_correlation_heatmap %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.indicator_correlation_heatmap }}" alt="Indicator correlations">
</div>
{% endif %}

<h2>3. PCA Analysis</h2>
<p>Components for variance thresholds: {{ pca_thresholds_str }}</p>

{% if figures.pca_scree %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.pca_scree }}" alt="PCA scree plot">
</div>
{% endif %}

{% if figures.pca_biplot %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.pca_biplot }}" alt="PCA biplot">
</div>
{% endif %}

<h2>4. Clustering Results</h2>
<p>Primary k: <strong>{{ primary_k }}</strong> |
   Data-driven k: <strong>{{ data_driven_k }}</strong> |
   Reported k values: <strong>{{ report_k_values_str }}</strong></p>
<p>Silhouette (primary): <strong>{{ silhouette_score }}</strong> |
   K-means vs Hierarchical ARI: <strong>{{ kmeans_hier_ari }}</strong></p>

{% if figures.optimal_k_metrics %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.optimal_k_metrics }}" alt="Optimal k metrics">
</div>
{% endif %}

{% if figures.dendrogram %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.dendrogram }}" alt="Dendrogram">
</div>
{% endif %}

<h2>5. Multi-k Comparison</h2>

{% if multi_k_table %}
<table>
  <tr>
    <th>k</th><th>Silhouette</th><th>Bootstrap ARI</th><th>Bootstrap 95% CI</th>
    <th>Supervised ARI</th><th>Hier ARI</th><th>Cluster Sizes</th>
  </tr>
  {% for row in multi_k_table %}
  <tr{% if row.primary %} style="background-color: #E8F5E9; font-weight: 600;"{% endif %}>
    <td>{{ row.k }}{% if row.primary %} *{% endif %}</td>
    <td>{{ row.silhouette }}</td>
    <td>{{ row.boot_ari }}</td>
    <td>{{ row.boot_ci }}</td>
    <td>{{ row.sup_ari }}</td>
    <td>{{ row.hier_ari }}</td>
    <td>{{ row.sizes }}</td>
  </tr>
  {% endfor %}
</table>
<p><em>* Primary k selected by best silhouette with parsimony tiebreak (+/- 0.02).</em></p>
{% endif %}

{% if figures.multi_k_comparison %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.multi_k_comparison }}" alt="Multi-k comparison">
</div>
{% endif %}

<h3>Silhouette Diagrams</h3>
{% for k_fig_key, k_fig_val in silhouette_figures.items() %}
<div class="figure">
  <img src="data:image/png;base64,{{ k_fig_val }}" alt="Silhouette diagram k={{ k_fig_key }}">
</div>
{% endfor %}

<h2>6. Cluster Characterisation (Primary k={{ primary_k }})</h2>

{% if figures.cluster_profiles_heatmap %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.cluster_profiles_heatmap }}" alt="Cluster profiles heatmap">
</div>
{% endif %}

{% if figures.cluster_parallel_coords %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.cluster_parallel_coords }}" alt="Parallel coordinates">
</div>
{% endif %}

{% if figures.cluster_boxplots %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.cluster_boxplots }}" alt="Cluster boxplots">
</div>
{% endif %}

<h2>7. Stability Analysis</h2>
{% if bootstrap_str %}
<p>Bootstrap stability (primary k): {{ bootstrap_str }}</p>
{% endif %}

<h3>7a. Specification Robustness</h3>
{% if spec_robustness_table %}
<p>Re-clustering under alternative indicator subsets (k={{ primary_k }}),
   compared with primary solution via Adjusted Rand Index (ARI) and
   Normalised Mutual Information (NMI).</p>
<table>
  <tr>
    <th>Specification</th><th># Features</th><th>ARI</th><th>NMI</th><th>Silhouette</th>
  </tr>
  {% for row in spec_robustness_table %}
  <tr>
    <td>{{ row.specification }}</td>
    <td>{{ row.n_features }}</td>
    <td>{{ row.ari }}</td>
    <td>{{ row.nmi }}</td>
    <td>{{ row.silhouette }}</td>
  </tr>
  {% endfor %}
</table>
{% if figures.specification_robustness %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.specification_robustness }}" alt="Specification robustness">
</div>
{% endif %}
{% else %}
<p><em>Specification robustness not computed (disabled or no specifications defined).</em></p>
{% endif %}

<h2>8. Supervised Comparison</h2>

{% if figures.cluster_crosstab_heatmap %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.cluster_crosstab_heatmap }}" alt="Cluster vs supervised">
</div>
{% endif %}

{% if figures.cluster_map %}
<h2>9. Spatial Map</h2>
<div class="figure">
  <img src="data:image/png;base64,{{ figures.cluster_map }}" alt="Cluster map">
</div>
{% endif %}

<h2>10. Relationship Analysis</h2>

{% if figures.correlation_heatmap_pearson %}
<h3>Pearson Correlations</h3>
<div class="figure">
  <img src="data:image/png;base64,{{ figures.correlation_heatmap_pearson }}" alt="Pearson correlations">
</div>
{% endif %}

{% if figures.correlation_heatmap_spearman %}
<h3>Spearman Correlations</h3>
<div class="figure">
  <img src="data:image/png;base64,{{ figures.correlation_heatmap_spearman }}" alt="Spearman correlations">
</div>
{% endif %}

{% if regression_str %}
<h3>OLS Regression</h3>
<p>{{ regression_str }}</p>
{% endif %}

{% if figures.regression_diagnostics %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.regression_diagnostics }}" alt="Regression diagnostics">
</div>
{% endif %}

{% if figures.regression_coefficients %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.regression_coefficients }}" alt="Regression coefficients">
</div>
{% endif %}

{% if spatial_str %}
<h3>Spatial Autocorrelation</h3>
<p>{{ spatial_str }}</p>
{% endif %}

{% if spatial_reg_table %}
<h3>Spatial Regression Model Comparison</h3>
<p>Dependent variable: <strong>{{ spatial_reg_dv }}</strong>. Moran's I on OLS
   residuals was significant, so spatial lag and spatial error models were fit
   for comparison.</p>
<table>
  <tr>
    <th>Model</th><th>R2 / Pseudo-R2</th><th>AIC</th><th>Spatial Coeff.</th><th>Log-Likelihood</th>
  </tr>
  {% for row in spatial_reg_table %}
  <tr{% if row.best %} style="background-color: #E8F5E9; font-weight: 600;"{% endif %}>
    <td>{{ row.model }}{% if row.best %} *{% endif %}</td>
    <td>{{ row.r2 }}</td>
    <td>{{ row.aic }}</td>
    <td>{{ row.spatial_coeff }}</td>
    <td>{{ row.logll }}</td>
  </tr>
  {% endfor %}
</table>
<p><em>* Best model by AIC (lower is better).</em></p>

{% if lm_diagnostics_str %}
<p><strong>LM Diagnostics:</strong> {{ lm_diagnostics_str }}</p>
{% endif %}

{% if figures.spatial_model_comparison %}
<div class="figure">
  <img src="data:image/png;base64,{{ figures.spatial_model_comparison }}" alt="Spatial model comparison">
</div>
{% endif %}
{% endif %}

{% if figures.subgroup_correlations %}
<h3>Subgroup Analysis</h3>
<div class="figure">
  <img src="data:image/png;base64,{{ figures.subgroup_correlations }}" alt="Subgroup correlations">
</div>
{% endif %}

<h2>Caveats and Limitations</h2>
<div class="warning">
<ul>
  <li>Sample size n={{ n_units }} limits statistical power for clustering stability,
      regression, and subgroup analysis.</li>
  <li>Cluster solutions with k>4 yield groups smaller than 15 units,
      which are too small for reliable regression.</li>
  <li>Demographics are static (single census snapshot) -- cannot capture
      temporal demographic change.</li>
  <li>Remote sensing trends are based on ~130 monthly observations per unit,
      but cloud cover introduces ~25-35% missing data.</li>
  <li>Ecological inference caveat: unit-level relationships do not
      necessarily hold at the individual level.</li>
</ul>
</div>

<div class="footer">
  <p>Generated by the typology pipeline. Shrinking Villages -- Aomori & Akita Prefectures.</p>
</div>

</body>
</html>
"""


def generate_report(summary: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """
    Generate HTML report and JSON summary.

    Args:
        summary: Collected results from all pipeline steps.
        cfg: Configuration dict.
    """
    if not cfg["report"].get("generate_html", True):
        print("HTML report generation disabled.")
        _save_json_summary(summary, cfg)
        return

    print("Generating HTML report...")
    figures_dir = Path(cfg["output"]["figures_dir"])

    # Embed figures as base64 (prefer PNG for HTML img tags)
    figure_map: Dict[str, str] = {}
    figure_files = {
        "indicator_distributions": "indicator_distributions",
        "indicator_correlation_heatmap": "indicator_correlation_heatmap",
        "pca_scree": "pca_scree",
        "pca_biplot": "pca_biplot",
        "optimal_k_metrics": "optimal_k_metrics",
        "dendrogram": "dendrogram",
        "multi_k_comparison": "multi_k_comparison",
        "cluster_profiles_heatmap": "cluster_profiles_heatmap",
        "cluster_parallel_coords": "cluster_parallel_coords",
        "cluster_boxplots": "cluster_boxplots",
        "cluster_crosstab_heatmap": "cluster_crosstab_heatmap",
        "cluster_map": "cluster_map",
        "correlation_heatmap_pearson": "correlation_heatmap_pearson",
        "correlation_heatmap_spearman": "correlation_heatmap_spearman",
        "regression_diagnostics": "regression_diagnostics",
        "regression_coefficients": "regression_coefficients",
        "subgroup_correlations": "subgroup_correlations",
        "specification_robustness": "specification_robustness",
        "spatial_model_comparison": "spatial_model_comparison",
    }

    for key, basename in figure_files.items():
        # Prefer PNG for HTML embedding
        for ext in ("png", "pdf"):
            fpath = figures_dir / f"{basename}.{ext}"
            if fpath.exists():
                if ext == "png":
                    figure_map[key] = _encode_image(fpath)
                break

    # Build template variables
    indicators = summary.get("indicators", {})
    clustering = summary.get("clustering", {})
    relationships = summary.get("relationships", {})

    # Quality notes
    quality = indicators.get("quality", {})
    quality_notes = []
    if quality.get("n_missing_total", 0) > 0:
        quality_notes.append(
            f"Missing values (imputed): {quality['n_missing_total']}"
        )
    if quality.get("near_constant_features"):
        quality_notes.append(
            f"Near-constant features: {', '.join(quality['near_constant_features'])}"
        )
    if quality.get("n_outlier_flags", 0) > 0:
        quality_notes.append(
            f"Outlier flags (>3 SD): {quality['n_outlier_flags']}"
        )

    # PCA thresholds
    pca_thresh = clustering.get("pca_thresholds", {})
    pca_str = ", ".join(f"{k}: {v} PCs" for k, v in pca_thresh.items())

    # Multi-k table
    multi_k = clustering.get("multi_k", {})
    primary_k = clustering.get("primary_k", clustering.get("optimal_k", "N/A"))
    data_driven_k = clustering.get("data_driven_k", primary_k)
    report_k_values = clustering.get("report_k_values", [primary_k])

    multi_k_table = []
    for k_val in sorted(multi_k.keys()):
        info = multi_k[k_val]
        ci = info.get("bootstrap_ci", ("N/A", "N/A"))
        sizes = info.get("cluster_sizes", {})
        sizes_str = ", ".join(f"{v}" for v in sorted(sizes.values(), reverse=True))
        multi_k_table.append({
            "k": k_val,
            "silhouette": f"{info.get('silhouette', 0):.3f}",
            "boot_ari": f"{info.get('bootstrap_ari', 0):.3f}",
            "boot_ci": f"[{ci[0]:.3f}, {ci[1]:.3f}]" if isinstance(ci, (list, tuple)) else str(ci),
            "sup_ari": f"{info.get('supervised_ari', 0):.3f}",
            "hier_ari": f"{info.get('hier_ari', 0):.3f}",
            "sizes": sizes_str,
            "primary": (k_val == primary_k),
        })

    # Silhouette diagram figures (per-k)
    silhouette_figures = {}
    for k_val in sorted(multi_k.keys()):
        sil_fname = f"silhouette_diagram_k{k_val}"
        for ext in ("png",):
            fpath = figures_dir / f"{sil_fname}.{ext}"
            if fpath.exists():
                silhouette_figures[k_val] = _encode_image(fpath)

    # Bootstrap
    bootstrap = clustering.get("bootstrap", {})
    if bootstrap:
        boot_str = (f"Mean ARI = {bootstrap.get('mean_ari', 'N/A')} "
                    f"(95% CI: {bootstrap.get('ci_95', 'N/A')})")
    else:
        boot_str = ""

    # Regression
    reg = relationships.get("regression", {})
    if reg.get("r_squared") is not None:
        reg_str = (f"R2 = {reg['r_squared']:.3f}, "
                   f"Adj R2 = {reg.get('adj_r_squared', 'N/A')}, "
                   f"F = {reg.get('f_stat', 'N/A')}, "
                   f"p = {reg.get('f_pvalue', 'N/A')}")
    else:
        reg_str = ""

    # Spatial
    spatial = relationships.get("spatial", {})
    if spatial.get("morans_i") is not None:
        spat_str = (f"Moran's I = {spatial['morans_i']:.4f}, "
                    f"p = {spatial.get('morans_p', 'N/A')}")
        if spatial.get("significant"):
            spat_str += " (significant spatial autocorrelation)"
    else:
        spat_str = spatial.get("reason", "Spatial analysis not performed")

    # Specification robustness table
    spec_rob = clustering.get("specification_robustness")
    spec_robustness_table = []
    if spec_rob is not None:
        if isinstance(spec_rob, pd.DataFrame):
            for _, row in spec_rob.iterrows():
                spec_robustness_table.append({
                    "specification": row.get("specification", ""),
                    "n_features": int(row.get("n_features", 0)),
                    "ari": f"{row.get('ari_vs_primary', 0):.3f}",
                    "nmi": f"{row.get('nmi_vs_primary', 0):.3f}",
                    "silhouette": f"{row.get('silhouette', 0):.3f}",
                })
        elif isinstance(spec_rob, list):
            for row in spec_rob:
                spec_robustness_table.append({
                    "specification": row.get("specification", ""),
                    "n_features": int(row.get("n_features", 0)),
                    "ari": f"{row.get('ari_vs_primary', 0):.3f}",
                    "nmi": f"{row.get('nmi_vs_primary', 0):.3f}",
                    "silhouette": f"{row.get('silhouette', 0):.3f}",
                })

    # Spatial regression table
    spatial_regression = spatial.get("spatial_regression", {}) or {}
    spatial_reg_table = []
    spatial_reg_dv = spatial_regression.get("dependent_var", "")
    lm_diagnostics_str = ""

    if not spatial_regression.get("skipped", True):
        best_model = spatial_regression.get("best_model", "")
        ols_info = spatial_regression.get("ols", {})
        lag_info = spatial_regression.get("spatial_lag", {})
        err_info = spatial_regression.get("spatial_error", {})

        if "aic" in ols_info:
            spatial_reg_table.append({
                "model": "OLS",
                "r2": f"{ols_info.get('r_squared', 0):.3f}",
                "aic": f"{ols_info['aic']:.1f}",
                "spatial_coeff": "--",
                "logll": f"{ols_info.get('log_likelihood', 0):.2f}",
                "best": best_model == "OLS",
            })
        if "aic" in lag_info:
            spatial_reg_table.append({
                "model": "Spatial Lag (ML)",
                "r2": f"{lag_info.get('pseudo_r_squared', 0):.3f}",
                "aic": f"{lag_info['aic']:.1f}",
                "spatial_coeff": f"rho = {lag_info.get('rho', 0):.3f}",
                "logll": f"{lag_info.get('log_likelihood', 0):.2f}",
                "best": best_model == "Spatial Lag",
            })
        if "aic" in err_info:
            spatial_reg_table.append({
                "model": "Spatial Error (ML)",
                "r2": f"{err_info.get('pseudo_r_squared', 0):.3f}",
                "aic": f"{err_info['aic']:.1f}",
                "spatial_coeff": f"lambda = {err_info.get('lambda', 0):.3f}",
                "logll": f"{err_info.get('log_likelihood', 0):.2f}",
                "best": best_model == "Spatial Error",
            })

        # LM diagnostics
        lm_parts = []
        lm_lag_info = spatial_regression.get("lm_lag", {})
        lm_err_info = spatial_regression.get("lm_error", {})
        if lm_lag_info:
            lm_parts.append(
                f"LM-Lag: stat={lm_lag_info.get('statistic', 0):.3f}, "
                f"p={lm_lag_info.get('p_value', 1):.4f}"
            )
        if lm_err_info:
            lm_parts.append(
                f"LM-Error: stat={lm_err_info.get('statistic', 0):.3f}, "
                f"p={lm_err_info.get('p_value', 1):.4f}"
            )
        lm_diagnostics_str = " | ".join(lm_parts) if lm_parts else ""

    # Report k values as string
    report_k_str = ", ".join(str(k) for k in sorted(report_k_values))

    # Render
    template = Template(REPORT_TEMPLATE)
    html = template.render(
        title=cfg["report"]["title"],
        author=cfg["report"]["author"],
        n_units=summary.get("n_units", 65),
        n_indicators=indicators.get("n_indicators", 0),
        n_physical=indicators.get("n_physical", 0),
        n_demographic=indicators.get("n_demographic", 0),
        class_labels=cfg["columns"]["class_labels"],
        quality_notes=quality_notes,
        pca_thresholds_str=pca_str if pca_str else "N/A",
        primary_k=primary_k,
        data_driven_k=data_driven_k,
        report_k_values_str=report_k_str,
        optimal_k=clustering.get("optimal_k", "N/A"),
        silhouette_score=f"{clustering.get('silhouette_mean', 0):.3f}",
        kmeans_hier_ari=f"{clustering.get('kmeans_hier_ari', 0):.3f}",
        multi_k_table=multi_k_table,
        silhouette_figures=silhouette_figures,
        bootstrap_str=boot_str,
        regression_str=reg_str,
        spatial_str=spat_str,
        spec_robustness_table=spec_robustness_table,
        spatial_reg_table=spatial_reg_table,
        spatial_reg_dv=spatial_reg_dv,
        lm_diagnostics_str=lm_diagnostics_str,
        figures=figure_map,
    )

    report_path = Path(cfg["output"]["reports_dir"]) / "typology_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"  Saved: {report_path}")

    # JSON summary
    _save_json_summary(summary, cfg)


def _save_json_summary(summary: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Save machine-readable JSON summary."""
    json_path = Path(cfg["output"]["reports_dir"]) / "typology_summary.json"
    clean = _make_serialisable(summary)
    write_json(str(json_path), clean)
    print(f"  Saved: {json_path}")


def _encode_image(path: Path) -> str:
    """Encode an image file as base64 string."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to native Python."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
