"""
Report generation for the EDA module.

Compiles analysis results into an HTML report with embedded figures,
a machine-readable JSON summary, and a data quality flags CSV.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from jinja2 import Template

from eda.src.utils import write_json


# ---------------------------------------------------------------------------
# HTML template (Jinja2)
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
  body {
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1200px; margin: 0 auto; padding: 20px;
    color: #333; background: #fafafa; line-height: 1.6;
  }
  h1 { color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }
  h2 { color: #283593; margin-top: 40px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
  h3 { color: #3949ab; }
  .meta { color: #666; font-size: 0.9em; margin-bottom: 30px; }
  .section { margin-bottom: 40px; }
  .figure { text-align: center; margin: 20px 0; }
  .figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
  .figure .caption { font-size: 0.85em; color: #666; margin-top: 4px; }
  table {
    border-collapse: collapse; width: 100%; margin: 15px 0;
    font-size: 0.9em;
  }
  th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
  th { background: #e8eaf6; font-weight: 600; }
  tr:nth-child(even) { background: #f5f5f5; }
  .flag { color: #c62828; font-weight: bold; }
  .ok { color: #2e7d32; }
  .warn { color: #ef6c00; }
  .note { background: #fff3e0; border-left: 4px solid #ff9800; padding: 10px 15px; margin: 15px 0; }
  .key-finding { background: #e8f5e9; border-left: 4px solid #4caf50; padding: 10px 15px; margin: 10px 0; }
  code { background: #f5f5f5; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
  ul { padding-left: 20px; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<div class="meta">
  <p>Author: {{ author }} | Generated: {{ timestamp }}</p>
  <p>Data: <code>{{ data_path }}</code></p>
</div>

<!-- 1. Data Overview -->
<div class="section">
<h2>1. Data Overview</h2>
{% if summary_stats %}
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Rows</td><td>{{ summary_stats.row_count }}</td></tr>
  <tr><td>Columns</td><td>{{ summary_stats.column_count }}</td></tr>
  <tr><td>Numeric features</td><td>{{ summary_stats.numeric_feature_count }}</td></tr>
  <tr><td>Units</td><td>{{ summary_stats.unit_count }}</td></tr>
  <tr><td>Prefectures</td><td>{{ summary_stats.prefectures | join(', ') }}</td></tr>
  {% if summary_stats.month_range %}
  <tr><td>Time range</td><td>{{ summary_stats.month_range.min }} to {{ summary_stats.month_range.max }}</td></tr>
  {% endif %}
</table>
{% endif %}
{% if schema_warnings %}
<div class="note">
  <strong>Schema warnings:</strong>
  <ul>{% for w in schema_warnings %}<li>{{ w }}</li>{% endfor %}</ul>
</div>
{% endif %}
</div>

<!-- 2. Missing Data -->
<div class="section">
<h2>2. Missing Data</h2>
{% if missing_data %}
<p>Overall missing rate: <strong>{{ missing_data.overall_missing_pct }}%</strong>
   (threshold: {{ missing_data.threshold_pct }}%)</p>
{% if missing_data.columns_above_threshold %}
<p class="warn">Features above threshold:
   {{ missing_data.columns_above_threshold | join(', ') }}</p>
{% else %}
<p class="ok">All features below missing threshold.</p>
{% endif %}
<div class="note">
  <strong>Note:</strong> {{ missing_data.glcm_note }}
</div>
{% endif %}
{{ figures.missing_heatmap | safe }}
{{ figures.missing_by_time | safe }}
</div>

<!-- 3. Distributions -->
<div class="section">
<h2>3. Feature Distributions</h2>
{% if distributions %}
<p>{{ distributions.n_analysed }} features analysed.
   {{ distributions.non_normal_features | length }} non-normal (Shapiro-Wilk p&lt;0.05).</p>
{% if distributions.features_needing_transform %}
<p class="warn">Features needing transform: {{ distributions.features_needing_transform | join(', ') }}</p>
{% endif %}
{% endif %}
{{ figures.distributions_rs | safe }}
{{ figures.distributions_demo | safe }}
{{ figures.boxplots_by_prefecture | safe }}
</div>

<!-- 4. Correlations -->
<div class="section">
<h2>4. Correlation Analysis</h2>
{% if correlations %}
<p>{{ correlations.n_features }} features analysed.
   {{ correlations.highly_correlated_pairs | length }} pairs above |r|&gt;{{ correlations.correlation_threshold }}.</p>
{% if correlations.highly_correlated_pairs %}
<h3>Highly Correlated Pairs</h3>
<table>
  <tr><th>Feature 1</th><th>Feature 2</th><th>r</th></tr>
  {% for p in correlations.highly_correlated_pairs %}
  <tr><td>{{ p.feature_1 }}</td><td>{{ p.feature_2 }}</td><td>{{ p.correlation }}</td></tr>
  {% endfor %}
</table>
{% endif %}
{% if correlations.top_rs_demo_associations %}
<h3>Top RS-Demographic Associations</h3>
<table>
  <tr><th>RS Feature</th><th>Demographic Feature</th><th>r</th></tr>
  {% for a in correlations.top_rs_demo_associations[:10] %}
  <tr><td>{{ a.rs_feature }}</td><td>{{ a.demo_feature }}</td><td>{{ a.correlation }}</td></tr>
  {% endfor %}
</table>
{% endif %}
{% endif %}
{{ figures.correlation_heatmap | safe }}
{{ figures.rs_demo_correlation | safe }}
</div>

<!-- 5. Temporal Patterns -->
<div class="section">
<h2>5. Temporal Patterns</h2>
{% if temporal and not temporal.get('skipped') %}
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Mean months per unit</td><td>{{ temporal.coverage_summary.mean_months_per_unit }}</td></tr>
  <tr><td>Min months</td><td>{{ temporal.coverage_summary.min_months }}</td></tr>
  <tr><td>Units with gaps</td><td>{{ temporal.coverage_summary.units_with_gaps }}</td></tr>
</table>
{% if temporal.seasonal_amplitude %}
<h3>Seasonal Amplitude</h3>
<table>
  <tr><th>Feature</th><th>Amplitude (max-min monthly mean)</th></tr>
  {% for feat, amp in temporal.seasonal_amplitude.items() %}
  <tr><td>{{ feat }}</td><td>{{ amp }}</td></tr>
  {% endfor %}
</table>
{% endif %}
{% endif %}
{{ figures.temporal_trends | safe }}
{{ figures.seasonal_patterns | safe }}
</div>

<!-- 6. Spatial Patterns -->
<div class="section">
<h2>6. Spatial Patterns</h2>
{% if spatial and not spatial.get('skipped') %}
<p>{{ spatial.n_units_mapped }} units mapped.</p>
{% if spatial.by_prefecture %}
<table>
  <tr><th>Prefecture</th><th>Units</th><th>Mean NDVI</th><th>Mean VIIRS</th><th>Mean Pop</th></tr>
  {% for pref, stats in spatial.by_prefecture.items() %}
  <tr>
    <td>{{ pref }}</td><td>{{ stats.n_units }}</td>
    <td>{{ stats.mean_ndvi }}</td><td>{{ stats.mean_viirs }}</td><td>{{ stats.mean_pop }}</td>
  </tr>
  {% endfor %}
</table>
{% endif %}
{% else %}
<p class="warn">Spatial analysis skipped: {{ spatial.get('reason', 'unknown') }}</p>
{% endif %}
{{ figures.spatial_ndvi_mean | safe }}
{{ figures.spatial_pop_total | safe }}
{{ figures.spatial_viirs_mean | safe }}
</div>

<!-- 7. Outliers -->
<div class="section">
<h2>7. Outlier Detection</h2>
{% if outliers %}
<p>Method: {{ outliers.method }}. Total outlier observations: {{ outliers.total_outlier_observations }}
   across {{ outliers.features_with_outliers }} features.</p>
{% if outliers.most_affected_features %}
<h3>Most Affected Features</h3>
<table>
  <tr><th>Feature</th><th>Outlier %</th></tr>
  {% for f in outliers.most_affected_features %}
  <tr><td>{{ f.feature }}</td><td>{{ f.outlier_pct }}%</td></tr>
  {% endfor %}
</table>
{% endif %}
{% if outliers.most_affected_units %}
<h3>Most Affected Units</h3>
<table>
  <tr><th>Unit ID</th><th>Outlier Count</th></tr>
  {% for u in outliers.most_affected_units %}
  <tr><td>{{ u.unit_id }}</td><td>{{ u.outlier_count }}</td></tr>
  {% endfor %}
</table>
{% endif %}
{% endif %}
{{ figures.outlier_boxplots | safe }}
</div>

<!-- 8. Feature Relationships -->
<div class="section">
<h2>8. RS-Demographic Relationships</h2>
{% if feature_relationships and feature_relationships.key_relationships %}
<table>
  <tr><th>X</th><th>Y</th><th>Pearson r</th></tr>
  {% for rel in feature_relationships.key_relationships %}
  <tr><td>{{ rel.x }}</td><td>{{ rel.y }}</td><td>{{ rel.r }}</td></tr>
  {% endfor %}
</table>
{% endif %}
{{ figures.ndvi_vs_population | safe }}
{{ figures.viirs_vs_population | safe }}
{{ figures.ndbi_vs_population | safe }}
{{ figures.viirs_vs_elderly_ratio | safe }}
</div>

<!-- 9. Data Quality Summary -->
<div class="section">
<h2>9. Data Quality Summary</h2>
<h3>Validation Checklist</h3>
<table>
  <tr><th>Check</th><th>Status</th></tr>
  {% for check in quality_checks %}
  <tr>
    <td>{{ check.description }}</td>
    <td class="{{ check.status }}">{{ check.result }}</td>
  </tr>
  {% endfor %}
</table>
</div>

<!-- 10. Key Findings -->
<div class="section">
<h2>10. Key Findings for Classification</h2>
{% for finding in key_findings %}
<div class="key-finding">
  <strong>{{ finding.finding }}</strong><br>
  Implication: {{ finding.implication }}
</div>
{% endfor %}
</div>

</body>
</html>
""")


def generate_report(
    summary: Dict[str, Any], cfg: Dict[str, Any], output_dir: str,
) -> None:
    """
    Compile all analysis results into HTML report, JSON summary, and quality flags.

    Args:
        summary: Dict of dicts returned by each analysis module.
        cfg: EDA configuration dict.
        output_dir: Base output directory.
    """
    print("Generating report...")
    reports_dir = Path(cfg["output"]["reports_dir"])
    figures_dir = Path(cfg["output"]["figures_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # --- Embed figures as base64 ---
    figure_html = _embed_all_figures(figures_dir)

    # --- Build quality checks ---
    quality_checks = _build_quality_checks(summary, cfg)

    # --- Build key findings ---
    key_findings = _build_key_findings(summary, cfg)

    # --- Render HTML ---
    from datetime import datetime, timezone
    html = _HTML_TEMPLATE.render(
        title=cfg["report"]["title"],
        author=cfg["report"].get("author", ""),
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        data_path=cfg["data"]["features_table"],
        summary_stats=summary.get("summary_stats"),
        schema_warnings=summary.get("schema_warnings", []),
        missing_data=summary.get("missing_data"),
        distributions=summary.get("distributions"),
        correlations=summary.get("correlations"),
        temporal=summary.get("temporal"),
        spatial=summary.get("spatial"),
        outliers=summary.get("outliers"),
        feature_relationships=summary.get("feature_relationships"),
        figures=figure_html,
        quality_checks=quality_checks,
        key_findings=key_findings,
    )

    html_path = reports_dir / "eda_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  Saved HTML report: {html_path}")

    # --- Optional PDF ---
    if cfg["report"].get("generate_pdf", False):
        _generate_pdf(html, reports_dir / "eda_report.pdf")

    # --- JSON summary ---
    json_path = str(reports_dir / "eda_summary.json")
    write_json(json_path, summary)
    print(f"  Saved JSON summary: {json_path}")

    # --- Data quality flags CSV ---
    flags = _build_quality_flags(summary, cfg)
    if flags:
        flags_df = pd.DataFrame(flags)
        flags_path = reports_dir / "data_quality_flags.csv"
        flags_df.to_csv(flags_path, index=False)
        print(f"  Saved quality flags: {flags_path} ({len(flags)} flags)")


def _embed_all_figures(figures_dir: Path) -> Dict[str, str]:
    """Read all PNG files in figures_dir and return as base64 HTML img tags."""
    result = {}
    for p in sorted(figures_dir.glob("*.png")):
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        name = p.stem
        result[name] = (
            f'<div class="figure">'
            f'<img src="data:image/png;base64,{b64}" alt="{name}">'
            f'<div class="caption">{name.replace("_", " ").title()}</div>'
            f'</div>'
        )
    return result


def _build_quality_checks(
    summary: Dict[str, Any], cfg: Dict[str, Any],
) -> list[Dict[str, str]]:
    """Build validation checklist rows."""
    checks = []

    # Schema
    warnings = summary.get("schema_warnings", [])
    checks.append({
        "description": "All expected columns present",
        "status": "ok" if not warnings else "warn",
        "result": "Yes" if not warnings else f"Warnings: {len(warnings)}",
    })

    # Missing data
    md = summary.get("missing_data", {})
    cols_above = md.get("columns_above_threshold", [])
    threshold = md.get("threshold_pct", 20)
    checks.append({
        "description": f"Missing data rates acceptable (<{threshold}%)",
        "status": "ok" if not cols_above else "warn",
        "result": "Yes" if not cols_above else f"Flagged: {', '.join(cols_above)}",
    })

    # GLCM
    checks.append({
        "description": "GLCM texture features available",
        "status": "ok",
        "result": md.get("glcm_note", "N/A"),
    })

    # Temporal coverage
    tc = summary.get("temporal", {})
    if not tc.get("skipped"):
        cs = tc.get("coverage_summary", {})
        checks.append({
            "description": "Temporal coverage sufficient per unit",
            "status": "ok" if cs.get("min_months", 0) >= 24 else "warn",
            "result": f"Min {cs.get('min_months', '?')} months, "
                      f"{cs.get('units_with_gaps', '?')} units with gaps",
        })

    # Outliers
    ol = summary.get("outliers", {})
    checks.append({
        "description": "Outliers identified and documented",
        "status": "ok" if ol else "warn",
        "result": f"{ol.get('total_outlier_observations', 0)} outlier observations"
                  if ol else "Not run",
    })

    # Correlations
    corr = summary.get("correlations", {})
    n_high = len(corr.get("highly_correlated_pairs", []))
    checks.append({
        "description": "High correlations documented (multicollinearity risk)",
        "status": "ok" if n_high == 0 else "warn",
        "result": f"{n_high} pairs above threshold" if corr else "Not run",
    })

    # Distributions
    dist = summary.get("distributions", {})
    n_transform = len(dist.get("features_needing_transform", []))
    checks.append({
        "description": "Feature distributions understood",
        "status": "ok",
        "result": f"{n_transform} features may need transformation",
    })

    return checks


def _build_key_findings(
    summary: Dict[str, Any], cfg: Dict[str, Any],
) -> list[Dict[str, str]]:
    """Synthesise key findings for classification from analysis results."""
    findings = []

    # Missing data
    md = summary.get("missing_data", {})
    overall = md.get("overall_missing_pct", 0)
    findings.append({
        "finding": f"Missing data: {overall}% overall",
        "implication": "Spectral and texture NaN from cloud cover — consider seasonal "
                       "composites or monthly imputation. GLCM texture features now "
                       "available for all units.",
    })

    # Distributions
    dist = summary.get("distributions", {})
    transforms = dist.get("features_needing_transform", [])
    if transforms:
        findings.append({
            "finding": f"Skewed features: {', '.join(transforms[:5])}",
            "implication": "Apply log1p or sqrt transform before distance-based models. "
                           "Robust scaling recommended for tree-based models.",
        })

    # Correlations
    corr = summary.get("correlations", {})
    high = corr.get("highly_correlated_pairs", [])
    if high:
        pairs_str = "; ".join(f"{p['feature_1']}/{p['feature_2']}" for p in high[:3])
        findings.append({
            "finding": f"Highly correlated features: {pairs_str}",
            "implication": "Consider dropping one feature per pair or using PCA "
                           "to reduce multicollinearity.",
        })

    # Temporal
    tc = summary.get("temporal", {})
    if not tc.get("skipped"):
        amp = tc.get("seasonal_amplitude", {})
        if amp:
            findings.append({
                "finding": f"Seasonal amplitude: {amp}",
                "implication": "Strong seasonality in NDVI suggests using summer composites "
                               "or including month/season as a feature.",
            })

    # RS-demographic relationships
    fr = summary.get("feature_relationships", {})
    rels = fr.get("key_relationships", [])
    if rels:
        top = rels[0]
        findings.append({
            "finding": f"Strongest RS-demographic link: {top['x']} vs {top['y']} (r={top['r']})",
            "implication": "Prioritise this feature pair for classification model.",
        })

    return findings


def _build_quality_flags(
    summary: Dict[str, Any], cfg: Dict[str, Any],
) -> list[Dict[str, str]]:
    """Build data quality flag rows for CSV export."""
    flags = []

    # Missing data flags
    md = summary.get("missing_data", {})
    for col in md.get("columns_above_threshold", []):
        flags.append({
            "type": "high_missing",
            "feature": col,
            "unit_id": "",
            "month": "",
            "detail": f"Missing rate above {md.get('threshold_pct', 20)}%",
        })

    # GLCM note (now available for all units)
    glcm_note = md.get("glcm_note", "")
    if glcm_note:
        flags.append({
            "type": "glcm_available",
            "feature": "S2_NDBI_contrast/entropy/homogeneity",
            "unit_id": "",
            "month": "",
            "detail": glcm_note,
        })

    # Outlier flags (top affected features)
    ol = summary.get("outliers", {})
    for f in ol.get("most_affected_features", []):
        if f["outlier_pct"] > 5:
            flags.append({
                "type": "high_outlier_rate",
                "feature": f["feature"],
                "unit_id": "",
                "month": "",
                "detail": f"{f['outlier_pct']}% outlier rate",
            })

    # Temporal coverage flags
    tc = summary.get("temporal", {})
    cs = tc.get("coverage_summary", {})
    if cs.get("units_with_gaps", 0) > 0:
        flags.append({
            "type": "temporal_gaps",
            "feature": "",
            "unit_id": "",
            "month": "",
            "detail": f"{cs['units_with_gaps']} units have temporal gaps",
        })

    return flags


def _generate_pdf(html: str, output_path: Path) -> None:
    """Generate PDF from HTML using weasyprint (optional dependency)."""
    try:
        from weasyprint import HTML as WeasyprintHTML
        WeasyprintHTML(string=html).write_pdf(str(output_path))
        print(f"  Saved PDF report: {output_path}")
    except ImportError:
        print("  WARNING: weasyprint not installed, skipping PDF generation.")
    except Exception as e:
        print(f"  WARNING: PDF generation failed: {e}")
