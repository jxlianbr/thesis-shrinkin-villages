"""Generate publication figures for the thesis from pipeline output artifacts.

Figures produced (thesis label in parentheses):
- mechanism_importance   mechanism-cluster SHAP bar chart (fig:results_mechanisms)
- decoupling_maps        physical typology vs demographic class map pair, aza scale
                         (section 5.1, ARI = 0.14 finding)

Outputs PNG (300 dpi) and PDF (vector, preferred for LaTeX) to
phase_b/outputs/figures/.

Usage:
    python phase_b/make_figures.py [--only NAME] [--variant SUFFIX]

--only renders a single figure by name (default: all).
--variant selects an alternative run artifact for the mechanism chart, e.g.
"--variant dw_bare_robust" reads mechanism_importance_dw_bare_robust.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PHASE_B_DIR = Path(__file__).resolve().parent
REPO_DIR = PHASE_B_DIR.parent

# Display names matching Table tab:results_mechanisms (thesis section 5.3).
CLUSTER_LABELS: Dict[str, str] = {
    "accessibility_medical": "Accessibility, medical",
    "accessibility_transit": "Accessibility, transit",
    "demographic_threshold": "Demographic threshold",
    "accessibility_did": "Accessibility, DID proximity",
    "accessibility_community": "Accessibility, community facilities",
    "accessibility_education": "Accessibility, education",
    "accessibility_hospital": "Accessibility, hospital",
    "institutional_merger": "Institutional, merger history",
    "institutional_fiscal": "Institutional, fiscal",
    "durability_housing": "Durability, housing stock",
    "institutional_policy": "Institutional, designation",
}

# One color per mechanism family (Okabe-Ito subset, CVD-validated). Identity is
# never color-alone: every bar carries its cluster name on the y axis.
FAMILY_COLORS: Dict[str, str] = {
    "accessibility": "#0072B2",
    "demographic": "#E69F00",
    "institutional": "#009E73",
    "durability": "#CC79A7",
}
FAMILY_LEGEND: Dict[str, str] = {
    "accessibility": "Accessibility",
    "demographic": "Demographic",
    "institutional": "Institutional",
    "durability": "Durability",
}

INK = "#333333"
INK_MUTED = "#666666"
GRID = "#dddddd"

# ColorBrewer 3-class sequential ramps (both panel scales are ordered), one
# hue per dimension so the two maps cannot be misread as one classification.
TYPE_COLORS: List[str] = ["#c6dbef", "#6baed6", "#2171b5"]      # Blues
CLASS_COLORS: List[str] = ["#fdd0a2", "#fd8d3c", "#d94801"]     # Oranges

# Units without census small-area data: white with a gray hatch, matching the
# typology cluster map style.
NO_DATA_STYLE: Dict[str, Any] = {
    "facecolor": "white",
    "edgecolor": "#999999",
    "hatch": "////",
}


def _save(fig: plt.Figure, name: str, tight: bool = False) -> Path:
    out_dir = PHASE_B_DIR / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs: Dict[str, Any] = {"bbox_inches": "tight"} if tight else {}
    png_path = out_dir / f"{name}.png"
    fig.savefig(png_path, dpi=300, **kwargs)
    # dpi also sets the resampling resolution of embedded rasters (basemap
    # tiles) in the otherwise-vector PDF; default would be figure dpi (100)
    fig.savefig(out_dir / f"{name}.pdf", dpi=600, **kwargs)
    plt.close(fig)
    print(f"[make_figures] wrote {png_path} (+ .pdf)")
    return png_path


def _plot_map_panel(
    ax: plt.Axes,
    gdf: Any,
    column: str,
    categories: List[Any],
    colors: List[str],
    labels: List[str],
    title: str,
    edge_lw: float,
) -> None:
    """One choropleth panel with explicit category colors and hatched no-data."""
    from matplotlib.patches import Patch

    for cat, color in zip(categories, colors):
        gdf[gdf[column] == cat].plot(
            ax=ax, facecolor=color, edgecolor="#666666", linewidth=edge_lw,
        )
    gdf[gdf[column].isna()].plot(ax=ax, linewidth=edge_lw, **NO_DATA_STYLE)

    handles = [Patch(facecolor=c, edgecolor="#666666", linewidth=0.5)
               for c in colors]
    handles.append(Patch(linewidth=0.5, **NO_DATA_STYLE))
    ax.legend(handles, labels + ["No data"], loc="lower left",
              frameon=False, fontsize="small", handlelength=1.4,
              handleheight=1.1)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_axis_off()


def make_decoupling_map_figure() -> Path:
    """Map pair: physical typology vs demographic shrinkage class (aza scale).

    Visual counterpart to the ARI = 0.14 result of thesis section 5.1. Type
    numbering (1 = physically strongest) is derived from the cluster profiles
    by mean built-up texture, so it survives cluster-id relabeling on rerun.
    """
    import geopandas as gpd

    tables_dir = REPO_DIR / "typology" / "outputs_aza" / "tables"
    assign = pd.read_csv(tables_dir / "unit_cluster_assignments.csv")
    profiles = pd.read_csv(tables_dir / "cluster_profiles.csv")
    type_order = profiles.sort_values(
        "S2_NDBI_contrast_mean_mean", ascending=False
    )["cluster"].tolist()

    gdf = gpd.read_file(
        REPO_DIR / "admin_demographics" / "boundaries" / "aza.gpkg"
    )
    gdf = gdf.merge(
        assign[["unit_id", "cluster", "shrinkage_class"]],
        on="unit_id", how="left",
    )

    plt.rcParams.update({"font.size": 8, "hatch.linewidth": 0.4})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 5.4), constrained_layout=True)

    _plot_map_panel(
        axes[0], gdf, "cluster",
        categories=type_order,
        colors=TYPE_COLORS,
        labels=["Type 1 (physically strongest)", "Type 2 (intermediate)",
                "Type 3 (physically weakest)"],
        title="Physical typology",
        edge_lw=0.03,
    )
    _plot_map_panel(
        axes[1], gdf, "shrinkage_class",
        categories=["stable", "shrinking", "severely_shrinking"],
        colors=CLASS_COLORS,
        labels=["Stable", "Shrinking", "Severely shrinking"],
        title="Demographic shrinkage class",
        edge_lw=0.03,
    )
    return _save(fig, "decoupling_maps")


def _family_of(mechanism: str) -> str:
    return mechanism.split("_")[0]


def make_mechanism_importance_figure(variant: str = "") -> Path:
    """Render the mechanism-cluster SHAP importance bar chart."""
    suffix = f"_{variant}" if variant else ""
    csv_path = PHASE_B_DIR / "outputs" / f"mechanism_importance{suffix}.csv"
    out_dir = PHASE_B_DIR / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).sort_values("mean_abs_shap", ascending=True)
    labels = [CLUSTER_LABELS.get(m, m) for m in df["mechanism"]]
    colors = [FAMILY_COLORS[_family_of(m)] for m in df["mechanism"]]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelcolor": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK,
            "axes.edgecolor": INK_MUTED,
        }
    )

    fig, ax = plt.subplots(figsize=(6.3, 3.6), constrained_layout=True)
    ax.barh(labels, df["mean_abs_shap"], height=0.62, color=colors, zorder=3)

    # Value label at each bar end (exact values also live in the thesis table).
    x_pad = df["mean_abs_shap"].max() * 0.012
    for y, value in enumerate(df["mean_abs_shap"]):
        ax.text(value + x_pad, y, f"{value:.3f}", va="center", ha="left",
                fontsize="small", color=INK_MUTED, zorder=3)

    ax.set_xlabel("Mean absolute SHAP attribution")
    ax.set_xlim(0, df["mean_abs_shap"].max() * 1.12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[fam])
        for fam in FAMILY_LEGEND
    ]
    ax.legend(handles, list(FAMILY_LEGEND.values()), loc="lower right",
              frameon=False, fontsize="small", title="Mechanism family",
              title_fontsize="small", alignment="left")

    png_path = out_dir / f"mechanism_importance{suffix}.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(out_dir / f"mechanism_importance{suffix}.pdf")
    plt.close(fig)
    print(f"[make_figures] wrote {png_path} (+ .pdf)")
    return png_path


# Display names for feature_matrix columns (SHAP beeswarm axis labels).
FEATURE_LABELS: Dict[str, str] = {
    "household_size": "Mean household size",
    "elderly_flag_severely": "Severe-aging threshold flag",
    "elderly_flag_shrinking": "Aging threshold flag",
    "household_size_small": "Small-household flag",
    "fin_local_tax": "Municipal local tax revenue",
    "fin_local_alloc_tax": "Local allocation tax",
    "fin_total_revenue": "Municipal total revenue",
    "fin_total_expenditure": "Municipal total expenditure",
    "fin_std_fiscal_revenue": "Standard fiscal revenue",
    "fin_std_fiscal_need": "Standard fiscal need",
    "fin_real_balance": "Real fiscal balance",
    "fin_fiscal_strength_index": "Financial capability index",
    "hls_pre1981_ratio": "Share of dwellings built before 1981",
    "hls_total_dwellings": "Total dwellings",
    "hls_missing": "Housing survey missing flag",
    "hls_vacancy_rate": "Vacancy rate",
    "hls_vacancy_other_rate": "Vacancy rate, other dwellings",
    "kaso_type": "Kaso designation type",
    "kaso_flag": "Kaso designation flag",
    "merged_flag": "Heisei merger flag",
    "years_since_merger": "Years since municipal merger",
    "dist_did_m": "Distance to DID",
    "in_did": "Within-DID flag",
    "dist_medical_m": "Distance to medical facility",
    "dist_hospital_m": "Distance to hospital",
    "n_medical_1000m": "Medical facilities within 1 km",
    "n_medical_3000m": "Medical facilities within 3 km",
    "n_medical_5000m": "Medical facilities within 5 km",
    "dist_school_m": "Distance to school",
    "dist_elementary_m": "Distance to elementary school",
    "n_school_1000m": "Schools within 1 km",
    "n_school_3000m": "Schools within 3 km",
    "n_school_5000m": "Schools within 5 km",
    "dist_bus_stop_m": "Distance to bus stop",
    "n_bus_stop_1000m": "Bus stops within 1 km",
    "n_bus_stop_3000m": "Bus stops within 3 km",
    "n_bus_stop_5000m": "Bus stops within 5 km",
    "dist_bus_route_m": "Distance to bus route",
    "dist_community_facility_m": "Distance to community facility",
    "n_community_facility_1000m": "Community facilities within 1 km",
    "n_community_facility_3000m": "Community facilities within 3 km",
    "n_community_facility_5000m": "Community facilities within 5 km",
}


def make_shap_beeswarm_figure(variant: str = "", max_display: int = 15) -> Path:
    """SHAP beeswarm of the top individual features (level model by default)."""
    import shap

    suffix = f"_{variant}" if variant else ""
    shap_df = pd.read_parquet(
        PHASE_B_DIR / "outputs" / f"shap_values{suffix}.parquet"
    )
    features = pd.read_parquet(
        PHASE_B_DIR / "outputs" / "feature_matrix.parquet"
    )[shap_df.columns]

    explanation = shap.Explanation(
        values=shap_df.to_numpy(),
        data=features.to_numpy(),
        feature_names=[FEATURE_LABELS.get(c, c) for c in shap_df.columns],
    )
    plt.rcParams.update({"font.size": 9})
    shap.plots.beeswarm(explanation, max_display=max_display, show=False,
                        plot_size=(6.3, 4.8))
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel("SHAP value (effect on predicted residual)", fontsize=9,
                  color=INK)
    ax.tick_params(labelsize=8)
    # shap adds a zero-height colorbar axis that breaks constrained layout,
    # so crop via bbox_inches instead
    return _save(fig, f"shap_beeswarm{suffix}", tight=True)


# Target variants in the row order of Table tab:results_cv_variants
# (thesis section 5.2): artifact suffix, display label, carried forward or not.
CV_VARIANTS: List[Any] = [
    ("", "Level of built-up texture", True),
    ("trajectory", "Trend of built-up texture", False),
    ("ndbi", "Trend of built-up index", False),
    ("dw_bare", "Bare ground trend, OLS", False),
    ("dw_bare_robust", "Bare ground trend, Theil-Sen", True),
]


def make_cv_variants_figure() -> Path:
    """Dot chart of CV R2 (mean +/- SD) across the decoupling target variants."""
    import json

    means, stds = [], []
    for suffix, _, _ in CV_VARIANTS:
        path = PHASE_B_DIR / "outputs" / (
            f"cv_results_{suffix}.json" if suffix else "cv_results.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)["xgboost"]
        means.append(metrics["mean_metrics"]["r2"])
        stds.append(metrics["std_metrics"]["r2"])

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelcolor": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK,
            "axes.edgecolor": INK_MUTED,
        }
    )

    fig, ax = plt.subplots(figsize=(6.3, 2.4), constrained_layout=True)
    ys = range(len(CV_VARIANTS) - 1, -1, -1)  # table order, top to bottom
    for y, mean, std, (_, label, carried) in zip(ys, means, stds, CV_VARIANTS):
        color = "#0072B2" if carried else "#999999"
        ax.errorbar(mean, y, xerr=std, fmt="o", markersize=5.5, color=color,
                    ecolor=color, elinewidth=1.4, capsize=3, zorder=3)
        ax.text(mean, y + 0.28, f"{mean:.2f} $\\pm$ {std:.2f}", ha="center",
                va="bottom", fontsize="small", color=INK_MUTED, zorder=3)

    ax.axvline(0, color=INK_MUTED, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(list(ys), [label for _, label, _ in CV_VARIANTS])
    ax.set_ylim(-0.6, len(CV_VARIANTS) - 0.2)
    ax.set_xlabel("Cross-validated $R^2$ (mean $\\pm$ SD, 25 grouped folds)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5.5,
                   color=color)
        for color in ("#0072B2", "#999999")
    ]
    ax.legend(handles, ["Carried forward", "Retired or superseded"],
              loc="lower left", frameon=False, fontsize="small")

    return _save(fig, "cv_target_variants")


def make_study_area_figure() -> Path:
    """Study-area map for fig:data_studyarea (thesis section 3.1).

    Aomori and Akita on an Esri World Topographic basemap: the inhabited aza
    sample as one semi-transparent fill (no unit borders), out-of-sample area
    hatched, prefecture boundaries, graticule labels, north arrow, scale bar,
    and a Japan locator inset (Natural Earth 1:50m outline, public domain).
    """
    import contextily as cx
    import geopandas as gpd
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.patheffects import withStroke
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    boundaries_dir = REPO_DIR / "admin_demographics" / "boundaries"
    gdf = gpd.read_file(boundaries_dir / "aza.gpkg")
    assign = pd.read_csv(
        REPO_DIR / "typology" / "outputs_aza" / "tables"
        / "unit_cluster_assignments.csv"
    )
    in_sample = gdf["unit_id"].isin(set(assign["unit_id"]))
    # Dissolve each layer so the transparent fill and the hatch render as one
    # continuous surface instead of per-aza patches (no interior borders)
    sample_geom = gdf[in_sample].dissolve()
    outside_geom = gdf[~in_sample].dissolve()
    prefs = gdf.dissolve(by="pref_name")
    # Close coverage slivers between aza polygons (500 m, in UTM 54N) so
    # .boundary yields clean prefecture outlines, not interior speckle
    prefs["geometry"] = (
        prefs.geometry.to_crs(epsg=32654).buffer(500).buffer(-500)
        .to_crs(gdf.crs)
    )
    japan = gpd.read_file(
        boundaries_dir / "japan_outline_ne50m.gpkg"
    ).to_crs(gdf.crs)

    plt.rcParams.update({"font.size": 9, "hatch.linewidth": 0.4})
    fig, ax = plt.subplots(figsize=(5.6, 6.4), constrained_layout=True)

    sample_color = "#6baed6"
    sample_geom.plot(ax=ax, facecolor=sample_color, edgecolor="none",
                     alpha=0.55)
    outside_geom.plot(ax=ax, facecolor="none", edgecolor="#999999",
                      linewidth=0.0, hatch="////")
    prefs.boundary.plot(ax=ax, color=INK, linewidth=0.9)

    cx.add_basemap(
        ax, crs=gdf.crs, source=cx.providers.Esri.WorldTopoMap, zoom=11,
        attribution="Basemap: Esri World Topographic Map",
        attribution_size=6,
    )

    halo = [withStroke(linewidth=2.5, foreground="white")]
    for pref, xy in {"Aomori": (140.55, 40.85), "Akita": (140.30, 39.70)}.items():
        ax.annotate(pref, xy, ha="center", va="center", fontsize=11,
                    color=INK, fontweight="bold", path_effects=halo)

    # Graticule labels on the frame
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:g}\N{DEGREE SIGN}E"))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:g}\N{DEGREE SIGN}N"))
    ax.tick_params(labelsize=7, colors=INK_MUTED, length=3)
    for spine in ax.spines.values():
        spine.set_color(INK_MUTED)
        spine.set_linewidth(0.6)

    # North arrow, upper right
    ax.annotate(
        "N", xy=(0.95, 0.97), xytext=(0.95, 0.89),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=11, fontweight="bold", color=INK,
        path_effects=halo,
        arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 1.4,
                    "path_effects": halo},
    )

    # 50 km scale bar (1 deg lon ~ 85.4 km at 40 deg N)
    bar_deg = 50.0 / 85.4
    x0, y0 = 139.55, 38.95
    ax.plot([x0, x0 + bar_deg], [y0, y0], color=INK, linewidth=1.5,
            path_effects=halo)
    ax.annotate("50 km", (x0 + bar_deg / 2, y0 + 0.05), ha="center",
                va="bottom", fontsize=8, color=INK, path_effects=halo)

    handles = [
        Patch(facecolor=sample_color, alpha=0.55, edgecolor="none"),
        Patch(facecolor="none", edgecolor="#999999", hatch="////"),
        Line2D([0], [0], color=INK, linewidth=0.9),
    ]
    ax.legend(
        handles,
        ["Inhabited aza (analytical sample)", "Not in sample",
         "Prefecture boundary"],
        loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2,
        frameon=False, fontsize="small",
        handlelength=1.4, handleheight=1.1,
    )

    # Japan locator inset, upper left (open sea in the main extent)
    inset = ax.inset_axes([0.02, 0.67, 0.28, 0.28])
    inset.set_facecolor("white")
    japan.plot(ax=inset, facecolor="#dddddd", edgecolor="#999999",
               linewidth=0.4)
    prefs.plot(ax=inset, facecolor="#2171b5", edgecolor="none")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color(INK_MUTED)
        spine.set_linewidth(0.6)

    return _save(fig, "study_area")


FIGURES: Dict[str, Any] = {
    "mechanism_importance": lambda args: make_mechanism_importance_figure(
        variant=args.variant),
    "decoupling_maps": lambda args: make_decoupling_map_figure(),
    "study_area": lambda args: make_study_area_figure(),
    "cv_target_variants": lambda args: make_cv_variants_figure(),
    "shap_beeswarm": lambda args: make_shap_beeswarm_figure(
        variant=args.variant),
}


def run_make_figures(args: argparse.Namespace) -> None:
    names = [args.only] if args.only else list(FIGURES)
    for name in names:
        FIGURES[name](args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", choices=["", *FIGURES],
                        help="render a single figure by name")
    parser.add_argument("--variant", default="",
                        help="artifact suffix, e.g. dw_bare_robust")
    run_make_figures(parser.parse_args())
