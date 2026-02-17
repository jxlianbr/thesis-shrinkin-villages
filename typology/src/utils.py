"""
Shared utility functions for the typology module.

Provides configuration loading, output directory management,
figure/table saving, JSON export, and plot style setup.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_config(
    path: str = "typology/config/typology_config.yaml",
) -> Dict[str, Any]:
    """Load typology configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(cfg: Dict[str, Any]) -> None:
    """Create all output directories specified in the config."""
    for key in ("figures_dir", "tables_dir", "reports_dir"):
        Path(cfg["output"][key]).mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, name: str, cfg: Dict[str, Any]) -> Path:
    """
    Save a matplotlib figure to the configured figures directory.

    Args:
        fig: Matplotlib figure to save.
        name: Filename without extension (e.g. "pca_scree").
        cfg: Configuration dict.

    Returns:
        Path to the first saved file.
    """
    figures_dir = Path(cfg["output"]["figures_dir"])
    dpi = cfg["plot"]["dpi"]
    saved_path = None
    for fmt in cfg["plot"]["save_formats"]:
        p = figures_dir / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        if saved_path is None:
            saved_path = p
    plt.close(fig)
    print(f"  Saved figure: {saved_path}")
    return saved_path


def save_table(df: pd.DataFrame, name: str, cfg: Dict[str, Any]) -> Path:
    """
    Save a DataFrame as CSV to the configured tables directory.

    Args:
        df: DataFrame to save.
        name: Filename without extension.
        cfg: Configuration dict.

    Returns:
        Path to the saved file.
    """
    tables_dir = Path(cfg["output"]["tables_dir"])
    p = tables_dir / f"{name}.csv"
    df.to_csv(p, index=True)
    print(f"  Saved table: {p}")
    return p


def write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write a dict to a JSON file with pretty-printing."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def setup_plot_style(cfg: Dict[str, Any]) -> None:
    """Configure matplotlib plot style from config."""
    plot_cfg = cfg["plot"]
    plt.style.use(plot_cfg["style"])
    plt.rcParams.update({
        "font.size": plot_cfg["font_size"],
        "axes.titlesize": plot_cfg["title_size"],
        "axes.labelsize": plot_cfg["font_size"],
        "xtick.labelsize": plot_cfg["font_size"] - 1,
        "ytick.labelsize": plot_cfg["font_size"] - 1,
        "legend.fontsize": plot_cfg["font_size"] - 1,
        "figure.dpi": 100,
        "savefig.dpi": plot_cfg["dpi"],
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })
