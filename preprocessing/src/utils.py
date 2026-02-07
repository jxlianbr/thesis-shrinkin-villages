"""
Shared utility functions for the preprocessing module.

Provides configuration loading, output directory management, and JSON export.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str = "preprocessing/config/preprocessing_config.yaml") -> Dict[str, Any]:
    """Load preprocessing configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(cfg: Dict[str, Any]) -> None:
    """Create all output directories specified in the config."""
    Path(cfg["output"]["output_dir"]).mkdir(parents=True, exist_ok=True)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write a dict to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
