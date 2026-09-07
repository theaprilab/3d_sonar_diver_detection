"""Configuration loading with explicit YAML inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input mapping."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a recipe and resolve its optional relative ``inherits`` field."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    parent = config.pop("inherits", None)
    if parent is None:
        return config
    return _merge(load_config(path.parent / parent), config)
