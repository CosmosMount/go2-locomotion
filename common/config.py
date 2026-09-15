"""YAML loading and deliberately small command-line override support."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return result


def apply_runtime_overrides(config: dict, **overrides) -> dict:
    """Apply only runner-authorized top-level runtime overrides."""

    result = deepcopy(config)
    allowed = {"seed", "device", "headless", "output_dir", "checkpoint"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"unsupported runtime overrides: {sorted(unknown)}")
    for name, value in overrides.items():
        if value is not None:
            result[name] = str(value) if isinstance(value, Path) else value
    return result
