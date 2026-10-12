"""Configuration loading and CLI override helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a YAML mapping.")
    return data


def _merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _namespace_to_dict(namespace: Namespace | None) -> dict[str, Any]:
    if namespace is None:
        return {}
    raw = vars(namespace)
    return {key: value for key, value in raw.items() if value is not None}


def load_and_resolve_config(
    default_config_path: str | Path,
    override_config_path: str | Path | None = None,
    cli_args: Namespace | None = None,
) -> dict[str, Any]:
    """Load default YAML, optional override YAML, and apply CLI values."""

    config = _read_yaml(default_config_path)
    if override_config_path:
        config = _merge_dicts(config, _read_yaml(override_config_path))
    config = _merge_dicts(config, _namespace_to_dict(cli_args))
    return config
