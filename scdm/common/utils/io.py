"""Input and output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def to_serializable(payload: Any) -> Any:
    """Convert common scientific Python objects to JSON-safe values."""

    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, np.generic):
        return payload.item()
    if isinstance(payload, dict):
        return {str(key): to_serializable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return [to_serializable(value) for value in payload]
    return payload


def save_json(path: str | Path, payload: Any) -> None:
    """Save a JSON-serializable object."""

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)


def save_yaml(path: str | Path, payload: Any) -> None:
    """Save a YAML object."""

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def save_numpy(path: str | Path, array: np.ndarray) -> None:
    """Save a NumPy array."""

    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, array)


def save_text(path: str | Path, content: str) -> None:
    """Save plain text content."""

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)
