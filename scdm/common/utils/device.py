"""Device resolution helpers."""

from __future__ import annotations

import torch


def resolve_device(device_name: str | None) -> torch.device:
    """Resolve the execution device."""

    if not device_name or device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
