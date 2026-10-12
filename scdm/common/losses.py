"""Forecasting loss used by SCDM."""

from __future__ import annotations

import torch
from torch import nn


def compute_forecasting_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """One-step forecasting MSE."""

    return nn.functional.mse_loss(y_pred, y_true)
