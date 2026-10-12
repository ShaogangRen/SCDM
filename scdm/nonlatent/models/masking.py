"""Diagonal-masking helper for [..., N, N] edge tensors."""

from __future__ import annotations

import torch


def apply_diagonal_mask(edge_values: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Zero (or fill) the diagonal entries of a [..., N, N] tensor."""

    num_nodes = edge_values.shape[-1]
    diagonal_mask = torch.eye(num_nodes, dtype=torch.bool, device=edge_values.device)
    view_shape = (1,) * (edge_values.ndim - 2) + diagonal_mask.shape
    return edge_values.masked_fill(diagonal_mask.view(view_shape), fill_value)
