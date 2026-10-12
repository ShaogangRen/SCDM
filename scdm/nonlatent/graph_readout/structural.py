"""Scalable structural graph-score readout (raw |W|).

This readout reads the shared causal-strength matrix W directly from the
trained model. Its cost is O(N^2) memory and O(1) forward passes, so it is the
readout used for the high-dimensional experiments where the source-isolated
readout (which needs O(N^2) forward passes per scoring episode) is impractical.

Index convention
----------------
Internal parameter: ``W[i, j]`` is the gate of SOURCE j on TARGET i.
Ground-truth adjacency: ``A[j, i] = 1`` means an edge j -> i (source-target).

Every matrix returned here is therefore TRANSPOSED into the source-target
convention:

    scores[j, i] = |W[i, j]|  -- strength of the directed edge j -> i
    binary[j, i] = 1          -- edge j -> i predicted
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def extract_structural_scores(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Read the structural scores from the shared W of a trained model.

    Args:
        model: a trained :class:`scdm.nonlatent.models.SCDMModel`.
        config: run configuration; ``graph_threshold``, ``top_k`` and
            ``allow_self_loops`` control the fixed-threshold binary graph.

    Returns:
        dict with

        - ``scores``: [N, N] float32, source-target continuous edge strengths.
        - ``W_signed``: [N, N] float32, source-target signed weights (K=1) or
          the unsigned edge strength when K > 1.
        - ``binary_graph``: [N, N] int64, fixed-threshold (or top-k) binary graph.
    """
    model.eval()

    with torch.no_grad():
        # Edge strength is [N, N] regardless of K.
        W_strength_ti = model.graph_head.get_edge_strength().cpu()
        W_abs = W_strength_ti.T  # source-target convention

        if model.w_components == 1:
            W_signed = model.graph_head.get_W_masked().cpu().T
        else:
            # K > 1: the unsigned per-edge norm is the canonical score.
            W_signed = W_abs

        threshold = float(config["graph_threshold"])
        top_k = config.get("top_k")
        allow_self_loops = bool(config.get("allow_self_loops", False))

        if top_k is not None:
            flat = W_abs.reshape(-1)
            k = min(int(top_k), flat.numel())
            cutoff = torch.topk(flat, k).values[-1]
            binary_graph = (W_abs >= cutoff).float()
        else:
            binary_graph = (W_abs > threshold).float()

        if not allow_self_loops:
            diag = torch.eye(W_abs.shape[0], dtype=torch.bool)
            binary_graph[diag] = 0.0

    return {
        "scores": W_abs.numpy().astype(np.float32),
        "W_signed": W_signed.numpy().astype(np.float32),
        "binary_graph": binary_graph.numpy().astype(np.int64),
    }
