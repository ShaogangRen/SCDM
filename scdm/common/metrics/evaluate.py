"""Evaluation orchestration for a single graph-score matrix."""

from __future__ import annotations

import numpy as np

from .graph_metrics import (
    compute_binary_graph_metrics,
    compute_graph_density,
    compute_matrix_value_stats,
    compute_ranking_metrics,
    compute_threshold_sweep,
    compute_topk_graph_metrics,
)


def evaluate_scores(
    scores: np.ndarray,
    gt_graph: np.ndarray | None,
    threshold_mode: str = "acc_max",
    graph_threshold: float = 0.3,
    top_k: int | None = None,
    allow_self_loops: bool = False,
) -> dict:
    """Evaluate one continuous score matrix against the ground-truth adjacency.

    Both ``scores`` and ``gt_graph`` must use the source-target convention:
    entry ``[j, i]`` refers to the directed edge j -> i.

    ``threshold_mode`` selects how the continuous scores are binarised:

    - ``acc_max``   : threshold maximising accuracy over a 200-point sweep.
    - ``f1_max``    : threshold maximising F1 over the same sweep.
    - ``fixed``     : the fixed ``graph_threshold`` value.
    - ``top_k``     : keep the ``top_k`` highest-magnitude off-diagonal entries.

    ``acc_max``, ``f1_max`` and ``top_k`` consult the ground truth when choosing
    the operating point; they are oracle-thresholded diagnostics, which is why
    the threshold-free ranking metrics (AUROC/AUPRC) are reported alongside.

    Returns:
        dict with ``ranking_metrics``, ``binary_metrics``, ``binary_graph``,
        ``threshold`` and score diagnostics.
    """
    scores = np.asarray(scores, dtype=float)
    result: dict = {
        "score_stats": compute_matrix_value_stats(scores),
        "threshold_mode": threshold_mode,
    }

    if gt_graph is None:
        binary = (np.abs(scores) > float(graph_threshold)).astype(np.int64)
        np.fill_diagonal(binary, 0)
        result.update({
            "ranking_metrics": {"auroc": None, "auprc": None},
            "binary_metrics": {"f1": None, "shd": None},
            "binary_graph": binary,
            "threshold": float(graph_threshold),
            "predicted_density": compute_graph_density(binary),
            "reason": "skipped_no_ground_truth_graph",
        })
        return result

    gt = np.asarray(gt_graph).astype(np.int64)
    result["ranking_metrics"] = compute_ranking_metrics(scores, gt)

    sweep = compute_threshold_sweep(scores, gt)
    result["acc_max_threshold"] = sweep.get("acc_max_threshold")
    result["f1_max_threshold"] = sweep.get("f1_max_threshold")

    mode = str(threshold_mode).lower()
    if mode == "top_k":
        k = int(top_k if top_k is not None else np.count_nonzero(gt) - np.count_nonzero(np.diag(gt)))
        binary_metrics = compute_topk_graph_metrics(
            scores, gt, k=k, allow_self_loops=allow_self_loops
        )
        from .graph_metrics import binary_graph_at_top_k

        binary = binary_graph_at_top_k(scores, k=k, allow_self_loops=allow_self_loops)
        threshold = None
    else:
        if mode == "acc_max":
            threshold = sweep.get("acc_max_threshold")
        elif mode == "f1_max":
            threshold = sweep.get("f1_max_threshold")
        elif mode == "fixed":
            threshold = float(graph_threshold)
        else:
            raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")
        if threshold is None:
            threshold = float(graph_threshold)
        threshold = float(threshold)
        binary = (np.abs(scores) > threshold).astype(np.int64)
        if not allow_self_loops:
            np.fill_diagonal(binary, 0)
        binary_metrics = compute_binary_graph_metrics(binary, gt)
        binary_metrics["threshold"] = threshold

    result["binary_metrics"] = binary_metrics
    result["binary_graph"] = binary
    result["threshold"] = threshold
    result["predicted_edges"] = int(binary.sum())
    result["predicted_density"] = compute_graph_density(binary)
    return result


def summarize(name: str, evaluation: dict) -> str:
    """One-line human-readable summary of an :func:`evaluate_scores` result."""
    ranking = evaluation.get("ranking_metrics") or {}
    binary = evaluation.get("binary_metrics") or {}

    def _fmt(value, digits=4):
        if value is None:
            return "n/a"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return f"{float(value):.{digits}f}"

    return (
        f"{name}: AUROC={_fmt(ranking.get('auroc'))} "
        f"AUPRC={_fmt(ranking.get('auprc'))} "
        f"F1={_fmt(binary.get('f1'))} "
        f"SHD={_fmt(binary.get('shd'))} "
        f"(edges={evaluation.get('predicted_edges')}, "
        f"threshold={_fmt(evaluation.get('threshold'), 6)})"
    )
