"""Metrics for forecasting and graph evaluation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _off_diagonal_mask(num_nodes: int) -> np.ndarray:
    return ~np.eye(num_nodes, dtype=bool)


def _safe_square_matrix(matrix: np.ndarray | None) -> np.ndarray | None:
    if matrix is None:
        return None
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return None
    return array


def _flatten_off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[_off_diagonal_mask(matrix.shape[0])]


def compute_graph_density(matrix: np.ndarray | None) -> float | None:
    """Compute off-diagonal density of a graph-like matrix."""

    square = _safe_square_matrix(matrix)
    if square is None:
        return None
    num_nodes = square.shape[0]
    denom = max(num_nodes * max(num_nodes - 1, 1), 1)
    return float(np.count_nonzero(_flatten_off_diagonal(square)) / denom)


def compute_matrix_value_stats(matrix: np.ndarray | None) -> dict[str, float | int | None]:
    """Summarize nonzero off-diagonal entries of a graph-like matrix."""

    square = _safe_square_matrix(matrix)
    if square is None:
        return {
            "num_nonzero": None,
            "density": None,
            "min": None,
            "max": None,
            "mean": None,
        }

    values = _flatten_off_diagonal(square.astype(float))
    nonzero = values[np.abs(values) > 1e-12]
    density = compute_graph_density(square)

    if nonzero.size == 0:
        return {
            "num_nonzero": 0,
            "density": density,
            "min": None,
            "max": None,
            "mean": None,
        }

    return {
        "num_nonzero": int(nonzero.size),
        "density": density,
        "min": float(nonzero.min()),
        "max": float(nonzero.max()),
        "mean": float(nonzero.mean()),
    }


def compute_forecasting_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard forecasting metrics."""

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = math.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}


def compute_binary_graph_metrics(pred_graph: np.ndarray, gt_graph: np.ndarray) -> dict[str, float]:
    """Compute binary graph recovery metrics."""

    pred = _safe_square_matrix(pred_graph)
    gt = _safe_square_matrix(gt_graph)
    if pred is None or gt is None or pred.shape != gt.shape:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
            "acc": None,
            "shd": None,
            "normalized_shd": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "tn": None,
        }

    pred = pred.astype(np.int64).copy()
    gt = gt.astype(np.int64).copy()
    np.fill_diagonal(pred, 0)
    np.fill_diagonal(gt, 0)

    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    tn = int(np.logical_and(pred == 0, gt == 0).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    shd = int(np.abs(pred - gt).sum())
    normalized_shd = shd / max(pred.shape[0] * max(pred.shape[0] - 1, 1), 1)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "acc": float(accuracy),
        "shd": shd,
        "normalized_shd": float(normalized_shd),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_ranking_metrics(score_matrix: np.ndarray | None, gt_graph: np.ndarray | None) -> dict[str, float | None]:
    """Compute AUROC and AUPRC from continuous scores against the GT graph."""

    scores = _safe_square_matrix(score_matrix)
    gt = _safe_square_matrix(gt_graph)
    if scores is None or gt is None or scores.shape != gt.shape:
        return {"auroc": None, "auprc": None}

    y_true = _flatten_off_diagonal(gt.astype(np.int64))
    y_score = _flatten_off_diagonal(scores.astype(float))

    unique_classes = np.unique(y_true)
    if unique_classes.size < 2:
        return {"auroc": None, "auprc": None}

    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = None

    try:
        auprc = float(average_precision_score(y_true, y_score))
    except Exception:
        auprc = None

    random_auprc = float(y_true.mean())
    auprc_gain = (float(auprc) / random_auprc) if auprc is not None and random_auprc > 0 else None

    return {
        "auroc": auroc,
        "auprc": auprc,
        "random_auprc_baseline": random_auprc,
        "auprc_gain_over_random": auprc_gain,
    }


def binary_graph_at_top_k(score_matrix: np.ndarray, k: int, allow_self_loops: bool = False) -> np.ndarray:
    scores = _safe_square_matrix(score_matrix)
    if scores is None:
        raise ValueError("score_matrix must be square for top-k thresholding.")
    values = np.abs(scores).astype(float).copy()
    if not allow_self_loops:
        np.fill_diagonal(values, -np.inf)
    flat = values.reshape(-1)
    k = max(0, min(int(k), int(np.isfinite(flat).sum())))
    binary = np.zeros_like(values, dtype=np.int64)
    if k == 0:
        return binary
    top_idx = np.argpartition(flat, -k)[-k:]
    binary.reshape(-1)[top_idx] = 1
    if not allow_self_loops:
        np.fill_diagonal(binary, 0)
    return binary


def compute_topk_graph_metrics(
    score_matrix: np.ndarray,
    gt_graph: np.ndarray,
    k: int,
    allow_self_loops: bool = False,
) -> dict[str, float]:
    binary = binary_graph_at_top_k(score_matrix, k=k, allow_self_loops=allow_self_loops)
    metrics = compute_binary_graph_metrics(binary, gt_graph)
    metrics["k"] = int(k)
    metrics["pred_edges"] = int(binary.sum())
    return metrics


def compute_threshold_sweep(
    score_matrix: np.ndarray,
    gt_graph: np.ndarray,
    num_points: int = 200,
) -> dict[str, object]:
    """Sweep thresholds on continuous scores to find ACC-max threshold.

    Returns dict with:
      - acc_max_threshold: threshold that maximises accuracy
      - acc_max_metrics: binary metrics at that threshold
      - sweep: list of (threshold, acc, f1, shd, precision, recall, edges) rows
    """
    scores = _safe_square_matrix(score_matrix)
    gt = _safe_square_matrix(gt_graph)
    if scores is None or gt is None or scores.shape != gt.shape:
        return {"acc_max_threshold": None, "acc_max_metrics": None, "sweep": []}

    gt_flat = _flatten_off_diagonal(gt.astype(np.int64))
    score_flat = _flatten_off_diagonal(np.abs(scores).astype(float))

    lo = float(score_flat.min())
    hi = float(score_flat.max())
    thresholds = np.linspace(lo, hi, num_points)

    best_acc = -1.0
    best_thr = 0.0
    best_f1 = -1.0
    best_f1_thr = 0.0
    sweep_rows = []

    for thr in thresholds:
        pred_flat = (score_flat > thr).astype(np.int64)
        tp = int(((pred_flat == 1) & (gt_flat == 1)).sum())
        fp = int(((pred_flat == 1) & (gt_flat == 0)).sum())
        fn = int(((pred_flat == 0) & (gt_flat == 1)).sum())
        tn = int(((pred_flat == 0) & (gt_flat == 0)).sum())
        total = tp + fp + fn + tn
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        edges = tp + fp
        shd = fp + fn

        sweep_rows.append({
            "threshold": float(thr),
            "acc": float(acc),
            "f1": float(f1),
            "shd": int(shd),
            "precision": float(prec),
            "recall": float(rec),
            "edges": int(edges),
        })

        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thr = float(thr)

    # Compute full binary metrics at ACC-max threshold
    best_binary = (np.abs(scores) > best_thr).astype(np.int64)
    np.fill_diagonal(best_binary, 0)
    acc_max_metrics = compute_binary_graph_metrics(best_binary, gt)
    acc_max_metrics["threshold"] = best_thr

    f1_binary = (np.abs(scores) > best_f1_thr).astype(np.int64)
    np.fill_diagonal(f1_binary, 0)
    f1_max_metrics = compute_binary_graph_metrics(f1_binary, gt)
    f1_max_metrics["threshold"] = best_f1_thr

    return {
        "acc_max_threshold": best_thr,
        "acc_max_metrics": acc_max_metrics,
        "f1_max_threshold": best_f1_thr,
        "f1_max_metrics": f1_max_metrics,
        "sweep": sweep_rows,
    }


def extract_top_edges(
    weight_graph: np.ndarray,
    node_names: Sequence[str] | None = None,
    k: int = 10,
) -> list[dict]:
    """Return the top-k weighted edges."""

    if node_names is None:
        node_names = [f"node_{idx}" for idx in range(weight_graph.shape[0])]

    edges = []
    for source_index in range(weight_graph.shape[0]):
        for target_index in range(weight_graph.shape[1]):
            weight = float(weight_graph[source_index, target_index])
            if weight <= 0:
                continue
            edges.append(
                {
                    "source": int(source_index),
                    "target": int(target_index),
                    "source_name": str(node_names[source_index]),
                    "target_name": str(node_names[target_index]),
                    "weight": weight,
                }
            )
    edges.sort(key=lambda item: item["weight"], reverse=True)
    return edges[:k]


def compute_weighted_graph_diagnostics(
    weight_graph: np.ndarray,
    node_names: Sequence[str] | None = None,
) -> dict:
    """Summarize weighted graph properties."""

    square = _safe_square_matrix(weight_graph)
    if square is None:
        return {
            "num_predicted_edges": None,
            "graph_density": None,
            "min_nonzero_weight": None,
            "max_nonzero_weight": None,
            "mean_nonzero_weight": None,
            "top_edges": [],
        }

    values = _flatten_off_diagonal(square)
    nonzero = values[np.abs(values) > 1e-12]
    density = compute_graph_density(square)
    diagnostics = {
        "num_predicted_edges": int(nonzero.size),
        "graph_density": density,
        "min_nonzero_weight": float(nonzero.min()) if nonzero.size else None,
        "max_nonzero_weight": float(nonzero.max()) if nonzero.size else None,
        "mean_nonzero_weight": float(nonzero.mean()) if nonzero.size else None,
        "top_edges": extract_top_edges(square, node_names=node_names, k=10),
    }
    return diagnostics
