"""Latent-graph evaluation with block metrics and latent-node matching.

Because q latent nodes are not identifiable by label (they are inferred,
not observed), predictions should be compared to the GT under all
permutations of latent node indices.  For q=2 there are only 2 permutations.

Nomenclature (source-target convention, A[src, tgt])
------------------------------------------------------
XX block:  A[:N_obs, :N_obs]  -- obs -> obs
ZX block:  A[N_obs:, :N_obs]  -- latent -> obs  (z -> x)
ZZ block:  A[N_obs:, N_obs:]  -- latent -> latent (z -> z)
OX block:  A[:N_obs, N_obs:]  -- obs -> latent   (forbidden in model, always 0)
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np

from scdm.common.metrics import compute_binary_graph_metrics, compute_ranking_metrics


def _permute_latent(
    matrix: np.ndarray,
    N_obs: int,
    latent_perm: list[int],
) -> np.ndarray:
    """Apply a permutation of latent node indices to a full [N_full, N_full] matrix.

    The permutation reorders both the latent rows and latent columns jointly
    so that latent node k maps to latent position latent_perm[k].

    Args:
        matrix: [N_full, N_full] matrix (source-target convention).
        N_obs: number of observed nodes.
        latent_perm: permutation of range(q_latent).

    Returns:
        [N_full, N_full] permuted matrix.
    """
    q = len(latent_perm)
    N_full = N_obs + q
    assert matrix.shape == (N_full, N_full), (
        f"Expected ({N_full}, {N_full}), got {matrix.shape}"
    )
    full_perm = list(range(N_obs)) + [N_obs + p for p in latent_perm]
    return matrix[np.ix_(full_perm, full_perm)]


def _block_metrics(
    score: np.ndarray,
    binary: np.ndarray,
    gt: np.ndarray,
    block_name: str,
) -> dict[str, Any]:
    """Compute binary and ranking metrics for a rectangular block.

    For very small blocks (e.g. 2x6 or 2x2 for q=2), AUROC/AUPRC may be
    None when only one class is present in the GT block.
    For non-square blocks, all entries are treated as edges (no diagonal mask).
    For square blocks, the diagonal is excluded.
    """
    is_square = (gt.ndim == 2 and gt.shape[0] == gt.shape[1])
    if is_square:
        # Use the standard binary metrics (which internally zero the diagonal)
        binary_m = compute_binary_graph_metrics(binary, gt)
        ranking_m = compute_ranking_metrics(score, gt)
        diag_vals_gt = np.diag(gt)
        diag_vals_b = np.diag(binary)
        gt_edges = int(np.count_nonzero(gt) - np.count_nonzero(diag_vals_gt))
        pred_edges = int(np.count_nonzero(binary) - np.count_nonzero(diag_vals_b))
    else:
        # Rectangular block: count all entries as potential edges
        gt_flat = gt.astype(np.int64).reshape(-1)
        binary_flat = binary.astype(np.int64).reshape(-1)
        score_flat = score.astype(float).reshape(-1)
        gt_edges = int(np.count_nonzero(gt_flat))
        pred_edges = int(np.count_nonzero(binary_flat))
        # Binary metrics manually
        tp = int(((binary_flat == 1) & (gt_flat == 1)).sum())
        fp = int(((binary_flat == 1) & (gt_flat == 0)).sum())
        fn = int(((binary_flat == 0) & (gt_flat == 1)).sum())
        tn = int(((binary_flat == 0) & (gt_flat == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)
        shd = fp + fn
        binary_m = {
            "precision": prec, "recall": rec, "f1": f1,
            "accuracy": acc, "acc": acc, "shd": shd,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        # Ranking metrics (may be None if only one class)
        from sklearn.metrics import roc_auc_score, average_precision_score
        unique = np.unique(gt_flat)
        if unique.size < 2:
            ranking_m = {"auroc": None, "auprc": None}
        else:
            try:
                auroc = float(roc_auc_score(gt_flat, score_flat))
            except Exception:
                auroc = None
            try:
                auprc = float(average_precision_score(gt_flat, score_flat))
            except Exception:
                auprc = None
            ranking_m = {"auroc": auroc, "auprc": auprc}
    return {
        "block": block_name,
        "shape": list(gt.shape),
        "gt_edges": gt_edges,
        "pred_edges": pred_edges,
        "auroc": ranking_m.get("auroc"),
        "auprc": ranking_m.get("auprc"),
        "f1": binary_m.get("f1"),
        "precision": binary_m.get("precision"),
        "recall": binary_m.get("recall"),
        "shd": binary_m.get("shd"),
        "tp": binary_m.get("tp"),
        "fp": binary_m.get("fp"),
        "fn": binary_m.get("fn"),
        "tn": binary_m.get("tn"),
    }


def evaluate_latent_graph(
    score_matrix: np.ndarray,
    binary_graph: np.ndarray,
    gt_graph: np.ndarray,
    N_obs: int,
    q_latent: int,
) -> dict[str, Any]:
    """Full latent-graph evaluation with block metrics and latent matching.

    All matrices use source-target convention: M[src, tgt].
    Observed nodes occupy indices 0..N_obs-1.
    Latent nodes occupy indices N_obs..N_obs+q_latent-1.

    Latent matching
    ---------------
    All q_latent! permutations of latent node indices are tried.
    The predicted graph is permuted accordingly (rows and columns together).
    The permutation that maximises full-graph F1 is reported as "best".
    Both raw (no permutation) and best-matched results are saved.

    Args:
        score_matrix: [N_full, N_full] continuous causal scores.
        binary_graph: [N_full, N_full] binary adjacency prediction.
        gt_graph:     [N_full, N_full] binary GT adjacency (obs first, latent last).
        N_obs:        number of observed nodes.
        q_latent:     number of latent nodes.

    Returns:
        dict containing:
          raw:         evaluation WITHOUT latent permutation (identity)
          best:        evaluation WITH best-matched latent permutation
          best_perm:   the best latent permutation as list[int]
          all_perms:   list of (perm, f1) for every permutation tried
          xx:          XX block metrics (no permutation needed; obs nodes are fixed)
          zx_best:     ZX block metrics under best latent permutation
          zz_best:     ZZ block metrics under best latent permutation
          ox_raw:      OX block (obs->latent, forbidden); pred is always 0 here
    """
    N_full = N_obs + q_latent
    assert score_matrix.shape == (N_full, N_full)
    assert binary_graph.shape == (N_full, N_full)
    assert gt_graph.shape == (N_full, N_full)

    # ---- XX block (no permutation needed) ----
    xx_metrics = _block_metrics(
        score=score_matrix[:N_obs, :N_obs],
        binary=binary_graph[:N_obs, :N_obs],
        gt=gt_graph[:N_obs, :N_obs],
        block_name="XX",
    )

    # ---- OX block: obs->latent (model always predicts 0 here) ----
    ox_metrics = _block_metrics(
        score=score_matrix[:N_obs, N_obs:],
        binary=binary_graph[:N_obs, N_obs:],
        gt=gt_graph[:N_obs, N_obs:],
        block_name="OX_forbidden",
    )

    # ---- Raw (identity permutation) ----
    raw_full_binary = compute_binary_graph_metrics(binary_graph, gt_graph)
    raw_full_ranking = compute_ranking_metrics(score_matrix, gt_graph)

    raw_zx = _block_metrics(
        score=score_matrix[N_obs:, :N_obs],
        binary=binary_graph[N_obs:, :N_obs],
        gt=gt_graph[N_obs:, :N_obs],
        block_name="ZX",
    )
    raw_zz = _block_metrics(
        score=score_matrix[N_obs:, N_obs:],
        binary=binary_graph[N_obs:, N_obs:],
        gt=gt_graph[N_obs:, N_obs:],
        block_name="ZZ",
    )

    raw_result = {
        "full_binary": raw_full_binary,
        "full_ranking": raw_full_ranking,
        "zx": raw_zx,
        "zz": raw_zz,
    }

    # ---- Latent matching: try all q_latent! permutations ----
    # Track two permutation choices for transparency:
    #   best_perm       = maximises full-graph F1 (kept for backward compatibility;
    #                     can tie across perms when ZX is sparse -> defaults to identity)
    #   best_zx_perm    = maximises ZX-block AUROC (reported alongside, not a replacement)
    all_perms_info: list[dict] = []
    best_f1 = -1.0
    best_perm = list(range(q_latent))

    # Pre-compute GT ZX flat (for per-perm ZX AUROC scoring).
    gt_zx_flat = gt_graph[N_obs:, :N_obs].astype(np.int64).reshape(-1)
    gt_zx_has_both = np.unique(gt_zx_flat).size >= 2

    best_zx_auroc = -1.0
    best_zx_perm = list(range(q_latent))

    for perm in permutations(range(q_latent)):
        perm = list(perm)
        score_perm = _permute_latent(score_matrix, N_obs, perm)
        binary_perm = _permute_latent(binary_graph, N_obs, perm)
        full_m = compute_binary_graph_metrics(binary_perm, gt_graph)
        f1 = float(full_m.get("f1") or 0.0)

        # Per-perm ZX AUROC
        if gt_zx_has_both:
            from sklearn.metrics import roc_auc_score
            try:
                zx_auroc_p = float(roc_auc_score(
                    gt_zx_flat, score_perm[N_obs:, :N_obs].astype(float).reshape(-1)
                ))
            except Exception:
                zx_auroc_p = None
        else:
            zx_auroc_p = None

        all_perms_info.append({
            "perm": perm,
            "f1": f1,
            "shd": full_m.get("shd"),
            "zx_auroc": zx_auroc_p,
        })
        if f1 > best_f1:
            best_f1 = f1
            best_perm = perm
        if zx_auroc_p is not None and zx_auroc_p > best_zx_auroc:
            best_zx_auroc = zx_auroc_p
            best_zx_perm = perm

    # ---- Best-matched metrics ----
    score_best = _permute_latent(score_matrix, N_obs, best_perm)
    binary_best = _permute_latent(binary_graph, N_obs, best_perm)

    best_full_binary = compute_binary_graph_metrics(binary_best, gt_graph)
    best_full_ranking = compute_ranking_metrics(score_best, gt_graph)

    best_zx = _block_metrics(
        score=score_best[N_obs:, :N_obs],
        binary=binary_best[N_obs:, :N_obs],
        gt=gt_graph[N_obs:, :N_obs],
        block_name="ZX",
    )
    best_zz = _block_metrics(
        score=score_best[N_obs:, N_obs:],
        binary=binary_best[N_obs:, N_obs:],
        gt=gt_graph[N_obs:, N_obs:],
        block_name="ZZ",
    )

    best_result = {
        "full_binary": best_full_binary,
        "full_ranking": best_full_ranking,
        "zx": best_zx,
        "zz": best_zz,
    }

    # ---- Identity-perm ZX block metrics (explicit alias of raw_zx) ----
    zx_identity = _block_metrics(
        score=score_matrix[N_obs:, :N_obs],
        binary=binary_graph[N_obs:, :N_obs],
        gt=gt_graph[N_obs:, :N_obs],
        block_name="ZX",
    )

    # ---- Best-ZX-perm ZX block metrics ----
    score_zx_best = _permute_latent(score_matrix, N_obs, best_zx_perm)
    binary_zx_best = _permute_latent(binary_graph, N_obs, best_zx_perm)
    zx_best_by_zx = _block_metrics(
        score=score_zx_best[N_obs:, :N_obs],
        binary=binary_zx_best[N_obs:, :N_obs],
        gt=gt_graph[N_obs:, :N_obs],
        block_name="ZX",
    )

    # Summary of ZX AUROC across all perms (for quick inspection).
    zx_auroc_values = [p["zx_auroc"] for p in all_perms_info if p.get("zx_auroc") is not None]
    zx_auroc_summary = {
        "min":    float(min(zx_auroc_values)) if zx_auroc_values else None,
        "max":    float(max(zx_auroc_values)) if zx_auroc_values else None,
        "mean":   float(sum(zx_auroc_values) / len(zx_auroc_values)) if zx_auroc_values else None,
        "count":  len(zx_auroc_values),
    }

    return {
        "raw": raw_result,
        "best": best_result,
        "best_perm": best_perm,
        "all_perms": all_perms_info,
        "xx": xx_metrics,
        "zx_best": best_zx,
        "zz_best": best_zz,
        "ox_raw": ox_metrics,
        # Added 2026-04-18 (opus47 audit follow-up): explicit per-perm ZX reporting.
        # best_perm above selects by full-graph F1 (ties default to identity when ZX is sparse);
        # the fields below expose the two ends of the ZX-AUROC distribution over all q! perms.
        "zx_identity":      zx_identity,         # ZX metrics under identity perm [0,1,...,q-1]
        "zx_best_by_zx":    zx_best_by_zx,       # ZX metrics under perm that maximises ZX AUROC
        "best_zx_perm":     best_zx_perm,        # the maximising perm
        "zx_auroc_summary": zx_auroc_summary,    # min/max/mean across all perms
    }
