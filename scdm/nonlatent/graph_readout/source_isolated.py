"""Source-isolated post-training graph-score readout.

For every (source i, target j) pair:

    score[i, j] = (loss_zero - loss_source_only) / (loss_zero + eps)

where, after adapting the predictor copy ``theta_j`` and only the target
structural row ``w_j = W[j, :]`` on the support window of a scoring episode,

    loss_zero:        query loss with ALL gates zeroed (w = 0)
    loss_source_only: query loss with ONLY source i's gate left active
                      (w_k = 0 for every k != i)

A positive score means that isolating source i reduces the prediction error for
target j, i.e. evidence for a directed edge i -> j.

Both losses use the same adapted ``theta_j`` and ``w_j`` state on the query
window. The zero baseline replaces the entire adapted gate row by zeros; the
source-only comparison keeps only the adapted source entry and zeros every
other source. Negative values are retained as negative evidence; scores are not
clipped, rectified, or absolute-valued.

Cost: ``num_episodes * N * (N + 1)`` forward passes plus ``num_episodes * N``
inner loops. This is the readout used for the low-dimensional experiments; for
large N use :func:`scdm.nonlatent.graph_readout.extract_structural_scores` instead.

The returned matrix is already in the source-target convention:
``score[i, j]`` is the strength of the directed edge i -> j.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from scdm.common.losses import compute_forecasting_loss

EPS = 1e-8


def compute_source_isolated_scores(
    trainer,
    scoring_episode_dataset,
    num_episodes: int = 30,
    eps: float = EPS,
) -> np.ndarray:
    """Compute the source-isolated score matrix using the trainer's model.

    Args:
        trainer: :class:`scdm.nonlatent.meta_training.Trainer` whose model is already at
            the checkpoint to be scored.
        scoring_episode_dataset: episode dataset used for scoring (validation
            split by default).
        num_episodes: number of episodes to average over.
        eps: constant guarding the normalisation denominator.

    Returns:
        [N, N] float32 score matrix in source-target convention.
    """
    model = trainer.model
    model.eval()
    N = model.num_nodes

    loader = DataLoader(scoring_episode_dataset, batch_size=1, shuffle=False, num_workers=0)

    accum = np.zeros((N, N), dtype=np.float64)
    cnt = 0

    for ep_idx, episode in enumerate(loader):
        if ep_idx >= num_episodes:
            break

        episode = trainer._move_batch_to_device(episode)
        x_sup = episode["support_hist"].squeeze(0)
        y_sup = episode["support_target"].squeeze(0)
        x_qry = episode["query_hist"].squeeze(0)
        y_qry = episode["query_target"].squeeze(0)

        for j in range(N):
            theta_j, w_j = trainer._inner_loop(x_sup, y_sup, j)
            y_true_j = y_qry[:, j]

            # Zero baseline: every gate switched off.
            w_zero = torch.zeros_like(w_j)
            with torch.no_grad():
                loss_zero = float(compute_forecasting_loss(
                    model.forward_task(x_qry, j, theta_j, w_zero), y_true_j).cpu())

            # Source-isolated: keep only source i active.
            for i in range(N):
                if i == j:
                    continue
                w_i_only = torch.zeros_like(w_j)
                w_i_only[i] = w_j[i]
                with torch.no_grad():
                    loss_i = float(compute_forecasting_loss(
                        model.forward_task(x_qry, j, theta_j, w_i_only), y_true_j).cpu())
                accum[i, j] += (loss_zero - loss_i) / (loss_zero + eps)

        cnt += 1

    if cnt > 0:
        accum /= cnt

    return accum.astype(np.float32)
