"""FOMAML trainer for the standard explicit-latent causal model.

Training objective (draft Eq. latent_obj)
------------------------------------------
  L = L_obs + λ_W R_W + λ_z R_z + λ_wzx_col R_wzx_col
      + λ_wzx_sparse R_wzx_sparse + λ_wzx_div R_wzx_div

Per task i on both support (inner loop) and query (outer loop):
  L_obs = MSE(ŷ_i, y_i)                    observed forecasting
  R_W   = ||w_i||_1 + ||W_zz_i||_1         structural sparsity (per-row + WZZ)
  R_z   = (1/L) Σ_t ||z_t||_2              latent magnitude regularization
  R_wzx_col = Σ_j ||W[:N_obs, N_obs+j]||_1 shared outer-loop WZX column sparsity
  R_wzx_sparse = ||W[:N_obs, N_obs:]||_1   shared outer-loop WZX entrywise sparsity
  R_wzx_div = ||offdiag(C^T C)||_F^2       shared outer-loop WZX column decorrelation
               where C column-normalizes W[:N_obs, N_obs:]

No hidden-state reconstruction loss.

Inner loop
  Adapts theta_i (MLP), w_i = W[target_idx, :], W_zz_i = WZZ copy, and z_ep.
  z_ep is initialised from z_global[seq_idx][episode_start] for training episodes
  and from zeros for validation episodes.
  Uses create_graph=False (first-order FOMAML; no second-order graph retention).

Outer loop
  Accumulates first-order FOMAML gradients across all tasks.
  theta grads: from adapted MLP leaf tensors.
  WZZ grads: from W_zz_i leaf tensors, accumulated into W_grad_accum[N_obs:, N_obs:].
  z grads: from the query-start z_ep leaf, accumulated into
           z_global[seq_idx][episode_start + l_sup].
  Updates shared Θ = {θ, W, Z} with Adam.

History keys logged per interval
  loss_total  loss_forecast  loss_rw  loss_rz
  loss_wzx_col  loss_wzx_sparse  loss_wzx_div
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam

from scdm.common.losses import compute_forecasting_loss
from scdm.common.callbacks import MetricHistoryCallback
from scdm.common.utils.io import ensure_dir


# Maps theta_params dict keys → model.named_parameters() keys (MLP only).
# z is handled episode-specifically via z_global — not in this map.
_THETA_KEY_MAP = {
    "l1.w":   "linear1.weight",
    "l1.b":   "linear1.bias",
    "l2.w":   "linear2.weight",
    "l2.b":   "linear2.bias",
    "l3.w":   "linear3.weight",
    "l3.b":   "linear3.bias",
}


def _infinite_episode_iter(dataloader):
    while True:
        for episode in dataloader:
            yield episode


class Trainer:
    """FOMAML trainer for LatentCausalModel (explicit-latent, per-sequence Z).

    Inner loop: adapts theta_i (MLP), w_i = W[i, :], W_zz_i (WZZ copy),
    and z_ep (episode-specific initial latent state).

    Outer loop: accumulates first-order FOMAML gradients over all tasks,
    updates shared model parameters (θ + W + Z) with Adam.

    Z update: the query-start z_ep gradient from each task is accumulated and
    written to z_global[seq_idx].grad at position [episode_start + l_sup]
    before optimizer.step().
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        device: torch.device,
        output_dir: str | Path,
        logger: logging.Logger,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = ensure_dir(output_dir)
        self.logger = logger
        self.history = MetricHistoryCallback()

        self.inner_loop_steps    = int(config.get("inner_loop_steps", 3))
        self.inner_loop_lr_theta = float(config.get("inner_loop_lr", 0.01))
        self.inner_loop_lr_w     = float(config.get("inner_loop_lr_w", 0.001))

        # Objective coefficients.
        # lambda_W_latent: separate (usually smaller) L1 weight for WZX and WZZ blocks.
        # Falls back to lambda_W if not set (original uniform behaviour).
        self.lambda_W          = float(config.get("lambda_W", 0.05))
        self.lambda_W_latent   = float(config.get("lambda_W_latent", config.get("lambda_W", 0.05)))
        self.lambda_wzx_col    = float(config.get("lambda_wzx_col", 0.0))
        self.lambda_wzx_sparse = float(config.get("lambda_wzx_sparse", 0.0))
        self.lambda_wzx_div    = float(config.get("lambda_wzx_div", 0.0))
        self.lambda_z          = float(config.get("lambda_z", 0.01))
        tps = config.get("tasks_per_outer_step", config.get("tasks_per_step", 0))
        self.tasks_per_outer_step = int(tps)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )

        self.best_val_loss = float("inf")
        self.best_step = 0
        self.current_step = 0
        self.best_epoch = 0  # backward-compat alias
        self.training_runtime_seconds: float | None = None  # set by fit() upon completion

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    @staticmethod
    def _param_grad_norm(param: torch.nn.Parameter) -> float:
        if param.grad is None:
            return 0.0
        return float(param.grad.detach().norm(2).cpu())

    @staticmethod
    def _compute_rz(z_seq: torch.Tensor) -> torch.Tensor:
        """R_z = (1/L) Σ_t ||z_t||_2  (L2 norm, not squared)."""
        return z_seq.norm(dim=1).mean()

    def _z_global_mean_norm(self) -> float:
        """Mean L2 norm across all z_global parameter matrices."""
        params = list(self.model.z_global)
        if not params:
            return 0.0
        return float(sum(z.detach().norm(2).cpu() for z in params) / len(params))

    def _latent_training_enabled(self) -> bool:
        return self.model.q_latent > 0

    def _use_wzz(self) -> bool:
        return bool(self.config.get("use_wzz", True)) and self._latent_training_enabled()

    def _wzz_diagonal_only(self) -> bool:
        return self._use_wzz() and bool(self.config.get("wzz_diagonal_only", False))

    def _zero_latent_sequence(self, length: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(length, self.model.q_latent, device=self.device, dtype=dtype)

    def _weighted_structural_penalty(
        self,
        w_i: torch.Tensor,
        W_zz_i: torch.Tensor | None,
    ) -> torch.Tensor:
        penalty = self.lambda_W * w_i[: self.model.num_obs].abs().sum()
        if self._latent_training_enabled() and W_zz_i is not None:
            penalty = penalty + self.lambda_W_latent * (
                w_i[self.model.num_obs :].abs().sum() + W_zz_i.abs().sum()
            )
        return penalty

    def _weighted_latent_penalty(self, z_seq: torch.Tensor | None) -> torch.Tensor:
        if not self._latent_training_enabled() or z_seq is None:
            return torch.zeros((), device=self.device)
        return self.lambda_z * self._compute_rz(z_seq)

    def _get_wzx_block(self) -> torch.Tensor:
        W_masked = self.model.graph_head.get_W_masked()
        return W_masked[: self.model.num_obs, self.model.num_obs :]

    def _weighted_wzx_column_penalty(self) -> torch.Tensor:
        if not self._latent_training_enabled() or self.lambda_wzx_col <= 0.0:
            return torch.zeros((), device=self.device)
        W_zx = self._get_wzx_block()
        # Penalize each latent-source column over observed-target rows only.
        return self.lambda_wzx_col * W_zx.abs().sum(dim=0).sum()

    def _weighted_wzx_sparse_penalty(self) -> torch.Tensor:
        if not self._latent_training_enabled() or self.lambda_wzx_sparse <= 0.0:
            return torch.zeros((), device=self.device)
        W_zx = self._get_wzx_block()
        # Additional WZX-only shrinkage for within-column tail support.
        return self.lambda_wzx_sparse * W_zx.abs().sum()

    def _weighted_wzx_diversity_penalty(self, eps: float = 1e-8) -> torch.Tensor:
        if not self._latent_training_enabled() or self.lambda_wzx_div <= 0.0:
            return torch.zeros((), device=self.device)
        if self.model.q_latent < 2:
            return torch.zeros((), device=self.device)

        W_zx = self._get_wzx_block()  # [N_obs, q] target-source latent->obs block
        col_norms = W_zx.norm(p=2, dim=0, keepdim=True).clamp_min(eps)
        W_zx_unit = W_zx / col_norms
        gram = W_zx_unit.transpose(0, 1) @ W_zx_unit  # [q, q]

        eye = torch.eye(self.model.q_latent, device=gram.device, dtype=gram.dtype)
        off_diag = gram.masked_fill(eye.bool(), 0.0)
        return self.lambda_wzx_div * off_diag.pow(2).sum()

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def _inner_loop(
        self,
        x_support: torch.Tensor,
        y_support: torch.Tensor,
        target_idx: int,
        z_ep_init: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """FOMAML inner loop for observed task i.

        Adapts theta_i (MLP), w_i = W[target_idx, :], W_zz_i (WZZ copy),
        and z_ep (episode-specific initial latent state) on the support set.

        create_graph=False throughout — first-order MAML approximation.

        Args:
            x_support:   [S, lag, N_obs]
            y_support:   [S, N_obs]
            target_idx:  observed target variable index
            z_ep_init:   [q]  initial latent state for this episode
                         (z_global[seq_idx][episode_start] for training,
                          zeros for validation)

        Returns:
            (theta_i', w_i', W_zz_i', z_ep')  — adapted leaf tensors
        """
        theta_i = self.model.get_theta_params()
        latent_enabled = self._latent_training_enabled()
        use_wzz = self._use_wzz()
        diag_only = self._wzz_diagonal_only()

        W_masked = self.model.graph_head.get_W_masked()
        w_i = (
            W_masked[target_idx, :]
            .clone().detach().requires_grad_(True)
        )
        # W_zz_i leaf tensor: None (no WZZ), [q] diagonal vector (diagonal mode),
        # or [q,q] full matrix (full WZZ mode).
        if not use_wzz:
            W_zz_i = None
        elif diag_only:
            W_zz_i = (
                W_masked[self.model.num_obs:, self.model.num_obs:].diag()
                .clone().detach().requires_grad_(True)
            )
        else:
            W_zz_i = (
                W_masked[self.model.num_obs:, self.model.num_obs:]
                .clone().detach().requires_grad_(True)
            )
        # Clone and detach z_ep_init to create an independent leaf tensor
        z_ep = z_ep_init.clone().detach().requires_grad_(True)

        l_sup = x_support.shape[0]
        N_obs = self.model.num_obs
        q = self.model.q_latent

        for _ in range(self.inner_loop_steps):
            # Compute latent sequence from episode-specific z_ep.
            # functional_z_seq handles all three modes: None / [q] / [q,q].
            if latent_enabled:
                z_seq = self.model.functional_z_seq(z_ep, W_zz_i, l_sup)
            else:
                z_seq = self._zero_latent_sequence(l_sup, x_support.dtype)

            y_pred = self.model.functional_forward(
                x_support, z_seq, w_i, theta_i, N_obs, q
            )
            L_pred = compute_forecasting_loss(y_pred, y_support[:, target_idx])

            R_W = self._weighted_structural_penalty(w_i, W_zz_i if use_wzz else None)
            R_z = self._weighted_latent_penalty(z_seq if latent_enabled else None)
            loss = L_pred + R_W + R_z

            n_theta = len(theta_i)
            all_params = list(theta_i.values()) + [w_i]
            if latent_enabled:
                if use_wzz:
                    all_params.extend([W_zz_i, z_ep])
                else:
                    all_params.append(z_ep)
            grads = torch.autograd.grad(loss, all_params, create_graph=False)

            new_theta = {}
            for (name, param), g in zip(theta_i.items(), grads[:n_theta]):
                new_theta[name] = (param - self.inner_loop_lr_theta * g).detach().requires_grad_(True)
            theta_i = new_theta
            w_i = (w_i - self.inner_loop_lr_w * grads[n_theta]).detach().requires_grad_(True)
            if latent_enabled:
                if use_wzz:
                    W_zz_i = (
                        W_zz_i - self.inner_loop_lr_w * grads[n_theta + 1]
                    ).detach().requires_grad_(True)
                    z_ep = (
                        z_ep - self.inner_loop_lr_theta * grads[n_theta + 2]
                    ).detach().requires_grad_(True)
                else:
                    z_ep = (
                        z_ep - self.inner_loop_lr_theta * grads[n_theta + 1]
                    ).detach().requires_grad_(True)

        return theta_i, w_i, W_zz_i, z_ep

    # ------------------------------------------------------------------
    # Outer step
    # ------------------------------------------------------------------

    def _outer_step(self, episode: dict[str, torch.Tensor]) -> dict[str, float]:
        """One FOMAML outer step — iterates over all N_obs observed tasks."""
        self.model.train()
        episode = self._move_batch_to_device(episode)
        x_support = episode["support_hist"].squeeze(0)     # [S, lag, N_obs]
        y_support = episode["support_target"].squeeze(0)   # [S, N_obs]
        x_query   = episode["query_hist"].squeeze(0)       # [Q, lag, N_obs]
        y_query   = episode["query_target"].squeeze(0)     # [Q, N_obs]
        seq_idx      = int(episode["sequence_id"].item())
        episode_start = int(episode["episode_start"].item())
        # support_hidden / query_hidden present but not used for training loss

        l_sup = x_support.shape[0]
        l_qry = x_query.shape[0]
        l_total = l_sup + l_qry

        N_obs = self.model.num_obs
        q = self.model.q_latent
        latent_enabled = self._latent_training_enabled()
        use_wzz = self._use_wzz()
        diag_only = self._wzz_diagonal_only()

        if self.tasks_per_outer_step > 0 and self.tasks_per_outer_step < N_obs:
            task_indices = random.sample(range(N_obs), self.tasks_per_outer_step)
        else:
            task_indices = list(range(N_obs))
        num_tasks = len(task_indices)

        episode_query_start = episode_start + l_sup
        if latent_enabled:
            # Support-window z: from z_global[seq_idx][episode_start].
            # Inner loop adapts a clone of this for support predictions.
            z_ep_sup_init = self.model.get_z_ep(seq_idx, episode_start)
            # Query-window z: from z_global[seq_idx][episode_start + l_sup].
            # This is the stored global latent state at the query start time.
            # It is NOT adapted in the inner loop; the outer step computes its
            # gradient directly from the query loss, avoiding the vanishing-gradient
            # problem that arises when rolling through l_sup WZZ steps from z_ep_sup.
            z_ep_qry_init = self.model.get_z_ep(seq_idx, episode_query_start)
        else:
            z_ep_sup_init = torch.zeros(q, device=self.device, dtype=x_support.dtype)
            z_ep_qry_init = torch.zeros(q, device=self.device, dtype=x_query.dtype)

        # Phase 1: inner-loop adaptation per task (support z only)
        adapted: dict[int, tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for i in task_indices:
            adapted[i] = self._inner_loop(x_support, y_support, i, z_ep_sup_init)

        # Phase 2: query loss + gradient accumulation
        theta_grad_accum: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if not name.startswith("graph_head.") and not name.startswith("z_global.")
        }
        W_grad_accum = torch.zeros_like(self.model.graph_head.W)
        z_ep_qry_grad_accum = torch.zeros(q, device=self.device) if latent_enabled else None

        total_forecast = total_rw = total_rz = 0.0

        for i in task_indices:
            theta_i, w_i, W_zz_i, _z_ep_sup = adapted[i]

            if latent_enabled:
                # Query z: fresh leaf from z_global[seq_idx][episode_query_start].
                # When use_wzz=True: WZZ rollout spans only l_qry steps.
                # When use_wzz=False: W_zz_i=None → constant z (no Z->Z).
                z_ep_qry = z_ep_qry_init.clone().detach().requires_grad_(True)
                z_seq_qry = self.model.functional_z_seq(z_ep_qry, W_zz_i, l_qry)  # [Q, q]
            else:
                z_ep_qry = None
                z_seq_qry = self._zero_latent_sequence(l_qry, x_query.dtype)

            y_pred  = self.model.functional_forward(x_query, z_seq_qry, w_i, theta_i, N_obs, q)
            L_pred  = compute_forecasting_loss(y_pred, y_query[:, i])
            R_W = self._weighted_structural_penalty(w_i, W_zz_i if use_wzz else None)
            R_z = self._weighted_latent_penalty(z_seq_qry if latent_enabled else None)

            L_qry_i = L_pred + R_W + R_z

            n_theta = len(theta_i)
            all_params = list(theta_i.values()) + [w_i]
            if latent_enabled:
                if use_wzz:
                    all_params.extend([W_zz_i, z_ep_qry])
                else:
                    all_params.append(z_ep_qry)
            grads = torch.autograd.grad(L_qry_i, all_params, create_graph=False)

            # Accumulate theta (MLP) gradients
            for (tname, _), g in zip(theta_i.items(), grads[:n_theta]):
                model_key = _THETA_KEY_MAP.get(tname, tname)
                if model_key in theta_grad_accum:
                    theta_grad_accum[model_key] = theta_grad_accum[model_key] + g

            # w_i grad → row i of W
            W_grad_accum[i, :] = W_grad_accum[i, :] + grads[n_theta]

            if latent_enabled:
                if use_wzz:
                    if diag_only:
                        # W_zz_i is [q] diagonal vector; grad goes only on diagonal of WZZ block.
                        wzz_diag_idx = torch.arange(q, device=self.device)
                        W_grad_accum[N_obs + wzz_diag_idx, N_obs + wzz_diag_idx] = (
                            W_grad_accum[N_obs + wzz_diag_idx, N_obs + wzz_diag_idx] + grads[n_theta + 1]
                        )
                    else:
                        # W_zz_i is [q,q]; full WZZ block gradient.
                        W_grad_accum[N_obs:, N_obs:] = W_grad_accum[N_obs:, N_obs:] + grads[n_theta + 1]
                    z_ep_qry_grad_accum = z_ep_qry_grad_accum + grads[n_theta + 2]
                else:
                    z_ep_qry_grad_accum = z_ep_qry_grad_accum + grads[n_theta + 1]

            total_forecast += float(L_pred.detach().cpu())
            total_rw       += float(R_W.detach().cpu())
            total_rz       += float(R_z.detach().cpu())

        wzx_col_penalty = self._weighted_wzx_column_penalty()
        wzx_sparse_penalty = self._weighted_wzx_sparse_penalty()
        wzx_div_penalty = self._weighted_wzx_diversity_penalty()
        if self._latent_training_enabled() and (
            self.lambda_wzx_col > 0.0
            or self.lambda_wzx_sparse > 0.0
            or self.lambda_wzx_div > 0.0
        ):
            wzx_penalty_grad = torch.autograd.grad(
                wzx_col_penalty + wzx_sparse_penalty + wzx_div_penalty,
                self.model.graph_head.W,
                create_graph=False,
            )[0]
        else:
            wzx_penalty_grad = torch.zeros_like(self.model.graph_head.W)

        # Phase 3: assign normalized grads and optimizer step
        self.optimizer.zero_grad(set_to_none=True)
        for name, param in self.model.named_parameters():
            if name in theta_grad_accum:
                param.grad = theta_grad_accum[name] / num_tasks
        self.model.graph_head.W.grad = (W_grad_accum / num_tasks) + wzx_penalty_grad

        if latent_enabled:
            # z_global[seq_idx][episode_query_start] gradient: sparse [T_k, q] tensor.
            # Note: z_global[seq_idx][episode_start] (support start) receives gradient
            # when it acts as episode_query_start in other randomly sampled episodes.
            z_k_grad = torch.zeros_like(self.model.z_global[seq_idx])
            z_k_grad[episode_query_start] = z_ep_qry_grad_accum / num_tasks
            self.model.z_global[seq_idx].grad = z_k_grad

        self.optimizer.step()

        avg_fc = total_forecast / num_tasks
        avg_rw = total_rw      / num_tasks
        avg_rz = total_rz      / num_tasks
        wzx_col_value = float(wzx_col_penalty.detach().cpu())
        wzx_sparse_value = float(wzx_sparse_penalty.detach().cpu())
        wzx_div_value = float(wzx_div_penalty.detach().cpu())
        return {
            "loss_total":    avg_fc + avg_rw + avg_rz + wzx_col_value + wzx_sparse_value + wzx_div_value,
            "loss_forecast": avg_fc,
            "loss_rw":       avg_rw,
            "loss_rz":       avg_rz,
            "loss_wzx_col":  wzx_col_value,
            "loss_wzx_sparse": wzx_sparse_value,
            "loss_wzx_div":  wzx_div_value,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(self, val_episode_loader, max_episodes: int = 0) -> dict[str, float]:
        self.model.eval()
        total_fc = total_rw = total_rz = 0.0
        num_episodes = 0
        q = self.model.q_latent
        latent_enabled = self._latent_training_enabled()
        wzx_col_value = float(self._weighted_wzx_column_penalty().detach().cpu())
        wzx_sparse_value = float(self._weighted_wzx_sparse_penalty().detach().cpu())
        wzx_div_value = float(self._weighted_wzx_diversity_penalty().detach().cpu())

        for episode in val_episode_loader:
            if max_episodes > 0 and num_episodes >= max_episodes:
                break

            episode = self._move_batch_to_device(episode)
            x_support = episode["support_hist"].squeeze(0)
            y_support = episode["support_target"].squeeze(0)
            x_query   = episode["query_hist"].squeeze(0)
            y_query   = episode["query_target"].squeeze(0)

            l_sup = x_support.shape[0]
            l_qry = x_query.shape[0]
            l_total = l_sup + l_qry

            N_obs = self.model.num_obs
            ep_fc = ep_rw = ep_rz = 0.0

            # Validation: support z adapted from zeros (no stored z_global for val windows).
            # Query z also starts from zeros — not rolled through l_sup WZZ steps,
            # matching the training scheme where query starts from its own z_global entry.
            z_ep_sup_val  = torch.zeros(q, device=self.device)
            z_ep_qry_val  = torch.zeros(q, device=self.device)

            for i in range(N_obs):
                theta_i, w_i, W_zz_i, _z_ep_sup = self._inner_loop(
                    x_support, y_support, i, z_ep_sup_val
                )
                # W_zz_i is None when use_wzz=False; functional_z_seq handles None → constant z.
                if latent_enabled:
                    wzz_for_rollout = W_zz_i.detach() if W_zz_i is not None else None
                    z_seq_qry = self.model.functional_z_seq(
                        z_ep_qry_val, wzz_for_rollout, l_qry
                    )
                else:
                    z_seq_qry = self._zero_latent_sequence(l_qry, x_query.dtype)

                with torch.no_grad():
                    y_pred = self.model.functional_forward(
                        x_query, z_seq_qry,
                        w_i.detach(), {k: v.detach() for k, v in theta_i.items()},
                        N_obs, q,
                    )
                    L_pred = compute_forecasting_loss(y_pred, y_query[:, i])
                    wzz_for_penalty = W_zz_i.detach() if (latent_enabled and W_zz_i is not None) else None
                    R_W = self._weighted_structural_penalty(
                        w_i.detach(), wzz_for_penalty
                    )
                    R_z = self._weighted_latent_penalty(
                        z_seq_qry if latent_enabled else None
                    )

                ep_fc  += float(L_pred.cpu())
                ep_rw  += float(R_W.cpu())
                ep_rz  += float(R_z.cpu())

            total_fc  += ep_fc  / N_obs
            total_rw  += ep_rw  / N_obs
            total_rz  += ep_rz  / N_obs
            num_episodes += 1

        if num_episodes == 0:
            raise ValueError("Empty validation episode loader.")

        avg_fc = total_fc / num_episodes
        avg_rw = total_rw / num_episodes
        avg_rz = total_rz / num_episodes
        return {
            "loss_total":    avg_fc + avg_rw + avg_rz + wzx_col_value + wzx_sparse_value + wzx_div_value,
            "loss_forecast": avg_fc,
            "loss_rw":       avg_rw,
            "loss_rz":       avg_rz,
            "loss_wzx_col":  wzx_col_value,
            "loss_wzx_sparse": wzx_sparse_value,
            "loss_wzx_div":  wzx_div_value,
            "num_episodes":  num_episodes,
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_payload(self, step: int, val_loss: float) -> dict[str, Any]:
        return {
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config":               self.config,
            "step":                 step,
            "epoch":                step,
            "validation_score":     val_loss,
        }

    def _save_checkpoint(self, filename: str, step: int, val_loss: float) -> Path:
        path = self.output_dir / filename
        torch.save(self._checkpoint_payload(step=step, val_loss=val_loss), path)
        return path

    def load_checkpoint(self, checkpoint_path: str | Path, load_optimizer: bool = False) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self, train_loader, val_episode_loader) -> Path:
        """FOMAML training loop — step-based budget."""
        total_outer_steps = int(self.config.get("total_outer_steps", 500))
        val_every    = int(self.config.get("val_every_outer_steps", 50))
        val_episodes = int(self.config.get("val_episodes", 50))
        patience     = int(self.config.get("patience", 0))

        best_checkpoint_path = self.output_dir / "best.ckpt"
        interval_losses = {
            k: 0.0
            for k in (
                "loss_total",
                "loss_forecast",
                "loss_rw",
                "loss_rz",
                "loss_wzx_col",
                "loss_wzx_sparse",
                "loss_wzx_div",
            )
        }
        interval_count = 0
        checks_without_improvement = 0

        episode_iter = _infinite_episode_iter(train_loader)

        n_z_params = len(list(self.model.z_global))
        self.logger.info(
            "Training (explicit-latent Z): steps=%d val_every=%d val_ep=%d tasks/step=%s "
            "patience=%d N_obs=%d q=%d n_train_seqs=%d",
            total_outer_steps, val_every, val_episodes,
            self.tasks_per_outer_step or "all",
            patience, self.model.num_obs, self.model.q_latent, n_z_params,
        )

        _train_start = time.time()
        for step in range(1, total_outer_steps + 1):
            self.current_step = step
            step_losses = self._outer_step(next(episode_iter))

            for k in interval_losses:
                interval_losses[k] += step_losses[k]
            interval_count += 1

            if step % val_every != 0 and step != total_outer_steps:
                continue

            train_metrics = {k: v / interval_count for k, v in interval_losses.items()}
            val_metrics   = self._run_validation(val_episode_loader, max_episodes=val_episodes)
            lr = self.optimizer.param_groups[0]["lr"]

            self._save_checkpoint("last.ckpt", step=step, val_loss=val_metrics["loss_forecast"])

            _periodic = set(self.config.get("periodic_ckpt_steps", []))
            if step in _periodic:
                self._save_checkpoint(
                    f"step_{step:04d}.ckpt", step=step, val_loss=val_metrics["loss_forecast"]
                )

            improved = val_metrics["loss_forecast"] < self.best_val_loss
            if improved:
                self.best_val_loss = val_metrics["loss_forecast"]
                self.best_step = step
                self.best_epoch = step
                best_checkpoint_path = self._save_checkpoint(
                    "best.ckpt", step=step, val_loss=val_metrics["loss_forecast"]
                )
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1

            W_abs_snap = self.model.graph_head.get_W_abs().detach()
            z_global_mean_norm = self._z_global_mean_norm()

            # WZX per-column L2 norms + pairwise (signed) cosine similarity.
            # Used to decide "columns specialising" (low off-diag cosine, uneven L2)
            # vs "columns shrinking uniformly" (similar L2, high off-diag cosine).
            wzx_col_l2_list: list[float] = []
            wzx_cos_abs_mean = 0.0
            wzx_cos_abs_max = 0.0
            q_val = self.model.q_latent
            if q_val > 0:
                with torch.no_grad():
                    W_zx_raw = self.model.graph_head.W[
                        :self.model.num_obs, self.model.num_obs:
                    ].detach()
                    col_l2 = W_zx_raw.norm(p=2, dim=0)                    # [q]
                    wzx_col_l2_list = [float(v) for v in col_l2.cpu()]
                    if q_val > 1:
                        Wn = W_zx_raw / col_l2.clamp_min(1e-12).unsqueeze(0)
                        G = Wn.transpose(0, 1) @ Wn                       # [q, q]
                        off_mask = ~torch.eye(q_val, dtype=torch.bool, device=G.device)
                        off = G[off_mask].abs()
                        wzx_cos_abs_mean = float(off.mean().cpu())
                        wzx_cos_abs_max  = float(off.max().cpu())

            record = {
                "outer_step":          step,
                "epoch":               step,
                "train_loss":          train_metrics["loss_total"],
                "train_forecast_loss": train_metrics["loss_forecast"],
                "train_rw_loss":       train_metrics["loss_rw"],
                "train_rz_loss":       train_metrics["loss_rz"],
                "train_wzx_col_loss":  train_metrics["loss_wzx_col"],
                "train_wzx_sparse_loss": train_metrics["loss_wzx_sparse"],
                "train_wzx_div_loss":  train_metrics["loss_wzx_div"],
                "val_loss":            val_metrics["loss_total"],
                "val_forecast_loss":   val_metrics["loss_forecast"],
                "val_rw_loss":         val_metrics["loss_rw"],
                "val_rz_loss":         val_metrics["loss_rz"],
                "val_wzx_col_loss":    val_metrics["loss_wzx_col"],
                "val_wzx_sparse_loss": val_metrics["loss_wzx_sparse"],
                "val_wzx_div_loss":    val_metrics["loss_wzx_div"],
                "val_episodes_used":   val_metrics["num_episodes"],
                "learning_rate":       lr,
                "best_step":           self.best_step,
                "best_epoch":          self.best_step,
                "grad_W_norm":         self._param_grad_norm(self.model.graph_head.W),
                "w_abs_max":           float(W_abs_snap.max()),
                "w_abs_mean":          float(W_abs_snap.mean()),
                "w_abs_std":           float(W_abs_snap.std()),
                "z_global_mean_norm":  z_global_mean_norm,
                "wzx_col_l2":          wzx_col_l2_list,
                "wzx_cos_abs_mean":    wzx_cos_abs_mean,
                "wzx_cos_abs_max":     wzx_cos_abs_max,
            }
            self.history.on_epoch_end(record)

            self.logger.info(
                "Step %d/%d | train %.6f (fc=%.4f rz=%.4f) | val %.6f | best %d | z_global_norm=%.4f",
                step, total_outer_steps,
                train_metrics["loss_total"],
                train_metrics["loss_forecast"],
                train_metrics["loss_rz"],
                val_metrics["loss_total"],
                self.best_step,
                z_global_mean_norm,
            )

            interval_losses = {k: 0.0 for k in interval_losses}
            interval_count = 0

            if patience > 0 and checks_without_improvement >= patience:
                self.logger.info(
                    "Early stopping at step %d (%d checks without improvement).",
                    step, patience,
                )
                break

        self.training_runtime_seconds = time.time() - _train_start
        self.history.save(self.output_dir)
        return best_checkpoint_path

    # ------------------------------------------------------------------
    # Graph extraction and prediction
    # ------------------------------------------------------------------

    def extract_graph(
        self,
        dataloader=None,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract the full [N_full x N_full] graph in source-target convention.

        Block layout (source-target, A[src, tgt]):
          [:N_obs, :N_obs]   XX  obs→obs
          [N_obs:, :N_obs]   ZX  latent→obs
          [:N_obs, N_obs:]   OX  obs→latent = 0 (forbidden, structural constraint)
          [N_obs:, N_obs:]   ZZ  latent→latent
        """
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        self.model.eval()

        with torch.no_grad():
            W_strength_ti = self.model.graph_head.get_edge_strength().cpu()  # [N_full, N_full] target-source
            W_abs    = W_strength_ti.T                                        # [N_full, N_full] source-target
            W_masked = self.model.graph_head.get_W_masked().cpu().T

            threshold  = float(self.config["graph_threshold"])
            top_k      = self.config.get("top_k")
            allow_self = bool(self.config.get("allow_self_loops", False))

            if top_k is not None:
                flat = W_abs.reshape(-1)
                k = min(int(top_k), flat.numel())
                cutoff = torch.topk(flat, k).values[-1]
                binary_graph = (W_abs >= cutoff).float()
            else:
                binary_graph = (W_abs > threshold).float()

            if not allow_self:
                diag = torch.eye(W_abs.shape[0], dtype=torch.bool)
                binary_graph[diag] = 0.0

            N_obs = self.model.num_obs
            binary_graph[:N_obs, N_obs:] = 0.0  # enforce obs→latent block = 0

        return {
            "W_masked":      W_masked.numpy(),
            "W_abs":         W_abs.numpy(),
            "W_norms":       W_abs.numpy(),
            "W_eff":         W_masked.numpy(),
            "sigmoid_score": W_abs.numpy(),
            "binary_graph":  binary_graph.numpy(),
        }

    def predict(
        self,
        dataloader,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, np.ndarray]:
        """Predict all observed variables (no per-task adaptation). Returns y_pred/y_true.

        Note: this non-adapted path uses z=0 for test sequences (since they have no
        entry in z_global).  For adapted predictions, use the inner-loop path.
        """
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        self.model.eval()

        predictions, targets = [], []
        with torch.no_grad():
            for batch in dataloader:
                batch  = self._move_batch_to_device(batch)
                y_pred = self.model.forward_all_tasks(batch["x_hist"])
                predictions.append(y_pred.detach().cpu().numpy())
                targets.append(batch["y_next"].detach().cpu().numpy())

        if not predictions:
            raise ValueError("Empty dataloader during prediction.")

        return {
            "y_pred": np.concatenate(predictions, axis=0),
            "y_true": np.concatenate(targets, axis=0),
        }


def train_latent_scdm(
    model,
    train_loader,
    val_episode_loader,
    config,
    device,
    output_dir,
    logger,
):
    """Construct a latent Trainer, fit the model, return the best checkpoint path."""
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        output_dir=output_dir,
        logger=logger,
    )
    best_checkpoint_path = trainer.fit(train_loader, val_episode_loader)
    return trainer, best_checkpoint_path
