"""SCDM episodic meta-training (first-order MAML over target-variable tasks).

Training budget is defined by ``total_outer_steps`` (not epochs).
Validation runs every ``val_every_outer_steps`` using a capped number of
episodes. Early stopping halts training when the validation forecast loss does
not improve for ``patience`` consecutive validation checks.

For each outer iteration (one episode = one support/query temporal window pair):

  1. Sample a target-task batch of size ``tasks_per_outer_step`` from the N
     target variables (see :meth:`Trainer._sample_task_indices`).
  2. For each sampled target task i:
     a. Clone the shared theta -> theta_i, clone the shared row W[i, :] -> w_i.
        These are TEMPORARY copies; the persistent shared parameters are never
        modified in place by the inner loop.
     b. Inner loop (K steps, ``create_graph=False`` -> first order):
          L_sup_i = MSE(y_pred, y_sup_i) + lambda_in * reg(w_i)
          theta_i <- theta_i - alpha_theta * dL_sup_i/dtheta_i
          w_i     <- w_i     - alpha_w     * dL_sup_i/dw_i     (alpha_w < alpha_theta)
     c. Query loss with the adapted parameters:
          L_qry_i = MSE(y_pred, y_qry_i) + lambda_out * reg(w_i')
     d. Collect first-order meta-gradients:
          g_theta_i = dL_qry_i / dtheta_i'
          g_w_i     = dL_qry_i / dw_i'
  3. Average the meta-gradients over the sampled tasks and write them onto the
     persistent shared parameters:
          theta.grad        = mean_i(g_theta_i)
          W.grad[i, :]      = g_w_i / num_tasks
  4. Clip the global gradient norm, then take one Adam step on (theta, W).

Validation reuses steps 2a-2c (adaptation on the support window, evaluation on
the query window) but skips gradient collection and the outer update. The
adapted query forecast loss is the checkpoint-selection criterion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam

from scdm.common.callbacks import MetricHistoryCallback
from scdm.common.losses import compute_forecasting_loss
from scdm.common.utils.io import ensure_dir


class Trainer:
    """First-order meta-trainer over target-variable tasks."""

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

        self.inner_loop_steps = int(config.get("inner_loop_steps", 3))
        self.inner_loop_lr_theta = float(config.get("inner_loop_lr", 0.01))
        self.inner_loop_lr_w = float(config.get("inner_loop_lr_w", 0.001))
        self.lambda_in = float(config.get("lambda_in", 1e-4))
        self.lambda_out = float(config.get("lambda_out", 1e-4))

        self.tasks_per_outer_step = int(config.get("tasks_per_outer_step", 0))
        self.target_sampling_mode = str(
            config.get("target_sampling_mode", "uniform_random")
        ).lower()
        if self.target_sampling_mode not in {"uniform_random", "coverage_aware"}:
            raise ValueError(
                f"Unsupported target_sampling_mode: {self.target_sampling_mode}"
            )
        self.val_tasks_per_episode = int(config.get("val_tasks_per_episode") or 0)
        self._target_sample_counts: np.ndarray | None = None
        self._rng = np.random.default_rng(int(config.get("seed", 0)))

        # The outer optimizer covers ALL persistent shared parameters (theta + W).
        self.optimizer = Adam(
            self.model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )

        meta_mode = str(config.get("meta_mode", "")).lower()
        if meta_mode == "full":
            self.training_mode = "fomaml"
        elif meta_mode == "no_meta":
            self.training_mode = "no_meta"
        else:
            self.training_mode = str(config.get("training_mode", "fomaml")).lower()
        if self.training_mode not in {"fomaml", "no_meta", "joint_supervised"}:
            raise ValueError(f"Unsupported training_mode: {self.training_mode}")

        self.best_val_loss = float("inf")
        self.best_step = 0
        self.current_step = 0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    @staticmethod
    def _param_grad_norm(param: torch.nn.Parameter) -> float:
        if param.grad is None:
            return 0.0
        return float(param.grad.detach().norm(2).cpu())

    def _sample_task_indices(
        self, num_nodes: int, requested: int, *, update_counts: bool
    ) -> list[int]:
        """Sub-sample the target-task batch for one outer step.

        ``requested <= 0`` or ``requested >= num_nodes`` means "use every target
        variable". ``coverage_aware`` sampling picks the least-frequently used
        targets first (ties broken by jitter), which keeps coverage uniform when
        N is large and only a small task batch fits in one step.
        """
        if requested <= 0 or requested >= num_nodes:
            indices = list(range(num_nodes))
            if update_counts and self.target_sampling_mode == "coverage_aware":
                if (
                    self._target_sample_counts is None
                    or self._target_sample_counts.shape[0] != num_nodes
                ):
                    self._target_sample_counts = np.zeros(num_nodes, dtype=np.int64)
                self._target_sample_counts += 1
            return indices

        if update_counts and self.target_sampling_mode == "coverage_aware":
            if (
                self._target_sample_counts is None
                or self._target_sample_counts.shape[0] != num_nodes
            ):
                self._target_sample_counts = np.zeros(num_nodes, dtype=np.int64)
            jitter = self._rng.random(num_nodes) * 1e-6
            order = np.lexsort((jitter, self._target_sample_counts))
            selected = order[:requested].astype(int).tolist()
            self._target_sample_counts[selected] += 1
            return selected

        return self._rng.choice(num_nodes, size=requested, replace=False).astype(int).tolist()

    # ------------------------------------------------------------------
    # Inner loop: temporary adaptation of theta_i and w_i on the support window
    # ------------------------------------------------------------------

    def _inner_loop(
        self,
        x_support: torch.Tensor,
        y_support: torch.Tensor,
        target_idx: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """First-order inner loop for target task ``target_idx``.

        Adapts theta_i (a full clone of the shared predictor) and, when
        ``use_structural_w=True``, w_i = W[target_idx, :]. Both are leaf tensors
        and ``create_graph=False`` throughout, so no second-order terms are kept.

        Returns:
            (theta_i', w_i') -- adapted leaf tensors with requires_grad=True.
        """
        theta_i = self.model.get_theta_params()
        w_trainable = self.model.graph_head.W.requires_grad
        w_i = self.model.graph_head.get_W_masked()[target_idx, :].clone().detach()
        if w_trainable:
            w_i = w_i.requires_grad_(True)

        for _step in range(self.inner_loop_steps):
            y_pred = self.model.forward_task(x_support, target_idx, theta_i, w_i)
            y_true_i = y_support[:, target_idx]

            if w_trainable:
                reg_row = self.model.graph_head.get_row_reg(target_idx, w_i)
                loss = compute_forecasting_loss(y_pred, y_true_i) + self.lambda_in * reg_row
                all_params = list(theta_i.values()) + [w_i]
                grads = torch.autograd.grad(loss, all_params, create_graph=False)
                theta_grads = grads[: len(theta_i)]
                w_grad = grads[-1]
                w_i = (w_i - self.inner_loop_lr_w * w_grad).detach().requires_grad_(True)
            else:
                loss = compute_forecasting_loss(y_pred, y_true_i)
                all_params = list(theta_i.values())
                grads = torch.autograd.grad(loss, all_params, create_graph=False)
                theta_grads = grads

            new_theta = {}
            for (name, param), grad in zip(theta_i.items(), theta_grads):
                new_theta[name] = (
                    (param - self.inner_loop_lr_theta * grad).detach().requires_grad_(True)
                )
            theta_i = new_theta

        return theta_i, w_i

    # ------------------------------------------------------------------
    # Outer step: first-order meta-update of the persistent shared parameters
    # ------------------------------------------------------------------

    def _outer_step(self, episode: dict[str, torch.Tensor]) -> dict[str, float]:
        """One first-order meta-update on a single episode."""
        self.model.train()
        episode = self._move_batch_to_device(episode)
        x_support = episode["support_hist"].squeeze(0)    # [S, lag, N]
        y_support = episode["support_target"].squeeze(0)  # [S, N]
        x_query = episode["query_hist"].squeeze(0)        # [Q, lag, N]
        y_query = episode["query_target"].squeeze(0)      # [Q, N]

        N = x_support.shape[2]

        task_indices = self._sample_task_indices(
            num_nodes=N,
            requested=self.tasks_per_outer_step,
            update_counts=True,
        )
        num_tasks = len(task_indices)

        # ---- Phase 1: inner-loop adaptation for every sampled target task ----
        adapted: dict[int, tuple[dict, torch.Tensor]] = {}
        for i in task_indices:
            theta_i, w_i = self._inner_loop(x_support, y_support, i)
            adapted[i] = (theta_i, w_i)

        # ---- Phase 2: query loss + first-order meta-gradient collection ----
        w_trainable = self.model.graph_head.W.requires_grad
        theta_grad_accum: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if not name.startswith("graph_head.")
        }
        if w_trainable:
            W_grad_accum = torch.zeros_like(self.model.graph_head.W)

        total_query_loss = 0.0
        total_l1 = 0.0

        for i in task_indices:
            theta_i, w_i = adapted[i]

            y_pred = self.model.forward_task(x_query, i, theta_i, w_i)
            y_true_i = y_query[:, i]
            mse_i = compute_forecasting_loss(y_pred, y_true_i)

            if w_trainable:
                reg_row = self.model.graph_head.get_row_reg(i, w_i)
                L_qry_i = mse_i + self.lambda_out * reg_row
                all_params = list(theta_i.values()) + [w_i]
                grads = torch.autograd.grad(L_qry_i, all_params, create_graph=False)
                theta_grads = grads[: len(theta_i)]
                w_grad = grads[-1]
                W_grad_accum[i, :] = W_grad_accum[i, :] + w_grad
                total_l1 += float(reg_row.detach().cpu())
            else:
                L_qry_i = mse_i
                all_params = list(theta_i.values())
                grads = torch.autograd.grad(L_qry_i, all_params, create_graph=False)
                theta_grads = grads

            for (tname, _), grad in zip(theta_i.items(), theta_grads):
                model_key = _theta_key_to_model_key(tname)
                if model_key in theta_grad_accum:
                    theta_grad_accum[model_key] = theta_grad_accum[model_key] + grad

            total_query_loss += float(mse_i.detach().cpu())

        # ---- Phase 3: write .grad onto the shared parameters, then step ----
        self.optimizer.zero_grad(set_to_none=True)

        for name, param in self.model.named_parameters():
            if name in theta_grad_accum:
                param.grad = theta_grad_accum[name] / num_tasks

        if w_trainable:
            self.model.graph_head.W.grad = W_grad_accum / num_tasks

        grad_clip = self.config.get("grad_clip_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

        self.optimizer.step()

        avg_query_loss = total_query_loss / num_tasks
        avg_l1 = total_l1 / num_tasks

        return {
            "loss_total": avg_query_loss + self.lambda_out * avg_l1,
            "loss_forecast": avg_query_loss,
            "loss_l1": avg_l1,
        }

    # ------------------------------------------------------------------
    # Ablation outer steps (no meta-adaptation)
    # ------------------------------------------------------------------

    def _outer_step_joint_supervised(self, episode: dict[str, torch.Tensor]) -> dict[str, float]:
        """Ablation: direct supervised step, no inner loop, no support/query split.

        Uses the shared theta and W directly and concatenates the support and
        query windows into one training batch.
        """
        self.model.train()
        episode = self._move_batch_to_device(episode)
        x_support = episode["support_hist"].squeeze(0)
        y_support = episode["support_target"].squeeze(0)
        x_query = episode["query_hist"].squeeze(0)
        y_query = episode["query_target"].squeeze(0)

        x_all = torch.cat([x_support, x_query], dim=0)   # [S+Q, lag, N]
        y_all = torch.cat([y_support, y_query], dim=0)   # [S+Q, N]
        N = x_all.shape[2]

        task_indices = self._sample_task_indices(
            num_nodes=N,
            requested=self.tasks_per_outer_step,
            update_counts=True,
        )
        num_tasks = len(task_indices)

        self.optimizer.zero_grad(set_to_none=True)
        total_mse = 0.0
        total_l1 = 0.0
        grad_clip = self.config.get("grad_clip_norm", 1.0)

        for i in task_indices:
            y_pred = self.model.forward_task(x_all, i)
            mse_i = compute_forecasting_loss(y_pred, y_all[:, i])
            reg_i = self.model.graph_head.get_row_reg(i)
            loss_i = (mse_i + self.lambda_out * reg_i) / num_tasks
            loss_i.backward()
            total_mse += float(mse_i.detach().cpu())
            total_l1 += float(reg_i.detach().cpu())

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        avg_mse = total_mse / num_tasks
        avg_l1 = total_l1 / num_tasks
        return {
            "loss_total": avg_mse + self.lambda_out * avg_l1,
            "loss_forecast": avg_mse,
            "loss_l1": avg_l1,
        }

    def _outer_step_no_meta(self, episode: dict[str, torch.Tensor]) -> dict[str, float]:
        """Ablation (K=0): query loss with shared theta and W, no support adaptation."""
        self.model.train()
        episode = self._move_batch_to_device(episode)
        x_query = episode["query_hist"].squeeze(0)
        y_query = episode["query_target"].squeeze(0)

        N = x_query.shape[2]
        task_indices = self._sample_task_indices(
            num_nodes=N,
            requested=self.tasks_per_outer_step,
            update_counts=True,
        )
        num_tasks = len(task_indices)

        self.optimizer.zero_grad(set_to_none=True)
        total_mse = 0.0
        total_l1 = 0.0
        grad_clip = self.config.get("grad_clip_norm", 1.0)

        for i in task_indices:
            y_pred = self.model.forward_task(x_query, i)
            mse_i = compute_forecasting_loss(y_pred, y_query[:, i])
            reg_i = self.model.graph_head.get_row_reg(i)
            loss_i = (mse_i + self.lambda_out * reg_i) / num_tasks
            loss_i.backward()
            total_mse += float(mse_i.detach().cpu())
            total_l1 += float(reg_i.detach().cpu())

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        avg_mse = total_mse / num_tasks
        avg_l1 = total_l1 / num_tasks
        return {
            "loss_total": avg_mse + self.lambda_out * avg_l1,
            "loss_forecast": avg_mse,
            "loss_l1": avg_l1,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(self, val_episode_loader, max_episodes: int = 0) -> dict[str, float]:
        """Validation with per-task inner-loop adaptation, matching training.

        Args:
            val_episode_loader: DataLoader yielding episodes.
            max_episodes: cap on episodes to evaluate (0 = use all).
        """
        self.model.eval()

        total_query_loss = 0.0
        total_l1 = 0.0
        num_episodes = 0

        for episode in val_episode_loader:
            if max_episodes > 0 and num_episodes >= max_episodes:
                break

            episode = self._move_batch_to_device(episode)
            x_support = episode["support_hist"].squeeze(0)
            y_support = episode["support_target"].squeeze(0)
            x_query = episode["query_hist"].squeeze(0)
            y_query = episode["query_target"].squeeze(0)

            N = x_support.shape[2]
            if 0 < self.val_tasks_per_episode < N:
                task_indices = self._sample_task_indices(
                    num_nodes=N,
                    requested=self.val_tasks_per_episode,
                    update_counts=False,
                )
            else:
                task_indices = list(range(N))
            num_tasks = len(task_indices)

            episode_query_loss = 0.0
            episode_l1 = 0.0

            for i in task_indices:
                theta_i, w_i = self._inner_loop(x_support, y_support, i)

                with torch.no_grad():
                    y_pred = self.model.forward_task(x_query, i, theta_i, w_i)
                    y_true_i = y_query[:, i]
                    reg_val = self.model.graph_head.get_row_reg(i, w_i)
                    mse_i = compute_forecasting_loss(y_pred, y_true_i)

                    episode_query_loss += float(mse_i.cpu())
                    episode_l1 += float(reg_val.cpu())

            total_query_loss += episode_query_loss / num_tasks
            total_l1 += episode_l1 / num_tasks
            num_episodes += 1

        if num_episodes == 0:
            raise ValueError("Empty validation episode loader.")

        avg_query_loss = total_query_loss / num_episodes
        avg_l1 = total_l1 / num_episodes
        return {
            "loss_total": avg_query_loss + self.lambda_out * avg_l1,
            "loss_forecast": avg_query_loss,
            "loss_l1": avg_l1,
            "num_episodes": num_episodes,
        }

    def _run_validation_joint(self, val_episode_loader, max_episodes: int = 0) -> dict[str, float]:
        """Validation without inner-loop adaptation (ablation modes)."""
        self.model.eval()
        total_query_loss = 0.0
        total_l1 = 0.0
        num_episodes = 0

        for episode in val_episode_loader:
            if max_episodes > 0 and num_episodes >= max_episodes:
                break
            episode = self._move_batch_to_device(episode)
            x_query = episode["query_hist"].squeeze(0)
            y_query = episode["query_target"].squeeze(0)
            N = x_query.shape[2]

            if 0 < self.val_tasks_per_episode < N:
                task_indices = self._sample_task_indices(
                    num_nodes=N,
                    requested=self.val_tasks_per_episode,
                    update_counts=False,
                )
            else:
                task_indices = list(range(N))
            num_tasks = len(task_indices)

            episode_loss = 0.0
            episode_l1 = 0.0
            with torch.no_grad():
                for i in task_indices:
                    y_pred = self.model.forward_task(x_query, i)
                    mse_i = compute_forecasting_loss(y_pred, y_query[:, i])
                    reg_i = self.model.graph_head.get_row_reg(i)
                    episode_loss += float(mse_i.cpu())
                    episode_l1 += float(reg_i.cpu())

            total_query_loss += episode_loss / num_tasks
            total_l1 += episode_l1 / num_tasks
            num_episodes += 1

        if num_episodes == 0:
            raise ValueError("Empty validation episode loader.")

        avg_query_loss = total_query_loss / num_episodes
        avg_l1 = total_l1 / num_episodes
        return {
            "loss_total": avg_query_loss + self.lambda_out * avg_l1,
            "loss_forecast": avg_query_loss,
            "loss_l1": avg_l1,
            "num_episodes": num_episodes,
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_payload(self, step: int, val_loss: float) -> dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "step": step,
            "validation_score": val_loss,
        }

    def _save_checkpoint(self, filename: str, step: int, val_loss: float) -> Path:
        path = self.output_dir / filename
        torch.save(self._checkpoint_payload(step=step, val_loss=val_loss), path)
        return path

    def load_checkpoint(
        self, checkpoint_path: str | Path, load_optimizer: bool = False
    ) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    # ------------------------------------------------------------------
    # Training loop (step-based budget)
    # ------------------------------------------------------------------

    def fit(self, train_loader, val_episode_loader) -> Path:
        """Run meta-training and return the path of the best checkpoint."""
        total_outer_steps = int(self.config.get("total_outer_steps", 500))
        val_every = int(self.config.get("val_every_outer_steps", 50))
        val_episodes = int(self.config.get("val_episodes", 50))
        patience = int(self.config.get("patience", 0))  # 0 = no early stopping

        best_checkpoint_path = self.output_dir / "best.ckpt"

        interval_losses = {"loss_total": 0.0, "loss_forecast": 0.0, "loss_l1": 0.0}
        interval_count = 0
        checks_without_improvement = 0

        episode_iter = _infinite_episode_iter(train_loader)

        self.logger.info(
            "Training: total_outer_steps=%d, val_every=%d, val_episodes=%d, "
            "tasks_per_outer_step=%s, target_sampling_mode=%s, val_tasks_per_episode=%s, patience=%d",
            total_outer_steps, val_every, val_episodes,
            self.tasks_per_outer_step or "all",
            self.target_sampling_mode,
            self.val_tasks_per_episode or "all",
            patience,
        )

        for step in range(1, total_outer_steps + 1):
            self.current_step = step

            episode = next(episode_iter)
            if self.training_mode == "joint_supervised":
                step_losses = self._outer_step_joint_supervised(episode)
            elif self.training_mode == "no_meta":
                step_losses = self._outer_step_no_meta(episode)
            else:
                step_losses = self._outer_step(episode)

            for key in interval_losses:
                interval_losses[key] += step_losses[key]
            interval_count += 1

            is_val_step = (step % val_every == 0) or (step == total_outer_steps)
            if not is_val_step:
                continue

            train_metrics = {key: val / interval_count for key, val in interval_losses.items()}
            if self.training_mode in {"joint_supervised", "no_meta"}:
                val_metrics = self._run_validation_joint(
                    val_episode_loader, max_episodes=val_episodes
                )
            else:
                val_metrics = self._run_validation(
                    val_episode_loader, max_episodes=val_episodes
                )
            lr = self.optimizer.param_groups[0]["lr"]

            self._save_checkpoint("last.ckpt", step=step, val_loss=val_metrics["loss_forecast"])

            improved = val_metrics["loss_forecast"] < self.best_val_loss
            if improved:
                self.best_val_loss = val_metrics["loss_forecast"]
                self.best_step = step
                best_checkpoint_path = self._save_checkpoint(
                    "best.ckpt", step=step, val_loss=val_metrics["loss_forecast"]
                )
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1

            W_abs_snap = self.model.graph_head.get_W_abs().detach()
            record = {
                "outer_step": step,
                "train_loss": train_metrics["loss_total"],
                "train_forecast_loss": train_metrics["loss_forecast"],
                "train_l1_loss": train_metrics["loss_l1"],
                "val_loss": val_metrics["loss_total"],
                "val_forecast_loss": val_metrics["loss_forecast"],
                "val_l1_loss": val_metrics["loss_l1"],
                "val_episodes_used": val_metrics["num_episodes"],
                "learning_rate": lr,
                "best_step": self.best_step,
                "grad_W_norm": self._param_grad_norm(self.model.graph_head.W),
                "w_abs_max": float(W_abs_snap.max()),
                "w_abs_mean": float(W_abs_snap.mean()),
                "w_abs_std": float(W_abs_snap.std()),
            }
            self.history.on_epoch_end(record)

            self.logger.info(
                "Step %d/%d | train %.6f | val %.6f (ep=%d) | best_step %d",
                step, total_outer_steps,
                train_metrics["loss_total"],
                val_metrics["loss_total"],
                val_metrics["num_episodes"],
                self.best_step,
            )

            interval_losses = {"loss_total": 0.0, "loss_forecast": 0.0, "loss_l1": 0.0}
            interval_count = 0

            if patience > 0 and checks_without_improvement >= patience:
                self.logger.info(
                    "Early stopping at step %d: no val improvement for %d checks.",
                    step, patience,
                )
                break

        self.history.save(self.output_dir)
        return best_checkpoint_path

    # ------------------------------------------------------------------
    # Forecasting readout
    # ------------------------------------------------------------------

    def predict(
        self,
        dataloader,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, np.ndarray]:
        """Predict all variables with the shared theta (no per-task adaptation)."""
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        self.model.eval()

        predictions = []
        targets = []
        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                y_pred = self.model.forward_all_tasks(batch["x_hist"])
                predictions.append(y_pred.detach().cpu().numpy())
                targets.append(batch["y_next"].detach().cpu().numpy())

        if not predictions:
            raise ValueError("Empty dataloader during prediction.")

        return {
            "y_pred": np.concatenate(predictions, axis=0),
            "y_true": np.concatenate(targets, axis=0),
        }


def train_scdm(
    model,
    train_loader,
    val_episode_loader,
    config: dict[str, Any],
    device,
    output_dir: str | Path,
    logger,
) -> tuple[Trainer, Path]:
    """Construct a trainer, fit the model, and return the best checkpoint path."""

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        output_dir=output_dir,
        logger=logger,
    )
    best_checkpoint_path = trainer.fit(train_loader, val_episode_loader)
    return trainer, best_checkpoint_path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_THETA_KEY_MAP = {
    "l1.w": "linear1.weight",
    "l1.b": "linear1.bias",
    "l2.w": "linear2.weight",
    "l2.b": "linear2.bias",
    "l3.w": "linear3.weight",
    "l3.b": "linear3.bias",
}


def _theta_key_to_model_key(tname: str) -> str:
    return _THETA_KEY_MAP.get(tname, tname)


def _infinite_episode_iter(dataloader):
    """Yield episodes forever by cycling through the dataloader."""
    while True:
        for episode in dataloader:
            yield episode
