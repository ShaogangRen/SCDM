"""Latent meta causal model — explicit-latent design per Chapter 3 draft.

Architecture (Θ = {θ, W, Z})
==============================
θ:   shared nonlinear predictor MLP  (linear1, linear2, linear3)
W:   structural matrix [N_full × N_full], block-constrained (see below)
Z:   explicit latent trajectories, one [T_k × q] parameter per training sequence
     (Z = {z_t}_{t=1}^T per draft Ch. 3; access by sequence index + timestep)

Structural matrix W (target-source convention, W[tgt, src]):
  W[:N_obs, :N_obs]   = WXX   (obs → obs)
  W[:N_obs, N_obs:]   = WZX   (latent → obs, i.e. W[obs_tgt, latent_src])
  W[N_obs:, :N_obs]   = 0     (obs → latent, FORBIDDEN: hidden vars have no observed parents)
  W[N_obs:, N_obs:]   = WZZ   (latent → latent, temporal dynamics)

Latent trajectory Z:
  Z = {z_t}_{t=1}^T is an explicit parameter set aligned to the global timeline.
  Each training sequence k has z_global[k] ∈ R^{T_k × q}.
  Episode at (seq_idx=k, episode_start=s): z_init_ep = z_global[k][s] ∈ R^q.
  WZZ rollout within the episode window: z_t = WZZ @ z_{t-1} for t = 1..L-1.
  The innovation term ξ_t is omitted (ξ_t = 0 within episode windows); any
  deviation from WZZ linearity in the stored z_global is absorbed into z_global
  optimisation across different episode starts.

Forward pass for observed target i (episode-specific):
  1. Weighted observed history:  H̃[t,j,ℓ] = W[i,j] * x[t-ℓ,j]   (WXX gating)
  2. Gated latent contribution:  z̃[t,k]   = W[i, N_obs+k] * z_t[k] (WZX gating)
  3. Concatenate and predict:    ŷ[t,i]   = f_θ(vec(H̃[t]), z̃[t])

Objective (draft Eq. latent_obj):
  L = L_obs + λ_W R_W + λ_z R_z
  L_obs = MSE(ŷ, y)            one-step observed forecasting
  R_W   = ||WXX||_1 + ||WZX||_1 + ||WZZ||_1  (L1 on admissible blocks)
  R_z   = (1/T) Σ_t ||z_t||_2                (latent magnitude, L2 not L2-sq)

No auxiliary encoder.
No hidden-state reconstruction loss.

Meta-learning: first-order FOMAML.
Per-task inner-loop adaptation of (θ_i, w_i = W[i,:], W_zz_i = WZZ copy, z_ep).
z_ep is initialised from z_global[seq_idx][episode_start] for training episodes,
and from zeros for validation/test episodes.

theta_params keys (inner-loop leaf tensors, MLP only — no z):
  "l1.w"   linear1.weight    "l1.b"   linear1.bias
  "l2.w"   linear2.weight    "l2.b"   linear2.bias
  "l3.w"   linear3.weight    "l3.b"   linear3.bias
z_ep is passed as a separate leaf tensor (not part of theta_params).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .graph_head import build_latent_graph_head


class LatentCausalModel(nn.Module):
    """Explicit-latent meta-learning model — no encoder.

    Shared meta-parameters (Θ):
      predictor MLP:    linear1, linear2, linear3
      structural matrix W [N_full, N_full]:  via graph_head (block-constrained)
      global latent trajectories: z_global[k] ∈ R^{T_k × q} for each training seq k

    Latent trajectory access:
      Training episode at (seq_idx=k, episode_start=s):
        z_init_ep = z_global[k][s]   — [q] vector
      WZZ rollout within episode window of length L:
        z_seq = functional_z_seq(z_init_ep, W_ZZ, L)   — [L, q]

    For each observed task i:
      1. Gate observed lagged history by W[i, :N_obs].
      2. Gate latent sequence by W[i, N_obs:].
      3. Concatenate and predict with shared MLP.
    """

    def __init__(
        self,
        num_obs: int,
        lag: int,
        hidden_dim: int,
        node_hidden_dim: int,
        graph_threshold: float,
        allow_self_loops: bool,
        q_latent: int = 2,
        top_k: int | None = None,
        w_init_min: float = 0.0,
        w_init_max: float = 0.3,
        w_init_min_latent: float | None = None,
        w_init_max_latent: float | None = None,
        wzz_init_diag: float | None = None,
        z_init_scale: float = 0.0,
        reg_mode: str = "l1",
        use_wzz: bool = True,
        wzz_diagonal_only: bool = False,
        train_sequence_lengths: list[int] | None = None,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.num_obs = num_obs
        self.q_latent = q_latent
        self.num_nodes = num_obs + q_latent   # N_full (Trainer compatibility)
        self.lag = lag
        self.hidden_dim = hidden_dim
        self.node_hidden_dim = node_hidden_dim
        self.use_wzz = use_wzz
        self.wzz_diagonal_only = wzz_diagonal_only and use_wzz

        # Structural weight matrix [N_full x N_full] with block constraint.
        # LatentBlockGraphHead permanently zeros the obs→latent block.
        self.graph_head = build_latent_graph_head(
            num_obs=num_obs,
            q_latent=q_latent,
            graph_threshold=graph_threshold,
            allow_self_loops=allow_self_loops,
            top_k=top_k,
            w_init_min=w_init_min,
            w_init_max=w_init_max,
            w_init_min_latent=w_init_min_latent,
            w_init_max_latent=w_init_max_latent,
            wzz_init_diag=wzz_init_diag,
            reg_mode=reg_mode,
            use_wzz=use_wzz,
            wzz_diagonal_only=wzz_diagonal_only,
        )

        # Predictor MLP: (N_obs*lag + q_latent) → hidden_dim → node_hidden_dim → 1
        pred_in = num_obs * lag + q_latent
        self.linear1 = nn.Linear(pred_in, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, node_hidden_dim)
        self.linear3 = nn.Linear(node_hidden_dim, 1)

        # Global latent trajectories: Z[k] ∈ R^{T_k × q}, one per training sequence.
        # Z = {z_t}_{t=1}^T (draft Ch. 3): each z_t is an explicit free parameter
        # aligned to the global timeline.  z_global[k][s] gives the initial latent
        # state for any episode starting at absolute time s in sequence k.
        # Initialised to zero (z_init_scale=0) or small random values; jointly
        # optimised with θ and W via L_obs.
        if train_sequence_lengths:
            def _make_z(T_k):
                if z_init_scale > 0.0:
                    return nn.Parameter(torch.empty(T_k, q_latent).uniform_(-z_init_scale, z_init_scale))
                return nn.Parameter(torch.zeros(T_k, q_latent))
            self.z_global = nn.ParameterList([_make_z(T_k) for T_k in train_sequence_lengths])
        else:
            self.z_global = nn.ParameterList([])

        self.w_components = 1  # Trainer compatibility

    # ------------------------------------------------------------------
    # Theta utilities  (MLP only — z is episode-specific via z_global)
    # ------------------------------------------------------------------

    def get_theta_params(self) -> dict[str, torch.Tensor]:
        """Clone MLP params as independent leaf tensors for inner loop.

        z is NOT included here — it is handled episode-specifically:
          train: z_ep = z_global[seq_idx][episode_start]
          val:   z_ep = zeros
        """
        return {
            "l1.w":   self.linear1.weight.clone().detach().requires_grad_(True),
            "l1.b":   self.linear1.bias.clone().detach().requires_grad_(True),
            "l2.w":   self.linear2.weight.clone().detach().requires_grad_(True),
            "l2.b":   self.linear2.bias.clone().detach().requires_grad_(True),
            "l3.w":   self.linear3.weight.clone().detach().requires_grad_(True),
            "l3.b":   self.linear3.bias.clone().detach().requires_grad_(True),
        }

    def get_z_ep(self, seq_idx: int, episode_start: int) -> torch.Tensor:
        """Return z_global[seq_idx][episode_start] as a [q] tensor (non-leaf view).

        The caller clones and detaches this to create an inner-loop leaf tensor.
        """
        return self.z_global[seq_idx][episode_start]

    # ------------------------------------------------------------------
    # Static functional methods for FOMAML inner/outer loops
    # ------------------------------------------------------------------

    @staticmethod
    def functional_z_seq(
        z_init: torch.Tensor,
        W_zz: torch.Tensor | None,
        L: int,
    ) -> torch.Tensor:
        """Unroll latent dynamics for t = 1..L-1.

        Three modes depending on W_zz:
          None     → static confounder: z_t = z_init for all t  (use_wzz=False)
          [q]      → diagonal: z_t = W_zz * z_{t-1}  (elementwise; wzz_diagonal_only=True)
          [q, q]   → full matrix: z_t = W_zz @ z_{t-1}  (standard mode)

        Args:
            z_init: [q]              initial latent state (leaf tensor in inner loop)
            W_zz:   [q]|[q,q]|None  dynamics; 1-D means diagonal-only mode
            L:      int              episode length

        Returns:
            z_seq: [L, q]
        """
        if W_zz is None:
            return z_init.unsqueeze(0).expand(L, -1).contiguous()
        z_list = [z_init]
        if W_zz.dim() == 1:
            # Diagonal-only: z_t = diag_vec * z_{t-1}  (no cross-latent coupling)
            for _ in range(1, L):
                z_list.append(W_zz * z_list[-1])
        else:
            # Full WZZ matrix
            for _ in range(1, L):
                z_list.append(torch.mv(W_zz, z_list[-1]))
        return torch.stack(z_list, dim=0)

    @staticmethod
    def functional_forward(
        x_ep: torch.Tensor,
        z_seq: torch.Tensor,
        w_i: torch.Tensor,
        theta_params: dict[str, torch.Tensor],
        num_obs: int,
        q_latent: int,
    ) -> torch.Tensor:
        """Functional forward for a single task using cloned theta_params.

        Args:
            x_ep:         [L, lag, N_obs]   observed lagged history for L timesteps
            z_seq:        [L, q]            pre-computed latent sequence
            w_i:          [N_full]          gate vector for task i (leaf tensor)
            theta_params: cloned MLP leaf tensors (keys: l1.w, l1.b, ... l3.b)
            num_obs:      N_obs
            q_latent:     q

        Returns:
            [L] scalar one-step predictions for task i
        """
        L, lag, N_obs = x_ep.shape

        # Weighted observed history: H̃[t,j,ℓ] = W[i,j] * x[t,ℓ,j]
        w_obs = w_i[:num_obs]                                         # [N_obs]
        x_weighted = x_ep * w_obs.unsqueeze(0).unsqueeze(0)          # [L, lag, N_obs]
        x_flat = x_weighted.reshape(L, N_obs * lag)                    # [L, N_obs*lag]

        # Gated latent: z̃[t,k] = W[i, N_obs+k] * z_t[k]
        w_lat = w_i[num_obs:]                                         # [q]
        z_gated = z_seq * w_lat.unsqueeze(0)                          # [L, q]

        inp = torch.cat([x_flat, z_gated], dim=1)                     # [L, N_obs*lag + q]

        h = F.relu(F.linear(inp, theta_params["l1.w"], theta_params["l1.b"]))
        h = F.relu(F.linear(h, theta_params["l2.w"], theta_params["l2.b"]))
        return F.linear(h, theta_params["l3.w"], theta_params["l3.b"]).squeeze(-1)

    # ------------------------------------------------------------------
    # Non-functional forward helpers (eval / perturbation)
    # ------------------------------------------------------------------

    def forward_task(
        self,
        x_hist: torch.Tensor,
        target_idx: int,
    ) -> torch.Tensor:
        """Forward for a single observed task (no inner-loop adaptation).

        Used only for the non-adapted predict/eval path (e.g., computing
        forecast MSE on test sequences after training).  Since test sequences
        are not in z_global, z is initialised to zeros here.  For a fully
        adapted prediction, use the inner-loop path in the Trainer.

        Args:
            x_hist:      [L, lag, N_obs]
            target_idx:  int in [0, num_obs)

        Returns:
            [L] predictions
        """
        L = x_hist.shape[0]
        W_masked = self.graph_head.get_W_masked()
        w_i = W_masked[target_idx, :]
        if not self.use_wzz:
            W_zz = None
        elif self.wzz_diagonal_only:
            W_zz = W_masked[self.num_obs:, self.num_obs:].diag()  # [q] diagonal vector
        else:
            W_zz = W_masked[self.num_obs:, self.num_obs:]          # [q, q] full matrix

        z_zero = torch.zeros(self.q_latent, device=x_hist.device)
        z_seq = self.functional_z_seq(z_zero, W_zz, L)

        w_obs = w_i[:self.num_obs]
        w_lat = w_i[self.num_obs:]
        x_weighted = x_hist * w_obs.unsqueeze(0).unsqueeze(0)
        x_flat = x_weighted.reshape(L, self.num_obs * self.lag)
        z_gated = z_seq * w_lat.unsqueeze(0)
        inp = torch.cat([x_flat, z_gated], dim=1)
        h = F.relu(self.linear1(inp))
        h = F.relu(self.linear2(h))
        return self.linear3(h).squeeze(-1)

    def forward_all_tasks(
        self,
        x_hist: torch.Tensor,
    ) -> torch.Tensor:
        """Forward for all N_obs observed tasks. Returns [L, N_obs]."""
        return torch.stack(
            [self.forward_task(x_hist, i) for i in range(self.num_obs)],
            dim=1,
        )

    def forward(self, x_hist: torch.Tensor) -> dict[str, torch.Tensor]:
        """Standard forward for evaluation / perturbation scoring."""
        y_pred = self.forward_all_tasks(x_hist)
        graph_outputs = self.graph_head()
        return {
            "y_pred":        y_pred,
            "W_abs":         graph_outputs["W_abs"],
            "W_masked":      graph_outputs["W_masked"],
            "binary_graph":  graph_outputs["binary_graph"],
        }
