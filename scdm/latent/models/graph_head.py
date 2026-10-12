"""Graph head for explicit-latent causal model (v5 latent extension).

LatentBlockGraphHead
--------------------
Full [N_full x N_full] structural weight matrix with a hard block constraint:

  W_internal [target, source] block layout:
    [ W_xx [N_obs x N_obs]    W_zx_col [N_obs x q]   ]
    [ 0 (forbidden) [q x N_obs]  W_zz [q x q]        ]

The forbidden block W[N_obs:, :N_obs] is permanently zeroed (obs -> latent
edges are not allowed by design assumption).

After transposing to source-target convention (as in extract_graph):
  W_abs[src, tgt]:
    [:N_obs, :N_obs]  -- obs -> obs     (XX)
    [N_obs:, :N_obs]  -- latent -> obs  (ZX)
    [:N_obs, N_obs:]  -- obs -> latent  = 0  (forbidden)
    [N_obs:, N_obs:]  -- latent -> lat  (ZZ)

Row-level regularization is only applied for observed target rows (0..N_obs-1)
since only those tasks receive prediction losses.
"""

from __future__ import annotations

import torch
from torch import nn


class LatentBlockGraphHead(nn.Module):
    """Direct W parameterization [N_full x N_full] with block constraint.

    W[i, j] is the gate weight of source j on target i.
    The block W[N_obs:, :N_obs] (obs->latent) is permanently masked to 0.
    """

    def __init__(
        self,
        num_obs: int,
        q_latent: int,
        graph_threshold: float,
        allow_self_loops: bool,
        top_k: int | None = None,
        w_init_min: float = 0.0,
        w_init_max: float = 0.3,
        w_init_min_latent: float | None = None,
        w_init_max_latent: float | None = None,
        wzz_init_diag: float | None = None,
        reg_mode: str = "l1",
        use_wzz: bool = True,
        wzz_diagonal_only: bool = False,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.num_obs = num_obs
        self.q_latent = q_latent
        self.num_nodes = num_obs + q_latent  # N_full
        self.graph_threshold = graph_threshold
        self.allow_self_loops = allow_self_loops
        self.top_k = top_k
        self.reg_mode = reg_mode
        self.w_components = 1  # scalar W (K=1)

        N_full = num_obs + q_latent

        # Initialize W [N_full x N_full].
        # WXX block uses w_init_min/max; WZX and WZZ blocks use w_init_min/max_latent
        # (falls back to the same range if not specified).
        lat_min = w_init_min_latent if w_init_min_latent is not None else w_init_min
        lat_max = w_init_max_latent if w_init_max_latent is not None else w_init_max

        W_init = torch.empty(N_full, N_full).uniform_(w_init_min, w_init_max)
        # Latent-source columns (WZX and WZZ) get their own init range.
        if q_latent > 0:
            W_init[:, num_obs:] = torch.empty(N_full, q_latent).uniform_(lat_min, lat_max)
        # WZZ near-identity init: when wzz_init_diag is set, override the entire WZZ block.
        # Zero the off-diagonal entries and set the diagonal to wzz_init_diag.
        # This gives spectral radius exactly = wzz_init_diag at initialization.
        # Without this, WZZ off-diagonal entries from lat_min/lat_max (e.g. uniform(0.3, 0.8))
        # combined with a large diagonal would give spectral radius >> 1.0, causing z to explode.
        # WZX (obs-target rows, latent-source cols) is NOT affected — it keeps lat_min/lat_max.
        if wzz_init_diag is not None and q_latent > 0:
            W_init[num_obs:, num_obs:] = 0.0   # zero full WZZ block (obs-target block unaffected)
            wzz_diag_idx = torch.arange(q_latent)
            W_init[num_obs + wzz_diag_idx, num_obs + wzz_diag_idx] = wzz_init_diag
        # Apply block constraint upfront: forbidden block = 0
        W_init[num_obs:, :num_obs] = 0.0
        # Zero obs-node self-loops only.
        # NOTE: allow_self_loops=False applies only to obs→obs (WXX diagonal).
        # The WZZ diagonal (latent→same-latent) represents temporal autocorrelation
        # and must NOT be zeroed — it is essential for z_seq to persist across timesteps.
        if not allow_self_loops:
            obs_diag_idx = torch.arange(num_obs)
            W_init[obs_diag_idx, obs_diag_idx] = 0.0

        # When use_wzz=False, zero the WZZ block at init (no Z->Z dynamics).
        self.use_wzz = use_wzz
        self.wzz_diagonal_only = wzz_diagonal_only and use_wzz  # diagonal only makes sense when wzz is on
        if not use_wzz and q_latent > 0:
            W_init[num_obs:, num_obs:] = 0.0

        # When wzz_diagonal_only=True, zero off-diagonal entries at init.
        if wzz_diagonal_only and use_wzz and q_latent > 0:
            wzz_off_diag_mask = ~torch.eye(q_latent, dtype=torch.bool)
            W_init[num_obs:, num_obs:][wzz_off_diag_mask] = 0.0

        self.W = nn.Parameter(W_init)

        # Forbidden block mask: obs->latent (W[N_obs:, :N_obs] = 0)
        block_mask = torch.zeros(N_full, N_full, dtype=torch.bool)
        block_mask[num_obs:, :num_obs] = True
        self.register_buffer("_block_mask", block_mask)

        # Diagonal mask: obs-node self-loops only (NOT WZZ diagonal).
        # WZZ diagonal = latent→same-latent temporal dynamics; always allowed.
        diag_mask = torch.zeros(N_full, N_full, dtype=torch.bool)
        if not allow_self_loops:
            obs_diag_idx = torch.arange(num_obs)
            diag_mask[obs_diag_idx, obs_diag_idx] = True
        self.register_buffer("_diag_mask", diag_mask)

        # WZZ mask: zeros the entire latent-latent block when use_wzz=False.
        wzz_mask = torch.zeros(N_full, N_full, dtype=torch.bool)
        if not use_wzz and q_latent > 0:
            wzz_mask[num_obs:, num_obs:] = True
        self.register_buffer("_wzz_mask", wzz_mask)

        # WZZ off-diagonal mask: zeros off-diagonal WZZ entries when wzz_diagonal_only=True.
        # Ensures graph extraction / evaluation never see spurious cross-latent edges.
        wzz_offdiag_mask = torch.zeros(N_full, N_full, dtype=torch.bool)
        if wzz_diagonal_only and use_wzz and q_latent > 0:
            off_diag = ~torch.eye(q_latent, dtype=torch.bool)
            wzz_offdiag_mask[num_obs:, num_obs:] = off_diag
        self.register_buffer("_wzz_offdiag_mask", wzz_offdiag_mask)

    def get_W_masked(self) -> torch.Tensor:
        """Return W with block constraint and diagonal zeroed. Shape [N_full, N_full]."""
        W = self.W.masked_fill(self._block_mask, 0.0)
        if not self.allow_self_loops:
            W = W.masked_fill(self._diag_mask, 0.0)
        if not self.use_wzz:
            W = W.masked_fill(self._wzz_mask, 0.0)
        if self.wzz_diagonal_only:
            W = W.masked_fill(self._wzz_offdiag_mask, 0.0)
        return W

    def get_edge_strength(self) -> torch.Tensor:
        """Absolute edge strength matrix [N_full, N_full]."""
        return self.get_W_masked().abs()

    def get_W_abs(self) -> torch.Tensor:
        return self.get_edge_strength()

    def get_row_reg(self, target_idx: int, w_i: torch.Tensor | None = None) -> torch.Tensor:
        """Row-level L1 regularization for task target_idx.

        For observed targets (target_idx < N_obs), w_i is the full [N_full] gate
        vector. We regularize all non-forbidden entries (the full row, since the
        forbidden entries are always 0).

        Args:
            target_idx: task index (must be < num_obs for prediction tasks)
            w_i: adapted gate vector [N_full]; uses W_masked row if None.

        Returns:
            scalar regularization loss
        """
        if w_i is None:
            w_i = self.get_W_masked()[target_idx]
        return w_i.abs().sum()

    # Backward compat alias
    def get_row_l1(self, target_idx: int) -> torch.Tensor:
        return self.get_row_reg(target_idx)

    def forward(self) -> dict[str, torch.Tensor]:
        W_abs = self.get_edge_strength()  # [N_full, N_full]

        if self.top_k is not None:
            flat = W_abs.reshape(-1)
            k = min(int(self.top_k), flat.numel())
            cutoff = torch.topk(flat, k).values[-1]
            binary_graph = (W_abs >= cutoff).float()
        else:
            binary_graph = (W_abs > self.graph_threshold).float()

        if not self.allow_self_loops:
            binary_graph = binary_graph.masked_fill(self._diag_mask, 0.0)

        # Enforce block constraint in binary graph too
        binary_graph = binary_graph.masked_fill(self._block_mask, 0.0)

        return {
            "W_masked": self.get_W_masked(),
            "W_abs": W_abs,
            "binary_graph": binary_graph,
        }


def build_latent_graph_head(
    num_obs: int,
    q_latent: int,
    graph_threshold: float,
    allow_self_loops: bool,
    top_k: int | None = None,
    w_init_min: float = 0.0,
    w_init_max: float = 0.3,
    w_init_min_latent: float | None = None,
    w_init_max_latent: float | None = None,
    wzz_init_diag: float | None = None,
    reg_mode: str = "l1",
    use_wzz: bool = True,
    wzz_diagonal_only: bool = False,
    **_kwargs,
) -> LatentBlockGraphHead:
    return LatentBlockGraphHead(
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
