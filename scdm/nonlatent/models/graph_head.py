"""Shared causal-strength matrix W (SCDM structural head).

Supports two modes controlled by ``w_components`` (K) and ``reg_mode``:

K=1 (default, used for all paper experiments):
  W is [N, N] scalar per directed pair.
  Gate for target task i = raw W[i, :], edge strength = |W[i, j]|.
  ``reg_mode`` is ignored (L1 and group lasso coincide for K=1).

K>1 (multi-component):
  W is [N, N, K] -- a K-component vector per directed pair.
  Gate for target task i = W[i, :, :] of shape [N, K], applied to K replicas
  of each source variable.
  Edge strength = ||W[i, j, :]||_2 (L2 norm per edge group).
  ``reg_mode="l1"``: element-wise L1 over all K components.
  ``reg_mode="group_lasso"``: sum_j ||W[i, j, :]||_2 (encourages whole edges
  to vanish jointly).

Index convention (internal parameter):
    W[i, j] is the structural weight of SOURCE j on TARGET i.
Readout functions transpose this into the source-target convention used for
evaluation; see ``scdm/nonlatent/graph_readout/`` and the README.
"""

from __future__ import annotations

import torch
from torch import nn

from .masking import apply_diagonal_mask  # noqa: F401  (re-exported helper)


class DirectWGraphHead(nn.Module):
    """Direct edge-weight matrix W in R^[N, N] or R^[N, N, K].

    W[i, j, ...] is the structural weight from source j to target i.
    """

    def __init__(
        self,
        num_nodes: int,
        graph_threshold: float,
        allow_self_loops: bool,
        top_k: int | None = None,
        w_init_min: float = 0.0,
        w_init_max: float = 0.3,
        w_components: int = 1,
        reg_mode: str = "l1",
        use_structural_w: bool = True,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.graph_threshold = graph_threshold
        self.allow_self_loops = allow_self_loops
        self.top_k = top_k
        self.w_components = w_components
        self.reg_mode = reg_mode
        self.use_structural_w = use_structural_w

        if not use_structural_w:
            # Ablation: fixed identity gate W = 1 (off-diagonal), not optimised.
            W_init = torch.ones(num_nodes, num_nodes)
            if not allow_self_loops:
                W_init.fill_diagonal_(0.0)
            self.W = nn.Parameter(W_init, requires_grad=False)
        elif w_components == 1:
            W_init = torch.empty(num_nodes, num_nodes).uniform_(w_init_min, w_init_max)
            if not allow_self_loops:
                diag_idx = torch.arange(num_nodes)
                W_init[diag_idx, diag_idx] = 0.0
            self.W = nn.Parameter(W_init)
        else:
            W_init = torch.empty(num_nodes, num_nodes, w_components).uniform_(
                w_init_min, w_init_max
            )
            if not allow_self_loops:
                diag_idx = torch.arange(num_nodes)
                W_init[diag_idx, diag_idx, :] = 0.0
            self.W = nn.Parameter(W_init)

        diag_mask = torch.eye(num_nodes, dtype=torch.bool)
        self.register_buffer("_diag_mask", diag_mask)

    def get_W_masked(self) -> torch.Tensor:
        """Return W with the diagonal zeroed. Shape [N, N] or [N, N, K]."""
        if not self.allow_self_loops:
            if self.w_components == 1:
                return self.W.masked_fill(self._diag_mask, 0.0)
            mask = self._diag_mask.unsqueeze(-1).expand_as(self.W)
            return self.W.masked_fill(mask, 0.0)
        return self.W

    def get_edge_strength(self) -> torch.Tensor:
        """Edge-strength matrix [N, N]: |W| for K=1, per-edge L2 norm for K>1."""
        W_m = self.get_W_masked()
        if self.w_components == 1:
            return W_m.abs()
        return W_m.norm(dim=-1)

    def get_W_abs(self) -> torch.Tensor:
        """Alias of :meth:`get_edge_strength`."""
        return self.get_edge_strength()

    def get_row_reg(self, target_idx: int, w_i: torch.Tensor | None = None) -> torch.Tensor:
        """Row-level sparsity regulariser for target task ``target_idx``.

        Args:
            target_idx: row index (used to read from W when ``w_i`` is None).
            w_i: adapted gate vector, shape [N] (K=1) or [N, K] (K>1).

        Returns:
            Scalar regularisation loss for row ``target_idx``.
        """
        if w_i is None:
            w_i = self.get_W_masked()[target_idx]

        if self.w_components == 1:
            return w_i.abs().sum()

        if self.reg_mode == "group_lasso":
            return w_i.norm(dim=-1).sum()
        return w_i.abs().sum()

    def get_row_l1(self, target_idx: int) -> torch.Tensor:
        """Alias of :meth:`get_row_reg` without an adapted row."""
        return self.get_row_reg(target_idx)

    def forward(self) -> dict[str, torch.Tensor]:
        W_abs = self.get_edge_strength()  # [N, N]

        if self.top_k is not None:
            flat = W_abs.reshape(-1)
            k = min(int(self.top_k), flat.numel())
            cutoff = torch.topk(flat, k).values[-1]
            binary_graph = (W_abs >= cutoff).float()
        else:
            binary_graph = (W_abs > self.graph_threshold).float()

        if not self.allow_self_loops:
            binary_graph = binary_graph.masked_fill(self._diag_mask, 0.0)

        return {
            "W_masked": self.get_W_masked(),
            "W_abs": W_abs,
            "binary_graph": binary_graph,
        }


def build_graph_head(
    num_nodes: int,
    graph_threshold: float,
    allow_self_loops: bool,
    top_k: int | None = None,
    w_init_min: float = 0.0,
    w_init_max: float = 0.3,
    w_components: int = 1,
    reg_mode: str = "l1",
    use_structural_w: bool = True,
    **_kwargs,
) -> nn.Module:
    return DirectWGraphHead(
        num_nodes=num_nodes,
        graph_threshold=graph_threshold,
        allow_self_loops=allow_self_loops,
        top_k=top_k,
        w_init_min=w_init_min,
        w_init_max=w_init_max,
        w_components=w_components,
        reg_mode=reg_mode,
        use_structural_w=use_structural_w,
    )
