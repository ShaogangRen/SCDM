"""SCDM model: shared temporal predictor + shared causal-strength matrix W.

Shared meta-parameters:
  - theta: parameters of the unified temporal predictor MLP
  - W:     continuous structural weights [N, N] (K=1) or [N, N, K] (K>1)

Each variable i defines one target task T_i (predict variable i one step ahead
from the lagged history of all variables):
  - gate    = W[i, :]  (raw row, no sigmoid)
  - x_gated = x_hist * gate, broadcast over the lag dimension
  - y_pred  = MLP(flatten(x_gated))

The inner loop adapts both theta and the task-relevant row w_i = W[i, :];
see ``scdm/nonlatent/meta_training/trainer.py``.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .graph_head import build_graph_head


class SCDMModel(nn.Module):
    """Shared temporal predictor gated by the shared causal-strength matrix W."""

    def __init__(
        self,
        num_nodes: int,
        lag: int,
        hidden_dim: int,
        node_hidden_dim: int,
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
        self.lag = lag
        self.hidden_dim = hidden_dim
        self.node_hidden_dim = node_hidden_dim
        self.w_components = w_components

        self.graph_head = build_graph_head(
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

        # Shared temporal predictor: d*p*K -> hidden_dim -> node_hidden_dim -> 1
        input_dim = num_nodes * lag * w_components
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, node_hidden_dim)
        self.linear3 = nn.Linear(node_hidden_dim, 1)

    # ------------------------------------------------------------------
    # Theta utilities (temporary inner-loop copies)
    # ------------------------------------------------------------------

    def get_theta_params(self) -> dict[str, torch.Tensor]:
        """Clone the predictor parameters as independent leaf tensors.

        The returned dict is the *temporary* per-task theta_i used by the inner
        loop. The persistent shared parameters are never modified in place.
        """
        return {
            "l1.w": self.linear1.weight.clone().detach().requires_grad_(True),
            "l1.b": self.linear1.bias.clone().detach().requires_grad_(True),
            "l2.w": self.linear2.weight.clone().detach().requires_grad_(True),
            "l2.b": self.linear2.bias.clone().detach().requires_grad_(True),
            "l3.w": self.linear3.weight.clone().detach().requires_grad_(True),
            "l3.b": self.linear3.bias.clone().detach().requires_grad_(True),
        }

    @staticmethod
    def functional_forward(
        x_flat: torch.Tensor,
        theta_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Functional predictor forward with task-adapted theta.

        Args:
            x_flat: [B, N*lag*K] gated, flattened input.
            theta_params: dict with keys l1.w, l1.b, l2.w, l2.b, l3.w, l3.b.

        Returns:
            [B] scalar predictions.
        """
        h = F.relu(F.linear(x_flat, theta_params["l1.w"], theta_params["l1.b"]))
        h = F.relu(F.linear(h, theta_params["l2.w"], theta_params["l2.b"]))
        return F.linear(h, theta_params["l3.w"], theta_params["l3.b"]).squeeze(-1)

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    def _gate_and_flatten(
        self,
        x_hist: torch.Tensor,
        w_i: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the target-task gate w_i to x_hist and flatten.

        Args:
            x_hist: [B, lag, N]
            w_i: [N] (K=1) or [N, K] (K>1)

        Returns:
            [B, N*lag] (K=1) or [B, N*lag*K] (K>1).
        """
        B, L, N = x_hist.shape
        if w_i.ndim == 1:
            x_gated = x_hist * w_i.unsqueeze(0).unsqueeze(0)  # [B, L, N]
            return x_gated.reshape(B, N * L)
        K = w_i.shape[1]
        x_gated = x_hist.unsqueeze(-1) * w_i.unsqueeze(0).unsqueeze(0)  # [B, L, N, K]
        return x_gated.reshape(B, N * L * K)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_task(
        self,
        x_hist: torch.Tensor,
        target_idx: int,
        theta_params: dict[str, torch.Tensor] | None = None,
        w_i: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for a single target task T_i.

        Args:
            x_hist: [B, lag, N]
            target_idx: index of the variable being predicted.
            theta_params: if given, use the functional forward with these params.
            w_i: if given, use as the gate; otherwise use W[target_idx, :].

        Returns:
            [B] predictions for variable ``target_idx``.
        """
        if w_i is None:
            w_i = self.graph_head.get_W_masked()[target_idx, :]

        x_flat = self._gate_and_flatten(x_hist, w_i)

        if theta_params is not None:
            return self.functional_forward(x_flat, theta_params)

        h = F.relu(self.linear1(x_flat))
        h = F.relu(self.linear2(h))
        return self.linear3(h).squeeze(-1)

    def forward_all_tasks(
        self,
        x_hist: torch.Tensor,
        theta_params: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward for all N target tasks at once (used for forecasting readout).

        Uses the shared W directly (no per-task adaptation).

        Returns:
            [B, N] predictions for all variables.
        """
        B, L, N = x_hist.shape
        W_masked = self.graph_head.get_W_masked()  # [N, N] or [N, N, K]
        K = self.w_components

        if K == 1:
            x_expanded = x_hist.unsqueeze(0)               # [1, B, L, N]
            gates = W_masked.unsqueeze(1).unsqueeze(2)     # [N, 1, 1, N]
            x_gated_all = x_expanded * gates               # [N, B, L, N]
            x_flat = x_gated_all.reshape(N * B, N * L)
        else:
            x_expanded = x_hist.unsqueeze(0).unsqueeze(-1)  # [1, B, L, N, 1]
            gates = W_masked.unsqueeze(1).unsqueeze(2)      # [N, 1, 1, N, K]
            x_gated_all = x_expanded * gates                # [N, B, L, N, K]
            x_flat = x_gated_all.reshape(N * B, N * L * K)

        if theta_params is not None:
            y_all = self.functional_forward(x_flat, theta_params)
        else:
            h = F.relu(self.linear1(x_flat))
            h = F.relu(self.linear2(h))
            y_all = self.linear3(h).squeeze(-1)             # [N*B]

        return y_all.reshape(N, B).T                        # [B, N]

    def forward(self, x_hist: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convenience forward returning predictions and structural outputs."""
        y_pred = self.forward_all_tasks(x_hist)
        graph_outputs = self.graph_head()
        return {
            "y_pred": y_pred,
            "W_abs": graph_outputs["W_abs"],
            "W_masked": graph_outputs["W_masked"],
            "binary_graph": graph_outputs["binary_graph"],
        }
