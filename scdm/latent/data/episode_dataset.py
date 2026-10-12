"""Dynamic episode dataset for episodic meta-learning (v5 framework).

v5 change vs v4:
- Episodes are sampled randomly at each __getitem__ call rather than being
  pre-built at construction time.
- This prevents meta-learner overfitting to a fixed support/query partition
  (framework §2.3 and §23).
- Temporal ordering is preserved: support always precedes query within each episode.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class DynamicEpisodeDataset(Dataset):
    """On-the-fly random temporal episode sampler.

    At each __getitem__ call, randomly selects a sequence and a contiguous
    temporal episode of length l_epi = l_sup + l_qry, then splits it
    chronologically into support and query subsets.

    Episode construction (0-indexed arrays, lag = p):
      - Valid supervised positions: t in {lag, lag+1, ..., T-1}
      - Episode start s: s in {lag, ..., T - l_epi}  (ensures full episode fits)
      - Support: supervised positions s, ..., s + l_sup - 1
      - Query:   supervised positions s + l_sup, ..., s + l_epi - 1
    """

    def __init__(
        self,
        sequences: list[np.ndarray],
        lag: int,
        l_sup: int,
        l_qry: int,
        steps_per_epoch: int,
        seed: int | None = None,
    ) -> None:
        self.lag = lag
        self.l_sup = l_sup
        self.l_qry = l_qry
        self.l_epi = l_sup + l_qry
        self.steps_per_epoch = steps_per_epoch

        # Pre-convert to float32 tensors
        self.sequences = [
            torch.from_numpy(seq.astype(np.float32)) for seq in sequences
        ]

        # Precompute valid start ranges per sequence
        # s_min = lag: first position with full lag history
        # s_max = T - l_epi: last start that keeps episode within sequence
        self.valid_ranges: list[tuple[int, int] | None] = []
        for seq in sequences:
            T = seq.shape[0]
            s_min = lag
            s_max = T - self.l_epi
            if s_max >= s_min:
                self.valid_ranges.append((s_min, s_max))
            else:
                self.valid_ranges.append(None)

        self._usable = [i for i, r in enumerate(self.valid_ranges) if r is not None]
        if not self._usable:
            raise ValueError(
                f"No sequence is long enough for episodes of l_epi={self.l_epi} "
                f"with lag={lag}. Need T >= lag + l_epi = {lag + self.l_epi}."
            )

        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Randomly pick a usable sequence
        seq_idx = self._usable[int(self._rng.integers(0, len(self._usable)))]
        seq = self.sequences[seq_idx]
        s_min, s_max = self.valid_ranges[seq_idx]

        # Randomly pick episode start
        s = int(self._rng.integers(s_min, s_max + 1))

        # Build support: supervised positions s, ..., s+l_sup-1
        # History for position s+i: seq[s+i-lag : s+i], shape [lag, N]
        support_hist = torch.stack(
            [seq[s + i - self.lag : s + i] for i in range(self.l_sup)]
        )                                             # [l_sup, lag, N]
        support_target = seq[s : s + self.l_sup]     # [l_sup, N]

        # Build query: supervised positions s+l_sup, ..., s+l_epi-1
        q0 = s + self.l_sup
        query_hist = torch.stack(
            [seq[q0 + i - self.lag : q0 + i] for i in range(self.l_qry)]
        )                                             # [l_qry, lag, N]
        query_target = seq[q0 : s + self.l_epi]      # [l_qry, N]

        return {
            "support_hist": support_hist,
            "support_target": support_target,
            "query_hist": query_hist,
            "query_target": query_target,
            "sequence_id": torch.tensor(seq_idx, dtype=torch.long),
        }


class LatentEpisodeDataset(Dataset):
    """Observed-only episode sampler for the latent-aware benchmark.

    Hidden trajectories are intentionally not stored or returned here. The
    latent-aware learner receives only observed histories and observed one-step
    targets; latent variables are retained in loader metadata only as graph
    ground truth for evaluation.
    """

    def __init__(
        self,
        obs_sequences: list[np.ndarray],
        hidden_sequences: list[np.ndarray] | None,
        lag: int,
        l_sup: int,
        l_qry: int,
        steps_per_epoch: int,
        seed: int | None = None,
    ) -> None:
        if hidden_sequences is not None and len(obs_sequences) != len(hidden_sequences):
            raise ValueError("obs_sequences and hidden_sequences must have the same length.")
        self.lag = lag
        self.l_sup = l_sup
        self.l_qry = l_qry
        self.l_epi = l_sup + l_qry
        self.steps_per_epoch = steps_per_epoch

        self.obs_seqs = [
            torch.from_numpy(seq.astype(np.float32)) for seq in obs_sequences
        ]

        self.valid_ranges: list[tuple[int, int] | None] = []
        for seq in obs_sequences:
            T = seq.shape[0]
            s_min = lag
            s_max = T - self.l_epi
            if s_max >= s_min:
                self.valid_ranges.append((s_min, s_max))
            else:
                self.valid_ranges.append(None)

        self._usable = [i for i, r in enumerate(self.valid_ranges) if r is not None]
        if not self._usable:
            raise ValueError(
                f"No sequence is long enough for episodes of l_epi={self.l_epi} "
                f"with lag={lag}. Need T >= lag + l_epi = {lag + self.l_epi}."
            )

        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx = self._usable[int(self._rng.integers(0, len(self._usable)))]
        obs = self.obs_seqs[seq_idx]
        s_min, s_max = self.valid_ranges[seq_idx]

        s = int(self._rng.integers(s_min, s_max + 1))

        support_hist = torch.stack(
            [obs[s + i - self.lag : s + i] for i in range(self.l_sup)]
        )                                                   # [l_sup, lag, N_obs]
        support_target = obs[s : s + self.l_sup]           # [l_sup, N_obs]

        q0 = s + self.l_sup
        query_hist = torch.stack(
            [obs[q0 + i - self.lag : q0 + i] for i in range(self.l_qry)]
        )                                                   # [l_qry, lag, N_obs]
        query_target = obs[q0 : s + self.l_epi]            # [l_qry, N_obs]

        return {
            "support_hist": support_hist,
            "support_target": support_target,
            "query_hist": query_hist,
            "query_target": query_target,
            "sequence_id":   torch.tensor(seq_idx, dtype=torch.long),
            "episode_start": torch.tensor(s,       dtype=torch.long),
        }
