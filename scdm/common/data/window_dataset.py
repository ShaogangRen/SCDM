"""Sliding-window datasets for multivariate time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WindowArrays:
    """Container for windowed histories and targets."""

    histories: np.ndarray
    targets: np.ndarray
    sequence_ids: np.ndarray
    target_indices: np.ndarray


def build_window_arrays(sequences: Iterable[np.ndarray], lag: int) -> WindowArrays:
    """Create chronological sliding windows from a list of sequences."""

    history_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []
    sequence_id_list: List[np.ndarray] = []
    target_index_list: List[np.ndarray] = []

    for sequence_id, sequence in enumerate(sequences):
        if sequence.ndim != 2:
            raise ValueError(f"Expected 2D sequence, got shape {sequence.shape}")
        if len(sequence) <= lag:
            raise ValueError(
                f"Sequence length {len(sequence)} is not larger than lag {lag}."
            )

        num_samples = len(sequence) - lag
        histories = np.stack([sequence[idx : idx + lag] for idx in range(num_samples)])
        targets = np.stack([sequence[idx + lag] for idx in range(num_samples)])

        history_list.append(histories.astype(np.float32))
        target_list.append(targets.astype(np.float32))
        sequence_id_list.append(np.full(num_samples, sequence_id, dtype=np.int64))
        target_index_list.append(np.arange(lag, lag + num_samples, dtype=np.int64))

    return WindowArrays(
        histories=np.concatenate(history_list, axis=0),
        targets=np.concatenate(target_list, axis=0),
        sequence_ids=np.concatenate(sequence_id_list, axis=0),
        target_indices=np.concatenate(target_index_list, axis=0),
    )


class SequenceWindowDataset(Dataset):
    """Torch dataset backed by precomputed sliding windows."""

    def __init__(
        self,
        histories: np.ndarray,
        targets: np.ndarray,
        sequence_ids: np.ndarray,
        target_indices: np.ndarray,
    ) -> None:
        if histories.shape[0] != targets.shape[0]:
            raise ValueError("Histories and targets must have the same number of samples.")
        self.histories = histories.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.sequence_ids = sequence_ids.astype(np.int64)
        self.target_indices = target_indices.astype(np.int64)

    @classmethod
    def from_sequences(cls, sequences: Iterable[np.ndarray], lag: int) -> "SequenceWindowDataset":
        window_arrays = build_window_arrays(sequences, lag)
        return cls(
            histories=window_arrays.histories,
            targets=window_arrays.targets,
            sequence_ids=window_arrays.sequence_ids,
            target_indices=window_arrays.target_indices,
        )

    def __len__(self) -> int:
        return self.histories.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x_hist": torch.from_numpy(self.histories[index]),
            "y_next": torch.from_numpy(self.targets[index]),
            "sequence_id": torch.tensor(self.sequence_ids[index], dtype=torch.long),
            "target_index": torch.tensor(self.target_indices[index], dtype=torch.long),
        }
