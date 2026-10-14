"""Dataset loading for the SCDM reviewer artifact.

Two generic input formats are supported. Neither is domain specific.

``csv_pair``
    A directory holding one or more pairs of CSV files::

        <prefix>_data_<k>.csv    [T, d]  observations, one column per variable
        <prefix>_struct_<k>.csv  [d, d]  ground-truth adjacency

    Files are matched by sorted order. Every ``*_struct_*.csv`` must be
    identical (one fixed ground-truth graph shared by all replicas). The NC8
    quick-start example uses this format.

``npz``
    A single ``.npz`` file with

        X : [T, d] or [R, T, d] float array  -- R replica sequences
        A : [d, d] integer array             -- ground-truth adjacency (optional)

    The high-dimensional quick-start example uses this format.

Shape conventions
-----------------
Every sequence is [T, d]: rows are time steps, columns are variables.
Adjacency matrices use the source-target convention: ``A[j, i] = 1`` means an
edge j -> i. Self-loops on the diagonal are ignored by all metrics.

Pipeline: chronological 70/15/15 split per sequence -> z-score statistics fit
on the training split only -> episodic and sliding-window datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .episode_dataset import DynamicEpisodeDataset
from .window_dataset import SequenceWindowDataset


def _default_variable_names(num_nodes: int) -> list[str]:
    return [f"X{index}" for index in range(num_nodes)]


def _normalize_mode(normalize: bool | str) -> str:
    if isinstance(normalize, bool):
        return "zscore" if normalize else "none"
    mode = str(normalize).lower()
    if mode not in {"none", "zscore"}:
        raise ValueError(f"Unsupported normalization mode: {normalize}")
    return mode


def _chronological_split(
    sequence: np.ndarray, split_ratio: tuple[float, float, float]
) -> dict[str, np.ndarray]:
    if len(split_ratio) != 3:
        raise ValueError("split_ratio must contain exactly three values.")
    total = float(sum(split_ratio))
    if total <= 0:
        raise ValueError("split_ratio must have positive sum.")

    normalized = [ratio / total for ratio in split_ratio]
    train_end = int(len(sequence) * normalized[0])
    val_end = train_end + int(len(sequence) * normalized[1])

    train = sequence[:train_end]
    val = sequence[train_end:val_end]
    test = sequence[val_end:]

    if min(len(train), len(val), len(test)) < 2:
        raise ValueError("Chronological split produced an empty or near-empty segment.")

    return {"train": train, "val": val, "test": test}


def _fit_zscore_stats(train_sequences: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    concatenated = np.concatenate(list(train_sequences), axis=0)
    mean = concatenated.mean(axis=0)
    std = concatenated.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_normalization(
    sequence: np.ndarray, mean: np.ndarray, std: np.ndarray, mode: str
) -> np.ndarray:
    if mode == "none":
        return sequence.astype(np.float32)
    if mode == "zscore":
        return ((sequence - mean) / std).astype(np.float32)
    raise ValueError(f"Unsupported normalization mode: {mode}")


# ----------------------------------------------------------------------
# Raw readers
# ----------------------------------------------------------------------


def _read_csv_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read a headed CSV into a float32 matrix plus its column names.

    ``pandas`` is used deliberately rather than ``numpy.loadtxt``: the array it
    returns has the same memory layout as in the original experiment code, and
    float32 reductions (the z-score statistics) are order sensitive, so a
    different layout perturbs the normalised data at the 1e-6 level.
    """
    frame = pd.read_csv(path)
    return frame.to_numpy(dtype=np.float32), [str(name) for name in frame.columns]


def _load_csv_pair_arrays(
    dataset_dir: Path,
    sequence_glob: str,
    structure_glob: str,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    data_paths = sorted(dataset_dir.glob(sequence_glob))
    struct_paths = sorted(dataset_dir.glob(structure_glob))
    if not data_paths or not struct_paths:
        raise FileNotFoundError(
            f"Missing CSV pairs under {dataset_dir} "
            f"(patterns: {sequence_glob!r}, {structure_glob!r})."
        )
    if len(data_paths) != len(struct_paths):
        raise ValueError("Data file count and structure file count do not match.")

    sequences: list[np.ndarray] = []
    graphs: list[np.ndarray] = []
    variable_names: list[str] | None = None

    for data_path, struct_path in zip(data_paths, struct_paths):
        data_values, data_columns = _read_csv_matrix(data_path)
        struct_values, _ = _read_csv_matrix(struct_path)

        if variable_names is None:
            variable_names = data_columns
        elif data_columns != variable_names:
            raise ValueError("Data files use inconsistent variable names.")

        sequences.append(data_values)
        graphs.append(struct_values)

    base_graph = graphs[0]
    for graph in graphs[1:]:
        if not np.array_equal(base_graph, graph):
            raise ValueError("Ground-truth graphs are inconsistent across replicas.")

    resolved_names = variable_names or _default_variable_names(base_graph.shape[0])
    return sequences, base_graph.astype(np.float32), resolved_names


def _load_npz_arrays(npz_path: Path) -> tuple[list[np.ndarray], np.ndarray | None, list[str]]:
    with np.load(npz_path) as payload:
        if "X" not in payload:
            raise KeyError(f"{npz_path} does not contain an 'X' array.")
        x = np.asarray(payload["X"], dtype=np.float32)
        gt = np.asarray(payload["A"]).astype(np.float32) if "A" in payload else None
        names = (
            [str(name) for name in payload["variable_names"]]
            if "variable_names" in payload
            else None
        )

    if x.ndim == 2:
        sequences = [x]
    elif x.ndim == 3:
        sequences = [np.ascontiguousarray(x[idx]) for idx in range(x.shape[0])]
    else:
        raise ValueError(f"Expected X with shape [T, d] or [R, T, d], got {x.shape}.")

    num_nodes = sequences[0].shape[1]
    if gt is not None and gt.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"Ground-truth shape {gt.shape} does not match data width {num_nodes}."
        )
    if not names or len(names) != num_nodes:
        names = _default_variable_names(num_nodes)
    return sequences, gt, names


def _resolve_source(config_dataset_path: str | Path, data_root: str | Path) -> Path:
    """Resolve ``dataset_path`` relative to ``data_root`` when it is not absolute."""
    path = Path(config_dataset_path).expanduser()
    if path.is_absolute():
        return path
    return (Path(data_root).expanduser() / path).resolve()


def load_raw_sequences(
    dataset_format: str,
    dataset_path: str | Path,
    data_root: str | Path,
    sequence_glob: str = "*_data_*.csv",
    structure_glob: str = "*_struct_*.csv",
) -> tuple[list[np.ndarray], np.ndarray | None, list[str], Path]:
    """Read the raw sequences and ground-truth adjacency from disk."""
    source = _resolve_source(dataset_path, data_root)
    fmt = str(dataset_format).lower()

    if fmt == "csv_pair":
        if not source.is_dir():
            raise FileNotFoundError(f"csv_pair dataset directory not found: {source}")
        sequences, gt, names = _load_csv_pair_arrays(source, sequence_glob, structure_glob)
    elif fmt == "npz":
        if not source.is_file():
            raise FileNotFoundError(f"npz dataset file not found: {source}")
        sequences, gt, names = _load_npz_arrays(source)
    else:
        raise ValueError(
            f"Unsupported dataset_format: {dataset_format!r} (expected 'csv_pair' or 'npz')."
        )

    widths = {seq.shape[1] for seq in sequences}
    if len(widths) != 1:
        raise ValueError(f"All sequences must have the same variable count, got {widths}.")
    if not names or len(names) != sequences[0].shape[1]:
        names = _default_variable_names(sequences[0].shape[1])
    return sequences, gt, names, source


def _ground_truth_metadata(ground_truth_graph: np.ndarray | None, source: Path) -> dict:
    if ground_truth_graph is None:
        return {
            "has_ground_truth_graph": False,
            "ground_truth_graph": None,
            "gt_num_edges": None,
            "gt_density": None,
        }

    gt_num_edges = int(
        np.count_nonzero(ground_truth_graph) - np.count_nonzero(np.diag(ground_truth_graph))
    )
    num_nodes = ground_truth_graph.shape[0]
    gt_density = float(gt_num_edges / max(num_nodes * max(num_nodes - 1, 1), 1))
    return {
        "has_ground_truth_graph": True,
        "ground_truth_graph": ground_truth_graph.astype(np.int64),
        "gt_num_edges": gt_num_edges,
        "gt_density": gt_density,
    }


def _auto_episodes(sequences: list[np.ndarray], l_epi: int, lag: int) -> int:
    """Non-overlapping episode count x 10 coverage multiplier."""
    independent = sum(max(1, (seq.shape[0] - lag) // l_epi) for seq in sequences)
    return max(independent * 10, 50)


def load_episode_dataset(
    dataset_format: str,
    dataset_path: str | Path,
    data_root: str | Path,
    lag: int,
    split_ratio: tuple[float, float, float],
    normalize: bool | str,
    seed: int,
    l_sup: int,
    l_qry: int,
    sequence_glob: str = "*_data_*.csv",
    structure_glob: str = "*_struct_*.csv",
    train_episodes_per_cycle: int | None = None,
    val_episodes_per_cycle: int | None = None,
) -> tuple[
    DynamicEpisodeDataset, DynamicEpisodeDataset, SequenceWindowDataset, SequenceWindowDataset, dict
]:
    """Build episodic train/val datasets plus val/test sliding-window datasets.

    Returns:
        ``(train_episodes, val_episodes, val_windows, test_windows, metadata)``.
    """
    if lag < 1:
        raise ValueError("lag must be at least 1.")

    mode = _normalize_mode(normalize)
    source = _resolve_source(dataset_path, data_root)
    split_sequences = None
    # Dataset releases preserve the original sequence partitions. This only
    # changes input loading; the model, training and graph readout are unchanged.
    if str(dataset_format).lower() == "npz" and source.is_file():
        with np.load(source, allow_pickle=False) as payload:
            split_keys = {name: "X_" + name for name in ("train", "val", "test")}
            if all(key in payload for key in split_keys.values()):
                split_sequences = {}
                for name, key in split_keys.items():
                    x = np.asarray(payload[key], dtype=np.float32)
                    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] <= lag:
                        raise ValueError(f"{key} must have shape [replicas, time, variables], with time > lag.")
                    split_sequences[name] = [np.ascontiguousarray(row) for row in x]
                sequences = [row for group in split_sequences.values() for row in group]
                widths = {row.shape[1] for row in sequences}
                if len(widths) != 1:
                    raise ValueError("Predefined partitions have inconsistent variable counts.")
                num_nodes = sequences[0].shape[1]
                ground_truth_graph = np.asarray(payload["A"], dtype=np.float32) if "A" in payload else None
                if ground_truth_graph is not None and ground_truth_graph.shape != (num_nodes, num_nodes):
                    raise ValueError("Ground-truth shape does not match predefined partition width.")
                variable_names = [str(v) for v in payload["variable_names"]] if "variable_names" in payload else _default_variable_names(num_nodes)
                if len(variable_names) != num_nodes:
                    raise ValueError("Variable-name count does not match predefined partition width.")
    predefined_partitions = split_sequences is not None
    if split_sequences is None:
        sequences, ground_truth_graph, variable_names, source = load_raw_sequences(
            dataset_format=dataset_format, dataset_path=dataset_path,
            data_root=data_root, sequence_glob=sequence_glob, structure_glob=structure_glob,
        )
        split_sequences = {"train": [], "val": [], "test": []}
        for sequence in sequences:
            splits = _chronological_split(sequence, split_ratio)
            for split_name, split_array in splits.items():
                if len(split_array) <= lag:
                    raise ValueError(
                        f"Split {split_name!r} length {len(split_array)} is not larger than lag {lag}."
                    )
                split_sequences[split_name].append(split_array)

    train_mean, train_std = _fit_zscore_stats(split_sequences["train"])

    normalized_splits: dict[str, list[np.ndarray]] = {}
    for split_name, arrays in split_sequences.items():
        normalized_splits[split_name] = [
            _apply_normalization(array, train_mean, train_std, mode) for array in arrays
        ]

    transformed_mean = np.zeros_like(train_mean) if mode == "zscore" else train_mean.copy()
    transformed_std = np.ones_like(train_std) if mode == "zscore" else train_std.copy()

    l_epi = l_sup + l_qry
    if train_episodes_per_cycle is None:
        train_episodes_per_cycle = _auto_episodes(normalized_splits["train"], l_epi, lag)
    if val_episodes_per_cycle is None:
        val_episodes_per_cycle = _auto_episodes(normalized_splits["val"], l_epi, lag)

    train_dataset = DynamicEpisodeDataset(
        sequences=normalized_splits["train"],
        lag=lag,
        l_sup=l_sup,
        l_qry=l_qry,
        steps_per_epoch=train_episodes_per_cycle,
        seed=seed,
    )
    val_episode_dataset = DynamicEpisodeDataset(
        sequences=normalized_splits["val"],
        lag=lag,
        l_sup=l_sup,
        l_qry=l_qry,
        steps_per_epoch=val_episodes_per_cycle,
        seed=seed + 1,
    )

    val_window_dataset = SequenceWindowDataset.from_sequences(normalized_splits["val"], lag=lag)
    test_dataset = SequenceWindowDataset.from_sequences(normalized_splits["test"], lag=lag)

    metadata = {
        "dataset_format": str(dataset_format),
        "dataset_source": str(source),
        "num_nodes": int(sequences[0].shape[1]),
        "num_timesteps": int(sum(sequence.shape[0] for sequence in sequences)),
        "sequence_length": int(sequences[0].shape[0]),
        "num_sequences": len(sequences),
        "lag": int(lag),
        "variable_names": variable_names,
        "normalization_mode": mode,
        "normalization_stats": {
            "raw_mean": train_mean.tolist(),
            "raw_std": train_std.tolist(),
            "transformed_mean": transformed_mean.tolist(),
            "transformed_std": transformed_std.tolist(),
        },
        "split_ratio": list(split_ratio),
        "split_protocol": "predefined_sequence_partitions" if predefined_partitions else "chronological_per_sequence",
        "split_sequence_counts": {name: len(group) for name, group in split_sequences.items()},
        "seed": int(seed),
        "l_sup": int(l_sup),
        "l_qry": int(l_qry),
        "train_episodes_per_cycle": int(train_episodes_per_cycle),
        "val_episodes_per_cycle": int(val_episodes_per_cycle),
    }
    metadata.update(_ground_truth_metadata(ground_truth_graph, source))

    return train_dataset, val_episode_dataset, val_window_dataset, test_dataset, metadata
