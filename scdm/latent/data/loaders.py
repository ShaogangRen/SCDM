"""Dataset loading for latent benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .episode_dataset import DynamicEpisodeDataset, LatentEpisodeDataset
from .latent_benchmark import build_nc8_latent_benchmark
from scdm.common.data.window_dataset import SequenceWindowDataset


def _default_variable_names(num_nodes: int) -> list[str]:
    return [f"X{index}" for index in range(num_nodes)]


def _normalize_mode(normalize: bool | str) -> str:
    if isinstance(normalize, bool):
        return "zscore" if normalize else "none"
    mode = str(normalize).lower()
    if mode not in {"none", "zscore"}:
        raise ValueError(f"Unsupported normalization mode: {normalize}")
    return mode


def _resolve_dataset_dir(dataset_name: str, data_root: str) -> Path:
    root = Path(data_root).expanduser().resolve()
    dataset_lower = dataset_name.lower()

    if dataset_lower in {"nc8", "nc8_latent"}:
        check_pattern = "nc8_data_*.csv"
        dir_candidates = [root / "nc8", root / "NC8", root]
    elif dataset_lower == "metr_la_128":
        check_pattern = "rep_0.npy"
        dir_candidates = [root / "metr_la_128", root / "METR_LA_128", root]
    elif dataset_lower in {"fmri_static", "fmri_latent"}:
        # Expect: data_root/{train,val,test}/ directly, or a known subdir
        dir_candidates = [
            root / "sim_fmri_64obs_4latent_compatible",
            root / "sim_fmri_68_static_semirealistic",
            root / "sim_fmri_68_static_large",
            root,
        ]
        for candidate in dir_candidates:
            if (candidate / "train").is_dir():
                return candidate
        raise FileNotFoundError(
            f"Could not find fmri split dirs under {root}. "
            f"Expected train/val/test subdirectories."
        )
    elif dataset_lower in {"eeg_static", "eeg_latent"}:
        dir_candidates = [root / "simulated_EEG_68_static_v2", root]
        for candidate in dir_candidates:
            if (candidate / "train").is_dir():
                return candidate
        raise FileNotFoundError(
            f"Could not find EEG split dirs under {root}. "
            f"Expected train/val/test subdirectories."
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    for candidate in dir_candidates:
        if list(candidate.glob(check_pattern)):
            return candidate

    raise FileNotFoundError(
        f"Could not find dataset files for {dataset_name!r} under {root}."
    )


def _load_fmri_static_presplit(dataset_dir: Path) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, list[str]
]:
    """Load a pre-split static dataset with shared GT stored at the dataset root.

    Reads X.npy from each sample directory in train/, val/, test/.
    GT adjacency A.npy is shared across all samples (static graph).

    Returns:
        train_seqs, val_seqs, test_seqs: lists of (T, N) float32 arrays
        gt_graph: (N, N) float32 binary adjacency
        variable_names: list of N ROI names
    """
    split_seqs: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    gt_graph: np.ndarray | None = None

    root_gt_path = dataset_dir / "A.npy"
    if root_gt_path.exists():
        gt_graph = np.load(root_gt_path).astype(np.float32)

    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Expected split directory not found: {split_dir}")
        for sample_dir in sorted(split_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            x_path = sample_dir / "X.npy"
            a_path = sample_dir / "A.npy"
            if not x_path.exists():
                continue
            split_seqs[split].append(np.load(x_path).astype(np.float32))
            if gt_graph is None and a_path.exists():
                gt_graph = np.load(a_path).astype(np.float32)

    if gt_graph is None:
        raise FileNotFoundError(f"Could not find A.npy under {dataset_dir}.")
    if not split_seqs["train"]:
        raise FileNotFoundError(f"No training samples found under {dataset_dir}/train/")

    N = gt_graph.shape[0]
    roi_names_path = dataset_dir / "roi_names.npy"
    if roi_names_path.exists():
        variable_names = [str(name) for name in np.load(roi_names_path, allow_pickle=True).tolist()]
    else:
        variable_names = [f"ROI{i}" for i in range(N)]
    return split_seqs["train"], split_seqs["val"], split_seqs["test"], gt_graph, variable_names


def _chronological_split(sequence: np.ndarray, split_ratio: tuple[float, float, float]) -> dict[str, np.ndarray]:
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
    if concatenated.ndim != 2:
        raise ValueError(f"Expected 2-D sequences for normalization, got shape {concatenated.shape}.")
    if concatenated.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    mean = concatenated.mean(axis=0)
    std = concatenated.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _summarize_hidden_selection(
    ground_truth_graph: np.ndarray,
    hidden_indices: list[int],
    variable_names: list[str] | None = None,
) -> dict:
    """Summarize whether a hidden set satisfies the latent benchmark assumption."""
    N = int(ground_truth_graph.shape[0])
    hidden_indices = list(hidden_indices)
    hidden_set = set(hidden_indices)
    if variable_names is None or len(variable_names) != N:
        variable_names = _default_variable_names(N)

    per_node = []
    for idx in hidden_indices:
        parents = np.where(ground_truth_graph[:, idx] > 0)[0].astype(int).tolist()
        children = np.where(ground_truth_graph[idx, :] > 0)[0].astype(int).tolist()
        observed_parents = [p for p in parents if p not in hidden_set]
        hidden_parents = [p for p in parents if p in hidden_set]
        out_degree = len(children)
        per_node.append({
            "index": int(idx),
            "name": str(variable_names[idx]),
            "in_degree": int(len(parents)),
            "out_degree": int(out_degree),
            "parent_indices": parents,
            "child_indices": children,
            "observed_parent_indices": observed_parents,
            "hidden_parent_indices": hidden_parents,
            "observed_parent_count": int(len(observed_parents)),
            "valid_no_observed_parents": len(observed_parents) == 0,
            "valid_out_degree_ge_2": out_degree >= 2,
            "valid_under_current_hidden_set": (len(observed_parents) == 0 and out_degree >= 2),
        })

    return {
        "hidden_indices": [int(i) for i in hidden_indices],
        "hidden_names": [str(variable_names[i]) for i in hidden_indices],
        "num_hidden": int(len(hidden_indices)),
        "all_hidden_nodes_valid": bool(all(node["valid_under_current_hidden_set"] for node in per_node)),
        "valid_hidden_node_count": int(sum(node["valid_under_current_hidden_set"] for node in per_node)),
        "per_hidden_node": per_node,
    }


def _apply_normalization(sequence: np.ndarray, mean: np.ndarray, std: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return sequence.astype(np.float32)
    if mode == "zscore":
        return ((sequence - mean) / std).astype(np.float32)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _load_nc8_arrays(dataset_dir: Path) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    data_paths = sorted(dataset_dir.glob("nc8_data_*.csv"))
    struct_paths = sorted(dataset_dir.glob("nc8_struct_*.csv"))
    if not data_paths or not struct_paths:
        raise FileNotFoundError(f"Missing NC8 CSV files under {dataset_dir}.")
    if len(data_paths) != len(struct_paths):
        raise ValueError("NC8 data file count and structure file count do not match.")

    sequences: list[np.ndarray] = []
    graphs: list[np.ndarray] = []
    variable_names: list[str] | None = None

    for data_path, struct_path in zip(data_paths, struct_paths):
        data_df = pd.read_csv(data_path)
        struct_df = pd.read_csv(struct_path)

        if variable_names is None:
            variable_names = list(data_df.columns)
        elif list(data_df.columns) != variable_names:
            raise ValueError("NC8 data files use inconsistent variable names.")

        sequences.append(data_df.to_numpy(dtype=np.float32))
        graphs.append(struct_df.to_numpy(dtype=np.float32))

    base_graph = graphs[0]
    for graph in graphs[1:]:
        if not np.array_equal(base_graph, graph):
            raise ValueError("NC8 ground-truth graphs are inconsistent across files.")

    resolved_names = variable_names or _default_variable_names(base_graph.shape[0])
    return sequences, base_graph.astype(np.float32), resolved_names


def _resolve_gt_path(dataset_name: str, dataset_dir: Path) -> str:
    if dataset_name == "metr_la_128":
        return str(dataset_dir / "road_adj_binary.npy")
    struct_files = sorted(dataset_dir.glob("*_struct_*.csv"))
    if struct_files:
        return str(struct_files[0])
    return str(dataset_dir)


def _load_metr_la_arrays(dataset_dir: Path) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    rep_paths = sorted(dataset_dir.glob("rep_*.npy"))
    if not rep_paths:
        raise FileNotFoundError(f"Missing rep_*.npy replica files under {dataset_dir}.")
    adj_path = dataset_dir / "road_adj_binary.npy"
    if not adj_path.exists():
        raise FileNotFoundError(f"Missing road_adj_binary.npy under {dataset_dir}.")

    sequences: list[np.ndarray] = []
    for rep_path in rep_paths:
        arr = np.load(rep_path).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D array in {rep_path}, got shape {arr.shape}.")
        sequences.append(arr)

    gt_graph = np.load(adj_path).astype(np.float32)
    num_nodes = gt_graph.shape[0]
    variable_names = [f"S{i}" for i in range(num_nodes)]
    return sequences, gt_graph, variable_names


def load_dataset(
    dataset_name: str,
    data_root: str,
    lag: int,
    split_ratio: tuple[float, float, float],
    normalize: bool | str,
    seed: int,
) -> tuple[SequenceWindowDataset, SequenceWindowDataset, SequenceWindowDataset, dict]:
    """Load train/validation/test standard window datasets and metadata."""

    if lag < 1:
        raise ValueError("lag must be at least 1.")

    dataset_name = dataset_name.lower()
    mode = _normalize_mode(normalize)
    dataset_dir = _resolve_dataset_dir(dataset_name, data_root)

    if dataset_name == "nc8":
        sequences, ground_truth_graph, variable_names = _load_nc8_arrays(dataset_dir)
    elif dataset_name == "metr_la_128":
        sequences, ground_truth_graph, variable_names = _load_metr_la_arrays(dataset_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not variable_names or len(variable_names) != sequences[0].shape[1]:
        variable_names = _default_variable_names(sequences[0].shape[1])
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

    normalized_splits = {"train": [], "val": [], "test": []}
    for split_name, arrays in split_sequences.items():
        normalized_splits[split_name] = [
            _apply_normalization(array, train_mean, train_std, mode) for array in arrays
        ]

    transformed_mean = np.zeros_like(train_mean) if mode == "zscore" else train_mean.copy()
    transformed_std = np.ones_like(train_std) if mode == "zscore" else train_std.copy()

    train_dataset = SequenceWindowDataset.from_sequences(normalized_splits["train"], lag=lag)
    val_window = SequenceWindowDataset.from_sequences(normalized_splits["val"], lag=lag)
    test_dataset = SequenceWindowDataset.from_sequences(normalized_splits["test"], lag=lag)

    num_timesteps = int(sum(sequence.shape[0] for sequence in sequences))
    gt_num_edges = int(np.count_nonzero(ground_truth_graph) - np.count_nonzero(np.diag(ground_truth_graph)))
    gt_density = float(gt_num_edges / max(sequences[0].shape[1] * max(sequences[0].shape[1] - 1, 1), 1))
    metadata = {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "num_nodes": int(sequences[0].shape[1]),
        "num_timesteps": num_timesteps,
        "sequence_length": int(sequences[0].shape[0]),
        "num_sequences": len(sequences),
        "lag": int(lag),
        "has_ground_truth_graph": True,
        "ground_truth_graph": ground_truth_graph.astype(np.int64),
        "ground_truth_graph_path": _resolve_gt_path(dataset_name, dataset_dir),
        "variable_names": variable_names,
        "gt_num_edges": gt_num_edges,
        "gt_density": gt_density,
        "normalization_mode": mode,
        "normalization_stats": {
            "raw_mean": train_mean.tolist(),
            "raw_std": train_std.tolist(),
            "transformed_mean": transformed_mean.tolist(),
            "transformed_std": transformed_std.tolist(),
        },
        "split_ratio": list(split_ratio),
        "seed": int(seed),
    }

    return train_dataset, val_window, test_dataset, metadata


def load_episode_dataset(
    dataset_name: str,
    data_root: str,
    lag: int,
    split_ratio: tuple[float, float, float],
    normalize: bool | str,
    seed: int,
    l_sup: int,
    l_qry: int,
    train_episodes_per_cycle: int | None = None,
    val_episodes_per_cycle: int | None = None,
) -> tuple[DynamicEpisodeDataset, DynamicEpisodeDataset, SequenceWindowDataset, SequenceWindowDataset, dict]:
    """Load episodic train/val datasets plus standard val/test window datasets.

    Args:
        l_sup: support episode length (number of supervised samples)
        l_qry: query episode length (number of supervised samples)
        train_episodes_per_cycle: episodes per dataloader cycle; auto if None.
        val_episodes_per_cycle: episodes per val dataloader cycle; auto if None.
    """
    if lag < 1:
        raise ValueError("lag must be at least 1.")

    dataset_name = dataset_name.lower()
    mode = _normalize_mode(normalize)
    dataset_dir = _resolve_dataset_dir(dataset_name, data_root)

    # ---- fmri_static: pre-split directory format ----
    if dataset_name == "fmri_static":
        return _load_episode_dataset_fmri_static(
            dataset_dir=dataset_dir,
            lag=lag,
            mode=mode,
            seed=seed,
            l_sup=l_sup,
            l_qry=l_qry,
            train_episodes_per_cycle=train_episodes_per_cycle,
            val_episodes_per_cycle=val_episodes_per_cycle,
        )

    if dataset_name == "nc8":
        sequences, ground_truth_graph, variable_names = _load_nc8_arrays(dataset_dir)
    elif dataset_name == "metr_la_128":
        sequences, ground_truth_graph, variable_names = _load_metr_la_arrays(dataset_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not variable_names or len(variable_names) != sequences[0].shape[1]:
        variable_names = _default_variable_names(sequences[0].shape[1])

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

    normalized_splits = {"train": [], "val": [], "test": []}
    for split_name, arrays in split_sequences.items():
        normalized_splits[split_name] = [
            _apply_normalization(array, train_mean, train_std, mode) for array in arrays
        ]

    transformed_mean = np.zeros_like(train_mean) if mode == "zscore" else train_mean.copy()
    transformed_std = np.ones_like(train_std) if mode == "zscore" else train_std.copy()

    # Auto-compute episodes per cycle based on non-overlapping coverage
    l_epi = l_sup + l_qry

    def _auto_episodes(sequences, l_epi, lag):
        """Non-overlapping episode count × 10 coverage multiplier."""
        independent = sum(
            max(1, (seq.shape[0] - lag) // l_epi)
            for seq in sequences
        )
        return max(independent * 10, 50)

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

    num_timesteps = int(sum(sequence.shape[0] for sequence in sequences))
    gt_num_edges = int(np.count_nonzero(ground_truth_graph) - np.count_nonzero(np.diag(ground_truth_graph)))
    gt_density = float(gt_num_edges / max(sequences[0].shape[1] * max(sequences[0].shape[1] - 1, 1), 1))
    metadata = {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "num_nodes": int(sequences[0].shape[1]),
        "num_timesteps": num_timesteps,
        "sequence_length": int(sequences[0].shape[0]),
        "num_sequences": len(sequences),
        "lag": int(lag),
        "has_ground_truth_graph": True,
        "ground_truth_graph": ground_truth_graph.astype(np.int64),
        "ground_truth_graph_path": _resolve_gt_path(dataset_name, dataset_dir),
        "variable_names": variable_names,
        "gt_num_edges": gt_num_edges,
        "gt_density": gt_density,
        "normalization_mode": mode,
        "normalization_stats": {
            "raw_mean": train_mean.tolist(),
            "raw_std": train_std.tolist(),
            "transformed_mean": transformed_mean.tolist(),
            "transformed_std": transformed_std.tolist(),
        },
        "split_ratio": list(split_ratio),
        "seed": int(seed),
    }

    return train_dataset, val_episode_dataset, val_window_dataset, test_dataset, metadata


def _load_episode_dataset_fmri_static(
    dataset_dir: Path,
    lag: int,
    mode: str,
    seed: int,
    l_sup: int,
    l_qry: int,
    train_episodes_per_cycle: int | None,
    val_episodes_per_cycle: int | None,
) -> tuple[DynamicEpisodeDataset, DynamicEpisodeDataset, SequenceWindowDataset, SequenceWindowDataset, dict]:
    """Episode dataset builder for fmri_static (pre-split directory format)."""
    train_seqs, val_seqs, test_seqs, gt_graph, variable_names = _load_fmri_static_presplit(dataset_dir)

    # Validate all sequences are long enough for episodes
    l_epi = l_sup + l_qry
    min_len_needed = lag + l_epi
    for seqs, split_name in ((train_seqs, "train"), (val_seqs, "val")):
        for i, seq in enumerate(seqs):
            if seq.shape[0] < min_len_needed:
                raise ValueError(
                    f"fmri_static {split_name} sample {i}: T={seq.shape[0]} < "
                    f"lag+l_epi={min_len_needed}. Reduce lag or l_sup/l_qry."
                )

    # Normalization: fit on train only (data is pre-standardized per channel,
    # but we apply zscore again if requested to be consistent across samples)
    train_mean, train_std = _fit_zscore_stats(train_seqs)
    normalized: dict[str, list[np.ndarray]] = {}
    for split_name, seqs in [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]:
        normalized[split_name] = [_apply_normalization(s, train_mean, train_std, mode) for s in seqs]

    transformed_mean = np.zeros_like(train_mean) if mode == "zscore" else train_mean.copy()
    transformed_std = np.ones_like(train_std) if mode == "zscore" else train_std.copy()

    def _auto_episodes(seqs, l_epi, lag):
        independent = sum(max(1, (s.shape[0] - lag) // l_epi) for s in seqs)
        return max(independent * 10, 50)

    if train_episodes_per_cycle is None:
        train_episodes_per_cycle = _auto_episodes(normalized["train"], l_epi, lag)
    if val_episodes_per_cycle is None:
        val_episodes_per_cycle = _auto_episodes(normalized["val"], l_epi, lag)

    train_dataset = DynamicEpisodeDataset(
        sequences=normalized["train"], lag=lag, l_sup=l_sup, l_qry=l_qry,
        steps_per_epoch=train_episodes_per_cycle, seed=seed,
    )
    val_episode_dataset = DynamicEpisodeDataset(
        sequences=normalized["val"], lag=lag, l_sup=l_sup, l_qry=l_qry,
        steps_per_epoch=val_episodes_per_cycle, seed=seed + 1,
    )
    val_window_dataset = SequenceWindowDataset.from_sequences(normalized["val"], lag=lag)
    test_dataset = SequenceWindowDataset.from_sequences(normalized["test"], lag=lag)

    all_seqs = train_seqs + val_seqs + test_seqs
    N = gt_graph.shape[0]
    gt_num_edges = int(np.count_nonzero(gt_graph) - np.count_nonzero(np.diag(gt_graph)))
    gt_density = float(gt_num_edges / max(N * (N - 1), 1))

    metadata = {
        "dataset_name": "fmri_static",
        "dataset_dir": str(dataset_dir),
        "num_nodes": N,
        "num_timesteps": int(sum(s.shape[0] for s in all_seqs)),
        "sequence_length": int(train_seqs[0].shape[0]),
        "num_sequences": len(all_seqs),
        "num_train_samples": len(train_seqs),
        "num_val_samples": len(val_seqs),
        "num_test_samples": len(test_seqs),
        "lag": int(lag),
        "has_ground_truth_graph": True,
        "ground_truth_graph": gt_graph.astype(np.int64),
        "variable_names": variable_names,
        "gt_num_edges": gt_num_edges,
        "gt_density": gt_density,
        "normalization_mode": mode,
        "normalization_stats": {
            "raw_mean": train_mean.tolist(),
            "raw_std": train_std.tolist(),
            "transformed_mean": transformed_mean.tolist(),
            "transformed_std": transformed_std.tolist(),
        },
        "split_ratio": [len(train_seqs), len(val_seqs), len(test_seqs)],
        "seed": int(seed),
    }
    return train_dataset, val_episode_dataset, val_window_dataset, test_dataset, metadata


def _load_episode_dataset_fmri_latent(
    dataset_dir: Path,
    lag: int,
    mode: str,
    seed: int,
    l_sup: int,
    l_qry: int,
    hidden_indices: list[int],
    train_episodes_per_cycle: int | None,
    val_episodes_per_cycle: int | None,
    max_train_sequences: int | None = None,
    dataset_label: str = "fmri_latent",
) -> tuple[LatentEpisodeDataset, LatentEpisodeDataset, SequenceWindowDataset, SequenceWindowDataset, dict]:
    """Episode dataset builder for pre-split latent benchmarks.

    max_train_sequences: if set, only the first N train sequences are used for
    z_global training. Useful when there are many sequences (e.g., 200) and
    z_global gets too few gradient updates per sequence.
    """
    train_seqs, val_seqs, test_seqs, gt_graph, variable_names = _load_fmri_static_presplit(dataset_dir)
    if max_train_sequences is not None and max_train_sequences < len(train_seqs):
        train_seqs = train_seqs[:max_train_sequences]

    N_full = gt_graph.shape[0]
    obs_indices = [i for i in range(N_full) if i not in hidden_indices]
    q = len(hidden_indices)
    N_obs = len(obs_indices)
    perm = obs_indices + hidden_indices

    # Reorder GT (source-target convention)
    gt_reordered = gt_graph[np.ix_(perm, perm)].astype(np.float32)
    gt_xx = gt_reordered[:N_obs, :N_obs]
    gt_obs_to_latent = gt_reordered[:N_obs, N_obs:]
    gt_zx = gt_reordered[N_obs:, :N_obs]
    gt_zz = gt_reordered[N_obs:, N_obs:]

    # Apply masking: split into observed and hidden columns per split
    def _mask_split(seqs):
        obs = [s[:, obs_indices] for s in seqs]
        hid = [s[:, hidden_indices] for s in seqs]
        return obs, hid

    obs_train, hid_train = _mask_split(train_seqs)
    obs_val, hid_val = _mask_split(val_seqs)
    obs_test, _ = _mask_split(test_seqs)

    # Validate episode lengths
    l_epi = l_sup + l_qry
    min_len_needed = lag + l_epi
    for seqs, split_name in ((obs_train, "train"), (obs_val, "val")):
        for i, seq in enumerate(seqs):
            if seq.shape[0] < min_len_needed:
                raise ValueError(
                    f"{dataset_label} {split_name} sample {i}: T={seq.shape[0]} < "
                    f"lag+l_epi={min_len_needed}. Reduce lag or l_sup/l_qry."
                )

    # Normalize observed and hidden independently (fit on train only)
    train_mean, train_std = _fit_zscore_stats(obs_train)
    hid_mean, hid_std = _fit_zscore_stats(hid_train)

    norm_obs_train = [_apply_normalization(s, train_mean, train_std, mode) for s in obs_train]
    norm_obs_val = [_apply_normalization(s, train_mean, train_std, mode) for s in obs_val]
    norm_obs_test = [_apply_normalization(s, train_mean, train_std, mode) for s in obs_test]
    norm_hid_train = [_apply_normalization(s, hid_mean, hid_std, mode) for s in hid_train]
    norm_hid_val = [_apply_normalization(s, hid_mean, hid_std, mode) for s in hid_val]

    def _auto_episodes(seqs, l_epi, lag):
        independent = sum(max(1, (s.shape[0] - lag) // l_epi) for s in seqs)
        return max(independent * 10, 50)

    if train_episodes_per_cycle is None:
        train_episodes_per_cycle = _auto_episodes(norm_obs_train, l_epi, lag)
    if val_episodes_per_cycle is None:
        val_episodes_per_cycle = _auto_episodes(norm_obs_val, l_epi, lag)

    train_dataset = LatentEpisodeDataset(
        obs_sequences=norm_obs_train, hidden_sequences=norm_hid_train,
        lag=lag, l_sup=l_sup, l_qry=l_qry,
        steps_per_epoch=train_episodes_per_cycle, seed=seed,
    )
    val_episode_dataset = LatentEpisodeDataset(
        obs_sequences=norm_obs_val, hidden_sequences=norm_hid_val,
        lag=lag, l_sup=l_sup, l_qry=l_qry,
        steps_per_epoch=val_episodes_per_cycle, seed=seed + 1,
    )
    val_window_dataset = SequenceWindowDataset.from_sequences(norm_obs_val, lag=lag)
    test_dataset = SequenceWindowDataset.from_sequences(norm_obs_test, lag=lag)

    obs_names = [variable_names[i] for i in obs_indices]
    latent_names = [f"Z{k}" for k in range(q)]
    full_names = obs_names + latent_names
    hidden_orig_names = [variable_names[i] for i in hidden_indices]

    gt_full = gt_reordered.astype(np.int64)
    gt_num_edges_full = int(np.count_nonzero(gt_full) - np.count_nonzero(np.diag(gt_full)))
    gt_density_full = float(gt_num_edges_full / max((N_obs + q) * (N_obs + q - 1), 1))
    forbidden_count = int(np.count_nonzero(gt_obs_to_latent))

    all_seqs = train_seqs + val_seqs + test_seqs
    train_sequence_lengths = [int(s.shape[0]) for s in norm_obs_train]
    transformed_mean = np.zeros_like(train_mean) if mode == "zscore" else train_mean.copy()
    transformed_std = np.ones_like(train_std) if mode == "zscore" else train_std.copy()

    hidden_selection_summary = _summarize_hidden_selection(
        ground_truth_graph=gt_graph.astype(np.int64),
        hidden_indices=hidden_indices,
        variable_names=variable_names,
    )

    metadata = {
        "dataset_name": dataset_label,
        "dataset_dir": str(dataset_dir),
        "num_nodes": N_obs,
        "num_obs": N_obs,
        "q_latent": q,
        "N_full": N_obs + q,
        "num_timesteps": int(sum(s.shape[0] for s in all_seqs)),
        "sequence_length": int(train_seqs[0].shape[0]),
        "num_sequences": len(all_seqs),
        "num_train_samples": len(train_seqs),
        "num_val_samples": len(val_seqs),
        "num_test_samples": len(test_seqs),
        "lag": int(lag),
        "has_ground_truth_graph": True,
        "ground_truth_graph": gt_full,
        "gt_full": gt_full,
        "gt_xx": gt_xx.astype(np.int64),
        "gt_zx": gt_zx.astype(np.int64),
        "gt_zz": gt_zz.astype(np.int64),
        "gt_obs_to_latent": gt_obs_to_latent.astype(np.int64),
        "gt_num_edges": gt_num_edges_full,
        "gt_density": gt_density_full,
        "variable_names": full_names,
        "obs_variable_names": obs_names,
        "latent_variable_names": latent_names,
        "hidden_original_variable_names": hidden_orig_names,
        "obs_indices": obs_indices,
        "hidden_indices": hidden_indices,
        "reorder_perm": perm,
        "forbidden_edges_in_gt": forbidden_count,
        "hidden_selection_summary": hidden_selection_summary,
        "normalization_mode": mode,
        "normalization_stats": {
            "raw_mean": train_mean.tolist(),
            "raw_std": train_std.tolist(),
            "transformed_mean": transformed_mean.tolist(),
            "transformed_std": transformed_std.tolist(),
        },
        "train_sequence_lengths": train_sequence_lengths,
        "split_ratio": [len(train_seqs), len(val_seqs), len(test_seqs)],
        "seed": int(seed),
    }

    return train_dataset, val_episode_dataset, val_window_dataset, test_dataset, metadata


def load_episode_dataset_latent(
    dataset_name: str,
    data_root: str,
    lag: int,
    split_ratio: tuple[float, float, float],
    normalize: bool | str,
    seed: int,
    l_sup: int,
    l_qry: int,
    hidden_indices: list[int],
    train_episodes_per_cycle: int | None = None,
    val_episodes_per_cycle: int | None = None,
    max_train_sequences: int | None = None,
) -> tuple[DynamicEpisodeDataset, DynamicEpisodeDataset, SequenceWindowDataset, SequenceWindowDataset, dict]:
    """Load episodic datasets for the latent benchmark.

    Supports nc8_latent plus pre-split latent datasets such as fMRI/EEG. Masks hidden_indices variables as latent,
    returns episode/window datasets over OBSERVED variables only.

    metadata includes:
      num_nodes = N_obs (model input dimension)
      num_obs, q_latent, N_full
      ground_truth_graph: [N_full x N_full] full GT (obs first, latent last)
      gt_xx, gt_zx, gt_zz: block sub-matrices
    """
    if lag < 1:
        raise ValueError("lag must be at least 1.")

    mode = _normalize_mode(normalize)
    dataset_lower = dataset_name.lower()
    if dataset_lower not in {"nc8", "nc8_latent", "fmri_latent", "eeg_latent"}:
        raise ValueError(
            f"load_episode_dataset_latent only supports nc8/nc8_latent/fmri_latent/eeg_latent, got {dataset_name!r}"
        )

    if dataset_lower in {"fmri_latent", "eeg_latent"}:
        dataset_dir = _resolve_dataset_dir(dataset_lower, data_root)
        return _load_episode_dataset_fmri_latent(
            dataset_dir=dataset_dir,
            lag=lag,
            mode=mode,
            seed=seed,
            l_sup=l_sup,
            l_qry=l_qry,
            hidden_indices=hidden_indices,
            train_episodes_per_cycle=train_episodes_per_cycle,
            val_episodes_per_cycle=val_episodes_per_cycle,
            max_train_sequences=max_train_sequences,
            dataset_label=dataset_lower,
        )

    dataset_dir = _resolve_dataset_dir("nc8", data_root)
    sequences, ground_truth_graph, variable_names = _load_nc8_arrays(dataset_dir)

    latent_info = build_nc8_latent_benchmark(
        sequences=sequences,
        ground_truth_graph=ground_truth_graph,
        variable_names=variable_names,
        hidden_indices=hidden_indices,
    )
    N_obs = latent_info["N_obs"]
    q = latent_info["q_latent"]

    obs_sequences = latent_info["obs_sequences"]
    hidden_sequences = latent_info["hidden_sequences"]

    # Split and normalise observed sequences (obs stats)
    split_obs: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_hid: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    for obs_seq, hid_seq in zip(obs_sequences, hidden_sequences):
        obs_splits = _chronological_split(obs_seq, split_ratio)
        hid_splits = _chronological_split(hid_seq, split_ratio)
        for split_name in ("train", "val", "test"):
            if len(obs_splits[split_name]) <= lag:
                raise ValueError(
                    f"Split {split_name!r} length {len(obs_splits[split_name])} <= lag {lag}."
                )
            split_obs[split_name].append(obs_splits[split_name])
            split_hid[split_name].append(hid_splits[split_name])

    train_mean, train_std = _fit_zscore_stats(split_obs["train"])
    hid_mean, hid_std = _fit_zscore_stats(split_hid["train"])

    normalized_splits: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    normalized_hid: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    for split_name in ("train", "val", "test"):
        normalized_splits[split_name] = [
            _apply_normalization(a, train_mean, train_std, mode) for a in split_obs[split_name]
        ]
        normalized_hid[split_name] = [
            _apply_normalization(a, hid_mean, hid_std, mode) for a in split_hid[split_name]
        ]

    transformed_mean = np.zeros_like(train_mean) if mode == "zscore" else train_mean.copy()
    transformed_std = np.ones_like(train_std) if mode == "zscore" else train_std.copy()

    l_epi = l_sup + l_qry

    def _auto_episodes(seqs, l_epi, lag):
        independent = sum(max(1, (s.shape[0] - lag) // l_epi) for s in seqs)
        return max(independent * 10, 50)

    if train_episodes_per_cycle is None:
        train_episodes_per_cycle = _auto_episodes(normalized_splits["train"], l_epi, lag)
    if val_episodes_per_cycle is None:
        val_episodes_per_cycle = _auto_episodes(normalized_splits["val"], l_epi, lag)

    train_dataset = LatentEpisodeDataset(
        obs_sequences=normalized_splits["train"],
        hidden_sequences=normalized_hid["train"],
        lag=lag, l_sup=l_sup, l_qry=l_qry,
        steps_per_epoch=train_episodes_per_cycle, seed=seed,
    )
    val_episode_dataset = LatentEpisodeDataset(
        obs_sequences=normalized_splits["val"],
        hidden_sequences=normalized_hid["val"],
        lag=lag, l_sup=l_sup, l_qry=l_qry,
        steps_per_epoch=val_episodes_per_cycle, seed=seed + 1,
    )
    val_window_dataset = SequenceWindowDataset.from_sequences(normalized_splits["val"], lag=lag)
    test_dataset = SequenceWindowDataset.from_sequences(normalized_splits["test"], lag=lag)

    gt_full = latent_info["gt_full"].astype(np.int64)
    N_full = N_obs + q
    gt_num_edges_full = int(np.count_nonzero(gt_full) - np.count_nonzero(np.diag(gt_full)))
    gt_density_full = float(gt_num_edges_full / max(N_full * (N_full - 1), 1))

    num_timesteps = int(sum(s.shape[0] for s in obs_sequences))
    train_sequence_lengths = [int(seq.shape[0]) for seq in normalized_splits["train"]]

    hidden_selection_summary = _summarize_hidden_selection(
        ground_truth_graph=ground_truth_graph.astype(np.int64),
        hidden_indices=latent_info["hidden_indices"],
        variable_names=variable_names,
    )

    metadata = {
        "dataset_name": "nc8_latent",
        "dataset_dir": str(dataset_dir),
        "num_nodes": N_obs,
        "num_obs": N_obs,
        "q_latent": q,
        "N_full": N_full,
        "num_timesteps": num_timesteps,
        "sequence_length": int(obs_sequences[0].shape[0]),
        "num_sequences": len(obs_sequences),
        "lag": int(lag),
        "has_ground_truth_graph": True,
        "ground_truth_graph": gt_full,
        "gt_full": gt_full,
        "gt_xx": latent_info["gt_xx"].astype(np.int64),
        "gt_zx": latent_info["gt_zx"].astype(np.int64),
        "gt_zz": latent_info["gt_zz"].astype(np.int64),
        "gt_obs_to_latent": latent_info["gt_obs_to_latent"].astype(np.int64),
        "gt_num_edges": gt_num_edges_full,
        "gt_density": gt_density_full,
        "variable_names": latent_info["full_variable_names"],
        "obs_variable_names": latent_info["obs_variable_names"],
        "latent_variable_names": latent_info["latent_variable_names"],
        "hidden_original_variable_names": latent_info["hidden_original_variable_names"],
        "obs_indices": latent_info["obs_indices"],
        "hidden_indices": latent_info["hidden_indices"],
        "reorder_perm": latent_info["reorder_perm"],
        "forbidden_edges_in_gt": latent_info["forbidden_edges_in_gt"],
        "hidden_selection_summary": hidden_selection_summary,
        "normalization_mode": mode,
        "normalization_stats": {
            "raw_mean": train_mean.tolist(),
            "raw_std": train_std.tolist(),
            "transformed_mean": transformed_mean.tolist(),
            "transformed_std": transformed_std.tolist(),
        },
        "train_sequence_lengths": train_sequence_lengths,
        "split_ratio": list(split_ratio),
        "seed": int(seed),
    }

    return train_dataset, val_episode_dataset, val_window_dataset, test_dataset, metadata
