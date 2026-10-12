"""NC8 latent benchmark construction.

Masks q real NC8 variables and treats them as hidden latent nodes.
The learner only receives the observed (unmasked) variables as input.
The full N_full-node GT is preserved for full-graph evaluation.

Masking protocol
----------------
1. Choose hidden_indices (default: last q variables).
2. Reorder: observed variables first, hidden (latent) variables last.
3. Return obs-only sequences for model input.
4. Return the full reordered GT for evaluation.

Block structure of the reordered GT (source-target convention, A[src, tgt]):
    [ GT_xx  GT_ox ]   obs source, obs/latent target
    [ GT_zx  GT_zz ]   latent source, obs/latent target

where GT_ox (obs->latent) may be non-zero in the raw NC8 GT
(the model sets this block to zero as a structural assumption).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def build_nc8_latent_benchmark(
    sequences: list[np.ndarray],
    ground_truth_graph: np.ndarray,
    variable_names: list[str],
    hidden_indices: Sequence[int],
) -> dict:
    """Construct the NC8 latent benchmark by masking selected variables.

    Args:
        sequences: list of [T, N_full] float32 arrays (full NC8 data, unnormalised).
        ground_truth_graph: [N_full, N_full] binary adjacency in source-target
            convention: GT[i, j] = 1 means edge i -> j.
        variable_names: length-N_full list of column names from the CSV.
        hidden_indices: indices of variables to hide (treat as latent nodes).
            All other variables are treated as observed.

    Returns:
        dict with fields:
          obs_sequences        list of [T, N_obs] arrays (observed only)
          full_sequences       list of [T, N_full] arrays (reordered: obs first)
          obs_indices          list[int] indices of observed variables in original
          hidden_indices       list[int] indices of latent variables in original
          obs_variable_names   list[str] names of observed nodes
          latent_variable_names list[str] names of latent slots (Z0, Z1, ...)
          full_variable_names  obs_names + latent_names
          gt_full              [N_full, N_full] GT reordered (obs first, latent last)
          gt_xx                [N_obs, N_obs] GT obs->obs block
          gt_zx                [q, N_obs]    GT latent->obs block
          gt_zz                [q, q]        GT latent->latent block
          gt_obs_to_latent     [N_obs, q]    GT obs->latent block (may be non-zero)
          N_obs                int
          q_latent             int
          reorder_perm         list[int] full permutation applied [obs..., hidden...]
          forbidden_edges_in_gt int count of GT edges in the obs->latent block
    """
    N_full = ground_truth_graph.shape[0]
    hidden_indices = list(hidden_indices)
    if len(hidden_indices) == 0:
        raise ValueError("hidden_indices must be non-empty.")
    for idx in hidden_indices:
        if idx < 0 or idx >= N_full:
            raise ValueError(f"hidden index {idx} out of range [0, {N_full}).")

    obs_indices = [i for i in range(N_full) if i not in hidden_indices]
    q = len(hidden_indices)
    N_obs = len(obs_indices)

    # Reorder permutation: obs first, latent last
    perm = obs_indices + hidden_indices

    # Reorder sequences
    full_sequences_reordered = [seq[:, perm] for seq in sequences]
    obs_sequences = [seq[:, :N_obs] for seq in full_sequences_reordered]

    # Reorder GT (source-target convention: GT[src, tgt])
    # np.ix_ applies the permutation to both row (source) and column (target) dimensions
    gt_reordered = ground_truth_graph[np.ix_(perm, perm)].astype(np.float32)

    # Extract blocks (source-target convention)
    gt_xx = gt_reordered[:N_obs, :N_obs]   # obs->obs
    gt_obs_to_latent = gt_reordered[:N_obs, N_obs:]  # obs->latent (forbidden in model)
    gt_zx = gt_reordered[N_obs:, :N_obs]   # latent->obs
    gt_zz = gt_reordered[N_obs:, N_obs:]   # latent->latent

    obs_names = [variable_names[i] for i in obs_indices]
    latent_names = [f"Z{k}" for k in range(q)]
    full_names = obs_names + latent_names
    # Original names of the masked variables (before renaming to Z0, Z1, ...)
    hidden_original_names = [variable_names[i] for i in hidden_indices]

    forbidden_count = int(np.count_nonzero(gt_obs_to_latent))

    hidden_sequences = [seq[:, N_obs:] for seq in full_sequences_reordered]

    return {
        "obs_sequences": obs_sequences,
        "hidden_sequences": hidden_sequences,
        "full_sequences": full_sequences_reordered,
        "obs_indices": obs_indices,
        "hidden_indices": hidden_indices,
        "obs_variable_names": obs_names,
        "latent_variable_names": latent_names,
        "hidden_original_variable_names": hidden_original_names,
        "full_variable_names": full_names,
        "gt_full": gt_reordered,
        "gt_xx": gt_xx.astype(np.float32),
        "gt_zx": gt_zx.astype(np.float32),
        "gt_zz": gt_zz.astype(np.float32),
        "gt_obs_to_latent": gt_obs_to_latent.astype(np.float32),
        "N_obs": N_obs,
        "q_latent": q,
        "reorder_perm": perm,
        "forbidden_edges_in_gt": forbidden_count,
    }
