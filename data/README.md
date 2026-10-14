# SCDM simulation benchmarks, version 1

Download the complete original simulation arrays from the
[datasets-v1 release](https://github.com/ShaogangRen/SCDM/releases/tag/datasets-v1).
These files contain simulations with known directed graphs, not recordings from
human participants. The 64-variable quick start is a separate lightweight example.

| Dataset | Variables | Trajectories | Time steps per trajectory | Directed edges |
|---|---:|---:|---:|---:|
| `fmri100` | 100 ROIs | 200 train / 50 validation / 50 test | 1,000 | 1,220 |
| `eeg100` | 100 source-space ROIs | 200 train / 50 validation / 50 test | 2,000 | 1,138 |
| `voxel1000` | 1,000 observed variables | 5 replicas | 800 | 7,142 |

## Download and run

From the repository root, after installing `requirements.txt`:

```bash
python scripts/download_datasets.py
python -m scdm.nonlatent.cli --config configs/nonlatent/fmri100.yaml --device cuda --seed 0
python -m scdm.nonlatent.cli --config configs/nonlatent/eeg100.yaml --device cuda --seed 0
python -m scdm.nonlatent.cli --config configs/nonlatent/voxel1000.yaml --device cuda --seed 0
```

Use `--device cpu` if needed; the full runs can be substantially slower.
Download one dataset with `python scripts/download_datasets.py --datasets fmri100`.
The downloader verifies file size and SHA-256 against `manifest.json` and reuses
verified local files. Data require approximately 348 MB of disk space after download.

## Array format

All packages are NumPy `.npz` files readable with `np.load(path, allow_pickle=False)`.

- `X_train`, `X_val`, `X_test`: original predefined sequence partitions for fMRI/EEG,
  shaped `[sequences, time, variables]`. The loader preserves these partitions.
- `X`: complete observed voxel trajectories, shaped `[5, 800, 1000]`. The loader
  splits each sequence chronologically in the ratio 70/15/15.
- `A`: binary adjacency `[source, target]`. `A[j, i]=1` means `j -> i`.
- `A_lags`: the lag-specific graph `[lag, source, target]`.
- `variable_names`: fixed-width Unicode array; no pickle is required.
- `coords`: ROI/voxel coordinates from the simulation metadata. For voxels, use
  `permuted_index_to_original_index` to map the saved column order back to spatial indices.
- `dataset_info_json`: UTF-8 JSON metadata stored as a Unicode scalar.

The original [fMRI ROI table](metadata/fmri100_rois.csv) and
[EEG ROI table](metadata/eeg100_rois.csv) map `Xj` to zero-based `roi_index=j`,
Schaefer atlas labels, Yeo network labels, hemispheres, and coordinates.
Atlas metadata should be attributed to the atlas authors cited below.

For example:

```python
import json
import numpy as np
with np.load("data/fmri100_v1.npz", allow_pickle=False) as data:
    train = data["X_train"]
    graph = data["A"]
    info = json.loads(str(data["dataset_info_json"]))
    print(train.shape, graph.shape, info["ground_truth_orientation"])
```

The fMRI package contains the original pre-normalized BOLD observation array
`X.npy` from the `fmri_v1b` full view. The EEG package contains the original full
source-space `X.npy` arrays; the 64-channel sensor projection is not the graph
input used by this benchmark. The voxel package contains the original observed
`X_obs.npy` arrays in their saved permutation. Its simulator also has four latent
regional drivers, so this is an empirical scalability benchmark.

Packaging changes storage format only: all observation and graph values are
preserved. It excludes checkpoints, private experiment reports, and logs.

## Evaluation

The fMRI configuration reports structural `|W|` graph scores; EEG reports the
source-isolated readout; voxel reports structural scores. AUROC and AUPRC use
off-diagonal directed pairs. Each configuration runs one specified training seed
and saves both scores and resolved configuration under `results/`.

The listed commands are full-budget single-seed runs using the manuscript's
hyperparameters, not the manuscript's multi-seed aggregates. Their runtime is not
a matched-wall-clock baseline comparison. The pipeline's F1/SHD threshold sweep
is an oracle diagnostic; use AUROC/AUPRC for the displayed ranking comparisons.

## Citation and reuse

The simulation arrays are released under the repository's MIT license. Cite the
SCDM paper when using these benchmarks and cite the versioned dataset release:

```bibtex
@misc{wang2026scdmdata,
  title   = {SCDM Simulation Benchmarks},
  author  = {Wang, Jingbo and Tang, Kegeng and Wang, Zihao and Ren, Shaogang},
  year    = {2026},
  version = {1},
  url     = {https://github.com/ShaogangRen/SCDM/releases/tag/datasets-v1},
  note    = {Simulated fMRI, EEG, and voxel-level temporal causal discovery data}
}
```

The ROI simulations use Schaefer-100 / Yeo network organization. When discussing
the anatomical atlas, also cite Schaefer et al. (2018), *Local-Global
Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity
MRI*, Cerebral Cortex 28(9), 3095–3114. No dataset DOI has been assigned to this
GitHub release.
