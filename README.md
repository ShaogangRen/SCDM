# SCDM

**Scalable Causal Discovery in Nonlinear Temporal Systems with Meta-Learning**

Jingbo Wang, Kegeng Tang, Zihao Wang, and Shaogang Ren · NeurIPS 2026

[Project webpage](https://shaogangren.github.io/SCDM/) · [Paper](https://openreview.net/forum?id=H2Bm4cicZc)

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CPU examples are provided below. For GPU execution, install the PyTorch build
appropriate for your CUDA environment and pass `--device cuda`.

## Quick start

Run an NC8 example with the bundled data:

```bash
python -m scdm.nonlatent.cli --config configs/nonlatent/nc8_quick.yaml --device cpu
```

Run the 64-variable synthetic example:

```bash
python -m scdm.nonlatent.cli --config configs/nonlatent/highdim_quick.yaml --device cpu
```

The example configurations specify their output directories under `results/`.
These commands demonstrate the implementation using compact training budgets.

A latent NC8 diagnostic is also included:

```bash
python -m scdm.latent.cli --config configs/latent/nc8_appendix_c_quick.yaml --device cpu
```

## Core implementation

SCDM treats each target variable as a related temporal prediction task. Support
adaptation creates temporary task parameters; query losses update the shared
predictor and structural parameters. Graph scores are read out after training.

| Component | Location |
| --- | --- |
| Shared utilities, data loading and metrics | `scdm/common/` |
| Fully observed models | `scdm/nonlatent/models/` |
| Episodic meta-training | `scdm/nonlatent/meta_training/trainer.py` |
| Structural and source-isolated readouts | `scdm/nonlatent/graph_readout/` |
| Fully observed experiment entry point | `scdm/nonlatent/cli.py` |
| Latent-aware implementation | `scdm/latent/` |
| Example configurations | `configs/` |
| Bundled example data | `examples/` |

## Citation

```bibtex
@inproceedings{wang2026scdm,
  title = {SCDM: Scalable Causal Discovery in Nonlinear Temporal Systems with Meta-Learning},
  author = {Wang, Jingbo and Tang, Kegeng and Wang, Zihao and Ren, Shaogang},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  url = {https://openreview.net/forum?id=H2Bm4cicZc}
}
```

## Project webpage

The repository also hosts the project webpage through GitHub Pages using
`main` and `/ (root)`. Website text and data are in `content.js`; layout is in
`index.html`, `styles.css` and `academic.css`; figures are in `assets/`.
