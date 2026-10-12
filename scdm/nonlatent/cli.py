"""Non-latent SCDM command-line entry point.

    python -m scdm.nonlatent.cli --config configs/nonlatent/<name>.yaml

This runs the FULLY OBSERVED setting. For the latent-aware setting use
``python -m scdm.latent.cli`` instead; the two never share a config file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scdm.nonlatent.pipeline import run_experiment
from scdm.common.utils import load_and_resolve_config, resolve_device, seed_everything

# Repository root = two levels above scdm/nonlatent/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "nonlatent" / "default.yaml"


def _str2bool(value: str) -> bool:
    lowered = str(value).lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SCDM: meta-learned causal discovery on multivariate time series.",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config layered on top of configs/default.yaml.")
    # Data
    parser.add_argument("--dataset-format", type=str, choices=["csv_pair", "npz"], default=None)
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Dataset directory (csv_pair) or .npz file, relative to --data-root.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory that --dataset-path is resolved against.")
    parser.add_argument("--sequence-glob", type=str, default=None)
    parser.add_argument("--structure-glob", type=str, default=None)
    parser.add_argument("--lag", type=int, default=None)
    parser.add_argument("--normalize", type=str, choices=["none", "zscore"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    # Model
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--node-hidden-dim", type=int, default=None)
    parser.add_argument("--w-init-min", type=float, default=None)
    parser.add_argument("--w-init-max", type=float, default=None)
    parser.add_argument("--w-components", type=int, default=None)
    parser.add_argument("--reg-mode", type=str, choices=["l1", "group_lasso"], default=None)
    parser.add_argument("--use-structural-w", type=_str2bool, default=None)
    parser.add_argument("--allow-self-loops", type=_str2bool, default=None)
    # Meta-training
    parser.add_argument("--training-mode", type=str,
                        choices=["fomaml", "no_meta", "joint_supervised"], default=None)
    parser.add_argument("--inner-loop-steps", type=int, default=None)
    parser.add_argument("--inner-loop-lr", type=float, default=None)
    parser.add_argument("--inner-loop-lr-w", type=float, default=None)
    parser.add_argument("--lambda-in", type=float, default=None)
    parser.add_argument("--lambda-out", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--total-outer-steps", type=int, default=None)
    parser.add_argument("--tasks-per-outer-step", type=int, default=None)
    parser.add_argument("--target-sampling-mode", type=str,
                        choices=["uniform_random", "coverage_aware"], default=None)
    parser.add_argument("--val-every-outer-steps", type=int, default=None)
    parser.add_argument("--val-episodes", type=int, default=None)
    parser.add_argument("--val-tasks-per-episode", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--l-sup", type=int, default=None)
    parser.add_argument("--l-qry", type=int, default=None)
    # Readout and evaluation
    parser.add_argument("--run-source-isolated-readout", type=_str2bool, default=None)
    parser.add_argument("--source-isolated-episodes", type=int, default=None)
    parser.add_argument("--threshold-mode", type=str,
                        choices=["acc_max", "f1_max", "fixed", "top_k"], default=None)
    parser.add_argument("--graph-threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    # Run control
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--readout-seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="'auto' (default), 'cpu', 'cuda', or 'cuda:N'.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    override_config_path = Path(args.config) if args.config else None
    config = load_and_resolve_config(
        default_config_path=DEFAULT_CONFIG,
        override_config_path=override_config_path,
        cli_args=args,
    )
    config.pop("config", None)

    seed_everything(int(config["seed"]))
    device = resolve_device(config.get("device"))
    run_experiment(config=config, project_root=PROJECT_ROOT, device=device)


if __name__ == "__main__":
    main()
