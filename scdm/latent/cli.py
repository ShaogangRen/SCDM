"""Latent-aware SCDM command-line entry point.

    python -m scdm.latent.cli --config configs/latent/<name>.yaml

This runs the PARTIALLY OBSERVED setting: the learner sees only the observed
variables and jointly infers q explicit latent nodes. For the fully observed
setting use ``python -m scdm.nonlatent.cli`` instead; the two never share a
config file, and neither package imports the other.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scdm.common.utils import load_and_resolve_config, resolve_device, seed_everything
from scdm.latent.pipeline import run_experiment

# Repository root = two levels above scdm/latent/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "latent" / "default.yaml"


def _str2bool(value: str) -> bool:
    lowered = str(value).lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _int_list(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Latent-aware SCDM: causal discovery with explicit latent nodes.",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config layered on configs/latent/default.yaml.")
    # Data / benchmark
    parser.add_argument("--dataset", type=str,
                        choices=["nc8_latent", "fmri_latent", "eeg_latent"], default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--hidden-indices", type=_int_list, default=None,
                        help="Indices of the withheld (latent) variables, e.g. '5,6'.")
    parser.add_argument("--q-latent", type=int, default=None,
                        help="Number of latent slots; must equal len(hidden_indices).")
    parser.add_argument("--lag", type=int, default=None)
    parser.add_argument("--normalize", type=str, choices=["none", "zscore"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    # Model
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--node-hidden-dim", type=int, default=None)
    parser.add_argument("--w-init-min", type=float, default=None)
    parser.add_argument("--w-init-max", type=float, default=None)
    parser.add_argument("--w-init-min-latent", type=float, default=None)
    parser.add_argument("--w-init-max-latent", type=float, default=None)
    parser.add_argument("--wzz-init-diag", type=float, default=None)
    parser.add_argument("--z-init-scale", type=float, default=None)
    parser.add_argument("--use-wzz", type=_str2bool, default=None)
    parser.add_argument("--wzz-diagonal-only", type=_str2bool, default=None)
    parser.add_argument("--allow-self-loops", type=_str2bool, default=None)
    # Latent objective
    parser.add_argument("--lambda-W", dest="lambda_W", type=float, default=None)
    parser.add_argument("--lambda-W-latent", dest="lambda_W_latent", type=float, default=None)
    parser.add_argument("--lambda-z", dest="lambda_z", type=float, default=None)
    parser.add_argument("--lambda-wzx-col", dest="lambda_wzx_col", type=float, default=None)
    parser.add_argument("--lambda-wzx-sparse", dest="lambda_wzx_sparse", type=float, default=None)
    parser.add_argument("--lambda-wzx-div", dest="lambda_wzx_div", type=float, default=None)
    # Meta-training
    parser.add_argument("--inner-loop-steps", type=int, default=None)
    parser.add_argument("--inner-loop-lr", type=float, default=None)
    parser.add_argument("--inner-loop-lr-w", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--total-outer-steps", type=int, default=None)
    parser.add_argument("--tasks-per-outer-step", type=int, default=None)
    parser.add_argument("--val-every-outer-steps", type=int, default=None)
    parser.add_argument("--val-episodes", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--l-sup", type=int, default=None)
    parser.add_argument("--l-qry", type=int, default=None)
    # Readout / evaluation
    parser.add_argument("--checkpoint-selection-rule", type=str,
                        choices=["forecast", "graph_aware"], default=None)
    parser.add_argument("--graph-threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    # Run control
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
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

    hidden_indices = list(config.get("hidden_indices") or [])
    if not hidden_indices:
        raise SystemExit(
            "hidden_indices must be non-empty: the latent-aware setting requires at "
            "least one withheld variable. Use the non-latent pipeline "
            "(python -m scdm.nonlatent.cli) for fully observed data."
        )
    if config.get("q_latent") is not None and int(config["q_latent"]) != len(hidden_indices):
        raise SystemExit(
            f"q_latent ({config['q_latent']}) must equal len(hidden_indices) "
            f"({len(hidden_indices)})."
        )

    seed_everything(int(config["seed"]))
    device = resolve_device(config.get("device"))
    run_experiment(config=config, project_root=PROJECT_ROOT, device=device)


if __name__ == "__main__":
    main()
