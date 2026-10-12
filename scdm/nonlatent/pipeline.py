"""End-to-end SCDM pipeline: load -> meta-train -> graph readout -> evaluate.

Stages
------
1. Data loading      chronological 70/15/15 split, z-score fit on train only,
                     episodic (support/query) and sliding-window datasets.
2. Meta-training     first-order meta-updates of the shared temporal predictor
                     and the shared causal-strength matrix W; the best
                     checkpoint is selected by minimum validation forecast loss.
3. Graph readout     structural readout (raw |W|, always) and, when enabled,
                     the source-isolated readout on validation episodes.
4. Evaluation        AUROC / AUPRC / F1 / SHD per readout, plus one-step
                     forecasting MSE / MAE / RMSE on the test split.

Every score and adjacency matrix written to disk uses the source-target
convention: entry [j, i] refers to the directed edge j -> i.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from scdm.common.data import load_episode_dataset
from scdm.nonlatent.graph_readout import compute_source_isolated_scores, extract_structural_scores
from scdm.nonlatent.meta_training import train_scdm
from scdm.common.metrics import compute_forecasting_metrics, evaluate_scores, summarize
from scdm.nonlatent.models import SCDMModel
from scdm.common.utils import (
    ensure_dir,
    save_json,
    save_numpy,
    save_text,
    save_yaml,
    seed_everything,
    setup_logger,
)

PRIMARY_READOUT_KEY = "graph_readout_primary"


def _resolve_path(project_root: Path, candidate: str | None, fallback: Path) -> Path:
    if not candidate:
        return fallback
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def _resolve_output_dir(project_root: Path, config: dict[str, Any]) -> Path:
    if config.get("output_dir"):
        return _resolve_path(project_root, config["output_dir"], project_root / "results")
    run_name = config.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "results" / str(run_name)


def _build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def run_experiment(
    config: dict[str, Any],
    project_root: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run the SCDM pipeline end to end and write all artifacts to disk."""

    total_start = time.perf_counter()
    project_root = Path(project_root).resolve()
    output_dir = ensure_dir(_resolve_output_dir(project_root, config))
    logger = setup_logger("scdm", output_dir)

    cuda_available = torch.cuda.is_available()
    if cuda_available and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    logger.info("Device: %s | CUDA available: %s", device, cuda_available)

    data_root = _resolve_path(project_root, config.get("data_root"), project_root)
    save_yaml(output_dir / "config_resolved.yaml", config)

    # ---- Stage 1: data ----
    (
        train_dataset,
        val_episode_dataset,
        val_window_dataset,
        test_dataset,
        metadata,
    ) = load_episode_dataset(
        dataset_format=str(config["dataset_format"]),
        dataset_path=str(config["dataset_path"]),
        data_root=str(data_root),
        lag=int(config["lag"]),
        split_ratio=tuple(config["split_ratio"]),
        normalize=str(config["normalize"]),
        seed=int(config["seed"]),
        l_sup=int(config["l_sup"]),
        l_qry=int(config["l_qry"]),
        sequence_glob=str(config.get("sequence_glob", "*_data_*.csv")),
        structure_glob=str(config.get("structure_glob", "*_struct_*.csv")),
        train_episodes_per_cycle=config.get("train_episodes_per_cycle"),
        val_episodes_per_cycle=config.get("val_episodes_per_cycle"),
    )
    num_nodes = int(metadata["num_nodes"])
    has_gt = bool(metadata["has_ground_truth_graph"])
    gt_graph = metadata["ground_truth_graph"]
    logger.info(
        "Dataset: %d variables, %d sequences, %d total timesteps, lag=%d, GT edges=%s",
        num_nodes, metadata["num_sequences"], metadata["num_timesteps"],
        metadata["lag"], metadata["gt_num_edges"],
    )
    save_json(output_dir / "dataset_metadata.json",
              {k: v for k, v in metadata.items() if k != "ground_truth_graph"})

    num_workers = int(config.get("num_workers", 0))
    train_loader = _build_dataloader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, seed=int(config["seed"]),
    )
    val_episode_loader = _build_dataloader(
        val_episode_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, seed=int(config["seed"]),
    )
    test_loader = _build_dataloader(
        test_dataset, batch_size=int(config["batch_size"]), shuffle=False,
        num_workers=num_workers, seed=int(config["seed"]),
    )

    model = SCDMModel(
        num_nodes=num_nodes,
        lag=int(config["lag"]),
        hidden_dim=int(config["hidden_dim"]),
        node_hidden_dim=int(config["node_hidden_dim"]),
        graph_threshold=float(config["graph_threshold"]),
        allow_self_loops=bool(config.get("allow_self_loops", False)),
        top_k=config.get("top_k"),
        w_init_min=float(config.get("w_init_min", 0.0)),
        w_init_max=float(config.get("w_init_max", 0.3)),
        w_components=int(config.get("w_components", 1)),
        reg_mode=str(config.get("reg_mode", "l1")),
        use_structural_w=bool(config.get("use_structural_w", True)),
    )

    # ---- Stage 2: meta-training ----
    train_start = time.perf_counter()
    trainer, best_checkpoint_path = train_scdm(
        model=model,
        train_loader=train_loader,
        val_episode_loader=val_episode_loader,
        config=config,
        device=device,
        output_dir=output_dir,
        logger=logger,
    )
    training_runtime_s = time.perf_counter() - train_start
    trainer.load_checkpoint(best_checkpoint_path)
    logger.info("Loaded best checkpoint from step %d.", trainer.best_step)

    # ---- Stage 3a: structural readout (scalable, always run) ----
    readout_start = time.perf_counter()
    structural = extract_structural_scores(trainer.model, config)
    save_numpy(output_dir / "structural_scores.npy", structural["scores"])
    save_numpy(output_dir / "W_signed.npy", structural["W_signed"])
    structural_runtime_s = time.perf_counter() - readout_start

    threshold_mode = str(config.get("threshold_mode", "acc_max"))
    structural_eval = evaluate_scores(
        scores=structural["scores"],
        gt_graph=gt_graph,
        threshold_mode=threshold_mode,
        graph_threshold=float(config["graph_threshold"]),
        top_k=config.get("top_k"),
        allow_self_loops=bool(config.get("allow_self_loops", False)),
    )
    save_numpy(output_dir / "structural_binary_graph.npy", structural_eval["binary_graph"])
    logger.info(summarize("structural readout (raw |W|)", structural_eval))

    # ---- Stage 3b: source-isolated readout (optional) ----
    configured_readout = str(config.get("graph_readout", "")).lower().strip()
    if configured_readout and configured_readout not in {"structural", "source_isolated"}:
        raise ValueError(
            f"Unsupported graph_readout: {configured_readout!r} "
            "(expected 'structural' or 'source_isolated')."
        )

    scoring_episodes = int(config.get("source_isolated_episodes", 0))
    run_source_isolated = bool(config.get("run_source_isolated_readout", True)) and scoring_episodes > 0

    source_isolated_eval = None
    source_isolated_runtime_s = None
    if run_source_isolated:
        logger.info("Running source-isolated readout (%d episodes)...", scoring_episodes)
        si_start = time.perf_counter()
        # Fixed scoring seed so the readout is reproducible independently of the
        # training seed, matching the configuration used for the paper runs.
        seed_everything(int(config.get("readout_seed", 0)))
        _, scoring_episode_dataset, _, _, _ = load_episode_dataset(
            dataset_format=str(config["dataset_format"]),
            dataset_path=str(config["dataset_path"]),
            data_root=str(data_root),
            lag=int(config["lag"]),
            split_ratio=tuple(config["split_ratio"]),
            normalize=str(config["normalize"]),
            seed=int(config.get("readout_seed", 0)),
            l_sup=int(config["l_sup"]),
            l_qry=int(config["l_qry"]),
            sequence_glob=str(config.get("sequence_glob", "*_data_*.csv")),
            structure_glob=str(config.get("structure_glob", "*_struct_*.csv")),
        )
        source_isolated_scores = compute_source_isolated_scores(
            trainer=trainer,
            scoring_episode_dataset=scoring_episode_dataset,
            num_episodes=scoring_episodes,
        )
        source_isolated_runtime_s = time.perf_counter() - si_start
        save_numpy(output_dir / "source_isolated_scores.npy", source_isolated_scores)

        source_isolated_eval = evaluate_scores(
            scores=source_isolated_scores,
            gt_graph=gt_graph,
            threshold_mode=threshold_mode,
            graph_threshold=float(config["graph_threshold"]),
            top_k=config.get("top_k"),
            allow_self_loops=bool(config.get("allow_self_loops", False)),
        )
        save_numpy(
            output_dir / "source_isolated_binary_graph.npy",
            source_isolated_eval["binary_graph"],
        )
        logger.info(summarize("source-isolated readout", source_isolated_eval))
    else:
        logger.info("Source-isolated readout disabled; using the structural readout only.")

    if gt_graph is not None:
        save_numpy(output_dir / "gt_binary_graph.npy", np.asarray(gt_graph).astype(np.int64))

    # ---- Stage 4: forecasting metrics ----
    predictions = trainer.predict(test_loader)
    forecast_metrics = compute_forecasting_metrics(predictions["y_true"], predictions["y_pred"])
    logger.info(
        "Test forecasting: MSE=%.6f MAE=%.6f RMSE=%.6f",
        forecast_metrics["mse"], forecast_metrics["mae"], forecast_metrics["rmse"],
    )

    if configured_readout == "source_isolated" and source_isolated_eval is None:
        raise ValueError(
            "graph_readout is 'source_isolated', but source-isolated readout did not run. "
            "Set run_source_isolated_readout: true and source_isolated_episodes > 0, "
            "or set graph_readout: structural."
        )
    primary_readout = (
        configured_readout
        if configured_readout
        else ("source_isolated" if source_isolated_eval is not None else "structural")
    )
    primary_eval = source_isolated_eval if source_isolated_eval is not None else structural_eval
    if primary_readout == "structural":
        primary_eval = structural_eval

    graph_output_manifest = {
        "orientation_note": (
            "The paper denotes W[source, target]. The implementation stores the "
            "equivalent matrix internally as W[target, source] so that one target "
            "task corresponds to one trainable row. All exported and evaluated "
            "matrices listed here are transposed back to source-target convention."
        ),
        "files": {
            "structural_scores.npy": {
                "orientation": "source-target",
                "entry": "score[source, target] = |W_internal[target, source]|",
                "self_edges": "zero when allow_self_loops is false",
            },
            "structural_binary_graph.npy": {
                "orientation": "source-target",
                "entry": "binary[source, target]",
                "self_edges": "zero when allow_self_loops is false",
            },
            "W_signed.npy": {
                "orientation": "source-target",
                "entry": "signed W_internal[target, source] for K=1; edge norm for K>1",
                "self_edges": "zero when allow_self_loops is false",
            },
            "gt_binary_graph.npy": {
                "orientation": "source-target",
                "entry": "ground_truth[source, target]",
                "self_edges": "metrics ignore diagonal",
            },
        },
    }
    if source_isolated_eval is not None:
        graph_output_manifest["files"].update({
            "source_isolated_scores.npy": {
                "orientation": "source-target",
                "entry": (
                    "score[source, target] = "
                    "(loss_zero(target) - loss_source_only(source, target)) / "
                    "(loss_zero(target) + eps)"
                ),
                "self_edges": "self-pairs skipped and left at zero",
            },
            "source_isolated_binary_graph.npy": {
                "orientation": "source-target",
                "entry": "binary[source, target]",
                "self_edges": "zero when allow_self_loops is false",
            },
        })
    save_json(output_dir / "graph_outputs_manifest.json", graph_output_manifest)

    total_runtime_s = time.perf_counter() - total_start
    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        if cuda_available and device.type == "cuda" else None
    )

    def _readout_payload(evaluation: dict | None) -> dict | None:
        if evaluation is None:
            return None
        return {
            "ranking_metrics": evaluation.get("ranking_metrics"),
            "binary_metrics": evaluation.get("binary_metrics"),
            "threshold": evaluation.get("threshold"),
            "threshold_mode": evaluation.get("threshold_mode"),
            "predicted_edges": evaluation.get("predicted_edges"),
            "predicted_density": evaluation.get("predicted_density"),
            "score_stats": evaluation.get("score_stats"),
        }

    result_payload = {
        "artifact_scope": (
            "Anonymous reviewer artifact: core SCDM implementation with lightweight "
            "fixed-data examples. Not a full reproduction of the paper tables."
        ),
        "dataset": {
            "format": metadata["dataset_format"],
            "num_variables": num_nodes,
            "num_sequences": metadata["num_sequences"],
            "num_timesteps": metadata["num_timesteps"],
            "lag": metadata["lag"],
            "has_ground_truth_graph": has_gt,
            "gt_num_edges": metadata["gt_num_edges"],
            "gt_density": metadata["gt_density"],
            "variable_names": metadata["variable_names"],
        },
        "config": config,
        "training": {
            "training_mode": trainer.training_mode,
            "best_step": trainer.best_step,
            "best_val_forecast_loss": trainer.best_val_loss,
            "total_outer_steps": int(config["total_outer_steps"]),
            "tasks_per_outer_step": trainer.tasks_per_outer_step,
            "target_sampling_mode": trainer.target_sampling_mode,
            "num_targets_used_per_step": (
                num_nodes
                if trainer.tasks_per_outer_step <= 0
                or trainer.tasks_per_outer_step >= num_nodes
                else trainer.tasks_per_outer_step
            ),
            "forecast_metrics": forecast_metrics,
        },
        "graph_readout_structural": {
            "description": (
                "Scalable structural readout: |W| read directly from the shared "
                "causal-strength matrix, transposed to source-target convention."
            ),
            "output_orientation": "source-target",
            **(_readout_payload(structural_eval) or {}),
        },
        "graph_readout_source_isolated": (
            {
                "description": (
                    "Source-isolated readout: score[i, j] = "
                    "(loss_zero - loss_source_only) / (loss_zero + eps) after "
                    "inner-loop adaptation on the support window of each episode."
                ),
                "output_orientation": "source-target",
                "scoring_episodes": scoring_episodes,
                **(_readout_payload(source_isolated_eval) or {}),
            }
            if source_isolated_eval is not None
            else {"enabled": False}
        ),
        PRIMARY_READOUT_KEY: {
            "readout": primary_readout,
            "configured_readout": configured_readout or None,
            "output_orientation": "source-target",
            "auroc": (primary_eval.get("ranking_metrics") or {}).get("auroc"),
            "auprc": (primary_eval.get("ranking_metrics") or {}).get("auprc"),
            "f1": (primary_eval.get("binary_metrics") or {}).get("f1"),
            "shd": (primary_eval.get("binary_metrics") or {}).get("shd"),
        },
        "runtime": {
            "training_runtime_s": training_runtime_s,
            "structural_readout_runtime_s": structural_runtime_s,
            "source_isolated_readout_runtime_s": source_isolated_runtime_s,
            "total_runtime_s": total_runtime_s,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "device": str(device),
        },
    }
    save_json(output_dir / "result.json", result_payload)

    summary_lines = [
        f"Variables: {num_nodes}   Lag: {metadata['lag']}   "
        f"GT edges: {metadata['gt_num_edges']}   GT density: {metadata['gt_density']}",
        f"Training mode: {trainer.training_mode}   "
        f"Outer steps: {config['total_outer_steps']}   "
        f"Tasks/step: {trainer.tasks_per_outer_step or 'all'}   "
        f"Best step: {trainer.best_step}",
        "",
        "=== Graph recovery (threshold mode: %s) ===" % threshold_mode,
        summarize("structural readout (raw |W|)", structural_eval),
    ]
    if source_isolated_eval is not None:
        summary_lines.append(summarize("source-isolated readout", source_isolated_eval))
    summary_lines += [
        f"Primary readout: {primary_readout}",
        "",
        "=== Forecasting (test split) ===",
        f"MSE={forecast_metrics['mse']:.6f}  MAE={forecast_metrics['mae']:.6f}  "
        f"RMSE={forecast_metrics['rmse']:.6f}",
        "",
        f"Total runtime: {total_runtime_s:.1f}s",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    save_text(output_dir / "summary.txt", summary_text)
    logger.info("Run summary\n%s", summary_text.rstrip())
    logger.info("Artifacts written to %s", output_dir)

    return {
        "output_dir": str(output_dir),
        "best_checkpoint_path": str(best_checkpoint_path),
        "result": result_payload,
        "summary": summary_text,
    }
