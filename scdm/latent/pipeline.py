"""End-to-end latent-aware SCDM pipeline.

Stages
------
1. Data loading            observed variables only; the withheld variables are
                           removed from the learner's input, while the full
                           ground-truth graph is retained for evaluation.
2. Meta-training           first-order meta-updates of the shared predictor
                           theta, the block-structured structural matrix W, and
                           the explicit latent trajectories Z.
3. Checkpoint selection    ``forecast`` (minimum validation forecast loss) or
                           ``graph_aware`` (rank periodic checkpoints by
                           latent-block recovery; see below).
4. Latent-aware evaluation full-graph recovery with latent-slot matching, plus
                           per-block XX / ZX / ZZ metrics and observed
                           one-step forecasting.

Block convention
----------------
The structural matrix is [N_full x N_full] with N_full = N_obs + q, latent
nodes last. Internally W[i, j] is the gate of source j on target i; the
obs -> latent block is permanently zero by design assumption. All matrices
written to disk are transposed into source-target convention, so entry [j, i]
scores the directed edge j -> i:

    [:N_obs, :N_obs]  XX   observed -> observed
    [N_obs:, :N_obs]  ZX   latent   -> observed
    [:N_obs, N_obs:]  --   observed -> latent  (structurally zero)
    [N_obs:, N_obs:]  ZZ   latent   -> latent

Latent slots are exchangeable, so full-graph metrics are computed under the
best matching permutation of the q latent slots; see
``scdm/latent/evaluation/latent_eval.py``.

The checkpoint-selection helpers below are copied verbatim from the source
implementation so that the selected checkpoint is identical.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from scdm.common.metrics import compute_forecasting_metrics, compute_threshold_sweep
from scdm.common.utils import (
    ensure_dir,
    save_json,
    save_numpy,
    save_text,
    save_yaml,
    setup_logger,
)
from scdm.latent.data import load_episode_dataset_latent
from scdm.latent.evaluation import evaluate_latent_graph
from scdm.latent.meta_training import train_latent_scdm
from scdm.latent.models import LatentCausalModel


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


# ----------------------------------------------------------------------
# Checkpoint selection (copied verbatim from the source implementation)
# ----------------------------------------------------------------------


def _metric_or_neg_inf(value: float | None) -> float:
    return float(value) if value is not None else float("-inf")


def _auto_periodic_ckpt_steps(config: dict[str, Any]) -> list[int]:
    total_outer_steps = int(config.get("total_outer_steps", 0))
    val_every = int(config.get("val_every_outer_steps", 0))
    if total_outer_steps <= 0 or val_every <= 0:
        return []
    steps = list(range(val_every, total_outer_steps + 1, val_every))
    if total_outer_steps not in steps:
        steps.append(total_outer_steps)
    return sorted(set(step for step in steps if step > 0))


def _maybe_enable_periodic_ckpts(config: dict[str, Any]) -> None:
    if str(config.get("checkpoint_selection_rule", "forecast")).lower() == "forecast":
        return
    if config.get("periodic_ckpt_steps"):
        return
    config["periodic_ckpt_steps"] = _auto_periodic_ckpt_steps(config)


def _evaluate_checkpoint_candidate(
    trainer,
    checkpoint_path: Path,
    gt_full: np.ndarray,
    N_obs: int,
    q_latent: int,
    allow_self_loops: bool,
) -> dict[str, Any]:
    graph_outputs = trainer.extract_graph(checkpoint_path=checkpoint_path)
    W_abs_np = np.asarray(graph_outputs["W_abs"]).astype(np.float32)
    sweep_result = compute_threshold_sweep(W_abs_np, gt_full)
    configured_thr = float(trainer.config.get("graph_threshold", 0.1))
    effective_threshold = sweep_result.get("acc_max_threshold") or configured_thr

    _test_pred = (W_abs_np > effective_threshold).astype(np.int64)
    if not allow_self_loops:
        np.fill_diagonal(_test_pred, 0)
    _test_pred[:N_obs, N_obs:] = 0
    if int(_test_pred.sum()) == 0:
        effective_threshold = configured_thr

    pred_binary_full = (W_abs_np > effective_threshold).astype(np.int64)
    if not allow_self_loops:
        np.fill_diagonal(pred_binary_full, 0)
    pred_binary_full[:N_obs, N_obs:] = 0

    latent_eval = evaluate_latent_graph(
        score_matrix=W_abs_np,
        binary_graph=pred_binary_full,
        gt_graph=gt_full,
        N_obs=N_obs,
        q_latent=q_latent,
    )

    xx_m = latent_eval["xx"]
    zx_m = latent_eval["zx_best"]
    best_full_binary = latent_eval["best"]["full_binary"]
    best_full_ranking = latent_eval["best"]["full_ranking"]

    return {
        "effective_threshold": float(effective_threshold),
        "best_latent_perm": latent_eval["best_perm"],
        "matched_full_auroc": best_full_ranking.get("auroc"),
        "matched_full_f1": best_full_binary.get("f1"),
        "xx_auroc": xx_m.get("auroc"),
        "xx_f1": xx_m.get("f1"),
        "xx_gt_edges": xx_m.get("gt_edges"),
        "xx_pred_edges": xx_m.get("pred_edges"),
        "zx_auroc": zx_m.get("auroc"),
        "zx_f1": zx_m.get("f1"),
        "zx_gt_edges": zx_m.get("gt_edges"),
        "zx_pred_edges": zx_m.get("pred_edges"),
    }


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    if int(candidate.get("zx_gt_edges") or 0) > 0:
        return (
            _metric_or_neg_inf(candidate.get("zx_auroc")),
            _metric_or_neg_inf(candidate.get("zx_f1")),
            _metric_or_neg_inf(candidate.get("xx_auroc")),
            _metric_or_neg_inf(candidate.get("matched_full_auroc")),
            -float(candidate.get("val_forecast_loss"))
            if candidate.get("val_forecast_loss") is not None
            else float("-inf"),
        )
    return (
        _metric_or_neg_inf(candidate.get("xx_auroc")),
        _metric_or_neg_inf(candidate.get("xx_f1")),
        _metric_or_neg_inf(candidate.get("matched_full_auroc")),
        -float(candidate.get("val_forecast_loss"))
        if candidate.get("val_forecast_loss") is not None
        else float("-inf"),
    )


def _select_checkpoint(
    trainer,
    output_dir: Path,
    config: dict[str, Any],
    gt_full: np.ndarray,
    N_obs: int,
    q_latent: int,
) -> dict[str, Any]:
    history_by_step = {
        int(r["outer_step"]): r for r in trainer.history.records if "outer_step" in r
    }
    candidate_paths: list[Path] = []
    for step in config.get("periodic_ckpt_steps", []):
        path = output_dir / f"step_{int(step):04d}.ckpt"
        if path.exists():
            candidate_paths.append(path)
    for name in ("best.ckpt", "last.ckpt"):
        path = output_dir / name
        if path.exists():
            candidate_paths.append(path)

    candidates_by_step: dict[int, dict[str, Any]] = {}
    allow_self_loops = bool(config.get("allow_self_loops", False))
    for path in candidate_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        step = int(payload.get("step", -1))
        if step < 0:
            continue
        graph_metrics = _evaluate_checkpoint_candidate(
            trainer=trainer,
            checkpoint_path=path,
            gt_full=gt_full,
            N_obs=N_obs,
            q_latent=q_latent,
            allow_self_loops=allow_self_loops,
        )
        val_record = history_by_step.get(step, {})
        candidate = {
            "step": step,
            "checkpoint_name": path.name,
            "checkpoint_path": str(path),
            "val_forecast_loss": val_record.get(
                "val_forecast_loss", payload.get("validation_score")
            ),
            **graph_metrics,
        }
        prev = candidates_by_step.get(step)
        if prev is None or _candidate_sort_key(candidate) > _candidate_sort_key(prev):
            candidates_by_step[step] = candidate

    if not candidates_by_step:
        best_path = output_dir / "best.ckpt"
        return {
            "selection_rule": "forecast",
            "selected_step": int(trainer.best_step),
            "selected_checkpoint_name": best_path.name,
            "selected_checkpoint_path": str(best_path),
            "candidates": [],
        }

    selection_rule = (
        "zx_auroc_then_zx_f1_then_xx_auroc_then_matched_full_auroc_then_val_forecast_loss"
        if q_latent > 0
        else "xx_auroc_then_xx_f1_then_matched_full_auroc_then_val_forecast_loss"
    )
    candidates = sorted(
        candidates_by_step.values(), key=lambda c: (c["step"], c["checkpoint_name"])
    )
    selected = max(candidates, key=_candidate_sort_key)
    return {
        "selection_rule": selection_rule,
        "selected_step": int(selected["step"]),
        "selected_checkpoint_name": selected["checkpoint_name"],
        "selected_checkpoint_path": selected["checkpoint_path"],
        "selected_metrics": selected,
        "candidates": candidates,
    }


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def run_experiment(
    config: dict[str, Any],
    project_root: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run the latent-aware SCDM pipeline end to end."""

    total_start = time.perf_counter()
    project_root = Path(project_root).resolve()
    output_dir = ensure_dir(_resolve_output_dir(project_root, config))
    logger = setup_logger("scdm_latent", output_dir)

    _maybe_enable_periodic_ckpts(config)
    logger.info("Device: %s | checkpoint_selection_rule=%s", device,
                config.get("checkpoint_selection_rule", "forecast"))

    data_root = _resolve_path(project_root, config.get("data_root"), project_root)
    save_yaml(output_dir / "config_resolved.yaml", config)

    # ---- Stage 1: data (observed only; full GT retained for evaluation) ----
    (
        train_dataset,
        val_episode_dataset,
        val_window_dataset,
        test_dataset,
        metadata,
    ) = load_episode_dataset_latent(
        dataset_name=str(config["dataset"]),
        data_root=str(data_root),
        lag=int(config["lag"]),
        split_ratio=tuple(config["split_ratio"]),
        normalize=str(config["normalize"]),
        seed=int(config["seed"]),
        l_sup=int(config["l_sup"]),
        l_qry=int(config["l_qry"]),
        hidden_indices=list(config["hidden_indices"]),
        train_episodes_per_cycle=config.get("train_episodes_per_cycle"),
        val_episodes_per_cycle=config.get("val_episodes_per_cycle"),
        max_train_sequences=config.get("max_train_sequences"),
    )
    N_obs = int(metadata["num_obs"])
    q_latent = int(metadata["q_latent"])
    gt_full = np.asarray(metadata["gt_full"]).astype(np.int64)
    logger.info(
        "Dataset %s: N_obs=%d, q_latent=%d, N_full=%d, full GT edges=%s",
        metadata["dataset_name"], N_obs, q_latent, N_obs + q_latent,
        metadata.get("gt_num_edges"),
    )
    save_json(
        output_dir / "dataset_metadata.json",
        {k: v for k, v in metadata.items()
         if k not in {"gt_full", "gt_xx", "gt_zx", "gt_zz", "gt_obs_to_latent",
                      "ground_truth_graph"}},
    )
    save_numpy(output_dir / "gt_full.npy", gt_full)

    num_workers = int(config.get("num_workers", 0))
    train_loader = _build_dataloader(train_dataset, 1, False, num_workers, int(config["seed"]))
    val_episode_loader = _build_dataloader(
        val_episode_dataset, 1, False, num_workers, int(config["seed"])
    )
    test_loader = _build_dataloader(
        test_dataset, int(config["batch_size"]), False, num_workers, int(config["seed"])
    )

    model = LatentCausalModel(
        num_obs=N_obs,
        lag=int(config["lag"]),
        hidden_dim=int(config["hidden_dim"]),
        node_hidden_dim=int(config["node_hidden_dim"]),
        graph_threshold=float(config["graph_threshold"]),
        allow_self_loops=bool(config.get("allow_self_loops", False)),
        q_latent=q_latent,
        top_k=config.get("top_k"),
        w_init_min=float(config.get("w_init_min", 0.0)),
        w_init_max=float(config.get("w_init_max", 0.3)),
        w_init_min_latent=config.get("w_init_min_latent"),
        w_init_max_latent=config.get("w_init_max_latent"),
        wzz_init_diag=config.get("wzz_init_diag"),
        z_init_scale=float(config.get("z_init_scale", 0.0)),
        reg_mode=str(config.get("reg_mode", "l1")),
        use_wzz=bool(config.get("use_wzz", True)),
        wzz_diagonal_only=bool(config.get("wzz_diagonal_only", False)),
        train_sequence_lengths=metadata.get("train_sequence_lengths"),
    )

    # ---- Stage 2: meta-training ----
    train_start = time.perf_counter()
    trainer, best_checkpoint_path = train_latent_scdm(
        model=model,
        train_loader=train_loader,
        val_episode_loader=val_episode_loader,
        config=config,
        device=device,
        output_dir=output_dir,
        logger=logger,
    )
    training_runtime_s = time.perf_counter() - train_start

    # ---- Stage 3: checkpoint selection ----
    rule = str(config.get("checkpoint_selection_rule", "forecast")).lower()
    if rule == "graph_aware":
        selection_info = _select_checkpoint(
            trainer=trainer, output_dir=output_dir, config=config,
            gt_full=gt_full, N_obs=N_obs, q_latent=q_latent,
        )
    else:
        selection_info = {
            "selection_rule": "forecast",
            "selected_step": int(trainer.best_step),
            "selected_checkpoint_name": Path(best_checkpoint_path).name,
            "selected_checkpoint_path": str(best_checkpoint_path),
            "candidates": [],
        }
    selected_ckpt = Path(selection_info["selected_checkpoint_path"])
    logger.info(
        "Checkpoint selection (%s): step %d (%s)",
        selection_info["selection_rule"],
        selection_info["selected_step"],
        selection_info["selected_checkpoint_name"],
    )
    save_json(output_dir / "checkpoint_selection.json", selection_info)

    # ---- Stage 4: latent-aware evaluation at the selected checkpoint ----
    graph_outputs = trainer.extract_graph(checkpoint_path=selected_ckpt)
    W_abs_np = np.asarray(graph_outputs["W_abs"]).astype(np.float32)

    sweep_result = compute_threshold_sweep(W_abs_np, gt_full)
    configured_thr = float(config.get("graph_threshold", 0.1))
    effective_threshold = sweep_result.get("acc_max_threshold") or configured_thr
    allow_self_loops = bool(config.get("allow_self_loops", False))

    _test_pred = (W_abs_np > effective_threshold).astype(np.int64)
    if not allow_self_loops:
        np.fill_diagonal(_test_pred, 0)
    _test_pred[:N_obs, N_obs:] = 0
    if int(_test_pred.sum()) == 0:
        effective_threshold = configured_thr

    pred_binary_full = (W_abs_np > effective_threshold).astype(np.int64)
    if not allow_self_loops:
        np.fill_diagonal(pred_binary_full, 0)
    pred_binary_full[:N_obs, N_obs:] = 0

    latent_eval = evaluate_latent_graph(
        score_matrix=W_abs_np,
        binary_graph=pred_binary_full,
        gt_graph=gt_full,
        N_obs=N_obs,
        q_latent=q_latent,
    )

    save_numpy(output_dir / "W_abs_full.npy", W_abs_np)
    save_numpy(output_dir / "pred_binary_graph_full.npy", pred_binary_full)
    save_json(output_dir / "graph_outputs_manifest.json", {
        "orientation_note": (
            "Internal W is target-source. Exported latent graph files use the paper "
            "source-target convention with observed nodes first and latent slots last."
        ),
        "blocks_source_target": {
            "XX_observed_to_observed": "[:N_obs, :N_obs]",
            "ZX_latent_to_observed": "[N_obs:, :N_obs]",
            "ZZ_latent_to_latent": "[N_obs:, N_obs:]",
            "OX_observed_to_latent_forbidden": "[:N_obs, N_obs:] hard-zeroed",
        },
        "files": {
            "gt_full.npy": {
                "orientation": "source-target",
                "entry": "ground_truth[source, target]",
            },
            "W_abs_full.npy": {
                "orientation": "source-target",
                "entry": "score[source, target] = |W_internal[target, source]|",
            },
            "pred_binary_graph_full.npy": {
                "orientation": "source-target",
                "entry": "binary[source, target]; observed-to-latent block is zero",
            },
        },
    })

    predictions = trainer.predict(test_loader)
    forecast_metrics = compute_forecasting_metrics(
        predictions["y_true"], predictions["y_pred"]
    )

    total_runtime_s = time.perf_counter() - total_start

    xx = latent_eval["xx"]
    zx = latent_eval["zx_best"]
    zz = latent_eval.get("zz_best") or {}
    best_full_ranking = latent_eval["best"]["full_ranking"]
    best_full_binary = latent_eval["best"]["full_binary"]

    result_payload = {
        "setting": "latent_aware",
        "artifact_scope": (
            "Anonymous reviewer artifact: latent-aware SCDM. Metrics depend on the "
            "benchmark and budget configured; see docs/EXPERIMENT_STATUS.md."
        ),
        "dataset": {
            "name": metadata["dataset_name"],
            "num_observed": N_obs,
            "q_latent": q_latent,
            "num_full": N_obs + q_latent,
            "lag": metadata["lag"],
            "hidden_indices": metadata.get("hidden_indices"),
            "gt_num_edges_full": metadata.get("gt_num_edges"),
            "gt_density_full": metadata.get("gt_density"),
            "forbidden_edges_in_gt": metadata.get("forbidden_edges_in_gt"),
        },
        "config": config,
        "training": {
            "best_step_forecast": trainer.best_step,
            "best_val_forecast_loss": trainer.best_val_loss,
            "total_outer_steps": int(config["total_outer_steps"]),
            "tasks_per_outer_step": int(config.get("tasks_per_outer_step", 0)),
            "forecast_metrics_observed": forecast_metrics,
        },
        "checkpoint_selection": selection_info,
        "graph_recovery": {
            "effective_threshold": float(effective_threshold),
            "best_latent_permutation": latent_eval["best_perm"],
            "full_matched": {
                "auroc": best_full_ranking.get("auroc"),
                "auprc": best_full_ranking.get("auprc"),
                "f1": best_full_binary.get("f1"),
                "shd": best_full_binary.get("shd"),
            },
            "block_xx": {
                "auroc": xx.get("auroc"), "auprc": xx.get("auprc"),
                "f1": xx.get("f1"), "shd": xx.get("shd"),
                "gt_edges": xx.get("gt_edges"), "pred_edges": xx.get("pred_edges"),
            },
            "block_zx": {
                "auroc": zx.get("auroc"), "auprc": zx.get("auprc"),
                "f1": zx.get("f1"), "shd": zx.get("shd"),
                "gt_edges": zx.get("gt_edges"), "pred_edges": zx.get("pred_edges"),
            },
            "block_zz": {
                "auroc": zz.get("auroc"), "auprc": zz.get("auprc"),
                "f1": zz.get("f1"), "shd": zz.get("shd"),
                "gt_edges": zz.get("gt_edges"), "pred_edges": zz.get("pred_edges"),
            },
        },
        "runtime": {
            "training_runtime_s": training_runtime_s,
            "total_runtime_s": total_runtime_s,
            "device": str(device),
        },
    }
    save_json(output_dir / "result.json", result_payload)

    summary_lines = [
        f"Setting: latent-aware SCDM",
        f"Dataset: {metadata['dataset_name']}   N_obs={N_obs}   q_latent={q_latent}   "
        f"lag={metadata['lag']}",
        f"Full GT edges: {metadata.get('gt_num_edges')}   "
        f"hidden_indices={metadata.get('hidden_indices')}",
        f"Checkpoint: step {selection_info['selected_step']} "
        f"({selection_info['selected_checkpoint_name']}, rule={selection_info['selection_rule']})",
        "",
        "=== Full-graph recovery (best latent-slot matching) ===",
        f"AUROC={_fmt(best_full_ranking.get('auroc'))}  "
        f"AUPRC={_fmt(best_full_ranking.get('auprc'))}  "
        f"F1={_fmt(best_full_binary.get('f1'))}  "
        f"SHD={_fmt(best_full_binary.get('shd'))}",
        "",
        "=== Per-block recovery ===",
        f"XX (obs->obs)    AUROC={_fmt(xx.get('auroc'))}  F1={_fmt(xx.get('f1'))}  "
        f"gt_edges={xx.get('gt_edges')}  pred_edges={xx.get('pred_edges')}",
        f"ZX (latent->obs) AUROC={_fmt(zx.get('auroc'))}  F1={_fmt(zx.get('f1'))}  "
        f"gt_edges={zx.get('gt_edges')}  pred_edges={zx.get('pred_edges')}",
        f"ZZ (latent->lat) AUROC={_fmt(zz.get('auroc'))}  F1={_fmt(zz.get('f1'))}  "
        f"gt_edges={zz.get('gt_edges')}  pred_edges={zz.get('pred_edges')}",
        "",
        "=== Observed one-step forecasting (test split) ===",
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
        "selected_checkpoint_path": str(selected_ckpt),
        "result": result_payload,
        "summary": summary_text,
    }
