from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from dataset import (
    build_training_arrays,
    compact_action_statistics,
    compute_action_statistics,
    load_dataset_splits_from_config,
    save_jsonl,
    write_action_statistics,
)
from model import NumpyBCMLPPolicy, NumpyLinearChunkPolicy
from trainer.artifacts import RunManifest
from trainer.config import ExperimentConfig
from trainer.trainer_utils import (
    append_jsonl,
    prepare_run_dir,
    write_csv,
    write_json,
    write_run_card,
)


ROOT = Path(__file__).resolve().parents[1]


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _dataset_config_for_runtime(config: ExperimentConfig) -> dict[str, Any]:
    dataset_config = dict(config.dataset)
    if dataset_config.get("source") == "jsonl":
        data_path = Path(str(dataset_config["path"]))
        if not data_path.is_absolute():
            dataset_config["path"] = str(ROOT / data_path)
    return dataset_config


def train_from_config(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    expected_policy_type: str | None = None,
) -> dict[str, Any]:
    """Run one deterministic NumPy training job from a strict config."""

    source_path = Path(config_path)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    config = ExperimentConfig.load(source_path)
    if expected_policy_type and config.policy["type"] != expected_policy_type:
        raise ValueError(
            f"this entry point requires policy.type={expected_policy_type}; "
            f"got {config.policy['type']}"
        )

    output_path = Path(str(config.artifacts["output_dir"]))
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_dir = prepare_run_dir(output_path, overwrite=overwrite)
    write_json(output_dir / "config.resolved.json", config.to_dict())

    splits = load_dataset_splits_from_config(_dataset_config_for_runtime(config))
    split_name = str(config.dataset["split"])
    records = splits[split_name]
    if not records:
        raise ValueError(f"selected dataset split {split_name!r} contains no records")
    for name, split_records in splits.items():
        if split_records:
            save_jsonl(split_records, output_dir / f"{name}_records.jsonl")
    data_path = output_dir / f"{split_name}_records.jsonl"

    chunk_size = int(config.policy["chunk_size"])
    action_dim = int(config.policy["action_dim"])
    arrays = build_training_arrays(
        records,
        chunk_size=chunk_size,
        instruction_dim=int(config.policy["instruction_dim"]),
    )
    expected_input_dim = int(config.policy["observation_dim"]) + int(
        config.policy["instruction_dim"]
    )
    if arrays.inputs.shape[1] != expected_input_dim:
        raise ValueError(
            f"dataset input dimension is {arrays.inputs.shape[1]}, expected {expected_input_dim}"
        )
    if arrays.targets.shape[2] != action_dim:
        raise ValueError(
            f"dataset action dimension is {arrays.targets.shape[2]}, expected {action_dim}"
        )

    action_stats = compute_action_statistics(
        records,
        source=f"{split_name}_records",
        action_dim=action_dim,
        clip_limit=float(config.eval["action_clip"]),
    )
    action_stats_path = output_dir / "action_statistics.json"
    write_action_statistics(action_stats_path, action_stats)
    action_stats_ref = _display_path(action_stats_path)

    train_config = config.training
    train_seed = int(train_config["seed"])
    rng = np.random.default_rng(train_seed)
    policy: NumpyLinearChunkPolicy | NumpyBCMLPPolicy
    if config.policy["type"] == NumpyLinearChunkPolicy.policy_name:
        policy = NumpyLinearChunkPolicy(
            input_dim=arrays.inputs.shape[1],
            action_dim=action_dim,
            chunk_size=chunk_size,
            seed=train_seed,
        )
    elif config.policy["type"] == NumpyBCMLPPolicy.policy_name:
        policy = NumpyBCMLPPolicy(
            input_dim=arrays.inputs.shape[1],
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=int(config.policy["hidden_dim"]),
            seed=train_seed,
        )
    else:  # ExperimentConfig already rejects this; keep the runtime invariant explicit.
        raise ValueError(f"unsupported policy.type: {config.policy['type']!r}")

    num_steps = int(train_config["num_steps"])
    batch_size = min(int(train_config["batch_size"]), len(arrays.inputs))
    learning_rate = float(train_config["learning_rate"])
    log_interval = int(train_config["log_interval"])
    loss_rows: list[dict[str, float | int]] = []
    metrics_path = output_dir / "metrics.jsonl"
    for step in range(1, num_steps + 1):
        indices = rng.choice(
            len(arrays.inputs), size=batch_size, replace=len(arrays.inputs) < batch_size
        )
        loss = policy.train_step(
            arrays.inputs[indices],
            arrays.targets[indices],
            learning_rate,
            valid_mask=arrays.valid_mask[indices],
        )
        if step == 1 or step % log_interval == 0 or step == num_steps:
            row = {"step": step, "loss": round(loss, 8)}
            loss_rows.append(row)
            append_jsonl(metrics_path, row)

    metadata = {
        **config.to_dict(),
        "config_path": _display_path(source_path),
        "policy_interface": {
            "policy_class": policy.__class__.__name__,
            "policy_name": policy.policy_name,
            "contract": "predict_chunk(sample)->ActionChunk; predict_action(sample)->first_action",
        },
        "action_stats": compact_action_statistics(action_stats, path=action_stats_ref),
    }
    checkpoint_path = output_dir / "checkpoint.json"
    policy.save(checkpoint_path, metadata=metadata)

    summary: dict[str, Any] = {
        "project_name": config.project_name,
        "policy_name": policy.policy_name,
        "policy_interface": "MiniVLAPolicyBase",
        "task": config.task,
        "dataset_source": config.dataset["source"],
        "dataset_path": config.dataset.get("path", "n/a"),
        "dataset_split": split_name,
        "checkpoint": _display_path(checkpoint_path),
        "records": len(records),
        "input_dim": int(arrays.inputs.shape[1]),
        "target_shape": list(arrays.targets.shape[1:]),
        "chunk_size": chunk_size,
        "action_stats_path": action_stats_ref,
        "action_mean": action_stats["action"]["mean"],
        "action_std": action_stats["action"]["std"],
        "action_min": action_stats["action"]["min"],
        "action_max": action_stats["action"]["max"],
        "num_steps": num_steps,
        "learning_rate": learning_rate,
        "final_loss": loss_rows[-1]["loss"],
    }
    if isinstance(policy, NumpyBCMLPPolicy):
        summary["hidden_dim"] = policy.hidden_dim
    write_csv(output_dir / "loss_curve.csv", loss_rows)
    write_json(output_dir / "training_summary.json", summary)
    write_run_card(output_dir / "run_card.md", "LunaVLA Training Run", summary)
    manifest = RunManifest.create(
        root=ROOT,
        config=config,
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        splits=splits,
        command=[sys.executable, *sys.argv],
        metrics={"training": {"final_loss": summary["final_loss"]}},
    )
    manifest.write(output_dir / "manifest.json")
    return summary
