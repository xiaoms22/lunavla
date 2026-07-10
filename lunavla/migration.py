"""One-way configuration migration from LunaVLA v1.1 to v2."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from lunavla.config import ExperimentConfig


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return copy.deepcopy(dict(value))


def migrate_v11_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a resolved v1.1 mapping and validate the v2 result."""

    payload = _mapping(source, "root")
    version = int(payload.get("schema_version", 0))
    if version == 2:
        return ExperimentConfig.from_mapping(payload).to_dict()
    if version != 1:
        raise ValueError(f"expected schema_version 1 or 2, got {version}")

    # Resolve v1 compatibility aliases before interpreting any selector. In
    # particular, v1 ``act`` meant the NumPy linear teaching policy, whereas
    # v2 reserves ``act`` for the capability-complete Transformer.
    from trainer.config import ExperimentConfig as V11ExperimentConfig

    payload = V11ExperimentConfig.from_mapping(payload).to_dict()

    policy = _mapping(payload.get("policy", {}), "policy")
    dataset = _mapping(payload.get("dataset", {}), "dataset")
    training = _mapping(payload.get("training", {}), "training")
    evaluation = _mapping(payload.get("eval", {}), "eval")
    artifacts = _mapping(payload.get("artifacts", {}), "artifacts")
    policy_type = str(policy.get("type", ""))
    if not policy_type:
        raise ValueError("v1.1 policy.type is required")

    dataset_parameters = {
        key: copy.deepcopy(value)
        for key, value in dataset.items()
        if key not in {"source", "split", "seed", "path", "num_episodes"}
    }
    task_parameters = {
        key: copy.deepcopy(value)
        for key, value in evaluation.items()
        if key
        in {
            "success_distance",
            "start_low",
            "start_high",
            "action_clip",
        }
    }
    v2: dict[str, Any] = {
        "schema_version": 2,
        "project_name": str(payload.get("project_name", "lunavla-v2-migrated")),
        "engine": "lunavla_v2",
        "policy": {
            "type": policy_type,
            "state_dim": int(policy.get("observation_dim", 4)),
            "instruction_dim": int(policy.get("instruction_dim", 0)),
            "action_dim": int(policy.get("action_dim", 2)),
            "chunk_size": int(policy.get("chunk_size", 1)),
            "device": str(training.get("device", "cpu")),
        },
        "task": {
            "id": str(payload.get("task", "pusht_style_point_reach")),
            "family": "point_reach",
            "max_steps": int(evaluation.get("rollout_steps", 40)),
            "goal": list(evaluation.get("goal", dataset.get("goal", [0.8, 0.2]))),
            "parameters": task_parameters,
        },
        "dataset": {
            "type": str(dataset.get("source", "memory")),
            "split": str(dataset.get("split", "train")),
            "seed": int(dataset.get("seed", 42)),
            "parameters": dataset_parameters,
        },
        "training": {
            "device": str(training.get("device", "cpu")),
            "seed": int(training.get("seed", 42)),
            "batch_size": int(training.get("batch_size", 32)),
            "steps": int(training.get("num_steps", 100)),
            "learning_rate": float(training.get("learning_rate", 0.04)),
            "kl_weight": 0.0,
        },
        "evaluation": {
            "execution_mode": str(evaluation.get("execution_mode", "receding_horizon")),
            "episodes": int(evaluation.get("episodes", 20)),
            "seed": int(evaluation.get("seed", 1000)),
            "language_ablation": "none",
            "image_ablation": "none",
            "parameters": {},
        },
        "artifacts": {
            "output_dir": str(artifacts.get("output_dir", "outputs/v2-migrated")),
            "checkpoint_name": str(artifacts.get("checkpoint_name", "checkpoint.json")),
        },
    }
    if "hidden_dim" in policy:
        v2["policy"]["hidden_dim"] = int(policy["hidden_dim"])
    if "path" in dataset:
        v2["dataset"]["path"] = str(dataset["path"])
    if "num_episodes" in dataset:
        v2["dataset"]["episode_count"] = int(dataset["num_episodes"])
    if "seeds" in evaluation:
        v2["evaluation"]["seeds"] = [int(value) for value in evaluation["seeds"]]
    if "report_path" in artifacts:
        v2["artifacts"]["report_path"] = str(artifacts["report_path"])
    return ExperimentConfig.from_mapping(v2).to_dict()


def migrate_v11_file(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    overwrite: bool = False,
) -> ExperimentConfig:
    source = Path(source_path)
    destination = Path(destination_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing config: {destination}")
    with source.open("r", encoding="utf-8-sig") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("configuration file must contain a mapping")
    config = ExperimentConfig.from_mapping(migrate_v11_mapping(payload))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.to_dict(), stream, sort_keys=False, allow_unicode=True)
    return config
