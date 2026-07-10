from __future__ import annotations

import copy
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from model.policy_io import canonical_policy_type


CONFIG_SCHEMA_VERSION = 1
CANONICAL_TASK = "pusht_style_point_reach"


def _reject_unknown(section: str, payload: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown field(s) in {section}: {', '.join(unknown)}")


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{section} must be a mapping")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if result < 0 if allow_zero else result <= 0:
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return result


@dataclass(frozen=True)
class ExperimentConfig:
    """Strict, versioned experiment configuration used by training and evaluation."""

    schema_version: int
    project_name: str
    framework: str
    task: str
    policy: dict[str, Any]
    dataset: dict[str, Any]
    training: dict[str, Any]
    eval: dict[str, Any]
    artifacts: dict[str, Any]

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "ExperimentConfig":
        payload = copy.deepcopy(dict(source))
        _reject_unknown(
            "root",
            payload,
            {
                "schema_version",
                "project_name",
                "framework",
                "task",
                "policy",
                "dataset",
                "training",
                "eval",
                "artifacts",
                "model",  # v1.0 migration input only
            },
        )
        schema_version = int(payload.get("schema_version", CONFIG_SCHEMA_VERSION))
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"unsupported config schema_version: {schema_version}")
        if "schema_version" not in payload:
            warnings.warn(
                "unversioned v1.0 config is deprecated; add schema_version: 1",
                DeprecationWarning,
                stacklevel=2,
            )

        legacy_model = _mapping(payload.get("model", {}), "model")
        _reject_unknown(
            "model",
            legacy_model,
            {
                "name",
                "policy_type",
                "observation_dim",
                "instruction_dim",
                "action_dim",
                "chunk_size",
            },
        )
        if legacy_model:
            warnings.warn(
                "model.* is deprecated; dimensions and policy selection now live in policy.*",
                DeprecationWarning,
                stacklevel=2,
            )

        policy = _mapping(payload.get("policy", {}), "policy")
        _reject_unknown(
            "policy",
            policy,
            {
                "type",
                "name",  # v1.0 migration input only
                "chunk_size",
                "hidden_dim",
                "observation_dim",
                "instruction_dim",
                "action_dim",
            },
        )
        raw_policy_candidates = [
            value
            for value in (
                policy.get("type"),
                policy.get("name"),
                legacy_model.get("policy_type"),
            )
            if value is not None
        ]
        if not raw_policy_candidates:
            raise ValueError("policy.type is required")
        canonical_candidates = [canonical_policy_type(str(value)) for value in raw_policy_candidates]
        if len(set(canonical_candidates)) != 1:
            raise ValueError(f"conflicting policy selectors: {raw_policy_candidates}")
        policy_type = canonical_candidates[0]
        supported = {"numpy_linear_chunk", "numpy_bc_mlp"}
        if policy_type not in supported:
            raise ValueError(f"unsupported policy.type: {policy_type!r}")
        if "name" in policy:
            warnings.warn(
                "policy.name is deprecated; use policy.type",
                DeprecationWarning,
                stacklevel=2,
            )

        chunk_candidates = [
            int(value)
            for value in (policy.get("chunk_size"), legacy_model.get("chunk_size"))
            if value is not None
        ]
        if len(set(chunk_candidates)) > 1:
            raise ValueError(f"conflicting chunk_size values: {chunk_candidates}")
        canonical_policy: dict[str, Any] = {
            "type": policy_type,
            "chunk_size": _positive_int(chunk_candidates[0] if chunk_candidates else 1, "policy.chunk_size"),
            "observation_dim": _positive_int(
                policy.get("observation_dim", legacy_model.get("observation_dim", 4)),
                "policy.observation_dim",
            ),
            "instruction_dim": int(
                policy.get("instruction_dim", legacy_model.get("instruction_dim", 8))
            ),
            "action_dim": _positive_int(
                policy.get("action_dim", legacy_model.get("action_dim", 2)),
                "policy.action_dim",
            ),
        }
        if canonical_policy["instruction_dim"] < 0:
            raise ValueError("policy.instruction_dim cannot be negative")
        if policy_type == "numpy_bc_mlp":
            canonical_policy["hidden_dim"] = _positive_int(
                policy.get("hidden_dim", 32), "policy.hidden_dim"
            )
        elif "hidden_dim" in policy:
            raise ValueError("policy.hidden_dim is only valid for numpy_bc_mlp")

        raw_task = str(payload.get("task", CANONICAL_TASK))
        if raw_task in {"pusht", "pusht_mock"}:
            warnings.warn(
                f"task alias {raw_task!r} is deprecated; use {CANONICAL_TASK!r}",
                DeprecationWarning,
                stacklevel=2,
            )
            task = CANONICAL_TASK
        else:
            task = raw_task
        if task != CANONICAL_TASK:
            raise ValueError(f"unsupported task: {task!r}")

        dataset = _mapping(payload.get("dataset", {}), "dataset")
        _reject_unknown(
            "dataset",
            dataset,
            {
                "source",
                "path",
                "num_episodes",
                "steps_per_episode",
                "seed",
                "split_seed",
                "train_fraction",
                "validation_fraction",
                "test_fraction",
                "split",
                "language_instruction",
                "goal",
                "start_low",
                "start_high",
                "action_gain",
                "action_clip",
                "action_noise_std",
                "success_distance",
            },
        )
        dataset.setdefault("source", "mock_pusht")
        if dataset["source"] not in {"mock_pusht", "generated", "jsonl"}:
            raise ValueError(f"unsupported dataset.source: {dataset['source']!r}")
        if dataset["source"] == "jsonl" and not dataset.get("path"):
            raise ValueError("dataset.source=jsonl requires dataset.path")
        dataset["seed"] = int(dataset.get("seed", 42))
        dataset["split_seed"] = int(dataset.get("split_seed", dataset["seed"]))
        dataset["train_fraction"] = float(dataset.get("train_fraction", 0.8))
        dataset["validation_fraction"] = float(dataset.get("validation_fraction", 0.1))
        dataset["test_fraction"] = float(dataset.get("test_fraction", 0.1))
        fractions = (
            dataset["train_fraction"]
            + dataset["validation_fraction"]
            + dataset["test_fraction"]
        )
        if abs(fractions - 1.0) > 1e-8:
            raise ValueError("dataset split fractions must sum to 1")
        if any(dataset[name] < 0 for name in ("train_fraction", "validation_fraction", "test_fraction")):
            raise ValueError("dataset split fractions cannot be negative")
        dataset["split"] = str(dataset.get("split", "train"))
        if dataset["split"] not in {"train", "validation", "test"}:
            raise ValueError("dataset.split must be train, validation, or test")
        if "num_episodes" in dataset:
            dataset["num_episodes"] = _positive_int(dataset["num_episodes"], "dataset.num_episodes")
        if "steps_per_episode" in dataset:
            dataset["steps_per_episode"] = _positive_int(
                dataset["steps_per_episode"], "dataset.steps_per_episode"
            )

        training = _mapping(payload.get("training", {}), "training")
        _reject_unknown(
            "training",
            training,
            {"device", "batch_size", "num_steps", "learning_rate", "seed", "log_interval"},
        )
        training = {
            "device": str(training.get("device", "cpu")).lower(),
            "batch_size": _positive_int(training.get("batch_size", 32), "training.batch_size"),
            "num_steps": _positive_int(training.get("num_steps", 100), "training.num_steps"),
            "learning_rate": _positive_float(
                training.get("learning_rate", 0.04), "training.learning_rate"
            ),
            "seed": int(training.get("seed", 42)),
            "log_interval": _positive_int(training.get("log_interval", 10), "training.log_interval"),
        }
        if training["device"] != "cpu":
            raise ValueError(
                f"{policy_type} is NumPy-only and requires training.device=cpu; "
                f"got {training['device']!r}"
            )

        evaluation = _mapping(payload.get("eval", {}), "eval")
        _reject_unknown(
            "eval",
            evaluation,
            {
                "episodes",
                "rollout_steps",
                "success_distance",
                "seed",
                "seeds",
                "goal",
                "start_low",
                "start_high",
                "action_clip",
                "execution_mode",
            },
        )
        execution_mode = str(
            evaluation.get(
                "execution_mode",
                "open_loop_chunk" if canonical_policy["chunk_size"] > 1 else "receding_horizon",
            )
        )
        if execution_mode not in {"open_loop_chunk", "receding_horizon"}:
            raise ValueError("eval.execution_mode must be open_loop_chunk or receding_horizon")
        evaluation = {
            "episodes": _positive_int(evaluation.get("episodes", 10), "eval.episodes"),
            "rollout_steps": _positive_int(
                evaluation.get("rollout_steps", 40), "eval.rollout_steps"
            ),
            "success_distance": _positive_float(
                evaluation.get("success_distance", dataset.get("success_distance", 0.10)),
                "eval.success_distance",
            ),
            "seed": int(evaluation.get("seed", 1000)),
            "goal": list(evaluation.get("goal", dataset.get("goal", [0.80, 0.20]))),
            "start_low": float(evaluation.get("start_low", dataset.get("start_low", 0.05))),
            "start_high": float(evaluation.get("start_high", dataset.get("start_high", 0.95))),
            "action_clip": _positive_float(
                evaluation.get("action_clip", dataset.get("action_clip", 0.12)),
                "eval.action_clip",
            ),
            "execution_mode": execution_mode,
        }
        if "seeds" in payload.get("eval", {}):
            seeds = list(payload["eval"]["seeds"])
            if not seeds:
                raise ValueError("eval.seeds cannot be empty")
            evaluation["seeds"] = [int(seed) for seed in seeds]
        if len(evaluation["goal"]) != 2:
            raise ValueError("eval.goal must contain two coordinates")
        if not 0 <= evaluation["start_low"] < evaluation["start_high"] <= 1:
            raise ValueError("eval start range must satisfy 0 <= start_low < start_high <= 1")

        artifacts = _mapping(payload.get("artifacts", {}), "artifacts")
        _reject_unknown(
            "artifacts", artifacts, {"output_dir", "checkpoint_name", "report_path"}
        )
        if "output_dir" not in artifacts:
            raise ValueError("artifacts.output_dir is required")
        checkpoint_name = str(artifacts.get("checkpoint_name", "checkpoint.json"))
        if checkpoint_name == "checkpoint.pt":
            warnings.warn(
                "artifacts.checkpoint_name=checkpoint.pt is deprecated and migrated to checkpoint.json",
                DeprecationWarning,
                stacklevel=2,
            )
            checkpoint_name = "checkpoint.json"
        if checkpoint_name != "checkpoint.json":
            raise ValueError("artifacts.checkpoint_name must be checkpoint.json")
        artifacts["checkpoint_name"] = checkpoint_name

        project_name = str(payload.get("project_name", "")).strip()
        if not project_name:
            raise ValueError("project_name is required")
        framework = str(payload.get("framework", "lunavla"))
        if framework != "lunavla":
            raise ValueError("framework must be 'lunavla'")
        return cls(
            schema_version=schema_version,
            project_name=project_name,
            framework=framework,
            task=task,
            policy=canonical_policy,
            dataset=dataset,
            training=training,
            eval=evaluation,
            artifacts=artifacts,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8-sig") as file:
            payload = yaml.safe_load(file)
        if not isinstance(payload, Mapping):
            raise TypeError("configuration file must contain a mapping")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_name": self.project_name,
            "framework": self.framework,
            "task": self.task,
            "policy": copy.deepcopy(self.policy),
            "dataset": copy.deepcopy(self.dataset),
            "training": copy.deepcopy(self.training),
            "eval": copy.deepcopy(self.eval),
            "artifacts": copy.deepcopy(self.artifacts),
        }

    def sha256(self) -> str:
        serialized = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
