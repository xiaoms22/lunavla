"""Strict configuration contract for the experimental v2 engine."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import yaml

from lunavla.contracts import normalize_device


CONFIG_SCHEMA_VERSION = 2

_ROOT_FIELDS = {
    "schema_version",
    "project_name",
    "engine",
    "policy",
    "task",
    "dataset",
    "training",
    "evaluation",
    "artifacts",
}
_POLICY_FIELDS = {
    "type",
    "state_dim",
    "instruction_dim",
    "image_shape",
    "action_dim",
    "chunk_size",
    "hidden_dim",
    "d_model",
    "nhead",
    "num_layers",
    "latent_dim",
    "dropout",
    "temporal_ensemble_decay",
    "device",
}
_TASK_FIELDS = {"id", "family", "max_steps", "goal", "render_size", "parameters"}
_DATASET_FIELDS = {"type", "split", "seed", "path", "episode_count", "parameters"}
_TRAINING_FIELDS = {
    "device",
    "seed",
    "batch_size",
    "steps",
    "learning_rate",
    "kl_weight",
}
_EVALUATION_FIELDS = {
    "execution_mode",
    "episodes",
    "seed",
    "seeds",
    "language_ablation",
    "image_ablation",
    "parameters",
}
_ARTIFACT_FIELDS = {"output_dir", "checkpoint_name", "report_path"}
_NUMPY_POLICIES = {"numpy_linear_chunk", "numpy_bc_mlp"}
_POLICIES = _NUMPY_POLICIES | {"transformer_chunk_cvae", "transformer_chunk", "act"}
_TASK_IDS = {
    "pusht_style_point_reach",
    "language_conditioned_point_reach",
    "rendered_visual_point_reach",
}
_DATASET_TYPES = {"memory", "mock_pusht", "generated", "jsonl", "lerobot"}
_VISUAL_FAMILIES = {"all", "direct_reach", "waypoint_reach"}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return copy.deepcopy(dict(value))


def _reject_unknown(name: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown field(s) in {name}: {', '.join(unknown)}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _integer(value: Any, name: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _output_path(value: Any, name: str, *, allow_root: bool = False) -> str:
    raw = str(value).strip()
    path = Path(raw)
    if (
        not raw
        or path.is_absolute()
        or "\\" in raw
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != "outputs"
        or (len(path.parts) == 1 and not allow_root)
    ):
        raise ValueError(f"{name} must be a normalized repository-relative path under outputs/")
    return path.as_posix()


def _validate_cross_section_contracts(
    policy: dict[str, Any],
    task: dict[str, Any],
    dataset: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    """Reject combinations that the train/evaluation engine cannot execute."""

    policy_type = str(policy["type"])
    task_id = str(task["id"])
    family = str(task["family"])
    language_ablation = str(evaluation["language_ablation"])
    image_ablation = str(evaluation["image_ablation"])
    image_shape = policy.get("image_shape")
    raw_state_only = dataset["parameters"].get("state_only", False)
    if not isinstance(raw_state_only, bool):
        raise TypeError("dataset.parameters.state_only must be boolean")
    state_only = raw_state_only

    if language_ablation != "none" and image_ablation != "none":
        raise ValueError("run one language or image ablation at a time")
    if (
        language_ablation != "none"
        and task_id != "language_conditioned_point_reach"
    ):
        raise ValueError(
            "evaluation.language_ablation requires "
            "task.id=language_conditioned_point_reach"
        )
    if image_ablation != "none" and task_id != "rendered_visual_point_reach":
        raise ValueError(
            "evaluation.image_ablation requires task.id=rendered_visual_point_reach"
        )
    if state_only and task_id != "rendered_visual_point_reach":
        raise ValueError(
            "dataset.parameters.state_only is only valid for "
            "task.id=rendered_visual_point_reach"
        )

    if policy_type == "transformer_chunk_cvae":
        d_model = int(policy.get("d_model", 64))
        nhead = int(policy.get("nhead", 4))
        if d_model % nhead:
            raise ValueError("policy.d_model must be divisible by policy.nhead")
        if policy["instruction_dim"] == 0:
            raise ValueError(
                "current v2 task environments provide instructions; Transformer "
                "policies require policy.instruction_dim > 0"
            )

    if policy["action_dim"] != 2:
        raise ValueError(f"task.id={task_id} requires policy.action_dim=2")

    if task_id == "pusht_style_point_reach":
        if family != "point_reach":
            raise ValueError(
                "task.id=pusht_style_point_reach requires task.family=point_reach"
            )
        if policy["state_dim"] != 4:
            raise ValueError(
                "task.id=pusht_style_point_reach requires policy.state_dim=4"
            )
        if image_shape is not None:
            raise ValueError("pusht_style_point_reach does not provide image observations")
        if language_ablation != "none" or image_ablation != "none":
            raise ValueError("pusht_style_point_reach does not support modality ablations")
        if "render_size" in task:
            raise ValueError("task.render_size is only valid for rendered visual tasks")
        return

    if task_id == "language_conditioned_point_reach":
        if family != "point_reach":
            raise ValueError(
                "task.id=language_conditioned_point_reach requires task.family=point_reach"
            )
        if policy["state_dim"] != 2:
            raise ValueError(
                "task.id=language_conditioned_point_reach requires policy.state_dim=2"
            )
        if policy["instruction_dim"] <= 0:
            raise ValueError(
                "language_conditioned_point_reach requires policy.instruction_dim > 0"
            )
        if image_shape is not None:
            raise ValueError(
                "language_conditioned_point_reach does not provide image observations"
            )
        split = str(task["parameters"].get("language_split", "heldout"))
        if split not in {"train", "heldout"}:
            raise ValueError(
                "task.parameters.language_split must be train or heldout"
            )
        if "render_size" in task:
            raise ValueError("task.render_size is only valid for rendered visual tasks")
        return

    if family not in _VISUAL_FAMILIES:
        raise ValueError(
            "rendered_visual_point_reach task.family must be all, direct_reach, "
            "or waypoint_reach"
        )
    render_size = int(task.get("render_size", 64))
    if render_size < 24:
        raise ValueError("rendered visual task.render_size must be at least 24")
    task["render_size"] = render_size
    if policy["instruction_dim"] <= 0:
        raise ValueError(
            "rendered_visual_point_reach requires policy.instruction_dim > 0"
        )

    if state_only:
        if image_shape is not None:
            raise ValueError(
                "visual state-only mode requires policy.image_shape=null"
            )
        if image_ablation != "state_only":
            raise ValueError(
                "dataset.parameters.state_only=true requires "
                "evaluation.image_ablation=state_only"
            )
        return

    if image_ablation == "state_only":
        raise ValueError(
            "evaluation.image_ablation=state_only requires "
            "dataset.parameters.state_only=true"
        )
    if image_shape is None:
        raise ValueError(
            "rendered visual image mode requires policy.image_shape=[H, W, 3]"
        )
    expected_shape = [render_size, render_size, 3]
    if image_shape != expected_shape:
        raise ValueError(
            "policy.image_shape must match the rendered RGB observation shape "
            f"{expected_shape}"
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved v2 configuration.

    The schema is intentionally strict: provider-specific options must be placed
    inside an explicitly named ``parameters`` mapping instead of being silently
    accepted at a section's top level.
    """

    schema_version: int
    project_name: str
    engine: str
    policy: dict[str, Any]
    task: dict[str, Any]
    dataset: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    artifacts: dict[str, Any]

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "ExperimentConfig":
        payload = _mapping(source, "root")
        _reject_unknown("root", payload, _ROOT_FIELDS)
        version = _integer(payload.get("schema_version", 0), "schema_version")
        if version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported v2 config schema_version: {version}; "
                "run scripts/migrate_v11_to_v2.py for a v1.1 config"
            )
        project_name = str(payload.get("project_name", "")).strip()
        if not project_name:
            raise ValueError("project_name is required")
        engine = str(payload.get("engine", "lunavla_v2"))
        if engine != "lunavla_v2":
            raise ValueError("engine must be 'lunavla_v2'")

        policy = _mapping(payload.get("policy", {}), "policy")
        _reject_unknown("policy", policy, _POLICY_FIELDS)
        provided_policy_fields = set(policy)
        raw_policy_type = str(policy.get("type", ""))
        if raw_policy_type not in _POLICIES:
            raise ValueError(f"unsupported policy.type: {raw_policy_type!r}")
        policy_type = {
            "transformer_chunk": "transformer_chunk_cvae",
            "act": "transformer_chunk_cvae",
        }.get(raw_policy_type, raw_policy_type)
        policy["type"] = policy_type
        policy["state_dim"] = _positive_int(policy.get("state_dim", 4), "policy.state_dim")
        policy["instruction_dim"] = _integer(
            policy.get("instruction_dim", 0), "policy.instruction_dim", nonnegative=True
        )
        if policy["instruction_dim"] < 0:
            raise ValueError("policy.instruction_dim cannot be negative")
        policy["action_dim"] = _positive_int(policy.get("action_dim", 2), "policy.action_dim")
        policy["chunk_size"] = _positive_int(policy.get("chunk_size", 1), "policy.chunk_size")
        policy["device"] = normalize_device(str(policy.get("device", "cpu")))
        image_shape = policy.get("image_shape")
        if image_shape is not None:
            if not isinstance(image_shape, (list, tuple)) or len(image_shape) != 3:
                raise ValueError("policy.image_shape must be [height, width, channels] or null")
            policy["image_shape"] = [
                _positive_int(part, f"policy.image_shape[{index}]")
                for index, part in enumerate(image_shape)
            ]
            if policy["image_shape"][-1] not in {1, 3, 4}:
                raise ValueError("policy.image_shape channel count must be 1, 3, or 4")
        if policy_type in _NUMPY_POLICIES and policy["device"] != "cpu":
            raise ValueError(f"{policy_type} is NumPy-only and requires policy.device=cpu")
        for name in ("hidden_dim", "d_model", "nhead", "num_layers", "latent_dim"):
            if name in policy:
                policy[name] = _positive_int(policy[name], f"policy.{name}")
        if "dropout" in policy:
            policy["dropout"] = _finite_float(policy["dropout"], "policy.dropout")
            if not 0 <= policy["dropout"] < 1:
                raise ValueError("policy.dropout must satisfy 0 <= dropout < 1")
        if "temporal_ensemble_decay" in policy:
            policy["temporal_ensemble_decay"] = _finite_float(
                policy["temporal_ensemble_decay"], "policy.temporal_ensemble_decay"
            )
            if not 0 < policy["temporal_ensemble_decay"] <= 1:
                raise ValueError("policy.temporal_ensemble_decay must satisfy 0 < value <= 1")
        transformer_only = {
            "d_model",
            "nhead",
            "num_layers",
            "latent_dim",
            "dropout",
            "temporal_ensemble_decay",
        }
        if policy_type in _NUMPY_POLICIES:
            invalid = sorted(provided_policy_fields & transformer_only)
            if invalid:
                raise ValueError(
                    f"{policy_type} does not accept Transformer field(s): {', '.join(invalid)}"
                )
            if policy.get("image_shape") is not None:
                raise ValueError(f"{policy_type} is state-only and requires image_shape=null")
        if policy_type != "numpy_bc_mlp" and "hidden_dim" in provided_policy_fields:
            raise ValueError("policy.hidden_dim is only valid for numpy_bc_mlp")

        task = _mapping(payload.get("task", {}), "task")
        _reject_unknown("task", task, _TASK_FIELDS)
        task_id = str(task.get("id", "")).strip()
        if not task_id:
            raise ValueError("task.id is required")
        if task_id not in _TASK_IDS:
            raise ValueError(f"unsupported task.id: {task_id!r}")
        task["id"] = task_id
        task["family"] = str(task.get("family", "point_reach"))
        task["max_steps"] = _positive_int(task.get("max_steps", 40), "task.max_steps")
        if "goal" in task:
            goal = list(task["goal"])
            if len(goal) != 2:
                raise ValueError("task.goal must contain two coordinates")
            task["goal"] = [_finite_float(value, f"task.goal[{index}]") for index, value in enumerate(goal)]
        if "render_size" in task:
            task["render_size"] = _positive_int(task["render_size"], "task.render_size")
        task["parameters"] = _mapping(task.get("parameters", {}), "task.parameters")

        dataset = _mapping(payload.get("dataset", {}), "dataset")
        _reject_unknown("dataset", dataset, _DATASET_FIELDS)
        dataset["type"] = str(dataset.get("type", "memory"))
        if dataset["type"] not in _DATASET_TYPES:
            raise ValueError(f"unsupported dataset.type: {dataset['type']!r}")
        if dataset["type"] in {"jsonl", "lerobot"} and not dataset.get("path"):
            raise ValueError(f"dataset.type={dataset['type']} requires dataset.path")
        dataset["split"] = str(dataset.get("split", "train"))
        if dataset["split"] not in {"train", "validation", "test"}:
            raise ValueError("dataset.split must be train, validation, or test")
        dataset["seed"] = _integer(
            dataset.get("seed", 42), "dataset.seed", nonnegative=True
        )
        if "episode_count" in dataset:
            dataset["episode_count"] = _positive_int(
                dataset["episode_count"], "dataset.episode_count"
            )
        dataset["parameters"] = _mapping(dataset.get("parameters", {}), "dataset.parameters")

        training = _mapping(payload.get("training", {}), "training")
        _reject_unknown("training", training, _TRAINING_FIELDS)
        training["device"] = normalize_device(
            str(training.get("device", policy["device"]))
        )
        if training["device"] != policy["device"]:
            raise ValueError("training.device and policy.device must match")
        if policy_type in _NUMPY_POLICIES and training["device"] != "cpu":
            raise ValueError(f"{policy_type} is NumPy-only and requires training.device=cpu")
        training["seed"] = _integer(
            training.get("seed", 42), "training.seed", nonnegative=True
        )
        training["batch_size"] = _positive_int(
            training.get("batch_size", 32), "training.batch_size"
        )
        training["steps"] = _positive_int(training.get("steps", 100), "training.steps")
        training["learning_rate"] = _finite_float(
            training.get("learning_rate", 1e-3), "training.learning_rate", positive=True
        )
        training["kl_weight"] = _finite_float(
            training.get("kl_weight", 0.0), "training.kl_weight"
        )
        if training["kl_weight"] < 0:
            raise ValueError("training.kl_weight cannot be negative")
        if policy_type in _NUMPY_POLICIES and training["kl_weight"] != 0:
            raise ValueError("training.kl_weight must be zero for NumPy policies")

        evaluation = _mapping(payload.get("evaluation", {}), "evaluation")
        _reject_unknown("evaluation", evaluation, _EVALUATION_FIELDS)
        evaluation["execution_mode"] = str(
            evaluation.get(
                "execution_mode",
                "open_loop_chunk" if policy["chunk_size"] > 1 else "receding_horizon",
            )
        )
        if evaluation["execution_mode"] not in {"receding_horizon", "open_loop_chunk"}:
            raise ValueError(
                "evaluation.execution_mode must be receding_horizon or open_loop_chunk"
            )
        evaluation["episodes"] = _positive_int(
            evaluation.get("episodes", 20), "evaluation.episodes"
        )
        evaluation["seed"] = _integer(
            evaluation.get("seed", 1000), "evaluation.seed", nonnegative=True
        )
        if "seeds" in evaluation:
            seeds = [
                _integer(value, "evaluation.seeds item", nonnegative=True)
                for value in list(evaluation["seeds"])
            ]
            if not seeds:
                raise ValueError("evaluation.seeds cannot be empty")
            if len(seeds) != evaluation["episodes"]:
                raise ValueError("evaluation.seeds must contain exactly evaluation.episodes values")
            evaluation["seeds"] = seeds
        for name in ("language_ablation", "image_ablation"):
            evaluation[name] = str(evaluation.get(name, "none"))
        if evaluation["language_ablation"] not in {
            "none",
            "mask",
            "shuffle",
            "counterfactual",
        }:
            raise ValueError(
                "evaluation.language_ablation must be none, mask, shuffle, or counterfactual"
            )
        if evaluation["image_ablation"] not in {
            "none",
            "occlusion",
            "shuffle",
            "state_only",
        }:
            raise ValueError(
                "evaluation.image_ablation must be none, occlusion, shuffle, or state_only"
            )
        evaluation["parameters"] = _mapping(
            evaluation.get("parameters", {}), "evaluation.parameters"
        )
        if (
            policy.get("temporal_ensemble_decay") is not None
            and evaluation["execution_mode"] != "receding_horizon"
        ):
            raise ValueError("temporal ensembling requires receding_horizon evaluation")

        _validate_cross_section_contracts(policy, task, dataset, evaluation)

        artifacts = _mapping(payload.get("artifacts", {}), "artifacts")
        _reject_unknown("artifacts", artifacts, _ARTIFACT_FIELDS)
        artifacts["output_dir"] = _output_path(
            artifacts.get("output_dir", ""),
            "artifacts.output_dir",
        )
        default_checkpoint = (
            "checkpoint.json" if policy_type in _NUMPY_POLICIES else "checkpoint.pt"
        )
        artifacts["checkpoint_name"] = str(
            artifacts.get("checkpoint_name", default_checkpoint)
        )
        checkpoint_name = artifacts["checkpoint_name"]
        if (
            not checkpoint_name
            or Path(checkpoint_name).name != checkpoint_name
            or "/" in checkpoint_name
            or "\\" in checkpoint_name
        ):
            raise ValueError("artifacts.checkpoint_name must be a plain file name")
        if policy_type in _NUMPY_POLICIES and artifacts["checkpoint_name"] != "checkpoint.json":
            raise ValueError("NumPy v2 policies require artifacts.checkpoint_name=checkpoint.json")
        if policy_type == "transformer_chunk_cvae" and not artifacts[
            "checkpoint_name"
        ].endswith(".pt"):
            raise ValueError("the Transformer policy requires a .pt checkpoint name")
        if "report_path" in artifacts:
            report_path = _output_path(artifacts["report_path"], "artifacts.report_path")
            output_prefix = artifacts["output_dir"] + "/"
            if not report_path.startswith(output_prefix):
                raise ValueError("artifacts.report_path must be inside artifacts.output_dir")
            artifacts["report_path"] = report_path

        return cls(
            schema_version=version,
            project_name=project_name,
            engine=engine,
            policy=policy,
            task=task,
            dataset=dataset,
            training=training,
            evaluation=evaluation,
            artifacts=artifacts,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path)
        try:
            with source.open("r", encoding="utf-8-sig") as stream:
                payload = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            problem = getattr(exc, "problem", None) or "malformed YAML"
            mark = getattr(exc, "problem_mark", None)
            location = (
                f" at line {mark.line + 1}, column {mark.column + 1}"
                if mark is not None
                else ""
            )
            raise ValueError(
                f"invalid YAML in {source}: {problem}{location}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise TypeError("configuration file must contain a mapping")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_name": self.project_name,
            "engine": self.engine,
            "policy": copy.deepcopy(self.policy),
            "task": copy.deepcopy(self.task),
            "dataset": copy.deepcopy(self.dataset),
            "training": copy.deepcopy(self.training),
            "evaluation": copy.deepcopy(self.evaluation),
            "artifacts": copy.deepcopy(self.artifacts),
        }

    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
