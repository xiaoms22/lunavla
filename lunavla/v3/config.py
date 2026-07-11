from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from lunavla.contracts import normalize_device

from .contracts import EmbodimentSpec, FeatureSchema


CONFIG_SCHEMA_VERSION = 3
_ROOT = {
    "schema_version", "project_name", "engine", "policy", "task", "dataset", "embodiment",
    "features", "training", "evaluation", "diagnostics", "artifacts",
}
_SECTION_FIELDS = {
    "policy": {"type", "parameters"},
    "task": {"id", "parameters"},
    "dataset": {"type", "split", "seed", "parameters"},
    "embodiment": {"id", "task_id", "control_rate_hz", "camera_mapping", "state_mapping", "action_mapping"},
    "training": {"device", "seed", "batch_size", "steps", "learning_rate"},
    "evaluation": {"execution_mode", "episodes", "seed", "seeds", "max_steps"},
    "diagnostics": {"enabled"},
    "artifacts": {"output_dir", "checkpoint_name"},
}
_POLICIES = {"numpy_linear_chunk", "numpy_bc_mlp", "transformer_chunk", "transformer_chunk_cvae", "act"}
_TASKS = {"fake_pusht", "fake_libero", "pusht_style_point_reach", "language_conditioned_point_reach", "rendered_visual_point_reach", "lerobot_pusht"}
_DATASETS = {"memory", "fake_pusht", "fake_libero", "v2_compat"}
_EXECUTION = {"open_loop_chunk", "receding_horizon"}
_NUMPY_PARAMETER_FIELDS = {
    "state_dim", "instruction_dim", "action_dim", "chunk_size", "hidden_dim",
    "state_feature", "unused_modalities",
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return copy.deepcopy(dict(value))


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown field(s) in {name}: {', '.join(unknown)}")


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True, init=False)
class ExperimentConfig:
    schema_version: int
    project_name: str
    engine: str
    policy: Mapping[str, Any]
    task: Mapping[str, Any]
    dataset: Mapping[str, Any]
    embodiment: Mapping[str, Any]
    features: Mapping[str, Any]
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    artifacts: Mapping[str, Any]

    def __init__(self, **_: Any) -> None:
        raise TypeError("ExperimentConfig must be created with from_mapping() or load()")

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "ExperimentConfig":
        root = _mapping(source, "config")
        _reject_unknown(root, _ROOT, "config")
        missing = sorted(_ROOT - set(root))
        if missing:
            raise ValueError("missing config field(s): " + ", ".join(missing))
        version = root["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("schema_version must be an integer")
        if version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
        project_name = root["project_name"]
        if not isinstance(project_name, str) or not project_name.strip():
            raise ValueError("project_name must be a non-empty string")
        if root["engine"] != "lunavla_v3":
            raise ValueError("engine must be lunavla_v3")

        sections: dict[str, dict[str, Any]] = {}
        for name, fields in _SECTION_FIELDS.items():
            section = _mapping(root[name], name)
            _reject_unknown(section, fields, name)
            missing_fields = sorted(fields - set(section))
            if missing_fields:
                raise ValueError(f"missing field(s) in {name}: {', '.join(missing_fields)}")
            sections[name] = section

        policy = sections["policy"]
        if policy["type"] not in _POLICIES:
            raise ValueError(f"unsupported policy.type {policy['type']!r}")
        policy["parameters"] = _mapping(policy["parameters"], "policy.parameters")
        if set(policy["parameters"]) != {"legacy"}:
            allowed_policy = _NUMPY_PARAMETER_FIELDS if policy["type"].startswith("numpy_") else set()
            _reject_unknown(policy["parameters"], allowed_policy, "policy.parameters")
            for name in ("state_dim", "instruction_dim", "action_dim", "chunk_size"):
                if name not in policy["parameters"]:
                    raise ValueError(f"policy.parameters.{name} is required")
            for name in ("state_dim", "action_dim", "chunk_size"):
                policy["parameters"][name] = _integer(
                    policy["parameters"][name], f"policy.parameters.{name}", positive=True
                )
            policy["parameters"]["instruction_dim"] = _integer(
                policy["parameters"]["instruction_dim"], "policy.parameters.instruction_dim"
            )
            if "hidden_dim" in policy["parameters"]:
                policy["parameters"]["hidden_dim"] = _integer(
                    policy["parameters"]["hidden_dim"], "policy.parameters.hidden_dim", positive=True
                )
            state_feature = policy["parameters"].get("state_feature", "state.proprioception")
            if not isinstance(state_feature, str) or not state_feature:
                raise ValueError("policy.parameters.state_feature must be a non-empty string")
            policy["parameters"]["state_feature"] = state_feature
            unused = policy["parameters"].get("unused_modalities", [])
            if isinstance(unused, (str, bytes, Mapping)) or not isinstance(unused, Sequence):
                raise TypeError("policy.parameters.unused_modalities must be a sequence")
            unused_values = list(unused)
            if any(item not in {"image", "instruction"} for item in unused_values):
                raise ValueError("unused_modalities supports only image and instruction")
            if len(unused_values) != len(set(unused_values)):
                raise ValueError("unused_modalities cannot contain duplicates")
            policy["parameters"]["unused_modalities"] = unused_values
        task = sections["task"]
        if task["id"] not in _TASKS:
            raise ValueError(f"unsupported task.id {task['id']!r}")
        task["parameters"] = _mapping(task["parameters"], "task.parameters")
        if set(task["parameters"]) != {"legacy"}:
            _reject_unknown(task["parameters"], set(), "task.parameters")
        dataset = sections["dataset"]
        if dataset["type"] not in _DATASETS:
            raise ValueError(f"unsupported dataset.type {dataset['type']!r}")
        if dataset["split"] not in {"train", "validation", "test"}:
            raise ValueError("dataset.split must be train, validation, or test")
        dataset["seed"] = _integer(dataset["seed"], "dataset.seed")
        dataset["parameters"] = _mapping(dataset["parameters"], "dataset.parameters")
        if dataset["type"] == "v2_compat":
            _reject_unknown(dataset["parameters"], {"legacy"}, "dataset.parameters")
            if "legacy" not in dataset["parameters"]:
                raise ValueError("v2_compat dataset requires dataset.parameters.legacy")
        else:
            _reject_unknown(
                dataset["parameters"], {"episode_count", "steps_per_episode"}, "dataset.parameters"
            )
            for name in ("episode_count", "steps_per_episode"):
                if name in dataset["parameters"]:
                    dataset["parameters"][name] = _integer(
                        dataset["parameters"][name], f"dataset.parameters.{name}", positive=True
                    )

        feature_schema = FeatureSchema.from_mapping(_mapping(root["features"], "features"))
        embodiment_section = sections["embodiment"]
        embodiment = EmbodimentSpec(
            embodiment_id=embodiment_section["id"],
            task_id=embodiment_section["task_id"],
            control_rate_hz=embodiment_section["control_rate_hz"],
            camera_mapping=_mapping(embodiment_section["camera_mapping"], "camera_mapping"),
            state_mapping=_mapping(embodiment_section["state_mapping"], "state_mapping"),
            action_mapping=_mapping(embodiment_section["action_mapping"], "action_mapping"),
        )
        if embodiment.task_id != task["id"]:
            raise ValueError("embodiment.task_id must match task.id")
        embodiment.validate_schema(feature_schema)

        training = sections["training"]
        training["device"] = normalize_device(training["device"])
        training["seed"] = _integer(training["seed"], "training.seed")
        training["batch_size"] = _integer(training["batch_size"], "training.batch_size", positive=True)
        training["steps"] = _integer(training["steps"], "training.steps", positive=True)
        training["learning_rate"] = _positive_float(training["learning_rate"], "training.learning_rate")
        if policy["type"].startswith("numpy_") and training["device"] != "cpu":
            raise ValueError("NumPy policies require CPU training")

        evaluation = sections["evaluation"]
        if evaluation["execution_mode"] not in _EXECUTION:
            raise ValueError("evaluation.execution_mode must be open_loop_chunk or receding_horizon")
        evaluation["episodes"] = _integer(evaluation["episodes"], "evaluation.episodes", positive=True)
        evaluation["seed"] = _integer(evaluation["seed"], "evaluation.seed")
        evaluation["max_steps"] = _integer(evaluation["max_steps"], "evaluation.max_steps", positive=True)
        seeds = evaluation["seeds"]
        if isinstance(seeds, (str, bytes, Mapping)) or not isinstance(seeds, Sequence):
            raise TypeError("evaluation.seeds must be a sequence")
        evaluation["seeds"] = [_integer(item, "evaluation.seeds item") for item in seeds]
        if len(evaluation["seeds"]) != evaluation["episodes"]:
            raise ValueError("evaluation.seeds must contain evaluation.episodes values")
        if len(set(evaluation["seeds"])) != len(evaluation["seeds"]):
            raise ValueError("evaluation.seeds cannot contain duplicates")

        diagnostics = sections["diagnostics"]
        if not isinstance(diagnostics["enabled"], bool):
            raise TypeError("diagnostics.enabled must be boolean")
        artifacts = sections["artifacts"]
        for name in ("output_dir", "checkpoint_name"):
            if not isinstance(artifacts[name], str) or not artifacts[name].strip():
                raise ValueError(f"artifacts.{name} must be a non-empty string")

        instance = object.__new__(cls)
        values = {
            "schema_version": version,
            "project_name": project_name.strip(),
            "engine": "lunavla_v3",
            "policy": policy,
            "task": task,
            "dataset": dataset,
            "embodiment": embodiment.to_dict(),
            "features": feature_schema.to_dict(),
            "training": training,
            "evaluation": evaluation,
            "diagnostics": diagnostics,
            "artifacts": artifacts,
        }
        for name, value in values.items():
            object.__setattr__(instance, name, _freeze(value))
        return instance

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        try:
            source = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            location = f" at line {mark.line + 1}" if mark is not None else ""
            raise ValueError(f"invalid YAML{location}: {exc}") from exc
        return cls.from_mapping(source)

    def to_dict(self) -> dict[str, Any]:
        return {name: _thaw(getattr(self, name)) for name in (
            "schema_version", "project_name", "engine", "policy", "task", "dataset", "embodiment",
            "features", "training", "evaluation", "diagnostics", "artifacts",
        )}

    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()

    @property
    def feature_schema(self) -> FeatureSchema:
        return FeatureSchema.from_mapping(self.features)

    @property
    def embodiment_spec(self) -> EmbodimentSpec:
        return EmbodimentSpec(
            embodiment_id=self.embodiment["id"],
            task_id=self.embodiment["task_id"],
            control_rate_hz=self.embodiment["control_rate_hz"],
            camera_mapping=self.embodiment["camera_mapping"],
            state_mapping=self.embodiment["state_mapping"],
            action_mapping=self.embodiment["action_mapping"],
        )
