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
    "training": {
        "device", "seed", "batch_size", "steps", "learning_rate", "optimizer",
        "scheduler", "precision", "gradient_clip_norm", "resume",
    },
    "evaluation": {"execution_mode", "episodes", "seed", "seeds", "max_steps"},
    "diagnostics": {"enabled"},
    "artifacts": {"output_dir", "checkpoint_name"},
}
_POLICIES = {
    "numpy_linear_chunk", "numpy_bc_mlp", "transformer_chunk", "transformer_chunk_cvae",
    "act", "act_v3",
}
_TASKS = {"fake_pusht", "fake_libero", "pusht_style_point_reach", "language_conditioned_point_reach", "rendered_visual_point_reach", "lerobot_pusht"}
_DATASETS = {"memory", "fake_pusht", "fake_libero", "v2_compat"}
_EXECUTION = {"open_loop_chunk", "receding_horizon"}
_NUMPY_PARAMETER_FIELDS = {
    "state_dim", "instruction_dim", "action_dim", "chunk_size", "hidden_dim",
    "state_feature", "unused_modalities", "history", "horizon", "execution_steps",
}
_ACT_PARAMETER_FIELDS = {
    "state_feature", "camera_feature", "instruction_dim", "chunk_size", "history",
    "horizon", "execution_steps", "d_model", "nhead", "num_encoder_layers",
    "num_decoder_layers", "dim_feedforward", "latent_dim", "dropout", "kl_weight",
    "sample_latent_during_training", "temporal_ensemble_decay",
}
_TRAINING_DEFAULTS: Mapping[str, Any] = {
    "optimizer": {"type": "sgd", "parameters": {}},
    "scheduler": {"type": "constant", "parameters": {}},
    "precision": "float32",
    "gradient_clip_norm": None,
    "resume": {"enabled": False, "checkpoint": None, "strict": True},
}
_RESERVED_ARTIFACT_NAMES = {
    "checkpoint.v3.json",
    "data_audit.json",
    "manifest.json",
    "metrics.json",
    "resolved_config.json",
    "rollouts",
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


def _checkpoint_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("artifacts.checkpoint_name must be a non-empty string")
    name = value.strip()
    if (
        name in {".", ".."}
        or Path(name).is_absolute()
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("artifacts.checkpoint_name must be a relative basename")
    if name in _RESERVED_ARTIFACT_NAMES:
        raise ValueError(f"artifacts.checkpoint_name is reserved: {name}")
    return name


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
            if name == "training":
                for field_name, default in _TRAINING_DEFAULTS.items():
                    section.setdefault(field_name, copy.deepcopy(default))
            missing_fields = sorted(fields - set(section))
            if missing_fields:
                raise ValueError(f"missing field(s) in {name}: {', '.join(missing_fields)}")
            sections[name] = section

        policy = sections["policy"]
        if policy["type"] not in _POLICIES:
            raise ValueError(f"unsupported policy.type {policy['type']!r}")
        policy["parameters"] = _mapping(policy["parameters"], "policy.parameters")
        if "legacy" in policy["parameters"]:
            if set(policy["parameters"]) != {"legacy", "compat_read_only"}:
                raise ValueError(
                    "legacy policy.parameters requires exactly legacy and compat_read_only"
                )
            if policy["parameters"]["compat_read_only"] is not True:
                raise ValueError("migrated legacy policies must be marked compat_read_only=true")
            policy["parameters"]["legacy"] = _mapping(
                policy["parameters"]["legacy"], "policy.parameters.legacy"
            )
        else:
            if not (policy["type"].startswith("numpy_") or policy["type"] == "act_v3"):
                raise ValueError(
                    f"policy.type {policy['type']!r} is migration-only in v3 Alpha 1"
                )
            allowed_policy = (
                _NUMPY_PARAMETER_FIELDS
                if policy["type"].startswith("numpy_")
                else _ACT_PARAMETER_FIELDS
            )
            _reject_unknown(policy["parameters"], allowed_policy, "policy.parameters")
            required_parameters = (
                ("state_dim", "instruction_dim", "action_dim", "chunk_size")
                if policy["type"].startswith("numpy_")
                else (
                    "state_feature", "instruction_dim", "chunk_size", "d_model", "nhead",
                    "num_encoder_layers", "num_decoder_layers", "dim_feedforward",
                    "latent_dim", "dropout", "kl_weight",
                )
            )
            for name in required_parameters:
                if name not in policy["parameters"]:
                    raise ValueError(f"policy.parameters.{name} is required")
            positive_integer_parameters = ["chunk_size"]
            if policy["type"].startswith("numpy_"):
                positive_integer_parameters.extend(["state_dim", "action_dim"])
            else:
                positive_integer_parameters.extend(
                    [
                        "d_model", "nhead", "num_encoder_layers", "num_decoder_layers",
                        "dim_feedforward", "latent_dim",
                    ]
                )
            for name in positive_integer_parameters:
                policy["parameters"][name] = _integer(
                    policy["parameters"][name], f"policy.parameters.{name}", positive=True
                )
            policy["parameters"]["instruction_dim"] = _integer(
                policy["parameters"]["instruction_dim"], "policy.parameters.instruction_dim"
            )
            if policy["type"].startswith("numpy_") and "hidden_dim" in policy["parameters"]:
                policy["parameters"]["hidden_dim"] = _integer(
                    policy["parameters"]["hidden_dim"], "policy.parameters.hidden_dim", positive=True
                )
            policy["parameters"]["history"] = _integer(
                policy["parameters"].get("history", 1),
                "policy.parameters.history",
                positive=True,
            )
            policy["parameters"]["horizon"] = _integer(
                policy["parameters"].get("horizon", policy["parameters"]["chunk_size"]),
                "policy.parameters.horizon",
                positive=True,
            )
            policy["parameters"]["execution_steps"] = _integer(
                policy["parameters"].get(
                    "execution_steps", policy["parameters"]["chunk_size"]
                ),
                "policy.parameters.execution_steps",
                positive=True,
            )
            if policy["parameters"]["chunk_size"] > policy["parameters"]["horizon"]:
                raise ValueError("policy.parameters.chunk_size cannot exceed horizon")
            if policy["parameters"]["execution_steps"] > policy["parameters"]["chunk_size"]:
                raise ValueError("policy.parameters.execution_steps cannot exceed chunk_size")
            state_feature = policy["parameters"].get("state_feature", "state.proprioception")
            if not isinstance(state_feature, str) or not state_feature:
                raise ValueError("policy.parameters.state_feature must be a non-empty string")
            policy["parameters"]["state_feature"] = state_feature
            if policy["type"] == "act_v3":
                camera_feature = policy["parameters"].get("camera_feature")
                if camera_feature is not None and (
                    not isinstance(camera_feature, str) or not camera_feature
                ):
                    raise ValueError("policy.parameters.camera_feature must be non-empty or null")
                policy["parameters"]["camera_feature"] = camera_feature
                if policy["parameters"]["d_model"] % policy["parameters"]["nhead"]:
                    raise ValueError("act_v3 d_model must be divisible by nhead")
                for name in ("dropout", "kl_weight"):
                    value = policy["parameters"][name]
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise TypeError(f"policy.parameters.{name} must be numeric")
                    value = float(value)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"policy.parameters.{name} must be finite and non-negative")
                    if name == "dropout" and value >= 1:
                        raise ValueError("policy.parameters.dropout must be below one")
                    policy["parameters"][name] = value
                sample_latent = policy["parameters"].get(
                    "sample_latent_during_training", True
                )
                if not isinstance(sample_latent, bool):
                    raise TypeError("sample_latent_during_training must be boolean")
                policy["parameters"]["sample_latent_during_training"] = sample_latent
                decay = policy["parameters"].get("temporal_ensemble_decay")
                if decay is not None:
                    decay = _positive_float(decay, "policy.parameters.temporal_ensemble_decay")
                policy["parameters"]["temporal_ensemble_decay"] = decay
            if policy["type"].startswith("numpy_"):
                unused = policy["parameters"].get("unused_modalities", [])
                if isinstance(unused, (str, bytes, Mapping)) or not isinstance(
                    unused, Sequence
                ):
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
            if dataset["parameters"].get("episode_count", 6) < 3:
                raise ValueError("fake datasets require at least three episodes")

        task_id = task["id"]
        dataset_type = dataset["type"]
        if task_id in {"fake_pusht", "fake_libero"} and dataset_type != task_id:
            raise ValueError("fake task.id and dataset.type must match")
        if dataset_type in {"fake_pusht", "fake_libero"} and task_id != dataset_type:
            raise ValueError("fake dataset.type and task.id must match")
        if dataset_type == "v2_compat" and "legacy" not in policy["parameters"]:
            raise ValueError("v2_compat datasets require a migrated compat_read_only policy")
        if task_id in {"fake_pusht", "fake_libero"} and dataset["split"] != "train":
            raise ValueError("runnable fake datasets require dataset.split=train")

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
        if "legacy" not in policy["parameters"]:
            state_feature_name = policy["parameters"]["state_feature"]
            state_features = {item.name: item for item in feature_schema.by_role("state")}
            action_features = feature_schema.by_role("action")
            if state_feature_name not in state_features:
                raise ValueError("policy state_feature is not declared by FeatureSchema")
            if policy["type"].startswith("numpy_") and state_features[state_feature_name].shape != (
                policy["parameters"]["state_dim"],
            ):
                raise ValueError("policy state_dim conflicts with FeatureSchema")
            if len(action_features) != 1:
                raise ValueError("runnable policies require exactly one action feature")
            if policy["type"].startswith("numpy_") and action_features[0].shape != (
                policy["parameters"]["action_dim"],
            ):
                raise ValueError("policy action_dim conflicts with FeatureSchema")
            if policy["type"] == "act_v3":
                camera_features = {item.name: item for item in feature_schema.by_role("image")}
                camera_feature = policy["parameters"]["camera_feature"]
                if camera_feature is not None and camera_feature not in camera_features:
                    raise ValueError("act_v3 camera_feature is not declared by FeatureSchema")
                if camera_feature is not None:
                    camera_shape = camera_features[camera_feature].shape
                    if len(camera_shape) != 3 or camera_shape[-1] != 3:
                        raise ValueError("act_v3 Alpha 2 requires an HWC RGB camera feature")
                if camera_feature is None and camera_features:
                    raise ValueError("act_v3 cannot silently discard declared image features")
                if len(camera_features) > 1:
                    raise ValueError("act_v3 Alpha 2 supports exactly zero or one camera")

        training = sections["training"]
        training["device"] = normalize_device(training["device"])
        training["seed"] = _integer(training["seed"], "training.seed")
        training["batch_size"] = _integer(training["batch_size"], "training.batch_size", positive=True)
        training["steps"] = _integer(training["steps"], "training.steps", positive=True)
        training["learning_rate"] = _positive_float(training["learning_rate"], "training.learning_rate")
        optimizer = _mapping(training["optimizer"], "training.optimizer")
        _reject_unknown(optimizer, {"type", "parameters"}, "training.optimizer")
        if set(optimizer) != {"type", "parameters"}:
            raise ValueError("training.optimizer requires type and parameters")
        if optimizer["type"] not in {"sgd", "adam", "adamw"}:
            raise ValueError("training.optimizer.type must be sgd, adam, or adamw")
        optimizer["parameters"] = _mapping(
            optimizer["parameters"], "training.optimizer.parameters"
        )
        _reject_unknown(
            optimizer["parameters"],
            {"momentum", "weight_decay", "betas", "eps"},
            "training.optimizer.parameters",
        )
        scheduler = _mapping(training["scheduler"], "training.scheduler")
        _reject_unknown(scheduler, {"type", "parameters"}, "training.scheduler")
        if set(scheduler) != {"type", "parameters"}:
            raise ValueError("training.scheduler requires type and parameters")
        if scheduler["type"] not in {"constant", "linear", "cosine"}:
            raise ValueError("training.scheduler.type must be constant, linear, or cosine")
        scheduler["parameters"] = _mapping(
            scheduler["parameters"], "training.scheduler.parameters"
        )
        _reject_unknown(
            scheduler["parameters"], {"warmup_steps", "min_learning_rate"},
            "training.scheduler.parameters",
        )
        if training["precision"] not in {"float32", "float16", "bfloat16"}:
            raise ValueError("training.precision must be float32, float16, or bfloat16")
        clip = training["gradient_clip_norm"]
        if clip is not None:
            clip = _positive_float(clip, "training.gradient_clip_norm")
        resume = _mapping(training["resume"], "training.resume")
        _reject_unknown(resume, {"enabled", "checkpoint", "strict"}, "training.resume")
        if set(resume) != {"enabled", "checkpoint", "strict"}:
            raise ValueError("training.resume requires enabled, checkpoint, and strict")
        if not isinstance(resume["enabled"], bool) or not isinstance(resume["strict"], bool):
            raise TypeError("training.resume enabled and strict must be boolean")
        checkpoint = resume["checkpoint"]
        if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
            raise ValueError("training.resume.checkpoint must be a non-empty string or null")
        if resume["enabled"] and checkpoint is None:
            raise ValueError("enabled resume requires training.resume.checkpoint")
        if not resume["enabled"] and checkpoint is not None:
            raise ValueError("disabled resume cannot declare training.resume.checkpoint")
        resume["checkpoint"] = checkpoint.strip() if isinstance(checkpoint, str) else None
        training["optimizer"] = optimizer
        training["scheduler"] = scheduler
        training["gradient_clip_norm"] = clip
        training["resume"] = resume
        if policy["type"].startswith("numpy_") and training["device"] != "cpu":
            raise ValueError("NumPy policies require CPU training")
        if policy["type"].startswith("numpy_") and optimizer["type"] != "sgd":
            raise ValueError("NumPy compatibility policies require optimizer.type=sgd")
        if policy["type"].startswith("numpy_") and scheduler["type"] != "constant":
            raise ValueError("NumPy compatibility policies require a constant scheduler")
        if policy["type"].startswith("numpy_") and training["precision"] != "float32":
            raise ValueError("NumPy compatibility policies require float32 precision")
        if policy["type"] == "act_v3" and optimizer["type"] != "adam":
            raise ValueError("act_v3 requires optimizer.type=adam")
        if policy["type"] == "act_v3" and optimizer["parameters"]:
            raise ValueError("act_v3 Alpha 2 requires empty optimizer.parameters")
        if policy["type"] == "act_v3" and scheduler["type"] != "constant":
            raise ValueError("act_v3 Alpha 2 requires scheduler.type=constant")
        if policy["type"] == "act_v3" and scheduler["parameters"]:
            raise ValueError("act_v3 Alpha 2 requires empty scheduler.parameters")
        if policy["type"] == "act_v3" and training["precision"] != "float32":
            raise ValueError("act_v3 Alpha 2 supports float32 precision only")
        if (
            policy["type"] == "act_v3"
            and policy["parameters"].get("temporal_ensemble_decay") is not None
            and sections["evaluation"]["execution_mode"] != "receding_horizon"
        ):
            raise ValueError("act_v3 temporal ensembling requires receding_horizon")

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
        if not isinstance(artifacts["output_dir"], str) or not artifacts["output_dir"].strip():
            raise ValueError("artifacts.output_dir must be a non-empty string")
        artifacts["output_dir"] = artifacts["output_dir"].strip()
        artifacts["checkpoint_name"] = _checkpoint_name(artifacts["checkpoint_name"])

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
