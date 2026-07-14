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
from .v31_contracts import TaskSuiteSpecV1, VLMBackendSpecV1


CONFIG_SCHEMA_VERSION = 3
CONFIG_CONTRACT_REVISION = 2
V31_CONFIG_CONTRACT_REVISION = 4
_ROOT_V1 = {
    "schema_version",
    "project_name",
    "engine",
    "policy",
    "task",
    "dataset",
    "embodiment",
    "features",
    "training",
    "evaluation",
    "diagnostics",
    "artifacts",
}
_ROOT_V2 = _ROOT_V1 | {"contract_revision", "prompt", "routing"}
_ROOT_V4 = _ROOT_V2 | {"vlm", "feature_cache", "task_suite", "trace"}
_SECTION_FIELDS = {
    "policy": {"type", "parameters"},
    "task": {"id", "parameters"},
    "dataset": {"type", "split", "seed", "parameters"},
    "embodiment": {
        "id",
        "task_id",
        "control_rate_hz",
        "camera_mapping",
        "state_mapping",
        "action_mapping",
    },
    "training": {
        "device",
        "seed",
        "batch_size",
        "steps",
        "learning_rate",
        "optimizer",
        "scheduler",
        "precision",
        "gradient_clip_norm",
        "resume",
    },
    "evaluation": {"execution_mode", "episodes", "seed", "seeds", "max_steps"},
    "diagnostics": {"enabled"},
    "artifacts": {"output_dir", "checkpoint_name"},
}
_POLICIES = {
    "numpy_linear_chunk",
    "numpy_bc_mlp",
    "transformer_chunk",
    "transformer_chunk_cvae",
    "act",
    "act_v3",
    "diffusion_v3",
    "lerobot_smolvla",
}
_TASKS = {
    "fake_pusht",
    "fake_libero",
    "pusht_style_point_reach",
    "language_conditioned_point_reach",
    "rendered_visual_point_reach",
    "lerobot_pusht",
    "synthetic_vlm_suite",
}
_DATASETS = {"memory", "fake_pusht", "fake_libero", "v2_compat", "v31_synthetic"}
_EXECUTION = {"open_loop_chunk", "receding_horizon"}
_NUMPY_PARAMETER_FIELDS = {
    "state_dim",
    "instruction_dim",
    "action_dim",
    "chunk_size",
    "hidden_dim",
    "state_feature",
    "unused_modalities",
    "history",
    "horizon",
    "execution_steps",
}
_ACT_PARAMETER_FIELDS = {
    "state_feature",
    "camera_feature",
    "instruction_dim",
    "chunk_size",
    "history",
    "horizon",
    "execution_steps",
    "d_model",
    "nhead",
    "num_encoder_layers",
    "num_decoder_layers",
    "dim_feedforward",
    "latent_dim",
    "dropout",
    "kl_weight",
    "sample_latent_during_training",
    "temporal_ensemble_decay",
    "condition_mode",
    "condition_input_dim",
    "feature_intervention",
    "feature_shuffle_seed",
}
_DIFFUSION_PARAMETER_FIELDS = {
    "state_feature",
    "camera_features",
    "unused_modalities",
    "chunk_size",
    "history",
    "horizon",
    "execution_steps",
    "n_action_steps",
    "noise_scheduler_type",
    "num_train_timesteps",
    "num_inference_steps",
    "prediction_type",
    "do_mask_loss_for_padding",
    "noise_seed",
    "down_dims",
    "kernel_size",
    "n_groups",
    "diffusion_step_embed_dim",
    "spatial_softmax_num_keypoints",
    "vision_backbone",
    "use_group_norm",
    "use_separate_rgb_encoder_per_camera",
    "pretrained_backbone_weights",
    "beta_schedule",
    "clip_sample",
    "clip_sample_range",
}
_SMOLVLA_PARAMETER_FIELDS = {
    "state_feature",
    "camera_features",
    "chunk_size",
    "history",
    "horizon",
    "execution_steps",
    "n_action_steps",
    "max_state_dim",
    "max_action_dim",
    "repo_id",
    "revision",
    "file_hashes",
    "license_status",
    "pretrained_enabled",
    "conformance_only",
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
_PROMPT_FIELDS = {
    "enabled",
    "renderer_id",
    "renderer_version",
    "assistant_target",
    "neutral_token",
    "camera_order",
    "public_slots",
}
_ROUTING_FIELDS = {"mode", "state_features"}
_STATE_ROUTES = {"none", "expert_only", "prompt_only", "dual"}


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


def _finite_json_mapping(value: Any, name: str) -> dict[str, Any]:
    mapping = _mapping(value, name)
    try:
        encoded = json.dumps(
            mapping,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc
    return json.loads(encoded)


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
    contract_revision: int
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
    prompt: Mapping[str, Any]
    routing: Mapping[str, Any]
    vlm: Mapping[str, Any]
    feature_cache: Mapping[str, Any]
    task_suite: Mapping[str, Any]
    trace: Mapping[str, Any]
    artifacts: Mapping[str, Any]

    def __init__(self, **_: Any) -> None:
        raise TypeError("ExperimentConfig must be created with from_mapping() or load()")

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "ExperimentConfig":
        root = _mapping(source, "config")
        version = root["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("schema_version must be an integer")
        if version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
        raw_revision = root.get("contract_revision", 1)
        if isinstance(raw_revision, bool) or not isinstance(raw_revision, int):
            raise TypeError("contract_revision must be an integer")
        if raw_revision not in {1, CONFIG_CONTRACT_REVISION, V31_CONFIG_CONTRACT_REVISION}:
            raise ValueError(
                "contract_revision must be 1, "
                f"{CONFIG_CONTRACT_REVISION}, or {V31_CONFIG_CONTRACT_REVISION}"
            )
        allowed_root = (
            _ROOT_V1
            if raw_revision == 1
            else _ROOT_V4
            if raw_revision == V31_CONFIG_CONTRACT_REVISION
            else _ROOT_V2
        )
        _reject_unknown(root, allowed_root, "config")
        missing = sorted(allowed_root - set(root))
        if missing:
            raise ValueError("missing config field(s): " + ", ".join(missing))
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
            if not (
                policy["type"].startswith("numpy_")
                or policy["type"] in {"act_v3", "diffusion_v3", "lerobot_smolvla"}
            ):
                raise ValueError(f"policy.type {policy['type']!r} is migration-only in v3 Alpha 1")
            allowed_policy = _NUMPY_PARAMETER_FIELDS
            if policy["type"] == "act_v3":
                allowed_policy = _ACT_PARAMETER_FIELDS
            elif policy["type"] == "diffusion_v3":
                allowed_policy = _DIFFUSION_PARAMETER_FIELDS
            elif policy["type"] == "lerobot_smolvla":
                allowed_policy = _SMOLVLA_PARAMETER_FIELDS
            _reject_unknown(policy["parameters"], allowed_policy, "policy.parameters")
            required_parameters: tuple[str, ...]
            if policy["type"].startswith("numpy_"):
                required_parameters = ("state_dim", "instruction_dim", "action_dim", "chunk_size")
            elif policy["type"] == "act_v3":
                required_parameters = (
                    "state_feature",
                    "instruction_dim",
                    "chunk_size",
                    "d_model",
                    "nhead",
                    "num_encoder_layers",
                    "num_decoder_layers",
                    "dim_feedforward",
                    "latent_dim",
                    "dropout",
                    "kl_weight",
                )
            elif policy["type"] == "diffusion_v3":
                required_parameters = (
                    "state_feature",
                    "camera_features",
                    "chunk_size",
                    "history",
                    "horizon",
                    "execution_steps",
                    "n_action_steps",
                    "noise_scheduler_type",
                    "num_train_timesteps",
                    "num_inference_steps",
                    "prediction_type",
                    "do_mask_loss_for_padding",
                    "noise_seed",
                    "down_dims",
                )
            else:
                required_parameters = (
                    "state_feature",
                    "camera_features",
                    "chunk_size",
                    "history",
                    "horizon",
                    "execution_steps",
                    "n_action_steps",
                    "max_state_dim",
                    "max_action_dim",
                    "repo_id",
                    "revision",
                    "file_hashes",
                    "license_status",
                    "pretrained_enabled",
                    "conformance_only",
                )
            for name in required_parameters:
                if name not in policy["parameters"]:
                    raise ValueError(f"policy.parameters.{name} is required")
            positive_integer_parameters = ["chunk_size"]
            if policy["type"].startswith("numpy_"):
                positive_integer_parameters.extend(["state_dim", "action_dim"])
            elif policy["type"] == "act_v3":
                positive_integer_parameters.extend(
                    [
                        "d_model",
                        "nhead",
                        "num_encoder_layers",
                        "num_decoder_layers",
                        "dim_feedforward",
                        "latent_dim",
                    ]
                )
            for name in positive_integer_parameters:
                policy["parameters"][name] = _integer(
                    policy["parameters"][name], f"policy.parameters.{name}", positive=True
                )
            if "instruction_dim" in policy["parameters"]:
                policy["parameters"]["instruction_dim"] = _integer(
                    policy["parameters"]["instruction_dim"], "policy.parameters.instruction_dim"
                )
            if policy["type"].startswith("numpy_") and "hidden_dim" in policy["parameters"]:
                policy["parameters"]["hidden_dim"] = _integer(
                    policy["parameters"]["hidden_dim"],
                    "policy.parameters.hidden_dim",
                    positive=True,
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
                policy["parameters"].get("execution_steps", policy["parameters"]["chunk_size"]),
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
                        raise ValueError(
                            f"policy.parameters.{name} must be finite and non-negative"
                        )
                    if name == "dropout" and value >= 1:
                        raise ValueError("policy.parameters.dropout must be below one")
                    policy["parameters"][name] = value
                sample_latent = policy["parameters"].get("sample_latent_during_training", True)
                if not isinstance(sample_latent, bool):
                    raise TypeError("sample_latent_during_training must be boolean")
                policy["parameters"]["sample_latent_during_training"] = sample_latent
                decay = policy["parameters"].get("temporal_ensemble_decay")
                if decay is not None:
                    decay = _positive_float(decay, "policy.parameters.temporal_ensemble_decay")
                policy["parameters"]["temporal_ensemble_decay"] = decay
                if raw_revision == V31_CONFIG_CONTRACT_REVISION or any(
                    name in policy["parameters"]
                    for name in ("condition_mode", "condition_input_dim")
                ):
                    condition_mode = policy["parameters"].get("condition_mode", "none")
                    if condition_mode not in {"none", "frozen_feature", "learned_null"}:
                        raise ValueError(
                            "act_v3 condition_mode must be none, frozen_feature, or learned_null"
                        )
                    condition_input_dim = policy["parameters"].get("condition_input_dim", 0)
                    condition_input_dim = _integer(
                        condition_input_dim,
                        "policy.parameters.condition_input_dim",
                    )
                    if condition_mode == "none" and condition_input_dim != 0:
                        raise ValueError("unconditioned act_v3 requires condition_input_dim=0")
                    if condition_mode != "none" and condition_input_dim <= 0:
                        raise ValueError("conditioned act_v3 requires positive condition_input_dim")
                    if (
                        condition_mode != "none"
                        and policy["parameters"]["instruction_dim"] != condition_input_dim
                    ):
                        raise ValueError(
                            "conditioned act_v3 instruction_dim must equal condition_input_dim"
                        )
                    policy["parameters"]["condition_mode"] = condition_mode
                    policy["parameters"]["condition_input_dim"] = condition_input_dim
                    intervention = policy["parameters"].get("feature_intervention", "control")
                    if intervention not in {
                        "control",
                        "feature_mask",
                        "feature_shuffle",
                    }:
                        raise ValueError("unsupported conditioned ACT feature_intervention")
                    shuffle_seed = _integer(
                        policy["parameters"].get("feature_shuffle_seed", 202701),
                        "policy.parameters.feature_shuffle_seed",
                    )
                    if condition_mode != "frozen_feature" and intervention != "control":
                        raise ValueError(
                            "feature interventions require condition_mode=frozen_feature"
                        )
                    policy["parameters"]["feature_intervention"] = intervention
                    policy["parameters"]["feature_shuffle_seed"] = shuffle_seed
            if policy["type"] == "diffusion_v3":
                parameters = policy["parameters"]
                cameras = parameters["camera_features"]
                if isinstance(cameras, (str, bytes, Mapping)) or not isinstance(cameras, Sequence):
                    raise TypeError("policy.parameters.camera_features must be a sequence")
                camera_values = list(cameras)
                if not camera_values or any(
                    not isinstance(item, str) or not item for item in camera_values
                ):
                    raise ValueError("diffusion_v3 requires non-empty camera feature names")
                if len(camera_values) != len(set(camera_values)):
                    raise ValueError("diffusion_v3 camera_features cannot contain duplicates")
                parameters["camera_features"] = camera_values
                for name in (
                    "n_action_steps",
                    "num_train_timesteps",
                    "num_inference_steps",
                    "noise_seed",
                    "kernel_size",
                    "n_groups",
                    "diffusion_step_embed_dim",
                    "spatial_softmax_num_keypoints",
                ):
                    default = {
                        "kernel_size": 3,
                        "n_groups": 8,
                        "diffusion_step_embed_dim": 32,
                        "spatial_softmax_num_keypoints": 8,
                    }.get(name)
                    if name not in parameters and default is not None:
                        parameters[name] = default
                    parameters[name] = _integer(
                        parameters[name], f"policy.parameters.{name}", positive=name != "noise_seed"
                    )
                if parameters["chunk_size"] != parameters["n_action_steps"]:
                    raise ValueError("diffusion_v3 chunk_size must equal n_action_steps")
                if parameters["n_action_steps"] != parameters["execution_steps"]:
                    raise ValueError("diffusion_v3 n_action_steps must equal execution_steps")
                if parameters["n_action_steps"] > parameters["horizon"] - parameters["history"] + 1:
                    raise ValueError("diffusion_v3 n_action_steps exceeds the usable horizon")
                if parameters["noise_scheduler_type"] != "DDIM":
                    raise ValueError("diffusion_v3 Alpha 2 requires the DDIM scheduler")
                if parameters["prediction_type"] not in {"epsilon", "sample"}:
                    raise ValueError("diffusion_v3 prediction_type must be epsilon or sample")
                if parameters["num_inference_steps"] > parameters["num_train_timesteps"]:
                    raise ValueError("diffusion inference steps cannot exceed training timesteps")
                for name in (
                    "do_mask_loss_for_padding",
                    "use_group_norm",
                    "use_separate_rgb_encoder_per_camera",
                    "clip_sample",
                ):
                    parameters.setdefault(name, name != "use_group_norm")
                    if not isinstance(parameters[name], bool):
                        raise TypeError(f"policy.parameters.{name} must be boolean")
                down_dims = parameters["down_dims"]
                if isinstance(down_dims, (str, bytes, Mapping)) or not isinstance(
                    down_dims, Sequence
                ):
                    raise TypeError("policy.parameters.down_dims must be a sequence")
                parameters["down_dims"] = [
                    _integer(item, "policy.parameters.down_dims item", positive=True)
                    for item in down_dims
                ]
                if not parameters["down_dims"]:
                    raise ValueError("policy.parameters.down_dims cannot be empty")
                factor = 2 ** len(parameters["down_dims"])
                if parameters["horizon"] % factor:
                    raise ValueError(
                        "diffusion_v3 horizon must match the U-Net downsampling factor"
                    )
                for name, default in (
                    ("vision_backbone", "resnet18"),
                    ("pretrained_backbone_weights", None),
                    ("beta_schedule", "squaredcos_cap_v2"),
                ):
                    parameters.setdefault(name, default)
                if parameters["vision_backbone"] != "resnet18":
                    raise ValueError("diffusion_v3 Alpha 2 requires vision_backbone=resnet18")
                if parameters["pretrained_backbone_weights"] is not None:
                    raise ValueError("diffusion_v3 smoke must not download backbone weights")
                if parameters["beta_schedule"] not in {"linear", "squaredcos_cap_v2"}:
                    raise ValueError("unsupported diffusion beta_schedule")
                clip_range = parameters.get("clip_sample_range", 1.0)
                parameters["clip_sample_range"] = _positive_float(
                    clip_range, "policy.parameters.clip_sample_range"
                )
                unused = parameters.get("unused_modalities")
                if unused != ["instruction"]:
                    raise ValueError(
                        "diffusion_v3 must explicitly declare unused_modalities=[instruction]"
                    )
            if policy["type"] == "lerobot_smolvla":
                parameters = policy["parameters"]
                cameras = parameters["camera_features"]
                if isinstance(cameras, (str, bytes, Mapping)) or not isinstance(cameras, Sequence):
                    raise TypeError("policy.parameters.camera_features must be a sequence")
                camera_values = list(cameras)
                if not camera_values or any(
                    not isinstance(item, str) or not item for item in camera_values
                ):
                    raise ValueError("lerobot_smolvla requires non-empty camera feature names")
                if len(camera_values) != len(set(camera_values)):
                    raise ValueError("lerobot_smolvla camera_features cannot contain duplicates")
                parameters["camera_features"] = camera_values
                for name in ("n_action_steps", "max_state_dim", "max_action_dim"):
                    parameters[name] = _integer(
                        parameters[name], f"policy.parameters.{name}", positive=True
                    )
                if parameters["history"] != 1:
                    raise ValueError("lerobot_smolvla Alpha 2 requires history=1")
                if not (
                    parameters["chunk_size"]
                    == parameters["horizon"]
                    == parameters["execution_steps"]
                    == parameters["n_action_steps"]
                ):
                    raise ValueError("lerobot_smolvla chunk, horizon, and action steps must match")
                fixed_strings = {
                    "repo_id": "lerobot/smolvla_base",
                    "revision": "d06fce6e38c25c04ac5a6319eefb9fae0e257cb2",
                    "license_status": "unverified",
                }
                for name, expected in fixed_strings.items():
                    if parameters[name] != expected:
                        raise ValueError(f"lerobot_smolvla {name} must be pinned to {expected}")
                hashes = _mapping(parameters["file_hashes"], "policy.parameters.file_hashes")
                if hashes != {
                    "model.safetensors": "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
                }:
                    raise ValueError("lerobot_smolvla model.safetensors hash is not pinned")
                parameters["file_hashes"] = hashes
                if parameters["pretrained_enabled"] is not False:
                    raise ValueError("unverified SmolVLA weights must remain disabled")
                if parameters["conformance_only"] is not True:
                    raise ValueError("lerobot_smolvla must remain conformance_only in Alpha 2")
            if policy["type"].startswith("numpy_"):
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
        elif dataset["type"] == "v31_synthetic":
            _reject_unknown(
                dataset["parameters"],
                {"train_per_task", "held_out_per_cell"},
                "dataset.parameters",
            )
            if set(dataset["parameters"]) != {"train_per_task", "held_out_per_cell"}:
                raise ValueError("v31_synthetic requires train_per_task and held_out_per_cell")
            for name in ("train_per_task", "held_out_per_cell"):
                dataset["parameters"][name] = _integer(
                    dataset["parameters"][name],
                    f"dataset.parameters.{name}",
                    positive=True,
                )
        else:
            _reject_unknown(
                dataset["parameters"],
                {"episode_count", "steps_per_episode", "instruction_variant"},
                "dataset.parameters",
            )
            for name in ("episode_count", "steps_per_episode"):
                if name in dataset["parameters"]:
                    dataset["parameters"][name] = _integer(
                        dataset["parameters"][name], f"dataset.parameters.{name}", positive=True
                    )
            if dataset["parameters"].get("episode_count", 6) < 3:
                raise ValueError("fake datasets require at least three episodes")
            instruction_variant = dataset["parameters"].get("instruction_variant")
            if instruction_variant is not None:
                if dataset["type"] != "fake_libero":
                    raise ValueError("instruction_variant is supported only by fake_libero")
                if instruction_variant != "region_instruction_v1":
                    raise ValueError("unsupported fake_libero instruction_variant")

        task_id = task["id"]
        dataset_type = dataset["type"]
        if task_id in {"fake_pusht", "fake_libero"} and dataset_type != task_id:
            raise ValueError("fake task.id and dataset.type must match")
        if dataset_type in {"fake_pusht", "fake_libero"} and task_id != dataset_type:
            raise ValueError("fake dataset.type and task.id must match")
        if dataset_type == "v2_compat" and "legacy" not in policy["parameters"]:
            raise ValueError("v2_compat datasets require a migrated compat_read_only policy")
        if (task_id == "synthetic_vlm_suite") != (dataset_type == "v31_synthetic"):
            raise ValueError(
                "synthetic_vlm_suite task and v31_synthetic dataset must be used together"
            )
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
            if policy["type"] == "diffusion_v3":
                camera_features = {item.name: item for item in feature_schema.by_role("image")}
                configured = tuple(policy["parameters"]["camera_features"])
                if configured != tuple(camera_features):
                    raise ValueError(
                        "diffusion_v3 camera_features must exactly match FeatureSchema order"
                    )
                for name in configured:
                    shape = camera_features[name].shape
                    if len(shape) != 3 or shape[-1] != 3:
                        raise ValueError("diffusion_v3 requires HWC RGB camera features")
            if policy["type"] == "lerobot_smolvla":
                camera_features = {item.name: item for item in feature_schema.by_role("image")}
                configured = tuple(policy["parameters"]["camera_features"])
                if configured != tuple(camera_features):
                    raise ValueError(
                        "lerobot_smolvla camera_features must exactly match FeatureSchema order"
                    )
                for name in configured:
                    shape = camera_features[name].shape
                    if len(shape) != 3 or shape[-1] != 3:
                        raise ValueError("lerobot_smolvla requires HWC RGB camera features")
                if (
                    state_features[state_feature_name].shape[0]
                    > policy["parameters"]["max_state_dim"]
                ):
                    raise ValueError("SmolVLA state feature exceeds max_state_dim")
                if action_features[0].shape[0] > policy["parameters"]["max_action_dim"]:
                    raise ValueError("SmolVLA action feature exceeds max_action_dim")

        training = sections["training"]
        training["device"] = normalize_device(training["device"])
        training["seed"] = _integer(training["seed"], "training.seed")
        training["batch_size"] = _integer(
            training["batch_size"], "training.batch_size", positive=True
        )
        training["steps"] = _integer(training["steps"], "training.steps", positive=True)
        training["learning_rate"] = _positive_float(
            training["learning_rate"], "training.learning_rate"
        )
        optimizer = _mapping(training["optimizer"], "training.optimizer")
        _reject_unknown(optimizer, {"type", "parameters"}, "training.optimizer")
        if set(optimizer) != {"type", "parameters"}:
            raise ValueError("training.optimizer requires type and parameters")
        if optimizer["type"] not in {"sgd", "adam", "adamw"}:
            raise ValueError("training.optimizer.type must be sgd, adam, or adamw")
        optimizer["parameters"] = _mapping(optimizer["parameters"], "training.optimizer.parameters")
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
        scheduler["parameters"] = _mapping(scheduler["parameters"], "training.scheduler.parameters")
        _reject_unknown(
            scheduler["parameters"],
            {"warmup_steps", "min_learning_rate"},
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
        if policy["type"] == "diffusion_v3" and optimizer["type"] != "adamw":
            raise ValueError("diffusion_v3 requires optimizer.type=adamw")
        if policy["type"] == "diffusion_v3":
            _reject_unknown(
                optimizer["parameters"],
                {"weight_decay", "betas", "eps"},
                "training.optimizer.parameters for diffusion_v3",
            )
            weight_decay = optimizer["parameters"].get("weight_decay", 1e-6)
            if (
                isinstance(weight_decay, bool)
                or not isinstance(weight_decay, (int, float))
                or not math.isfinite(float(weight_decay))
                or float(weight_decay) < 0
            ):
                raise ValueError(
                    "diffusion_v3 optimizer weight_decay must be finite and non-negative"
                )
            optimizer["parameters"]["weight_decay"] = float(weight_decay)
            optimizer["parameters"]["eps"] = _positive_float(
                optimizer["parameters"].get("eps", 1e-8),
                "diffusion_v3 optimizer eps",
            )
            betas = optimizer["parameters"].get("betas", [0.95, 0.999])
            if isinstance(betas, (str, bytes, Mapping)) or not isinstance(betas, Sequence):
                raise TypeError("diffusion_v3 optimizer betas must be a sequence")
            if len(betas) != 2 or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or not 0 <= float(item) < 1
                for item in betas
            ):
                raise ValueError("diffusion_v3 optimizer betas must contain two values in [0, 1)")
            optimizer["parameters"]["betas"] = [float(item) for item in betas]
        if policy["type"] == "diffusion_v3" and scheduler["type"] != "constant":
            raise ValueError("diffusion_v3 Alpha 2 requires scheduler.type=constant")
        if policy["type"] == "diffusion_v3" and scheduler["parameters"]:
            raise ValueError("diffusion_v3 Alpha 2 requires empty scheduler.parameters")
        if policy["type"] == "diffusion_v3" and training["precision"] != "float32":
            raise ValueError("diffusion_v3 Alpha 2 supports float32 precision only")
        if policy["type"] == "lerobot_smolvla" and optimizer["type"] != "adamw":
            raise ValueError("lerobot_smolvla conformance config requires optimizer.type=adamw")
        if policy["type"] == "lerobot_smolvla" and optimizer["parameters"]:
            raise ValueError(
                "lerobot_smolvla conformance config requires empty optimizer parameters"
            )
        if policy["type"] == "lerobot_smolvla" and scheduler["type"] != "constant":
            raise ValueError("lerobot_smolvla conformance config requires a constant scheduler")
        if policy["type"] == "lerobot_smolvla" and scheduler["parameters"]:
            raise ValueError(
                "lerobot_smolvla conformance config requires empty scheduler parameters"
            )
        if policy["type"] == "lerobot_smolvla" and training["precision"] != "float32":
            raise ValueError("lerobot_smolvla conformance config requires float32")
        if (
            policy["type"] == "lerobot_smolvla"
            and policy["parameters"]["pretrained_enabled"]
            and not training["device"].startswith("cuda")
        ):
            raise ValueError("enabled lerobot_smolvla weights require CUDA training")
        if (
            policy["type"] == "act_v3"
            and policy["parameters"].get("temporal_ensemble_decay") is not None
            and sections["evaluation"]["execution_mode"] != "receding_horizon"
        ):
            raise ValueError("act_v3 temporal ensembling requires receding_horizon")

        evaluation = sections["evaluation"]
        if evaluation["execution_mode"] not in _EXECUTION:
            raise ValueError(
                "evaluation.execution_mode must be open_loop_chunk or receding_horizon"
            )
        evaluation["episodes"] = _integer(
            evaluation["episodes"], "evaluation.episodes", positive=True
        )
        evaluation["seed"] = _integer(evaluation["seed"], "evaluation.seed")
        evaluation["max_steps"] = _integer(
            evaluation["max_steps"], "evaluation.max_steps", positive=True
        )
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

        state_names = tuple(item.name for item in feature_schema.by_role("state"))
        parameters = policy["parameters"]
        policy_parameters = dict(parameters.get("legacy", parameters))
        expected_state = (str(policy_parameters.get("state_feature", "state.proprioception")),)
        if raw_revision == 1:
            prompt = {
                "enabled": False,
                "renderer_id": "lunavla.canonical_json",
                "renderer_version": 1,
                "assistant_target": "action_chunk",
                "neutral_token": "[MASKED]",
                "camera_order": [],
                "public_slots": {},
            }
            routing = {"mode": "expert_only", "state_features": list(expected_state)}
        else:
            prompt = _mapping(root["prompt"], "prompt")
            routing = _mapping(root["routing"], "routing")
            _reject_unknown(prompt, _PROMPT_FIELDS, "prompt")
            _reject_unknown(routing, _ROUTING_FIELDS, "routing")
            missing_prompt = sorted(_PROMPT_FIELDS - set(prompt))
            missing_routing = sorted(_ROUTING_FIELDS - set(routing))
            if missing_prompt:
                raise ValueError("missing field(s) in prompt: " + ", ".join(missing_prompt))
            if missing_routing:
                raise ValueError("missing field(s) in routing: " + ", ".join(missing_routing))
        if not isinstance(prompt["enabled"], bool):
            raise TypeError("prompt.enabled must be boolean")
        if prompt["renderer_id"] != "lunavla.canonical_json":
            raise ValueError("prompt.renderer_id must be lunavla.canonical_json")
        renderer_version = prompt["renderer_version"]
        if isinstance(renderer_version, bool) or renderer_version != 1:
            raise ValueError("prompt.renderer_version must be integer 1")
        if prompt["assistant_target"] != "action_chunk":
            raise ValueError("prompt.assistant_target must be action_chunk")
        if not isinstance(prompt["neutral_token"], str) or not prompt["neutral_token"].strip():
            raise ValueError("prompt.neutral_token must be a non-empty string")
        prompt["neutral_token"] = prompt["neutral_token"].strip()
        cameras = prompt["camera_order"]
        if isinstance(cameras, (str, bytes, Mapping)) or not isinstance(cameras, Sequence):
            raise TypeError("prompt.camera_order must be a sequence")
        camera_values = list(cameras)
        if any(not isinstance(item, str) or not item for item in camera_values):
            raise ValueError("prompt.camera_order must contain non-empty strings")
        if len(camera_values) != len(set(camera_values)):
            raise ValueError("prompt.camera_order cannot contain duplicates")
        prompt["camera_order"] = camera_values
        prompt["public_slots"] = _finite_json_mapping(prompt["public_slots"], "prompt.public_slots")
        mode = routing["mode"]
        if mode not in _STATE_ROUTES:
            raise ValueError(f"unsupported routing.mode {mode!r}")
        route_features = routing["state_features"]
        if isinstance(route_features, (str, bytes, Mapping)) or not isinstance(
            route_features, Sequence
        ):
            raise TypeError("routing.state_features must be a sequence")
        route_values = list(route_features)
        routing["state_features"] = route_values
        if tuple(route_values) != expected_state:
            raise ValueError("routing.state_features must exactly match policy state order")
        if any(item not in state_names for item in route_values):
            raise ValueError("routing.state_features must reference declared state features")

        expected_cameras: tuple[str, ...] = ()
        if policy["type"] == "act_v3" and policy_parameters.get("camera_feature") is not None:
            expected_cameras = (str(policy_parameters["camera_feature"]),)
        elif policy["type"] in {"diffusion_v3", "lerobot_smolvla"}:
            expected_cameras = tuple(str(item) for item in policy_parameters["camera_features"])
        elif "legacy" in parameters and policy_parameters.get("image_shape") is not None:
            expected_cameras = ("camera.primary",)
        if (
            raw_revision in {CONFIG_CONTRACT_REVISION, V31_CONFIG_CONTRACT_REVISION}
            and tuple(camera_values) != expected_cameras
        ):
            raise ValueError("prompt.camera_order must exactly match policy camera order")
        consumes_instruction = False
        if policy["type"] == "lerobot_smolvla":
            consumes_instruction = True
        elif policy["type"] != "diffusion_v3":
            consumes_instruction = int(policy_parameters.get("instruction_dim", 0)) > 0
        if mode in {"prompt_only", "dual"} and not consumes_instruction:
            raise ValueError("prompt state routing requires an instruction-consuming policy")
        if mode in {"prompt_only", "dual"} and not prompt["enabled"]:
            raise ValueError("prompt state routing requires prompt.enabled=true")
        if policy["type"] == "diffusion_v3" and (
            prompt["enabled"] or mode not in {"none", "expert_only"}
        ):
            raise ValueError("diffusion_v3 does not support prompt diagnostics")
        if diagnostics["enabled"]:
            if raw_revision not in {CONFIG_CONTRACT_REVISION, V31_CONFIG_CONTRACT_REVISION}:
                raise ValueError("diagnostics require config contract_revision=2 or 4")
            if not prompt["enabled"]:
                raise ValueError("diagnostics require prompt.enabled=true")
            if evaluation["execution_mode"] != "receding_horizon":
                raise ValueError("prompt/state diagnostics require receding_horizon")
            if (
                policy["type"] == "lerobot_smolvla"
                and policy_parameters.get("conformance_only") is True
            ):
                raise ValueError("conformance-only SmolVLA cannot run diagnostic training")
            if dataset["parameters"].get("instruction_variant") != "region_instruction_v1":
                raise ValueError(
                    "diagnostics require fake_libero instruction_variant=region_instruction_v1"
                )
        artifacts = sections["artifacts"]
        if not isinstance(artifacts["output_dir"], str) or not artifacts["output_dir"].strip():
            raise ValueError("artifacts.output_dir must be a non-empty string")
        artifacts["output_dir"] = artifacts["output_dir"].strip()
        artifacts["checkpoint_name"] = _checkpoint_name(artifacts["checkpoint_name"])

        if raw_revision == V31_CONFIG_CONTRACT_REVISION:
            vlm = VLMBackendSpecV1.from_mapping(_mapping(root["vlm"], "vlm")).to_dict()
            task_suite = TaskSuiteSpecV1.from_mapping(
                _mapping(root["task_suite"], "task_suite")
            ).to_dict()
            feature_cache = _finite_json_mapping(root["feature_cache"], "feature_cache")
            _reject_unknown(
                feature_cache,
                {"enabled", "root", "backend_spec_sha256", "read_only"},
                "feature_cache",
            )
            if set(feature_cache) != {"enabled", "root", "backend_spec_sha256", "read_only"}:
                raise ValueError(
                    "feature_cache requires enabled, root, backend_spec_sha256, read_only"
                )
            if not isinstance(feature_cache["enabled"], bool) or not isinstance(
                feature_cache["read_only"], bool
            ):
                raise TypeError("feature_cache enabled and read_only must be boolean")
            cache_root = feature_cache["root"]
            if not isinstance(cache_root, str) or not cache_root.strip():
                raise ValueError("feature_cache.root must be a non-empty string")
            cache_path = Path(cache_root)
            if cache_path.is_absolute() or cache_root in {".", ".."} or ".." in cache_path.parts:
                raise ValueError("feature_cache.root must be a contained relative path")
            feature_cache["root"] = cache_root.strip()
            backend_hash = feature_cache["backend_spec_sha256"]
            if backend_hash != VLMBackendSpecV1.from_mapping(vlm).sha256():
                raise ValueError("feature_cache.backend_spec_sha256 does not match vlm")
            trace = _finite_json_mapping(root["trace"], "trace")
            _reject_unknown(trace, {"enabled", "output_dir", "languages", "offline"}, "trace")
            if set(trace) != {"enabled", "output_dir", "languages", "offline"}:
                raise ValueError("trace requires enabled, output_dir, languages, and offline")
            if not isinstance(trace["enabled"], bool) or trace["offline"] is not True:
                raise ValueError("trace.enabled must be boolean and trace.offline must be true")
            trace_path = trace["output_dir"]
            if not isinstance(trace_path, str) or not trace_path.strip():
                raise ValueError("trace.output_dir must be a non-empty string")
            parsed_trace_path = Path(trace_path)
            if (
                parsed_trace_path.is_absolute()
                or trace_path in {".", ".."}
                or ".." in parsed_trace_path.parts
            ):
                raise ValueError("trace.output_dir must be a contained relative path")
            trace["output_dir"] = trace_path.strip()
            if trace["languages"] != ["en", "zh-CN"]:
                raise ValueError("trace.languages must be ordered as [en, zh-CN]")
            if feature_cache["enabled"] is not True or feature_cache["read_only"] is not True:
                raise ValueError("v3.1 training requires an enabled read-only frozen feature cache")
            if policy["type"] != "act_v3":
                raise ValueError("contract_revision=4 requires policy.type=act_v3")
            if policy["parameters"]["condition_mode"] not in {
                "frozen_feature",
                "learned_null",
            }:
                raise ValueError("contract_revision=4 requires an explicit conditioned ACT arm")
            if policy["parameters"]["d_model"] != 64:
                raise ValueError("v3.1 conditioned ACT requires d_model=64")
        else:
            if (
                policy["type"] == "act_v3"
                and policy["parameters"].get("condition_mode", "none") != "none"
            ):
                raise ValueError("conditioned ACT requires contract_revision=4")
            vlm = {}
            feature_cache = {}
            task_suite = {}
            trace = {}

        instance = object.__new__(cls)
        values = {
            "schema_version": version,
            "contract_revision": raw_revision,
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
            "prompt": prompt,
            "routing": routing,
            "vlm": vlm,
            "feature_cache": feature_cache,
            "task_suite": task_suite,
            "trace": trace,
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
        result = {
            name: _thaw(getattr(self, name))
            for name in (
                "schema_version",
                "project_name",
                "engine",
                "policy",
                "task",
                "dataset",
                "embodiment",
                "features",
                "training",
                "evaluation",
                "diagnostics",
                "artifacts",
            )
        }
        if self.contract_revision in {CONFIG_CONTRACT_REVISION, V31_CONFIG_CONTRACT_REVISION}:
            result["contract_revision"] = self.contract_revision
            result["prompt"] = _thaw(self.prompt)
            result["routing"] = _thaw(self.routing)
        if self.contract_revision == V31_CONFIG_CONTRACT_REVISION:
            result["vlm"] = _thaw(self.vlm)
            result["feature_cache"] = _thaw(self.feature_cache)
            result["task_suite"] = _thaw(self.task_suite)
            result["trace"] = _thaw(self.trace)
        return result

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
