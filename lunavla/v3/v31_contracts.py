from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ID = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SPLITS = {"train", "validation", "test"}


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _identifier(value: object, name: str) -> str:
    result = _string(value, name)
    if not _ID.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase identifier")
    return result


def _sha256(value: object, name: str) -> str:
    result = _string(value, name)
    if not _SHA256.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _revision(value: object, name: str) -> str:
    result = _string(value, name)
    if not _REVISION.fullmatch(result):
        raise ValueError(f"{name} must be an immutable 40-character Git revision")
    return result


def _version(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError(f"{name} must be integer 1")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{name} must be {qualifier}finite")
    return result


def _tuple_strings(value: object, name: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(_string(item, f"{name} item") for item in value)
    if nonempty and not result:
        raise ValueError(f"{name} cannot be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


def _relative_path(value: object, name: str) -> str:
    result = _string(value, name)
    path = PurePosixPath(result)
    if path.is_absolute() or result in {".", ".."} or ".." in path.parts or "\\" in result:
        raise ValueError(f"{name} must be a contained POSIX relative path")
    return result


def _freeze_json(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains NaN or infinite values")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} keys must be strings")
            frozen[key] = _freeze_json(item, f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{name}[]") for item in value)
    raise TypeError(f"{name} contains unsupported type {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _strict_mapping(value: Mapping[str, Any], fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing {name} field(s): {', '.join(missing)}")
    return dict(value)


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VLMBackendSpecV1:
    backend_id: str
    repo_id: str
    revision: str
    spdx_license: str
    license_scope: str
    license_evidence_sha256: str
    processor_class: str
    processor_config_sha256: str
    model_config_sha256: str
    hidden_layer: int
    pooling: str
    image_token_layout: str
    camera_order: tuple[str, ...]
    model_dtype: str
    device: str
    offload_plan: str
    deterministic: bool
    evidence_role: str
    weight_files: Mapping[str, str]
    total_weight_bytes: int
    frozen: bool = True
    quantized: bool = False
    finetuned: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        _version(self.schema_version, "VLMBackendSpecV1.schema_version")
        object.__setattr__(self, "backend_id", _identifier(self.backend_id, "backend_id"))
        object.__setattr__(self, "repo_id", _string(self.repo_id, "repo_id"))
        object.__setattr__(self, "revision", _revision(self.revision, "revision"))
        object.__setattr__(self, "spdx_license", _string(self.spdx_license, "spdx_license"))
        if self.license_scope != "model_weights":
            raise ValueError("license_scope must explicitly cover model_weights")
        object.__setattr__(self, "license_evidence_sha256", _sha256(self.license_evidence_sha256, "license_evidence_sha256"))
        object.__setattr__(self, "processor_class", _string(self.processor_class, "processor_class"))
        object.__setattr__(self, "processor_config_sha256", _sha256(self.processor_config_sha256, "processor_config_sha256"))
        object.__setattr__(self, "model_config_sha256", _sha256(self.model_config_sha256, "model_config_sha256"))
        if isinstance(self.hidden_layer, bool) or self.hidden_layer != -1:
            raise ValueError("hidden_layer must be -1 (the final hidden layer)")
        if self.pooling != "attention_mask_mean":
            raise ValueError("pooling must be attention_mask_mean")
        if self.image_token_layout not in {"processor_native", "video_frames_first"}:
            raise ValueError("unsupported image_token_layout")
        object.__setattr__(self, "camera_order", _tuple_strings(self.camera_order, "camera_order", nonempty=True))
        if self.model_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported model_dtype")
        if self.device not in {"cpu", "mps", "cuda"}:
            raise ValueError("device must be cpu, mps, or cuda")
        object.__setattr__(self, "offload_plan", _string(self.offload_plan, "offload_plan"))
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be boolean")
        if self.evidence_role not in {"claim_bearing", "observational"}:
            raise ValueError("evidence_role must be claim_bearing or observational")
        files: dict[str, str] = {}
        for path, digest in self.weight_files.items():
            files[_relative_path(path, "weight file")] = _sha256(digest, "weight file hash")
        if not files:
            raise ValueError("weight_files cannot be empty")
        object.__setattr__(self, "weight_files", MappingProxyType(dict(sorted(files.items()))))
        object.__setattr__(self, "total_weight_bytes", _positive_integer(self.total_weight_bytes, "total_weight_bytes"))
        if self.frozen is not True or self.quantized is not False or self.finetuned is not False:
            raise ValueError("v3.1 VLM backends must be frozen, unquantized, and not finetuned")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "backend_id": self.backend_id,
            "repo_id": self.repo_id, "revision": self.revision,
            "spdx_license": self.spdx_license,
            "license_scope": self.license_scope,
            "license_evidence_sha256": self.license_evidence_sha256,
            "processor_class": self.processor_class,
            "processor_config_sha256": self.processor_config_sha256,
            "model_config_sha256": self.model_config_sha256, "hidden_layer": self.hidden_layer,
            "pooling": self.pooling, "image_token_layout": self.image_token_layout,
            "camera_order": list(self.camera_order), "model_dtype": self.model_dtype,
            "device": self.device, "offload_plan": self.offload_plan,
            "deterministic": self.deterministic, "evidence_role": self.evidence_role,
            "weight_files": dict(self.weight_files),
            "total_weight_bytes": self.total_weight_bytes, "frozen": self.frozen,
            "quantized": self.quantized, "finetuned": self.finetuned,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VLMBackendSpecV1":
        data = _strict_mapping(value, set(cls.__dataclass_fields__), "VLMBackendSpecV1")
        return cls(**data)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class FrozenFeatureManifestV1:
    backend_spec_sha256: str
    processor_sha256: str
    prompt_renderer_sha256: str
    image_sha256: str
    sample_id: str
    episode_id: str
    step_index: int
    split: str
    task_id: str
    held_out_stratum: str
    hidden_layer: int
    pooling: str
    dtype: str
    device_environment_sha256: str
    output_shape: tuple[int, ...]
    finite: bool
    feature_sha256: str
    deterministic: bool
    generation_command: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _version(self.schema_version, "FrozenFeatureManifestV1.schema_version")
        for name in ("backend_spec_sha256", "processor_sha256", "prompt_renderer_sha256", "image_sha256", "device_environment_sha256", "feature_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("sample_id", "episode_id", "task_id", "held_out_stratum"):
            object.__setattr__(self, name, _string(getattr(self, name), name))
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        if self.split not in _SPLITS:
            raise ValueError("split must be train, validation, or test")
        if self.hidden_layer != -1 or self.pooling != "attention_mask_mean":
            raise ValueError("feature extraction must use final-layer attention-mask mean pooling")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported feature dtype")
        shape = tuple(_positive_integer(item, "output_shape item") for item in self.output_shape)
        if len(shape) != 1:
            raise ValueError("output_shape must describe one pooled feature vector")
        object.__setattr__(self, "output_shape", shape)
        if self.finite is not True:
            raise ValueError("feature manifest must declare finite=true")
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be boolean")
        object.__setattr__(self, "generation_command", _tuple_strings(self.generation_command, "generation_command", nonempty=True))

    @property
    def typed_identity(self) -> tuple[str, str, str, int]:
        return (self.split, self.task_id, self.episode_id, self.step_index)

    def to_dict(self) -> dict[str, Any]:
        return {name: (list(value) if isinstance(value, tuple) else value) for name, value in self.__dict__.items()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenFeatureManifestV1":
        return cls(**_strict_mapping(value, set(cls.__dataclass_fields__), "FrozenFeatureManifestV1"))

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class FeatureCacheIndexV1:
    backend_spec_sha256: str
    manifest_hashes: tuple[str, ...]
    expected_identities: tuple[str, ...]
    observed_identities: tuple[str, ...]
    task_ids: tuple[str, ...]
    held_out_strata: tuple[str, ...]
    split_counts: Mapping[str, int]
    total_feature_bytes: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        _version(self.schema_version, "FeatureCacheIndexV1.schema_version")
        object.__setattr__(self, "backend_spec_sha256", _sha256(self.backend_spec_sha256, "backend_spec_sha256"))
        hashes = tuple(_sha256(item, "manifest_hash") for item in self.manifest_hashes)
        if not hashes or len(hashes) != len(set(hashes)):
            raise ValueError("manifest_hashes must be non-empty and unique")
        object.__setattr__(self, "manifest_hashes", hashes)
        expected = _tuple_strings(self.expected_identities, "expected_identities", nonempty=True)
        observed = _tuple_strings(self.observed_identities, "observed_identities", nonempty=True)
        if expected != observed:
            raise ValueError("expected and observed cache identities must match exactly")
        object.__setattr__(self, "expected_identities", expected)
        object.__setattr__(self, "observed_identities", observed)
        object.__setattr__(self, "task_ids", _tuple_strings(self.task_ids, "task_ids", nonempty=True))
        object.__setattr__(self, "held_out_strata", _tuple_strings(self.held_out_strata, "held_out_strata", nonempty=True))
        if set(self.split_counts) != _SPLITS:
            raise ValueError("split_counts must contain train, validation, and test")
        counts = {key: _positive_integer(value, f"split_counts.{key}") for key, value in self.split_counts.items()}
        if sum(counts.values()) != len(expected):
            raise ValueError("split_counts must equal the identity count")
        object.__setattr__(self, "split_counts", MappingProxyType(counts))
        object.__setattr__(self, "total_feature_bytes", _positive_integer(self.total_feature_bytes, "total_feature_bytes"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "backend_spec_sha256": self.backend_spec_sha256,
            "manifest_hashes": list(self.manifest_hashes),
            "expected_identities": list(self.expected_identities),
            "observed_identities": list(self.observed_identities), "task_ids": list(self.task_ids),
            "held_out_strata": list(self.held_out_strata), "split_counts": dict(self.split_counts),
            "total_feature_bytes": self.total_feature_bytes,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeatureCacheIndexV1":
        return cls(**_strict_mapping(value, set(cls.__dataclass_fields__), "FeatureCacheIndexV1"))

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class TaskSuiteSpecV1:
    suite_id: str
    task_ids: tuple[str, ...]
    geometry_generator: str
    visible_modalities: tuple[str, ...]
    instruction_generator: str
    held_out_strata: tuple[str, ...]
    success_conditions: Mapping[str, Any]
    image_shape: tuple[int, int, int]
    state_fields: tuple[str, ...]
    action_fields: tuple[str, ...]
    action_min: float
    action_max: float
    control_rate_hz: float
    max_steps: int
    oracle_excluded_fields: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _version(self.schema_version, "TaskSuiteSpecV1.schema_version")
        object.__setattr__(self, "suite_id", _identifier(self.suite_id, "suite_id"))
        task_ids = _tuple_strings(self.task_ids, "task_ids", nonempty=True)
        required = {"direct_pick_place", "waypoint_sequence", "failure_recovery"}
        if set(task_ids) != required:
            raise ValueError("v3.1 task_ids must be the three registered teaching tasks")
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "geometry_generator", _string(self.geometry_generator, "geometry_generator"))
        object.__setattr__(self, "visible_modalities", _tuple_strings(self.visible_modalities, "visible_modalities", nonempty=True))
        object.__setattr__(self, "instruction_generator", _string(self.instruction_generator, "instruction_generator"))
        object.__setattr__(self, "held_out_strata", _tuple_strings(self.held_out_strata, "held_out_strata", nonempty=True))
        object.__setattr__(self, "success_conditions", _freeze_json(self.success_conditions, "success_conditions"))
        shape = tuple(_positive_integer(item, "image_shape item") for item in self.image_shape)
        if shape != (96, 96, 3):
            raise ValueError("v3.1 image_shape must be [96, 96, 3]")
        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "state_fields", _tuple_strings(self.state_fields, "state_fields", nonempty=True))
        object.__setattr__(self, "action_fields", _tuple_strings(self.action_fields, "action_fields", nonempty=True))
        minimum = _finite_number(self.action_min, "action_min")
        maximum = _finite_number(self.action_max, "action_max")
        if (minimum, maximum) != (-1.0, 1.0):
            raise ValueError("v3.1 action bounds must be [-1, 1]")
        object.__setattr__(self, "action_min", minimum)
        object.__setattr__(self, "action_max", maximum)
        rate = _finite_number(self.control_rate_hz, "control_rate_hz", positive=True)
        if rate != 10.0:
            raise ValueError("v3.1 control_rate_hz must be 10")
        object.__setattr__(self, "control_rate_hz", rate)
        if _positive_integer(self.max_steps, "max_steps") != 64:
            raise ValueError("v3.1 max_steps must be 64")
        object.__setattr__(self, "oracle_excluded_fields", _tuple_strings(self.oracle_excluded_fields, "oracle_excluded_fields", nonempty=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "suite_id": self.suite_id,
            "task_ids": list(self.task_ids), "geometry_generator": self.geometry_generator,
            "visible_modalities": list(self.visible_modalities),
            "instruction_generator": self.instruction_generator,
            "held_out_strata": list(self.held_out_strata),
            "success_conditions": _thaw_json(self.success_conditions),
            "image_shape": list(self.image_shape), "state_fields": list(self.state_fields),
            "action_fields": list(self.action_fields), "action_min": self.action_min,
            "action_max": self.action_max, "control_rate_hz": self.control_rate_hz,
            "max_steps": self.max_steps, "oracle_excluded_fields": list(self.oracle_excluded_fields),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskSuiteSpecV1":
        return cls(**_strict_mapping(value, set(cls.__dataclass_fields__), "TaskSuiteSpecV1"))

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class TraceBundleManifestV1:
    evidence_manifest_sha256: str
    run_manifest_hashes: tuple[str, ...]
    paired_identity_hash: str
    static_files: Mapping[str, str]
    privacy_report_sha256: str
    languages: tuple[str, ...]
    offline: bool
    csp_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _version(self.schema_version, "TraceBundleManifestV1.schema_version")
        for name in ("evidence_manifest_sha256", "paired_identity_hash", "privacy_report_sha256", "csp_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        hashes = tuple(_sha256(item, "run manifest hash") for item in self.run_manifest_hashes)
        if not hashes or len(hashes) != len(set(hashes)):
            raise ValueError("run_manifest_hashes must be non-empty and unique")
        object.__setattr__(self, "run_manifest_hashes", hashes)
        files: dict[str, str] = {}
        for path, digest in self.static_files.items():
            files[_relative_path(path, "static file")] = _sha256(digest, "static file hash")
        if not files:
            raise ValueError("static_files cannot be empty")
        object.__setattr__(self, "static_files", MappingProxyType(dict(sorted(files.items()))))
        languages = _tuple_strings(self.languages, "languages", nonempty=True)
        if languages != ("en", "zh-CN"):
            raise ValueError("Trace Lab languages must be ordered as en, zh-CN")
        object.__setattr__(self, "languages", languages)
        if self.offline is not True:
            raise ValueError("Trace Lab must be offline")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "run_manifest_hashes": list(self.run_manifest_hashes),
            "paired_identity_hash": self.paired_identity_hash,
            "static_files": dict(self.static_files),
            "privacy_report_sha256": self.privacy_report_sha256,
            "languages": list(self.languages), "offline": self.offline,
            "csp_sha256": self.csp_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TraceBundleManifestV1":
        return cls(**_strict_mapping(value, set(cls.__dataclass_fields__), "TraceBundleManifestV1"))

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())
