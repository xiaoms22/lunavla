from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import numpy.typing as npt

from lunavla.contracts import normalize_device
from model.policy_base import ActionChunk

from .contracts import ObservationV3


Array = npt.NDArray[np.generic]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODALITIES = {"image", "state", "instruction"}
_LICENSE_STATES = {"verified", "unverified", "not_required"}


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _name(value: object, name: str) -> str:
    result = _string(value, name)
    if not _NAME.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase contract identifier")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _readonly(value: Array, *, dtype: npt.DTypeLike | None = None) -> Array:
    array = np.asarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown:
        raise ValueError(f"unknown field(s) in {name}: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing field(s) in {name}: {', '.join(missing)}")
    return dict(value)


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class ModelSourceContractV1:
    repo_id: str
    revision: str
    file_hashes: Mapping[str, str]
    license_status: str
    pretrained_enabled: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ModelSourceContractV1 schema_version must be integer 1")
        repo_id = _string(self.repo_id, "repo_id")
        revision = _string(self.revision, "revision")
        if not isinstance(self.file_hashes, Mapping):
            raise TypeError("file_hashes must be a mapping")
        hashes: dict[str, str] = {}
        for path, digest in self.file_hashes.items():
            filename = _string(path, "file_hashes path")
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise ValueError("model source file paths must be relative and contained")
            hashes[filename] = _sha(digest, f"file_hashes.{filename}")
        if self.license_status not in _LICENSE_STATES:
            raise ValueError(f"unsupported license_status {self.license_status!r}")
        if not isinstance(self.pretrained_enabled, bool):
            raise TypeError("pretrained_enabled must be boolean")
        if self.pretrained_enabled and self.license_status != "verified":
            raise ValueError("pretrained weights require license_status=verified")
        object.__setattr__(self, "repo_id", repo_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "file_hashes", MappingProxyType(dict(sorted(hashes.items()))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "file_hashes": dict(self.file_hashes),
            "license_status": self.license_status,
            "pretrained_enabled": self.pretrained_enabled,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelSourceContractV1":
        payload = _exact(
            value,
            {
                "schema_version",
                "repo_id",
                "revision",
                "file_hashes",
                "license_status",
                "pretrained_enabled",
            },
            "model source contract",
        )
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class PolicySpecV3:
    policy_id: str
    backend: str
    model_source: ModelSourceContractV1
    required_modalities: tuple[str, ...]
    camera_order: tuple[str, ...]
    state_order: tuple[str, ...]
    history: int
    chunk_size: int
    horizon: int
    execution_steps: int
    normalization: Mapping[str, str]
    device: str
    deterministic: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PolicySpecV3 schema_version must be integer 1")
        policy_id = _name(self.policy_id, "policy_id")
        backend = _name(self.backend, "backend")
        if not isinstance(self.model_source, ModelSourceContractV1):
            raise TypeError("model_source must be ModelSourceContractV1")
        modalities = tuple(_name(item, "required modality") for item in self.required_modalities)
        if not modalities or any(item not in _MODALITIES for item in modalities):
            raise ValueError("required_modalities must use image, state, or instruction")
        if len(modalities) != len(set(modalities)):
            raise ValueError("required_modalities cannot contain duplicates")
        cameras = tuple(_name(item, "camera_order item") for item in self.camera_order)
        states = tuple(_name(item, "state_order item") for item in self.state_order)
        if len(cameras) != len(set(cameras)) or len(states) != len(set(states)):
            raise ValueError("camera_order and state_order cannot contain duplicates")
        if ("image" in modalities) != bool(cameras):
            raise ValueError("image modality and camera_order must be declared together")
        if ("state" in modalities) != bool(states):
            raise ValueError("state modality and state_order must be declared together")
        history = _positive_integer(self.history, "history")
        chunk = _positive_integer(self.chunk_size, "chunk_size")
        horizon = _positive_integer(self.horizon, "horizon")
        execution = _positive_integer(self.execution_steps, "execution_steps")
        if chunk > horizon:
            raise ValueError("chunk_size cannot exceed horizon")
        if execution > chunk:
            raise ValueError("execution_steps cannot exceed chunk_size")
        if not isinstance(self.normalization, Mapping):
            raise TypeError("normalization must be a mapping")
        normalization = {
            _name(feature, "normalization feature"): _name(mode, "normalization mode")
            for feature, mode in self.normalization.items()
        }
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be boolean")
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "required_modalities", modalities)
        object.__setattr__(self, "camera_order", cameras)
        object.__setattr__(self, "state_order", states)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "chunk_size", chunk)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "execution_steps", execution)
        object.__setattr__(self, "normalization", MappingProxyType(normalization))
        object.__setattr__(self, "device", normalize_device(self.device))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "backend": self.backend,
            "model_source": self.model_source.to_dict(),
            "required_modalities": list(self.required_modalities),
            "camera_order": list(self.camera_order),
            "state_order": list(self.state_order),
            "history": self.history,
            "chunk_size": self.chunk_size,
            "horizon": self.horizon,
            "execution_steps": self.execution_steps,
            "normalization": dict(self.normalization),
            "device": self.device,
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PolicySpecV3":
        payload = _exact(
            value,
            {
                "schema_version",
                "policy_id",
                "backend",
                "model_source",
                "required_modalities",
                "camera_order",
                "state_order",
                "history",
                "chunk_size",
                "horizon",
                "execution_steps",
                "normalization",
                "device",
                "deterministic",
            },
            "policy spec",
        )
        payload["model_source"] = ModelSourceContractV1.from_mapping(payload["model_source"])
        for name in ("required_modalities", "camera_order", "state_order"):
            raw = payload[name]
            if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Sequence):
                raise TypeError(f"policy spec {name} must be a sequence")
            payload[name] = tuple(raw)
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True, eq=False)
class PolicySampleV3:
    observation_history: tuple[ObservationV3, ...]
    history_mask: BoolArray
    action_chunk: Float32Array | None
    valid_mask: BoolArray | None
    episode_id: str | int
    step_index: int

    def __post_init__(self) -> None:
        history = tuple(self.observation_history)
        if not history or any(not isinstance(item, ObservationV3) for item in history):
            raise ValueError("observation_history must contain ObservationV3 values")
        mask_raw = np.asarray(self.history_mask)
        if mask_raw.dtype.kind != "b" or mask_raw.shape != (len(history),):
            raise ValueError("history_mask must be boolean with one value per history step")
        mask = _readonly(mask_raw, dtype=bool)
        if not np.any(mask) or not bool(mask[-1]):
            raise ValueError("history_mask must include the current observation")
        current = history[-1]
        if current.episode_id != self.episode_id or current.step_index != self.step_index:
            raise ValueError("sample identity must match the current observation")
        if (self.action_chunk is None) != (self.valid_mask is None):
            raise ValueError("action_chunk and valid_mask must be provided together")
        chunk: Float32Array | None = None
        valid: BoolArray | None = None
        if self.action_chunk is not None and self.valid_mask is not None:
            chunk_raw = np.asarray(self.action_chunk)
            valid_raw = np.asarray(self.valid_mask)
            if chunk_raw.dtype.kind not in "fiu" or chunk_raw.ndim != 2:
                raise ValueError("action_chunk must be numeric [chunk, action]")
            if valid_raw.dtype.kind != "b" or valid_raw.shape != chunk_raw.shape[:1]:
                raise ValueError("valid_mask must match the action chunk length")
            if not np.any(valid_raw):
                raise ValueError("valid_mask must contain at least one valid action")
            values = np.asarray(chunk_raw, dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise ValueError("action_chunk contains NaN or infinite values")
            chunk = _readonly(values, dtype=np.float32)  # type: ignore[assignment]
            valid = _readonly(valid_raw, dtype=bool)  # type: ignore[assignment]
        object.__setattr__(self, "observation_history", history)
        object.__setattr__(self, "history_mask", mask)
        object.__setattr__(self, "action_chunk", chunk)
        object.__setattr__(self, "valid_mask", valid)

    @property
    def observation(self) -> ObservationV3:
        return self.observation_history[-1]


@dataclass(frozen=True)
class PolicyBatchV3:
    samples: tuple[PolicySampleV3, ...]
    device: str = "cpu"

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples or any(not isinstance(item, PolicySampleV3) for item in samples):
            raise ValueError("samples must contain PolicySampleV3 values")
        shapes = {
            (
                len(item.observation_history),
                None if item.action_chunk is None else item.action_chunk.shape,
            )
            for item in samples
        }
        if len(shapes) != 1:
            raise ValueError("all policy samples in a batch must have matching shapes")
        if any(item.action_chunk is None for item in samples):
            raise ValueError("training batches require action supervision")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "device", normalize_device(self.device))

    @property
    def batch_size(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class TrainStepResultV3:
    loss: float
    loss_components: Mapping[str, float]
    gradient_norm: float | None
    learning_rate: float
    step: int
    finite: bool
    timing_ms: Mapping[str, float]

    def __post_init__(self) -> None:
        loss = float(self.loss)
        learning_rate = float(self.learning_rate)
        if not math.isfinite(loss) or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("loss and learning_rate must be finite; learning_rate must be positive")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if not isinstance(self.finite, bool):
            raise TypeError("finite must be boolean")
        components = {str(name): float(value) for name, value in self.loss_components.items()}
        timing = {str(name): float(value) for name, value in self.timing_ms.items()}
        if not all(math.isfinite(value) for value in (*components.values(), *timing.values())):
            raise ValueError("loss components and timing must be finite")
        if any(value < 0 for value in timing.values()):
            raise ValueError("timing values cannot be negative")
        gradient = None if self.gradient_norm is None else float(self.gradient_norm)
        if gradient is not None and (not math.isfinite(gradient) or gradient < 0):
            raise ValueError("gradient_norm must be finite and non-negative")
        if not self.finite:
            raise ValueError("non-finite train steps cannot cross the public contract")
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "loss_components", MappingProxyType(components))
        object.__setattr__(self, "gradient_norm", gradient)
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "timing_ms", MappingProxyType(timing))


@runtime_checkable
class VLAPolicyV3(Protocol):
    @property
    def spec(self) -> PolicySpecV3: ...

    def reset(self, seed: int) -> None: ...

    def train_step(
        self, batch: PolicyBatchV3, *, learning_rate: float, step: int
    ) -> TrainStepResultV3: ...

    def predict_chunk(self, sample: PolicySampleV3) -> ActionChunk: ...

    def save_checkpoint(self, path: Path, *, metadata: Mapping[str, Any]) -> Path: ...
