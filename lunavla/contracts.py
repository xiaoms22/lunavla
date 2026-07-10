from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import numpy.typing as npt

from model.policy_base import ActionChunk


Array = npt.NDArray[np.generic]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

_DEVICE_PATTERN = re.compile(r"^(?:cpu|mps(?::0)?|cuda(?::[0-9]+)?)$")


def normalize_device(device: str) -> str:
    """Return a canonical device string or fail before work is started."""

    value = str(device).strip().lower()
    if not _DEVICE_PATTERN.fullmatch(value):
        raise ValueError("device must be 'cpu', 'mps', 'cuda', or an indexed accelerator")
    if value == "cuda":
        return "cuda:0"
    if value == "mps":
        return "mps:0"
    return value


def _state_array(value: Array, *, name: str) -> Float32Array:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must have a numeric dtype")
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty rank-1 array; got {array.shape}")
    result = np.array(array, dtype=np.float32, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def _image_array(value: Array) -> Array:
    array = np.asarray(value)
    if array.ndim not in (2, 3) or any(size <= 0 for size in array.shape):
        raise ValueError(f"image must be non-empty HW or HWC; got {array.shape}")
    if array.ndim == 3 and array.shape[-1] not in (1, 3, 4):
        raise ValueError("an HWC image must have 1, 3, or 4 channels")
    if array.dtype == np.uint8:
        result: Array = np.array(array, copy=True)
    elif array.dtype.kind == "f":
        float_image = np.array(array, dtype=np.float32, copy=True)
        if not np.all(np.isfinite(float_image)):
            raise ValueError("image contains NaN or infinite values")
        result = float_image
    else:
        raise TypeError("image dtype must be uint8 or floating point")
    return result


def _freeze_info_value(value: Any, *, name: str) -> Any:
    if isinstance(value, np.ndarray):
        return _freeze_info_value(value.tolist(), name=name)
    if isinstance(value, np.generic):
        return _freeze_info_value(value.item(), name=name)
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
            frozen[key] = _freeze_info_value(item, name=f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_info_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{name} contains unsupported value type {type(value).__name__}")


@dataclass(frozen=True)
class Observation:
    """A modality-preserving observation shared by every v2 policy and task."""

    state: Array
    instruction: str | None = None
    image: Array | None = None

    def __post_init__(self) -> None:
        if self.instruction is not None and not isinstance(self.instruction, str):
            raise TypeError("instruction must be a string or None")
        object.__setattr__(self, "state", _state_array(self.state, name="state"))
        if self.image is not None:
            object.__setattr__(self, "image", _image_array(self.image))


@dataclass(frozen=True)
class Transition:
    """One environment or demonstration transition at the public v2 boundary."""

    observation: Observation
    action: Array
    reward: float
    next_observation: Observation
    terminated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, Observation):
            raise TypeError("observation must be an Observation")
        if not isinstance(self.next_observation, Observation):
            raise TypeError("next_observation must be an Observation")
        action = _state_array(self.action, name="action")
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if not isinstance(self.terminated, (bool, np.bool_)):
            raise TypeError("terminated must be boolean")
        if not isinstance(self.info, Mapping):
            raise TypeError("info must be a mapping")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "info", _freeze_info_value(self.info, name="info"))


@dataclass(frozen=True)
class PolicyBatch:
    """A raw-modality batch with padded action-chunk supervision."""

    observations: tuple[Observation, ...]
    targets: Array
    valid_mask: Array
    device: str = "cpu"

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        if not observations or any(not isinstance(item, Observation) for item in observations):
            raise ValueError("observations must contain at least one Observation")

        raw_targets = np.asarray(self.targets)
        if raw_targets.dtype.kind not in "fiu":
            raise TypeError("targets must have a numeric dtype")
        if raw_targets.ndim != 3:
            raise ValueError(f"targets must be [batch, chunk, action]; got {raw_targets.shape}")
        targets = np.array(raw_targets, dtype=np.float32, copy=True)
        if not np.all(np.isfinite(targets)):
            raise ValueError("targets contain NaN or infinite values")

        raw_mask = np.asarray(self.valid_mask)
        if raw_mask.dtype.kind != "b":
            raise TypeError("valid_mask must have boolean dtype")
        mask = np.array(raw_mask, dtype=bool, copy=True)
        if mask.shape != targets.shape[:2]:
            raise ValueError(
                f"valid_mask must have shape {targets.shape[:2]}; got {mask.shape}"
            )
        if len(observations) != targets.shape[0]:
            raise ValueError("observations and targets must have the same batch size")
        if targets.shape[1] <= 0 or targets.shape[2] <= 0:
            raise ValueError("chunk and action dimensions must be positive")
        if np.any(~np.any(mask, axis=1)):
            raise ValueError("each sample must contain at least one valid action")

        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "device", normalize_device(self.device))

    @property
    def batch_size(self) -> int:
        return len(self.observations)


@runtime_checkable
class VLAPolicy(Protocol):
    @property
    def policy_id(self) -> str:
        ...

    @property
    def device(self) -> str:
        ...

    @property
    def action_dim(self) -> int:
        ...

    @property
    def chunk_size(self) -> int:
        ...

    def train_batch(self, batch: PolicyBatch, *, learning_rate: float) -> float:
        ...

    def predict_chunk(self, observation: Observation) -> ActionChunk:
        ...

    def save_checkpoint(
        self,
        path: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        ...


@runtime_checkable
class TaskEnv(Protocol):
    def reset(self, *, seed: int | None = None) -> Observation:
        ...

    def step(self, action: Array) -> Transition:
        ...


@runtime_checkable
class DatasetSource(Protocol):
    def load(self) -> Sequence[Transition]:
        ...
