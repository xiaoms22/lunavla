from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.generic]
Scalar = TypeVar("Scalar", bound=np.generic)
_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")
_ROLES = {"image", "state", "action", "instruction", "reward", "termination", "metadata"}
_DTYPES = {"uint8", "float32", "float64", "int64", "bool"}
_NORMALIZATIONS = {"none", "standard", "minmax", "dataset"}


def _readonly_copy(value: npt.NDArray[Scalar]) -> npt.NDArray[Scalar]:
    result = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
    result.setflags(write=False)
    return result


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_shape(value: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of positive integers")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise TypeError(f"{name}[{index}] must be an integer")
        if int(item) <= 0:
            raise ValueError(f"{name}[{index}] must be positive")
        result.append(int(item))
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return tuple(result)


def _freeze_json(value: Any, name: str) -> Any:
    if isinstance(value, np.ndarray):
        return _freeze_json(value.tolist(), name)
    if isinstance(value, np.generic):
        return _freeze_json(value.item(), name)
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
        return tuple(_freeze_json(item, f"{name}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{name} contains unsupported value type {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _array_equal(left: Array, right: Array) -> bool:
    return bool(left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right))


def _mapping_arrays_equal(left: Mapping[str, Array], right: Mapping[str, Array]) -> bool:
    return bool(
        tuple(left) == tuple(right)
        and all(_array_equal(left[name], right[name]) for name in left)
    )


def _image(value: Array, name: str) -> Array:
    raw = np.asarray(value)
    if raw.ndim not in (2, 3) or any(size <= 0 for size in raw.shape):
        raise ValueError(f"{name} must be a non-empty HW or HWC image")
    if raw.ndim == 3 and raw.shape[-1] not in (1, 3, 4):
        raise ValueError(f"{name} HWC images must have 1, 3, or 4 channels")
    if raw.dtype == np.uint8:
        return _readonly_copy(np.array(raw, copy=True))
    if raw.dtype.kind != "f":
        raise TypeError(f"{name} must have uint8 or floating dtype")
    result = np.array(raw, dtype=np.float32, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return _readonly_copy(result)


def _vector(value: Array, name: str) -> npt.NDArray[np.float32]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty numeric rank-1 array")
    result = np.array(raw, dtype=np.float32, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return _readonly_copy(result)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    role: str
    dtype: str
    shape: tuple[int, ...]
    unit: str
    frame: str
    rate_hz: float | None
    normalization: str
    source_key: str
    required_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _non_empty_string(self.name, "name")
        if not _NAME.fullmatch(name):
            raise ValueError("name must use lowercase letters, digits, '.', '_' or '-'")
        if self.role not in _ROLES:
            raise ValueError(f"unsupported feature role {self.role!r}")
        if self.dtype not in _DTYPES:
            raise ValueError(f"unsupported feature dtype {self.dtype!r}")
        shape = _positive_shape(self.shape, "shape")
        unit = _non_empty_string(self.unit, "unit")
        frame = _non_empty_string(self.frame, "frame")
        source_key = _non_empty_string(self.source_key, "source_key")
        if self.normalization not in _NORMALIZATIONS:
            raise ValueError(f"unsupported normalization {self.normalization!r}")
        rate = self.rate_hz
        if rate is not None:
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise TypeError("rate_hz must be numeric or None")
            rate = float(rate)
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("rate_hz must be positive and finite")
        required = tuple(_non_empty_string(item, "required_by item") for item in self.required_by)
        if len(required) != len(set(required)):
            raise ValueError("required_by cannot contain duplicates")
        if self.role == "image" and self.dtype not in {"uint8", "float32"}:
            raise ValueError("image features must use uint8 or float32")
        if self.role in {"state", "action"} and len(shape) != 1:
            raise ValueError(f"{self.role} features must be rank-1")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "source_key", source_key)
        object.__setattr__(self, "rate_hz", rate)
        object.__setattr__(self, "required_by", required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "unit": self.unit,
            "frame": self.frame,
            "rate_hz": self.rate_hz,
            "normalization": self.normalization,
            "source_key": self.source_key,
            "required_by": list(self.required_by),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeatureSpec":
        allowed = {
            "name", "role", "dtype", "shape", "unit", "frame", "rate_hz",
            "normalization", "source_key", "required_by",
        }
        unknown = sorted(set(value) - allowed)
        missing = sorted(allowed - {"required_by"} - set(value))
        if unknown:
            raise ValueError("unknown FeatureSpec field(s): " + ", ".join(unknown))
        if missing:
            raise ValueError("missing FeatureSpec field(s): " + ", ".join(missing))
        return cls(**dict(value))


@dataclass(frozen=True)
class FeatureSchema:
    features: tuple[FeatureSpec, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("FeatureSchema schema_version must be integer 1")
        features = tuple(self.features)
        if not features or any(not isinstance(item, FeatureSpec) for item in features):
            raise ValueError("features must contain at least one FeatureSpec")
        names = [item.name for item in features]
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        object.__setattr__(self, "features", features)

    def by_role(self, role: str) -> tuple[FeatureSpec, ...]:
        if role not in _ROLES:
            raise ValueError(f"unsupported feature role {role!r}")
        return tuple(item for item in self.features if item.role == role)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "items": [item.to_dict() for item in self.features]}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeatureSchema":
        unknown = sorted(set(value) - {"schema_version", "items"})
        if unknown:
            raise ValueError("unknown FeatureSchema field(s): " + ", ".join(unknown))
        if set(value) != {"schema_version", "items"}:
            raise ValueError("FeatureSchema requires schema_version and items")
        items = value["items"]
        if isinstance(items, (str, bytes, Mapping)) or not isinstance(items, Sequence):
            raise TypeError("features.items must be a sequence")
        return cls(tuple(FeatureSpec.from_mapping(item) for item in items), value["schema_version"])

    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def validate_observation(self, observation: "ObservationV3") -> None:
        expected_images = self.by_role("image")
        expected_state = self.by_role("state")
        if tuple(observation.images) != tuple(item.name for item in expected_images):
            raise ValueError("observation image order/names do not match FeatureSchema")
        if tuple(observation.state) != tuple(item.name for item in expected_state):
            raise ValueError("observation state order/names do not match FeatureSchema")
        for item in (*expected_images, *expected_state):
            value = observation.images[item.name] if item.role == "image" else observation.state[item.name]
            if tuple(value.shape) != item.shape or value.dtype.name != item.dtype:
                raise ValueError(f"feature {item.name} does not match declared dtype/shape")

    def validate_action(self, action: Array) -> None:
        actions = self.by_role("action")
        if len(actions) != 1:
            raise ValueError("Alpha action validation requires exactly one action feature")
        value = np.asarray(action)
        if value.dtype.name != actions[0].dtype or tuple(value.shape) != actions[0].shape:
            raise ValueError("action does not match declared dtype/shape")
        if value.dtype.kind in "f" and not np.all(np.isfinite(value)):
            raise ValueError("action contains NaN or infinite values")


def _freeze_string_mapping(value: Mapping[str, str], name: str) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        result[_non_empty_string(key, f"{name} key")] = _non_empty_string(item, f"{name}.{key}")
    if len(result.values()) != len(set(result.values())):
        raise ValueError(f"{name} feature targets must be unique")
    return MappingProxyType(result)


@dataclass(frozen=True)
class EmbodimentSpec:
    embodiment_id: str
    task_id: str
    control_rate_hz: float | None
    camera_mapping: Mapping[str, str]
    state_mapping: Mapping[str, str]
    action_mapping: Mapping[str, str]

    def __post_init__(self) -> None:
        embodiment_id = _non_empty_string(self.embodiment_id, "embodiment_id")
        task_id = _non_empty_string(self.task_id, "task_id")
        rate = self.control_rate_hz
        if rate is None:
            if not embodiment_id.startswith("v2_compat/"):
                raise ValueError("control_rate_hz may be None only for v2_compat embodiments")
        else:
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise TypeError("control_rate_hz must be numeric or None")
            rate = float(rate)
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("control_rate_hz must be positive and finite")
        camera = _freeze_string_mapping(self.camera_mapping, "camera_mapping")
        state = _freeze_string_mapping(self.state_mapping, "state_mapping")
        action = _freeze_string_mapping(self.action_mapping, "action_mapping")
        if not action:
            raise ValueError("action_mapping cannot be empty")
        object.__setattr__(self, "embodiment_id", embodiment_id)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "control_rate_hz", rate)
        object.__setattr__(self, "camera_mapping", camera)
        object.__setattr__(self, "state_mapping", state)
        object.__setattr__(self, "action_mapping", action)

    def validate_schema(self, schema: FeatureSchema) -> None:
        roles = {item.name: item.role for item in schema.features}
        for mapping, role in (
            (self.camera_mapping, "image"), (self.state_mapping, "state"), (self.action_mapping, "action")
        ):
            for target in mapping.values():
                if roles.get(target) != role:
                    raise ValueError(f"mapping target {target!r} is not a declared {role} feature")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.embodiment_id,
            "task_id": self.task_id,
            "control_rate_hz": self.control_rate_hz,
            "camera_mapping": dict(self.camera_mapping),
            "state_mapping": dict(self.state_mapping),
            "action_mapping": dict(self.action_mapping),
        }


@dataclass(frozen=True, eq=False)
class ObservationV3:
    images: Mapping[str, Array]
    state: Mapping[str, Array]
    instruction: str | None
    timestamp_s: float
    episode_id: str | int
    step_index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.instruction is not None and (not isinstance(self.instruction, str) or not self.instruction.strip()):
            raise ValueError("instruction must be a non-empty string or None")
        if isinstance(self.timestamp_s, bool) or not isinstance(self.timestamp_s, (int, float)):
            raise TypeError("timestamp_s must be numeric")
        timestamp = float(self.timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if isinstance(self.episode_id, bool) or not isinstance(self.episode_id, (str, int)):
            raise TypeError("episode_id must be a string or integer")
        if isinstance(self.episode_id, str) and not self.episode_id:
            raise ValueError("episode_id cannot be empty")
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, (int, np.integer)) or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        images = MappingProxyType({name: _image(value, f"images.{name}") for name, value in self.images.items()})
        state = MappingProxyType({name: _vector(value, f"state.{name}") for name, value in self.state.items()})
        if any(not isinstance(name, str) or not name for name in (*images, *state)):
            raise ValueError("image/state names must be non-empty strings")
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "instruction", self.instruction.strip() if self.instruction else None)
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "step_index", int(self.step_index))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObservationV3):
            return NotImplemented
        return bool(
            _mapping_arrays_equal(self.images, other.images)
            and _mapping_arrays_equal(self.state, other.state)
            and self.instruction == other.instruction
            and self.timestamp_s == other.timestamp_s
            and self.episode_id == other.episode_id
            and self.step_index == other.step_index
            and self.metadata == other.metadata
        )


@dataclass(frozen=True, eq=False)
class TransitionV3:
    observation: ObservationV3
    action: Array
    reward: float
    next_observation: ObservationV3
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ObservationV3) or not isinstance(self.next_observation, ObservationV3):
            raise TypeError("observation and next_observation must be ObservationV3")
        action = _vector(self.action, "action")
        if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
            raise TypeError("reward must be numeric")
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if not isinstance(self.terminated, (bool, np.bool_)) or not isinstance(self.truncated, (bool, np.bool_)):
            raise TypeError("terminated and truncated must be boolean")
        if self.terminated and self.truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(self, "info", _freeze_json(self.info, "info"))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransitionV3):
            return NotImplemented
        return bool(
            self.observation == other.observation
            and _array_equal(self.action, other.action)
            and self.reward == other.reward
            and self.next_observation == other.next_observation
            and self.terminated == other.terminated
            and self.truncated == other.truncated
            and self.info == other.info
        )


@dataclass(frozen=True)
class EpisodeRecordV3:
    episode_id: str | int
    transitions: tuple[TransitionV3, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        if not transitions:
            raise ValueError("transitions cannot be empty")
        for index, transition in enumerate(transitions):
            if not isinstance(transition, TransitionV3):
                raise TypeError("transitions must contain TransitionV3 values")
            if transition.observation.episode_id != self.episode_id:
                raise ValueError("transition observation episode_id does not match episode")
            if transition.observation.step_index != index:
                raise ValueError("episode timesteps must be contiguous and start at zero")
            if transition.next_observation.episode_id != self.episode_id:
                raise ValueError("next_observation episode_id does not match episode")
            if transition.next_observation.step_index != index + 1:
                raise ValueError("next_observation timestep must be current timestep plus one")
            if index < len(transitions) - 1 and (transition.terminated or transition.truncated):
                raise ValueError("only the final transition may end an episode")
        if not (transitions[-1].terminated or transitions[-1].truncated):
            raise ValueError("the final transition must be terminated or truncated")
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "steps": len(self.transitions),
            "metadata": _thaw_json(self.metadata),
        }


@runtime_checkable
class DatasetSourceV3(Protocol):
    def load(self) -> Sequence[EpisodeRecordV3]: ...


@runtime_checkable
class TaskEnvV3(Protocol):
    def reset(self, *, seed: int | None = None) -> ObservationV3: ...

    def step(self, action: Array) -> TransitionV3: ...

    def close(self) -> None: ...
