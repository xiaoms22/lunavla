from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .contracts import EpisodeRecordV3, FeatureSchema


Float32Array = npt.NDArray[np.float32]
_MODES = {"none", "standard", "minmax"}


def _readonly(value: npt.ArrayLike) -> Float32Array:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("normalization values must be finite")
    result = np.frombuffer(array.tobytes(order="C"), dtype=np.float32).reshape(array.shape)
    result.setflags(write=False)
    return result


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True, eq=False)
class FeatureNormalizationV1:
    feature_name: str
    mode: str
    sample_count: int
    offset: Float32Array
    scale: Float32Array

    def __post_init__(self) -> None:
        if not isinstance(self.feature_name, str) or not self.feature_name:
            raise ValueError("feature_name must be a non-empty string")
        if self.mode not in _MODES:
            raise ValueError(f"unsupported normalization mode {self.mode!r}")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int):
            raise TypeError("sample_count must be an integer")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        offset = _readonly(self.offset)
        scale = _readonly(self.scale)
        if offset.shape != scale.shape or offset.ndim != 1 or offset.size == 0:
            raise ValueError("normalization offset and scale must be matching rank-1 arrays")
        if np.any(scale <= 0):
            raise ValueError("normalization scale must be positive")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "scale", scale)

    def normalize(self, value: npt.ArrayLike) -> Float32Array:
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1:] != self.offset.shape:
            raise ValueError(f"feature {self.feature_name} normalization shape mismatch")
        result = (array - self.offset) / self.scale
        return _readonly(result)

    def denormalize(self, value: npt.ArrayLike) -> Float32Array:
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1:] != self.offset.shape:
            raise ValueError(f"feature {self.feature_name} denormalization shape mismatch")
        return _readonly(array * self.scale + self.offset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "mode": self.mode,
            "sample_count": self.sample_count,
            "offset": self.offset.tolist(),
            "scale": self.scale.tolist(),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureNormalizationV1):
            return NotImplemented
        return bool(
            self.feature_name == other.feature_name
            and self.mode == other.mode
            and self.sample_count == other.sample_count
            and np.array_equal(self.offset, other.offset)
            and np.array_equal(self.scale, other.scale)
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeatureNormalizationV1":
        fields = {"feature_name", "mode", "sample_count", "offset", "scale"}
        if set(value) != fields:
            unknown = sorted(set(value) - fields)
            missing = sorted(fields - set(value))
            raise ValueError(
                "invalid FeatureNormalizationV1 fields; "
                f"unknown={unknown}, missing={missing}"
            )
        return cls(
            feature_name=value["feature_name"],
            mode=value["mode"],
            sample_count=value["sample_count"],
            offset=np.asarray(value["offset"], dtype=np.float32),
            scale=np.asarray(value["scale"], dtype=np.float32),
        )


@dataclass(frozen=True, eq=False)
class NormalizationStatsV1:
    feature_schema_sha256: str
    source_split: str
    features: Mapping[str, FeatureNormalizationV1]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("NormalizationStatsV1 schema_version must be integer 1")
        if (
            not isinstance(self.feature_schema_sha256, str)
            or len(self.feature_schema_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.feature_schema_sha256)
        ):
            raise ValueError("feature_schema_sha256 must be a lowercase SHA-256")
        if self.source_split != "train":
            raise ValueError("normalization statistics must come from the train split")
        if not isinstance(self.features, Mapping):
            raise TypeError("features must be a mapping")
        normalized: dict[str, FeatureNormalizationV1] = {}
        for name, stats in self.features.items():
            if not isinstance(stats, FeatureNormalizationV1) or name != stats.feature_name:
                raise ValueError("normalization feature keys must match their records")
            normalized[name] = stats
        object.__setattr__(self, "features", MappingProxyType(dict(sorted(normalized.items()))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_schema_sha256": self.feature_schema_sha256,
            "source_split": self.source_split,
            "features": [item.to_dict() for item in self.features.values()],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NormalizationStatsV1":
        fields = {"schema_version", "feature_schema_sha256", "source_split", "features"}
        if set(value) != fields:
            unknown = sorted(set(value) - fields)
            missing = sorted(fields - set(value))
            raise ValueError(
                f"invalid NormalizationStatsV1 fields; unknown={unknown}, missing={missing}"
            )
        raw = value["features"]
        if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Sequence):
            raise TypeError("normalization features must be a sequence")
        records = tuple(FeatureNormalizationV1.from_mapping(item) for item in raw)
        return cls(
            schema_version=value["schema_version"],
            feature_schema_sha256=value["feature_schema_sha256"],
            source_split=value["source_split"],
            features={item.feature_name: item for item in records},
        )

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NormalizationStatsV1):
            return NotImplemented
        return bool(
            self.feature_schema_sha256 == other.feature_schema_sha256
            and self.source_split == other.source_split
            and tuple(self.features) == tuple(other.features)
            and all(self.features[name] == other.features[name] for name in self.features)
            and self.schema_version == other.schema_version
        )


def fit_normalization_stats(
    episodes: Sequence[EpisodeRecordV3], feature_schema: FeatureSchema
) -> NormalizationStatsV1:
    records = tuple(episodes)
    if not records:
        raise ValueError("normalization requires non-empty train episodes")
    result: dict[str, FeatureNormalizationV1] = {}
    for feature in feature_schema.features:
        values: list[Float32Array] = []
        if feature.role == "state":
            values = [
                np.asarray(transition.observation.state[feature.name], dtype=np.float32)
                for episode in records
                for transition in episode.transitions
            ]
        elif feature.role == "action":
            values = [
                np.asarray(transition.action, dtype=np.float32)
                for episode in records
                for transition in episode.transitions
            ]
        elif feature.role == "image" and feature.normalization != "none":
            raise ValueError("image normalization belongs to the policy processor contract")
        else:
            continue
        stacked = np.stack(values)
        if stacked.ndim != 2 or stacked.shape[1:] != feature.shape:
            raise ValueError(f"normalization data shape mismatch for {feature.name}")
        if not np.all(np.isfinite(stacked)):
            raise ValueError(f"normalization data contains non-finite values for {feature.name}")
        if feature.normalization in {"none", "dataset"}:
            mode = "none" if feature.normalization == "none" else "standard"
        else:
            mode = feature.normalization
        if mode == "standard":
            offset = np.mean(stacked, axis=0, dtype=np.float64).astype(np.float32)
            scale = np.std(stacked, axis=0, dtype=np.float64).astype(np.float32)
        elif mode == "minmax":
            offset = np.min(stacked, axis=0)
            scale = np.max(stacked, axis=0) - offset
        else:
            offset = np.zeros(stacked.shape[1], dtype=np.float32)
            scale = np.ones(stacked.shape[1], dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)
        offset = np.asarray(offset, dtype=np.float32).reshape(stacked.shape[1])
        scale = scale.reshape(stacked.shape[1])
        scale[~np.isfinite(scale) | (scale <= np.finfo(np.float32).eps)] = 1.0
        if not all(math.isfinite(float(item)) for item in np.ravel(offset)):
            raise ValueError(f"normalization offset is non-finite for {feature.name}")
        result[feature.name] = FeatureNormalizationV1(
            feature.name,
            mode,
            int(stacked.shape[0]),
            offset,
            scale,
        )
    return NormalizationStatsV1(feature_schema.sha256(), "train", result)
