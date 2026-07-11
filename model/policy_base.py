from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.generic]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
PolicyBatch = dict[str, Array]
PolicySample = Union[dict[str, Any], Array]


def _readonly_owned_array(value: Array) -> Array:
    """Detach an array onto a bytes-backed buffer that cannot become writable."""

    contiguous = np.ascontiguousarray(value)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _array_values_equal(left: Array, right: Array) -> bool:
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(left, right)
    )


@dataclass(frozen=True, eq=False)
class ActionChunk:
    """A policy prediction with explicit action and padding dimensions."""

    values: Float32Array
    valid_mask: BoolArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        mask = np.asarray(self.valid_mask)
        if values.dtype.kind not in "fiu":
            raise TypeError("ActionChunk.values must have a numeric dtype")
        if values.ndim != 2 or any(size <= 0 for size in values.shape):
            raise ValueError(f"ActionChunk.values must be [chunk, action]; got {values.shape}")
        if mask.dtype.kind != "b":
            raise TypeError("ActionChunk.valid_mask must have boolean dtype")
        if mask.shape != values.shape[:1]:
            raise ValueError(
                f"ActionChunk.valid_mask must have shape {values.shape[:1]}; got {mask.shape}"
            )
        values = np.array(values, dtype=np.float32, copy=True)
        mask = np.array(mask, dtype=bool, copy=True)
        if not np.all(np.isfinite(values)):
            raise ValueError("ActionChunk.values contain NaN or infinite values")
        if not np.any(mask):
            raise ValueError("ActionChunk.valid_mask must mark at least one action as valid")
        object.__setattr__(self, "values", _readonly_owned_array(values))
        object.__setattr__(self, "valid_mask", _readonly_owned_array(mask))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActionChunk):
            return NotImplemented
        return bool(
            _array_values_equal(self.values, other.values)
            and _array_values_equal(self.valid_mask, other.valid_mask)
        )


class MiniVLAPolicyBase(ABC):
    """Minimal policy contract shared by training and rollout evaluation."""

    policy_name = "base"

    @abstractmethod
    def forward(self, batch: PolicyBatch) -> dict[str, float]:
        """Return loss values for a training batch without changing weights."""

    @abstractmethod
    def predict_chunk(self, sample: PolicySample) -> ActionChunk:
        """Return a ``[chunk_size, action_dim]`` prediction and its validity mask."""

    def predict_action(self, sample: PolicySample) -> Float32Array:
        """Compatibility adapter returning the first valid action only."""

        chunk = self.predict_chunk(sample)
        first_valid = int(np.flatnonzero(chunk.valid_mask)[0])
        return chunk.values[first_valid].copy()

    @abstractmethod
    def save_pretrained(self, run_dir: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Save enough state for later policy-agnostic evaluation."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, run_dir: str | Path) -> tuple["MiniVLAPolicyBase", dict[str, Any]]:
        """Load a policy and its metadata from a saved run directory."""
