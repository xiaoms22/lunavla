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


@dataclass(frozen=True)
class ActionChunk:
    """A policy prediction with explicit action and padding dimensions."""

    values: Float32Array
    valid_mask: BoolArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        mask = np.asarray(self.valid_mask)
        if values.dtype.kind not in "fiu":
            raise TypeError("ActionChunk.values must have a numeric dtype")
        if values.ndim != 2:
            raise ValueError(f"ActionChunk.values must be [chunk, action]; got {values.shape}")
        if mask.shape != values.shape[:1]:
            raise ValueError(
                f"ActionChunk.valid_mask must have shape {values.shape[:1]}; got {mask.shape}"
            )
        values = values.astype(np.float32, copy=False)
        mask = mask.astype(bool, copy=False)
        if not np.all(np.isfinite(values)):
            raise ValueError("ActionChunk.values contain NaN or infinite values")
        if not np.any(mask):
            raise ValueError("ActionChunk.valid_mask must mark at least one action as valid")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", mask)


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
