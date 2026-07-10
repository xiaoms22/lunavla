"""Dependency-light temporal ensembling for overlapping action chunks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from model.policy_base import ActionChunk


Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class _HistoricalChunk:
    start_step: int
    values: Float32Array
    valid_mask: BoolArray


class TemporalEnsembler:
    """Exponentially combine aligned predictions from overlapping action chunks."""

    def __init__(
        self,
        *,
        decay: float = 0.01,
        action_dim: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        if not math.isfinite(decay) or decay < 0.0:
            raise ValueError("decay must be finite and non-negative")
        if action_dim is not None and action_dim <= 0:
            raise ValueError("action_dim must be positive when supplied")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive when supplied")
        self.decay = float(decay)
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self._step = 0
        self._history: list[_HistoricalChunk] = []

    @property
    def step(self) -> int:
        return self._step

    @property
    def history_size(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self._step = 0
        self._history.clear()

    def update(self, chunk: ActionChunk) -> Float32Array:
        if not isinstance(chunk, ActionChunk):
            raise TypeError("chunk must be a model.policy_base.ActionChunk")
        chunk_size, action_dim = chunk.values.shape
        if self.action_dim is None:
            self.action_dim = action_dim
        elif action_dim != self.action_dim:
            raise ValueError(f"expected action_dim={self.action_dim}, got {action_dim}")
        if self.chunk_size is None:
            self.chunk_size = chunk_size
        elif chunk_size != self.chunk_size:
            raise ValueError(f"expected chunk_size={self.chunk_size}, got {chunk_size}")

        self._history.append(
            _HistoricalChunk(
                start_step=self._step,
                values=np.array(chunk.values, dtype=np.float32, copy=True),
                valid_mask=np.array(chunk.valid_mask, dtype=bool, copy=True),
            )
        )
        candidates: list[Float32Array] = []
        ages: list[int] = []
        retained: list[_HistoricalChunk] = []
        for historical in self._history:
            relative_step = self._step - historical.start_step
            if relative_step < len(historical.values):
                retained.append(historical)
                if historical.valid_mask[relative_step]:
                    candidates.append(historical.values[relative_step])
                    ages.append(relative_step)
        self._history = retained
        if not candidates:
            raise ValueError("no valid historical prediction is available for the current step")

        weights = np.exp(-self.decay * np.asarray(ages, dtype=np.float64))
        stacked = np.stack(candidates).astype(np.float64)
        action = np.average(stacked, axis=0, weights=weights).astype(np.float32)
        self._step += 1
        return action

    def add(self, chunk: ActionChunk) -> Float32Array:
        """Compatibility synonym for :meth:`update`."""

        return self.update(chunk)

    def __call__(self, chunk: ActionChunk) -> Float32Array:
        return self.update(chunk)
