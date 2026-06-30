from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import numpy as np


PolicyBatch = dict[str, np.ndarray]
PolicySample = Union[dict[str, Any], np.ndarray]


class MiniVLAPolicyBase(ABC):
    """Minimal policy contract shared by training and rollout evaluation."""

    policy_name = "base"

    @abstractmethod
    def forward(self, batch: PolicyBatch) -> dict[str, float]:
        """Return loss values for a training batch without changing weights."""

    @abstractmethod
    def predict_action(self, sample: PolicySample) -> np.ndarray:
        """Return one predicted action chunk for a single sample."""

    @abstractmethod
    def save_pretrained(self, run_dir: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Save enough state for later policy-agnostic evaluation."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, run_dir: str | Path) -> tuple["MiniVLAPolicyBase", dict[str, Any]]:
        """Load a policy and its metadata from a saved run directory."""
