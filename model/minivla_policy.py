from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class MiniVLAPolicy:
    """A tiny linear VLA policy for smoke tests and repository scaffolding.

    The model predicts an action chunk from observation + instruction features.
    It is intentionally small so the GitHub repo can be verified without a robot
    simulator or a GPU.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 2,
        chunk_size: int = 1,
        seed: int = 42,
        weights: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.output_dim = action_dim * chunk_size
        rng = np.random.default_rng(seed)
        self.weights = (
            weights.astype(np.float32)
            if weights is not None
            else rng.normal(0.0, 0.02, size=(input_dim, self.output_dim)).astype(np.float32)
        )
        self.bias = (
            bias.astype(np.float32)
            if bias is not None
            else np.zeros(self.output_dim, dtype=np.float32)
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        return inputs @ self.weights + self.bias

    def train_step(self, inputs: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
        predictions = self.predict(inputs)
        error = predictions - targets
        loss = float(np.mean(error**2))
        grad = (2.0 / len(inputs)) * error
        grad_w = inputs.T @ grad
        grad_b = np.mean(grad, axis=0)
        self.weights -= learning_rate * grad_w.astype(np.float32)
        self.bias -= learning_rate * grad_b.astype(np.float32)
        return loss

    def save(self, path: str | Path, metadata: dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "metadata": metadata,
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> tuple["MiniVLAPolicy", dict[str, Any]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = cls(
            input_dim=int(payload["input_dim"]),
            action_dim=int(payload["action_dim"]),
            chunk_size=int(payload["chunk_size"]),
            weights=np.asarray(payload["weights"], dtype=np.float32),
            bias=np.asarray(payload["bias"], dtype=np.float32),
        )
        return policy, payload.get("metadata", {})
