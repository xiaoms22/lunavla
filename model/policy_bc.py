from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .policy_base import MiniVLAPolicyBase, PolicyBatch, PolicySample


class BCMLPPolicy(MiniVLAPolicyBase):
    """A tiny NumPy MLP behavior cloning policy."""

    policy_name = "bc_mlp"

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 2,
        chunk_size: int = 1,
        hidden_dim: int = 32,
        seed: int = 42,
        weights: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.output_dim = action_dim * chunk_size
        rng = np.random.default_rng(seed)
        if weights is None:
            init_scale = 0.02
            self.w1 = rng.normal(0.0, init_scale, size=(input_dim, hidden_dim)).astype(np.float32)
            self.b1 = np.zeros(hidden_dim, dtype=np.float32)
            self.w2 = rng.normal(0.0, init_scale, size=(hidden_dim, self.output_dim)).astype(np.float32)
            self.b2 = np.zeros(self.output_dim, dtype=np.float32)
        else:
            self.w1 = weights["w1"].astype(np.float32)
            self.b1 = weights["b1"].astype(np.float32)
            self.w2 = weights["w2"].astype(np.float32)
            self.b2 = weights["b2"].astype(np.float32)

    def _forward_arrays(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(inputs @ self.w1 + self.b1)
        predictions = hidden @ self.w2 + self.b2
        return hidden, predictions

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        _, predictions = self._forward_arrays(inputs.astype(np.float32))
        return predictions

    def forward(self, batch: PolicyBatch) -> dict[str, float]:
        inputs = np.asarray(batch["inputs"], dtype=np.float32)
        targets = np.asarray(batch["targets"], dtype=np.float32)
        _, predictions = self._forward_arrays(inputs)
        loss = float(np.mean((predictions - targets) ** 2))
        return {"loss": loss, "mse_loss": loss}

    def predict_action(self, sample: PolicySample) -> np.ndarray:
        if isinstance(sample, dict):
            inputs = np.asarray(sample["inputs"], dtype=np.float32)
        else:
            inputs = np.asarray(sample, dtype=np.float32)
        return self.predict(inputs)[0]

    def train_step(self, inputs: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
        inputs = inputs.astype(np.float32)
        targets = targets.astype(np.float32)
        hidden, predictions = self._forward_arrays(inputs)
        error = predictions - targets
        loss = float(np.mean(error**2))

        grad_predictions = (2.0 / max(error.size, 1)) * error
        grad_w2 = hidden.T @ grad_predictions
        grad_b2 = np.sum(grad_predictions, axis=0)
        grad_hidden = grad_predictions @ self.w2.T
        grad_hidden_pre = grad_hidden * (1.0 - hidden**2)
        grad_w1 = inputs.T @ grad_hidden_pre
        grad_b1 = np.sum(grad_hidden_pre, axis=0)

        self.w1 -= learning_rate * grad_w1.astype(np.float32)
        self.b1 -= learning_rate * grad_b1.astype(np.float32)
        self.w2 -= learning_rate * grad_w2.astype(np.float32)
        self.b2 -= learning_rate * grad_b2.astype(np.float32)
        return loss

    def save(self, path: str | Path, metadata: dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_name": self.policy_name,
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "hidden_dim": self.hidden_dim,
            "weights": {
                "w1": self.w1.tolist(),
                "b1": self.b1.tolist(),
                "w2": self.w2.tolist(),
                "b2": self.b2.tolist(),
            },
            "metadata": metadata,
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_pretrained(self, run_dir: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        run_path = Path(run_dir)
        checkpoint_path = run_path if run_path.suffix else run_path / "checkpoint.pt"
        self.save(checkpoint_path, metadata=metadata or {})
        return checkpoint_path

    @classmethod
    def load(cls, path: str | Path) -> tuple["BCMLPPolicy", dict[str, Any]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in payload["weights"].items()
        }
        policy = cls(
            input_dim=int(payload["input_dim"]),
            action_dim=int(payload["action_dim"]),
            chunk_size=int(payload["chunk_size"]),
            hidden_dim=int(payload["hidden_dim"]),
            weights=weights,
        )
        return policy, payload.get("metadata", {})

    @classmethod
    def from_pretrained(cls, run_dir: str | Path) -> tuple["BCMLPPolicy", dict[str, Any]]:
        run_path = Path(run_dir)
        checkpoint_path = run_path if run_path.suffix else run_path / "checkpoint.pt"
        return cls.load(checkpoint_path)
