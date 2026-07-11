from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .checkpoint_contract import (
    load_json_object,
    safe_metadata,
    validate_legacy_checkpoint,
    validate_versioned_checkpoint,
)
from .losses import masked_mse, masked_mse_gradient
from .minivla_policy import CHECKPOINT_FORMAT, CHECKPOINT_SCHEMA_VERSION
from .policy_base import ActionChunk, MiniVLAPolicyBase, PolicyBatch, PolicySample


Array = npt.NDArray[np.generic]
Float32Array = npt.NDArray[np.float32]


class NumpyBCMLPPolicy(MiniVLAPolicyBase):
    """A small CPU-only NumPy MLP behavior-cloning policy."""

    policy_name = "numpy_bc_mlp"

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 2,
        chunk_size: int = 1,
        hidden_dim: int = 32,
        seed: int = 42,
        weights: dict[str, Array] | None = None,
    ) -> None:
        if min(input_dim, action_dim, chunk_size, hidden_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = self.action_dim * self.chunk_size
        self.w1: Float32Array
        self.b1: Float32Array
        self.w2: Float32Array
        self.b2: Float32Array
        rng = np.random.default_rng(seed)
        if weights is None:
            init_scale = 0.02
            self.w1 = rng.normal(0.0, init_scale, size=(input_dim, hidden_dim)).astype(np.float32)
            self.b1 = np.zeros(hidden_dim, dtype=np.float32)
            self.w2 = rng.normal(0.0, init_scale, size=(hidden_dim, self.output_dim)).astype(np.float32)
            self.b2 = np.zeros(self.output_dim, dtype=np.float32)
        else:
            self.w1 = np.asarray(weights["w1"], dtype=np.float32)
            self.b1 = np.asarray(weights["b1"], dtype=np.float32)
            self.w2 = np.asarray(weights["w2"], dtype=np.float32)
            self.b2 = np.asarray(weights["b2"], dtype=np.float32)
        expected_shapes = {
            "w1": (self.input_dim, self.hidden_dim),
            "b1": (self.hidden_dim,),
            "w2": (self.hidden_dim, self.output_dim),
            "b2": (self.output_dim,),
        }
        for name, expected in expected_shapes.items():
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}; got {value.shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or infinite values")

    def _inputs(self, inputs: Array) -> Float32Array:
        raw = np.asarray(inputs)
        if raw.dtype.kind not in "fiu":
            raise TypeError("inputs must have a numeric dtype")
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.ndim != 2 or raw.shape[1] != self.input_dim:
            raise ValueError(f"inputs must have shape [batch, {self.input_dim}]; got {raw.shape}")
        result = raw.astype(np.float32, copy=False)
        if not np.all(np.isfinite(result)):
            raise ValueError("inputs contain NaN or infinite values")
        return result

    def _targets(self, targets: Array, batch_size: int) -> Float32Array:
        raw = np.asarray(targets)
        if raw.dtype.kind not in "fiu":
            raise TypeError("targets must have a numeric dtype")
        if raw.shape == (batch_size, self.output_dim):
            raw = raw.reshape(batch_size, self.chunk_size, self.action_dim)
        expected = (batch_size, self.chunk_size, self.action_dim)
        if raw.shape != expected:
            raise ValueError(f"targets must have shape {expected}; got {raw.shape}")
        result = raw.astype(np.float32, copy=False)
        if not np.all(np.isfinite(result)):
            raise ValueError("targets contain NaN or infinite values")
        return result

    def _forward_arrays(self, inputs: Float32Array) -> tuple[Float32Array, Float32Array]:
        hidden = np.tanh(inputs @ self.w1 + self.b1)
        predictions = hidden @ self.w2 + self.b2
        return hidden, predictions

    def predict(self, inputs: Array) -> Float32Array:
        inputs_array = self._inputs(inputs)
        _, predictions = self._forward_arrays(inputs_array)
        return predictions.astype(np.float32)

    def _prediction_tensor(self, inputs: Array) -> Float32Array:
        flat = self.predict(inputs)
        return flat.reshape(len(flat), self.chunk_size, self.action_dim)

    def forward(self, batch: PolicyBatch) -> dict[str, float]:
        inputs = self._inputs(batch["inputs"])
        targets = self._targets(batch["targets"], len(inputs))
        predictions = self._prediction_tensor(inputs)
        loss = masked_mse(predictions, targets, batch.get("valid_mask"))
        return {"loss": loss, "mse_loss": loss}

    def predict_chunk(self, sample: PolicySample) -> ActionChunk:
        inputs = np.asarray(sample["inputs"] if isinstance(sample, dict) else sample)
        prediction = self._prediction_tensor(inputs)
        if len(prediction) != 1:
            raise ValueError("predict_chunk accepts exactly one sample")
        return ActionChunk(prediction[0], np.ones(self.chunk_size, dtype=bool))

    def train_step(
        self,
        inputs: Array,
        targets: Array,
        learning_rate: float,
        valid_mask: Array | None = None,
    ) -> float:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite value")
        inputs_array = self._inputs(inputs)
        targets_array = self._targets(targets, len(inputs_array))
        hidden, flat_predictions = self._forward_arrays(inputs_array)
        predictions = flat_predictions.reshape(len(inputs_array), self.chunk_size, self.action_dim)
        loss = masked_mse(predictions, targets_array, valid_mask)
        grad_predictions = masked_mse_gradient(predictions, targets_array, valid_mask)
        grad_flat = grad_predictions.reshape(len(inputs_array), self.output_dim)
        grad_w2 = hidden.T @ grad_flat
        grad_b2 = np.sum(grad_flat, axis=0)
        grad_hidden = grad_flat @ self.w2.T
        grad_hidden_pre = grad_hidden * (1.0 - hidden**2)
        grad_w1 = inputs_array.T @ grad_hidden_pre
        grad_b1 = np.sum(grad_hidden_pre, axis=0)
        self.w1 -= float(learning_rate) * grad_w1.astype(np.float32)
        self.b1 -= float(learning_rate) * grad_b1.astype(np.float32)
        self.w2 -= float(learning_rate) * grad_w2.astype(np.float32)
        self.b2 -= float(learning_rate) * grad_b2.astype(np.float32)
        return loss

    def _checkpoint_payload(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "format": CHECKPOINT_FORMAT,
            "policy": {
                "type": self.policy_name,
                "input_dim": self.input_dim,
                "action_dim": self.action_dim,
                "chunk_size": self.chunk_size,
                "hidden_dim": self.hidden_dim,
                "parameters": {
                    "w1": self.w1.tolist(),
                    "b1": self.b1.tolist(),
                    "w2": self.w2.tolist(),
                    "b2": self.b2.tolist(),
                },
            },
            "metadata": safe_metadata(metadata),
        }

    def save(self, path: str | Path, metadata: dict[str, Any]) -> None:
        target = Path(path)
        if target.suffix != ".json":
            raise ValueError("new checkpoints must use the checkpoint.json format")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                self._checkpoint_payload(metadata),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    def save_pretrained(
        self, run_dir: str | Path, metadata: dict[str, Any] | None = None
    ) -> Path:
        run_path = Path(run_dir)
        checkpoint_path = run_path if run_path.suffix else run_path / "checkpoint.json"
        self.save(checkpoint_path, metadata=metadata or {})
        return checkpoint_path

    @classmethod
    def load(cls, path: str | Path) -> tuple["NumpyBCMLPPolicy", dict[str, Any]]:
        payload = load_json_object(path)
        if "schema_version" in payload:
            policy_payload, parameters, metadata = validate_versioned_checkpoint(
                payload,
                policy_name=cls.policy_name,
            )
            policy = cls(
                input_dim=policy_payload["input_dim"],
                action_dim=policy_payload["action_dim"],
                chunk_size=policy_payload["chunk_size"],
                hidden_dim=policy_payload["hidden_dim"],
                weights={
                    key: np.asarray(value, dtype=np.float32)
                    for key, value in parameters.items()
                },
            )
            return policy, metadata

        parameters, metadata = validate_legacy_checkpoint(
            payload,
            policy_name=cls.policy_name,
            accepted_names=frozenset({cls.policy_name, "bc", "bc_mlp"}),
        )
        policy = cls(
            input_dim=payload["input_dim"],
            action_dim=payload["action_dim"],
            chunk_size=payload["chunk_size"],
            hidden_dim=payload["hidden_dim"],
            weights={
                key: np.asarray(value, dtype=np.float32)
                for key, value in parameters.items()
            },
        )
        return policy, metadata

    @classmethod
    def from_pretrained(
        cls, run_dir: str | Path
    ) -> tuple["NumpyBCMLPPolicy", dict[str, Any]]:
        run_path = Path(run_dir)
        if run_path.suffix:
            checkpoint_path = run_path
        elif (run_path / "checkpoint.json").exists():
            checkpoint_path = run_path / "checkpoint.json"
        else:
            checkpoint_path = run_path / "checkpoint.pt"
        return cls.load(checkpoint_path)


BCMLPPolicy = NumpyBCMLPPolicy
