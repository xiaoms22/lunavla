from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from lunavla.artifact_contracts import (
    NUMPY_CHECKPOINT_FORMAT as CHECKPOINT_FORMAT,
)
from lunavla.artifact_contracts import (
    NUMPY_CHECKPOINT_SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION,
)

from .checkpoint_contract import (
    load_json_object,
    safe_metadata,
    validate_legacy_checkpoint,
    validate_versioned_checkpoint,
)
from .losses import masked_mse, masked_mse_gradient
from .policy_base import ActionChunk, MiniVLAPolicyBase, PolicyBatch, PolicySample


Array = npt.NDArray[np.generic]
Float32Array = npt.NDArray[np.float32]


class NumpyLinearChunkPolicy(MiniVLAPolicyBase):
    """Small CPU-only linear action-chunk policy implemented with NumPy."""

    policy_name = "numpy_linear_chunk"

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 2,
        chunk_size: int = 1,
        seed: int = 42,
        weights: Array | None = None,
        bias: Array | None = None,
    ) -> None:
        if input_dim <= 0 or action_dim <= 0 or chunk_size <= 0:
            raise ValueError("input_dim, action_dim, and chunk_size must be positive")
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.output_dim = self.action_dim * self.chunk_size
        rng = np.random.default_rng(seed)
        self.weights = (
            np.asarray(weights, dtype=np.float32)
            if weights is not None
            else rng.normal(0.0, 0.02, size=(self.input_dim, self.output_dim)).astype(np.float32)
        )
        self.bias = (
            np.asarray(bias, dtype=np.float32)
            if bias is not None
            else np.zeros(self.output_dim, dtype=np.float32)
        )
        if self.weights.shape != (self.input_dim, self.output_dim):
            raise ValueError(
                f"weights must have shape {(self.input_dim, self.output_dim)}; "
                f"got {self.weights.shape}"
            )
        if self.bias.shape != (self.output_dim,):
            raise ValueError(f"bias must have shape {(self.output_dim,)}; got {self.bias.shape}")
        if not np.all(np.isfinite(self.weights)) or not np.all(np.isfinite(self.bias)):
            raise ValueError("policy parameters contain NaN or infinite values")

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

    def predict(self, inputs: Array) -> Float32Array:
        """Compatibility helper returning flattened chunks for a batch."""

        inputs_array = self._inputs(inputs)
        return (inputs_array @ self.weights + self.bias).astype(np.float32)

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
        return ActionChunk(
            values=prediction[0],
            valid_mask=np.ones(self.chunk_size, dtype=bool),
        )

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
        predictions = self._prediction_tensor(inputs_array)
        loss = masked_mse(predictions, targets_array, valid_mask)
        grad_predictions = masked_mse_gradient(predictions, targets_array, valid_mask)
        grad_flat = grad_predictions.reshape(len(inputs_array), self.output_dim)
        grad_w = inputs_array.T @ grad_flat
        grad_b = np.sum(grad_flat, axis=0)
        self.weights -= float(learning_rate) * grad_w.astype(np.float32)
        self.bias -= float(learning_rate) * grad_b.astype(np.float32)
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
                "parameters": {
                    "weights": self.weights.tolist(),
                    "bias": self.bias.tolist(),
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
    def load(cls, path: str | Path) -> tuple["NumpyLinearChunkPolicy", dict[str, Any]]:
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
                weights=np.asarray(parameters["weights"], dtype=np.float32),
                bias=np.asarray(parameters["bias"], dtype=np.float32),
            )
            return policy, metadata

        # Read-only compatibility for JSON payloads historically named checkpoint.pt.
        parameters, metadata = validate_legacy_checkpoint(
            payload,
            policy_name=cls.policy_name,
            accepted_names=frozenset(
                {cls.policy_name, "act", "tiny_linear", "linear_smoke"}
            ),
        )
        policy = cls(
            input_dim=payload["input_dim"],
            action_dim=payload["action_dim"],
            chunk_size=payload["chunk_size"],
            weights=np.asarray(parameters["weights"], dtype=np.float32),
            bias=np.asarray(parameters["bias"], dtype=np.float32),
        )
        return policy, metadata

    @classmethod
    def from_pretrained(
        cls, run_dir: str | Path
    ) -> tuple["NumpyLinearChunkPolicy", dict[str, Any]]:
        run_path = Path(run_dir)
        if run_path.suffix:
            checkpoint_path = run_path
        elif (run_path / "checkpoint.json").exists():
            checkpoint_path = run_path / "checkpoint.json"
        else:
            checkpoint_path = run_path / "checkpoint.pt"
        return cls.load(checkpoint_path)


# Source compatibility for imports used by the v1.0 lessons.
MiniVLAPolicy = NumpyLinearChunkPolicy
