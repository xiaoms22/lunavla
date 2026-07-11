from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import numpy.typing as npt

from dataset import instruction_features
from model import NumpyBCMLPPolicy, NumpyLinearChunkPolicy, load_policy
from model.policy_base import ActionChunk, MiniVLAPolicyBase

from .contracts import Observation, PolicyBatch, normalize_device

if TYPE_CHECKING:
    from .registry import PolicyRegistry


Float32Array = npt.NDArray[np.float32]
_NUMPY_POLICY_TYPES = (NumpyLinearChunkPolicy, NumpyBCMLPPolicy)
_COMMON_CONFIG_KEYS = {
    "state_dim",
    "instruction_dim",
    "action_dim",
    "chunk_size",
    "seed",
    "device",
}


def _positive_int(config: Mapping[str, Any], key: str, default: int | None = None) -> int:
    if key not in config:
        if default is None:
            raise ValueError(f"{key} is required")
        value = default
    else:
        value = int(config[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _adapter_config(config: Mapping[str, Any], *, bc: bool) -> dict[str, Any]:
    allowed = _COMMON_CONFIG_KEYS | ({"hidden_dim"} if bc else set())
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError("unknown NumPy policy config field(s): " + ", ".join(unknown))
    state_dim = _positive_int(config, "state_dim")
    instruction_dim = int(config.get("instruction_dim", 0))
    if instruction_dim < 0:
        raise ValueError("instruction_dim cannot be negative")
    result: dict[str, Any] = {
        "state_dim": state_dim,
        "instruction_dim": instruction_dim,
        "action_dim": _positive_int(config, "action_dim", 2),
        "chunk_size": _positive_int(config, "chunk_size", 1),
        "seed": int(config.get("seed", 0)),
        "device": normalize_device(str(config.get("device", "cpu"))),
    }
    if bc:
        result["hidden_dim"] = _positive_int(config, "hidden_dim", 32)
    if result["device"] != "cpu":
        raise ValueError("v1 NumPy policy adapters support only device='cpu'")
    return result


class NumpyPolicyAdapter:
    """Expose a v1 NumPy policy through the modality-preserving v2 protocol."""

    def __init__(
        self,
        policy: MiniVLAPolicyBase,
        *,
        state_dim: int,
        instruction_dim: int = 0,
        device: str = "cpu",
    ) -> None:
        if not isinstance(policy, _NUMPY_POLICY_TYPES):
            raise TypeError("NumpyPolicyAdapter requires a supported v1 NumPy policy")
        if state_dim <= 0 or instruction_dim < 0:
            raise ValueError("state_dim must be positive and instruction_dim non-negative")
        normalized_device = normalize_device(device)
        if normalized_device != "cpu":
            raise ValueError("v1 NumPy policy adapters support only device='cpu'")
        expected_input = int(state_dim) + int(instruction_dim)
        if int(policy.input_dim) != expected_input:
            raise ValueError(
                f"wrapped policy input_dim={policy.input_dim}, expected "
                f"state_dim + instruction_dim = {expected_input}"
            )

        self.policy = policy
        self.policy_id = str(policy.policy_name)
        self.device = normalized_device
        self.state_dim = int(state_dim)
        self.instruction_dim = int(instruction_dim)
        self.action_dim = int(policy.action_dim)
        self.chunk_size = int(policy.chunk_size)

    def _encode(self, observation: Observation) -> Float32Array:
        if not isinstance(observation, Observation):
            raise TypeError("policy input must be an Observation")
        if observation.image is not None:
            raise ValueError(
                f"{self.policy_id} is state-only and cannot consume Observation.image"
            )
        if observation.state.shape != (self.state_dim,):
            raise ValueError(
                f"observation state must have shape {(self.state_dim,)}; "
                f"got {observation.state.shape}"
            )
        instruction = instruction_features(observation.instruction, self.instruction_dim)
        return np.concatenate([observation.state, instruction]).astype(np.float32, copy=False)

    def train_batch(self, batch: PolicyBatch, *, learning_rate: float) -> float:
        if not isinstance(batch, PolicyBatch):
            raise TypeError("batch must be a PolicyBatch")
        if batch.device != self.device:
            raise ValueError(
                f"batch device {batch.device!r} does not match policy device {self.device!r}"
            )
        expected = (self.chunk_size, self.action_dim)
        if batch.targets.shape[1:] != expected:
            raise ValueError(
                f"batch targets must have trailing shape {expected}; got {batch.targets.shape[1:]}"
            )
        inputs = np.stack([self._encode(item) for item in batch.observations])
        return float(
            self.policy.train_step(
                inputs,
                batch.targets,
                float(learning_rate),
                valid_mask=batch.valid_mask,
            )
        )

    def predict_chunk(self, observation: Observation) -> ActionChunk:
        chunk = self.policy.predict_chunk(self._encode(observation))
        if chunk.values.shape != (self.chunk_size, self.action_dim):
            raise ValueError(
                "wrapped policy returned an invalid ActionChunk shape: "
                f"{chunk.values.shape}"
            )
        return chunk

    def save_checkpoint(
        self,
        path: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        target = Path(path)
        if target.suffix and target.suffix != ".json":
            raise ValueError("v2 NumPy checkpoints must use a .json file")
        payload = dict(metadata or {})
        payload["v2_adapter"] = {
            "policy_id": self.policy_id,
            "state_dim": self.state_dim,
            "instruction_dim": self.instruction_dim,
            "device": self.device,
        }
        return self.policy.save_pretrained(target, metadata=payload)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        config: Mapping[str, Any] | None = None,
    ) -> "NumpyPolicyAdapter":
        policy, metadata = load_policy(path)
        if not isinstance(policy, _NUMPY_POLICY_TYPES):
            raise TypeError("checkpoint does not contain a supported NumPy policy")
        supplied = dict(config or {})
        unknown = sorted(set(supplied) - {"state_dim", "instruction_dim", "device"})
        if unknown:
            raise ValueError("unknown NumPy checkpoint config field(s): " + ", ".join(unknown))

        adapter = metadata.get("v2_adapter", {})
        if not isinstance(adapter, Mapping):
            adapter = {}
        policy_metadata = metadata.get("policy", {})
        if not isinstance(policy_metadata, Mapping):
            policy_metadata = {}
        instruction_dim = int(
            supplied.get(
                "instruction_dim",
                adapter.get("instruction_dim", policy_metadata.get("instruction_dim", 0)),
            )
        )
        state_dim = int(
            supplied.get(
                "state_dim",
                adapter.get(
                    "state_dim",
                    policy_metadata.get("observation_dim", int(policy.input_dim) - instruction_dim),
                ),
            )
        )
        device = str(supplied.get("device", adapter.get("device", "cpu")))
        return cls(
            policy,
            state_dim=state_dim,
            instruction_dim=instruction_dim,
            device=device,
        )


def create_numpy_linear_chunk(config: Mapping[str, Any]) -> NumpyPolicyAdapter:
    parsed = _adapter_config(config, bc=False)
    policy = NumpyLinearChunkPolicy(
        input_dim=parsed["state_dim"] + parsed["instruction_dim"],
        action_dim=parsed["action_dim"],
        chunk_size=parsed["chunk_size"],
        seed=parsed["seed"],
    )
    return NumpyPolicyAdapter(
        policy,
        state_dim=parsed["state_dim"],
        instruction_dim=parsed["instruction_dim"],
        device=parsed["device"],
    )


def create_numpy_bc_mlp(config: Mapping[str, Any]) -> NumpyPolicyAdapter:
    parsed = _adapter_config(config, bc=True)
    policy = NumpyBCMLPPolicy(
        input_dim=parsed["state_dim"] + parsed["instruction_dim"],
        action_dim=parsed["action_dim"],
        chunk_size=parsed["chunk_size"],
        hidden_dim=parsed["hidden_dim"],
        seed=parsed["seed"],
    )
    return NumpyPolicyAdapter(
        policy,
        state_dim=parsed["state_dim"],
        instruction_dim=parsed["instruction_dim"],
        device=parsed["device"],
    )


def _load_numpy(path: Path, config: Mapping[str, Any]) -> NumpyPolicyAdapter:
    return NumpyPolicyAdapter.from_checkpoint(path, config)


def register_numpy_policies(registry: "PolicyRegistry") -> None:
    registry.register(
        "numpy_linear_chunk",
        create_numpy_linear_chunk,
        loader=_load_numpy,
        aliases=("numpy_linear", "linear_chunk", "tiny_linear", "linear_smoke"),
    )
    registry.register(
        "numpy_bc_mlp",
        create_numpy_bc_mlp,
        loader=_load_numpy,
        aliases=("numpy_bc", "bc", "bc_mlp"),
    )
