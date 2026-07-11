from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch
from model.policy_base import ActionChunk

from .config import ExperimentConfig
from .normalization import NormalizationStatsV1
from .policy import (
    ModelSourceContractV1,
    PolicyBatchV3,
    PolicySampleV3,
    PolicySpecV3,
    TrainStepResultV3,
    VLAPolicyV3,
)
from .registry import PolicyFactoryV3, PolicyRegistryV3, PolicyRestorerV3


POLICY_ID = "lerobot_smolvla"
MODEL_REPO_ID = "lerobot/smolvla_base"
MODEL_REVISION = "d06fce6e38c25c04ac5a6319eefb9fae0e257cb2"
MODEL_SHA256 = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"


class SmolVLAPublicPolicy(Protocol):
    def reset(self) -> None: ...

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Mapping[str, Any]]: ...

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor: ...

    def save_pretrained(
        self, save_directory: str | Path, *, push_to_hub: bool = False
    ) -> object: ...


class SmolVLAPublicProcessor(Protocol):
    def __call__(self, value: Any) -> Any: ...

    def reset(self) -> None: ...

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        config_filename: str | None = None,
        push_to_hub: bool = False,
    ) -> object: ...


def smolvla_policy_spec(config: ExperimentConfig) -> PolicySpecV3:
    if config.policy["type"] != POLICY_ID:
        raise ValueError("smolvla_policy_spec requires policy.type=lerobot_smolvla")
    parameters = config.policy["parameters"]
    state_feature = str(parameters["state_feature"])
    cameras = tuple(str(item) for item in parameters["camera_features"])
    return PolicySpecV3(
        policy_id=POLICY_ID,
        backend="lerobot_smolvla_adapter",
        model_source=ModelSourceContractV1(
            repo_id=str(parameters["repo_id"]),
            revision=str(parameters["revision"]),
            file_hashes=dict(parameters["file_hashes"]),
            license_status=str(parameters["license_status"]),
            pretrained_enabled=bool(parameters["pretrained_enabled"]),
        ),
        required_modalities=("image", "state", "instruction"),
        camera_order=cameras,
        state_order=(state_feature,),
        history=int(parameters["history"]),
        chunk_size=int(parameters["chunk_size"]),
        horizon=int(parameters["horizon"]),
        execution_steps=int(parameters["execution_steps"]),
        normalization={
            item.name: item.normalization
            for item in config.feature_schema.features
            if item.name == state_feature or item.role == "action"
        },
        device=config.training["device"],
        deterministic=False,
    )


def _camera_key(name: str) -> str:
    return f"observation.images.{name}"


class SmolVLAAdapterV3:
    """Public-API-only adapter; Alpha 2 intentionally performs no optimizer step."""

    def __init__(
        self,
        policy: SmolVLAPublicPolicy,
        preprocessor: SmolVLAPublicProcessor,
        postprocessor: SmolVLAPublicProcessor,
        *,
        spec: PolicySpecV3,
        normalization: NormalizationStatsV1,
    ) -> None:
        for name in ("reset", "forward", "predict_action_chunk", "save_pretrained"):
            if not callable(getattr(policy, name, None)):
                raise TypeError(f"SmolVLA public policy is missing {name}()")
        for processor in (preprocessor, postprocessor):
            for name in ("__call__", "reset", "save_pretrained"):
                if not callable(getattr(processor, name, None)):
                    raise TypeError(f"SmolVLA public processor is missing {name}()")
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.spec = spec
        self.normalization = normalization

    def reset(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("policy reset seed must be a non-negative integer")
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def _raw(self, sample: PolicySampleV3, *, include_action: bool) -> dict[str, Any]:
        observation = sample.observation
        if tuple(observation.images) != self.spec.camera_order:
            raise ValueError("SmolVLA observation camera order does not match PolicySpecV3")
        if not observation.instruction:
            raise ValueError("SmolVLA requires a non-empty instruction")
        raw: dict[str, Any] = {
            "observation.state": torch.from_numpy(
                np.array(observation.state[self.spec.state_order[0]], copy=True)
            ).to(torch.float32),
            "task": observation.instruction,
        }
        for name in self.spec.camera_order:
            image = np.array(observation.images[name], copy=True)
            raw[_camera_key(name)] = torch.from_numpy(
                np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
            )
        if include_action:
            if sample.action_chunk is None or sample.valid_mask is None:
                raise ValueError("SmolVLA conformance samples require action supervision")
            raw["action"] = torch.from_numpy(
                np.array(sample.action_chunk, dtype=np.float32, copy=True)
            )
            raw["action_is_pad"] = torch.from_numpy(
                ~np.array(sample.valid_mask, dtype=bool, copy=True)
            )
        return raw

    def _processed(
        self, sample: PolicySampleV3, *, include_action: bool
    ) -> dict[str, torch.Tensor]:
        value = self.preprocessor(self._raw(sample, include_action=include_action))
        if not isinstance(value, Mapping):
            raise TypeError("SmolVLA preprocessor must return a mapping")
        tensors = {name: item for name, item in value.items() if isinstance(item, torch.Tensor)}
        if not tensors:
            raise ValueError("SmolVLA preprocessor returned no tensors")
        return tensors

    def _batch(self, batch: PolicyBatchV3) -> dict[str, torch.Tensor]:
        processed = [self._processed(sample, include_action=True) for sample in batch.samples]
        keys = tuple(processed[0])
        if any(tuple(item) != keys for item in processed):
            raise ValueError("SmolVLA processed batch keys are inconsistent")
        return {name: torch.cat([item[name] for item in processed], dim=0) for name in keys}

    def train_step(
        self, batch: PolicyBatchV3, *, learning_rate: float, step: int
    ) -> TrainStepResultV3:
        started = time.perf_counter()
        with torch.no_grad():
            loss, components = self.policy.forward(self._batch(batch))
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not torch.isfinite(loss):
            raise FloatingPointError("SmolVLA public forward returned a non-finite scalar loss")
        value = float(loss.detach().cpu())
        normalized_components = {
            str(name): float(item)
            for name, item in components.items()
            if isinstance(item, (int, float)) and math.isfinite(float(item))
        }
        normalized_components["total"] = value
        return TrainStepResultV3(
            value,
            normalized_components,
            None,
            learning_rate,
            step,
            True,
            {"conformance_forward": (time.perf_counter() - started) * 1000.0},
        )

    def predict_chunk(self, sample: PolicySampleV3) -> ActionChunk:
        processed = self._processed(sample, include_action=False)
        values = self.policy.predict_action_chunk(processed)
        if not isinstance(values, torch.Tensor) or values.ndim != 3 or values.shape[0] != 1:
            raise ValueError("SmolVLA public predict_action_chunk must return [1, chunk, action]")
        action = self.postprocessor(values[0])
        if not isinstance(action, torch.Tensor):
            raise TypeError("SmolVLA postprocessor must return a tensor")
        array = action.detach().cpu().numpy().astype(np.float32)
        return ActionChunk(array, np.ones(array.shape[0], dtype=bool))

    def save_checkpoint(self, path: Path, *, metadata: Mapping[str, Any]) -> Path:
        path.mkdir(parents=True, exist_ok=False)
        self.policy.save_pretrained(path / "model", push_to_hub=False)
        self.preprocessor.save_pretrained(
            path / "processors", config_filename="preprocessor.json", push_to_hub=False
        )
        self.postprocessor.save_pretrained(
            path / "processors", config_filename="postprocessor.json", push_to_hub=False
        )
        marker = path / "adapter_contract.json"
        marker.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "policy_id": POLICY_ID,
                    "policy_spec_sha256": self.spec.sha256(),
                    "normalization_sha256": self.normalization.sha256(),
                    "license_status": self.spec.model_source.license_status,
                    "pretrained_enabled": self.spec.model_source.pretrained_enabled,
                    "optimizer_step_verified": False,
                    "metadata": dict(metadata),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return marker


def blocked_smolvla_factory(
    config: ExperimentConfig, spec: PolicySpecV3, normalization: NormalizationStatsV1
) -> VLAPolicyV3:
    del config, spec, normalization
    raise RuntimeError(
        "SmolVLA pretrained and optimizer gates are closed; inject a public-API conformance fixture"
    )


def blocked_smolvla_restorer(
    checkpoint: Path,
    config: ExperimentConfig,
    spec: PolicySpecV3,
    normalization: NormalizationStatsV1,
) -> VLAPolicyV3:
    del checkpoint, config, spec, normalization
    raise RuntimeError("SmolVLA checkpoint restore is closed until the pretrained gate is verified")


def smolvla_conformance_factory(
    policy_factory: Callable[
        [ExperimentConfig, PolicySpecV3, NormalizationStatsV1], SmolVLAPublicPolicy
    ],
    processor_factory: Callable[
        [ExperimentConfig, PolicySpecV3, NormalizationStatsV1],
        tuple[SmolVLAPublicProcessor, SmolVLAPublicProcessor],
    ],
) -> PolicyFactoryV3:
    def create(
        config: ExperimentConfig, spec: PolicySpecV3, normalization: NormalizationStatsV1
    ) -> VLAPolicyV3:
        preprocessor, postprocessor = processor_factory(config, spec, normalization)
        return SmolVLAAdapterV3(
            policy_factory(config, spec, normalization),
            preprocessor,
            postprocessor,
            spec=spec,
            normalization=normalization,
        )

    return create


def register_smolvla_policy(
    registry: PolicyRegistryV3,
    *,
    factory: PolicyFactoryV3 = blocked_smolvla_factory,
    restorer: PolicyRestorerV3 = blocked_smolvla_restorer,
) -> None:
    registry.register(POLICY_ID, factory, restorer)
