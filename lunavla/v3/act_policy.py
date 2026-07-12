from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import torch

from lunavla.contracts import Observation, PolicyBatch
from lunavla.temporal import TemporalEnsembler
from lunavla.transformer_policy import TransformerChunkCVAEPolicy, TransformerPolicyConfig
from model.policy_base import ActionChunk

from .config import ExperimentConfig
from .normalization import NormalizationStatsV1
from .policy import (
    ModelSourceContractV1,
    PolicyBatchV3,
    PolicySampleV3,
    PolicySpecV3,
    TrainStepResultV3,
)
from .registry import PolicyRegistryV3


POLICY_ID = "act_v3"


def act_policy_spec(config: ExperimentConfig) -> PolicySpecV3:
    if config.policy["type"] != POLICY_ID:
        raise ValueError("act_policy_spec requires policy.type=act_v3")
    parameters = config.policy["parameters"]
    state_feature = str(parameters["state_feature"])
    camera_features = tuple(
        str(item) for item in parameters.get("camera_features", ())
    )
    if not camera_features and parameters["camera_feature"] is not None:
        camera_features = (str(parameters["camera_feature"]),)
    required = ["state"]
    if int(parameters["instruction_dim"]) > 0:
        required.append("instruction")
    if camera_features:
        required.insert(0, "image")
    return PolicySpecV3(
        policy_id=POLICY_ID,
        backend="torch_native",
        model_source=ModelSourceContractV1(
            "lunavla/native", "act_v3-alpha2", {}, "not_required", False
        ),
        required_modalities=tuple(required),
        camera_order=camera_features,
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
        deterministic=True,
    )


def _transformer_config(config: ExperimentConfig, spec: PolicySpecV3) -> TransformerPolicyConfig:
    parameters = config.policy["parameters"]
    state_feature = config.feature_schema.by_role("state")
    state_by_name = {item.name: item for item in state_feature}
    action_feature = config.feature_schema.by_role("action")[0]
    image_shape = None
    if spec.camera_order:
        image_by_name = {item.name: item for item in config.feature_schema.by_role("image")}
        shapes = [image_by_name[name].shape for name in spec.camera_order]
        if len({(shape[0], shape[2]) for shape in shapes}) != 1:
            raise ValueError("ACT multi-camera composition requires equal heights and channels")
        image_shape = (shapes[0][0], sum(shape[1] for shape in shapes), shapes[0][2])
    clip = config.training["gradient_clip_norm"]
    return TransformerPolicyConfig(
        state_dim=state_by_name[spec.state_order[0]].shape[0],
        action_dim=action_feature.shape[0],
        chunk_size=spec.chunk_size,
        d_model=int(parameters["d_model"]),
        nhead=int(parameters["nhead"]),
        num_encoder_layers=int(parameters["num_encoder_layers"]),
        num_decoder_layers=int(parameters["num_decoder_layers"]),
        dim_feedforward=int(parameters["dim_feedforward"]),
        latent_dim=int(parameters["latent_dim"]),
        instruction_dim=int(parameters["instruction_dim"]),
        image_shape=image_shape,
        dropout=float(parameters["dropout"]),
        kl_weight=float(parameters["kl_weight"]),
        max_grad_norm=10.0 if clip is None else float(clip),
        sample_latent_during_training=bool(parameters["sample_latent_during_training"]),
        seed=int(config.training["seed"]),
        device=spec.device,
    )


class ActPolicyV3:
    def __init__(
        self,
        policy: TransformerChunkCVAEPolicy,
        *,
        spec: PolicySpecV3,
        normalization: NormalizationStatsV1,
        state_feature: str,
        camera_features: tuple[str, ...],
        temporal_ensemble_decay: float | None,
    ) -> None:
        self.policy = policy
        self.spec = spec
        self.normalization = normalization
        self.state_feature = state_feature
        self.camera_features = camera_features
        self._ensembler = (
            None
            if temporal_ensemble_decay is None
            else TemporalEnsembler(
                decay=temporal_ensemble_decay,
                action_dim=policy.action_dim,
                chunk_size=policy.chunk_size,
            )
        )

    def reset(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("policy reset seed must be a non-negative integer")
        if self._ensembler is not None:
            self._ensembler.reset()

    def _observation(self, sample: PolicySampleV3) -> Observation:
        observation = sample.observation
        if self.state_feature not in observation.state:
            raise ValueError(f"missing ACT state feature {self.state_feature}")
        state = np.asarray(observation.state[self.state_feature])
        stats = self.normalization.features.get(self.state_feature)
        if stats is not None:
            state = stats.normalize(state)
        image = None
        if self.camera_features:
            if tuple(observation.images) != self.camera_features:
                raise ValueError("ACT observation camera order does not match PolicySpecV3")
            image = np.concatenate(
                [observation.images[name] for name in self.camera_features], axis=1
            )
        elif observation.images:
            raise ValueError("ACT cannot silently discard image features")
        return Observation(state, instruction=observation.instruction, image=image)

    def train_step(
        self, batch: PolicyBatchV3, *, learning_rate: float, step: int
    ) -> TrainStepResultV3:
        observations: list[Observation] = []
        targets: list[npt.NDArray[np.float32]] = []
        masks: list[npt.NDArray[np.bool_]] = []
        for sample in batch.samples:
            if sample.action_chunk is None or sample.valid_mask is None:
                raise ValueError("ACT training samples require action supervision")
            observations.append(self._observation(sample))
            targets.append(sample.action_chunk)
            masks.append(sample.valid_mask)
        started = time.perf_counter()
        loss = self.policy.train_batch(
            PolicyBatch(
                tuple(observations),
                np.stack(targets),
                np.stack(masks),
                device=batch.device,
            ),
            learning_rate=learning_rate,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        squared_norm = 0.0
        for parameter in self.policy.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("ACT produced a non-finite gradient")
                squared_norm += float(torch.sum(parameter.grad.detach() ** 2).cpu())
        gradient_norm = math.sqrt(squared_norm)
        if not math.isfinite(loss):
            raise FloatingPointError("ACT produced a non-finite loss")
        return TrainStepResultV3(
            loss,
            {"total": loss},
            gradient_norm,
            learning_rate,
            step,
            True,
            {"train_step": elapsed_ms},
        )

    def predict_chunk(self, sample: PolicySampleV3) -> ActionChunk:
        chunk = self.policy.predict_chunk(self._observation(sample))
        if self._ensembler is None:
            return chunk
        values = np.array(chunk.values, copy=True)
        values[0] = self._ensembler.update(chunk)
        return ActionChunk(values, chunk.valid_mask)

    def save_checkpoint(self, path: Path, *, metadata: Mapping[str, Any]) -> Path:
        merged = {
            **dict(metadata),
            "v3_policy_id": POLICY_ID,
            "policy_spec_sha256": self.spec.sha256(),
            "normalization_sha256": self.normalization.sha256(),
        }
        return self.policy.save_checkpoint(path, metadata=merged)


def _create(
    config: ExperimentConfig, spec: PolicySpecV3, normalization: NormalizationStatsV1
) -> ActPolicyV3:
    parameters = config.policy["parameters"]
    policy = TransformerChunkCVAEPolicy(_transformer_config(config, spec))
    return ActPolicyV3(
        policy,
        spec=spec,
        normalization=normalization,
        state_feature=str(parameters["state_feature"]),
        camera_features=spec.camera_order,
        temporal_ensemble_decay=parameters["temporal_ensemble_decay"],
    )


def _restore(
    checkpoint: Path,
    config: ExperimentConfig,
    spec: PolicySpecV3,
    normalization: NormalizationStatsV1,
) -> ActPolicyV3:
    target = checkpoint
    if checkpoint.is_dir():
        target = checkpoint / "policy" / str(config.artifacts["checkpoint_name"])
    policy = TransformerChunkCVAEPolicy.load_checkpoint(target, device=spec.device)
    if policy.config != _transformer_config(config, spec):
        raise ValueError("ACT checkpoint config does not match resolved ExperimentConfig")
    metadata = policy.checkpoint_metadata
    if metadata.get("v3_policy_id") != POLICY_ID:
        raise ValueError("v2 Transformer checkpoints cannot be relabeled as act_v3")
    if metadata.get("policy_spec_sha256") != spec.sha256():
        raise ValueError("ACT checkpoint PolicySpecV3 hash does not match")
    if metadata.get("normalization_sha256") != normalization.sha256():
        raise ValueError("ACT checkpoint normalization hash does not match")
    parameters = config.policy["parameters"]
    return ActPolicyV3(
        policy,
        spec=spec,
        normalization=normalization,
        state_feature=str(parameters["state_feature"]),
        camera_features=spec.camera_order,
        temporal_ensemble_decay=parameters["temporal_ensemble_decay"],
    )


def register_act_policy(registry: PolicyRegistryV3) -> None:
    registry.register(POLICY_ID, _create, _restore)
