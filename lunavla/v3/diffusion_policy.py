from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
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


POLICY_ID = "diffusion_v3"
LEROBOT_REVISION = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
LEROBOT_WHEEL_SHA256 = "b38a564fbc441d98380576863bf68635dde5fc2c42ddc2a39d0486640dc9e9a8"


def _upstream_camera_key(name: str) -> str:
    return f"observation.images.{name}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"checkpoint tree is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def diffusion_policy_spec(config: ExperimentConfig) -> PolicySpecV3:
    if config.policy["type"] != POLICY_ID:
        raise ValueError("diffusion_policy_spec requires policy.type=diffusion_v3")
    parameters = config.policy["parameters"]
    state_feature = str(parameters["state_feature"])
    cameras = tuple(str(item) for item in parameters["camera_features"])
    return PolicySpecV3(
        policy_id=POLICY_ID,
        backend="lerobot_diffusion",
        model_source=ModelSourceContractV1(
            repo_id="huggingface/lerobot",
            revision=LEROBOT_REVISION,
            file_hashes={"lerobot-0.6.0-py3-none-any.whl": LEROBOT_WHEEL_SHA256},
            license_status="verified",
            pretrained_enabled=False,
        ),
        required_modalities=("image", "state"),
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
        deterministic=config.training["device"] == "cpu",
    )


def _normalization_mode(mode: str) -> NormalizationMode:
    return {
        "none": NormalizationMode.IDENTITY,
        "dataset": NormalizationMode.MEAN_STD,
        "standard": NormalizationMode.MEAN_STD,
        "minmax": NormalizationMode.MIN_MAX,
    }[mode]


def _dataset_stats(
    spec: PolicySpecV3, normalization: NormalizationStatsV1
) -> dict[str, dict[str, npt.NDArray[np.float32]]]:
    state_name = spec.state_order[0]
    action_name = next(name for name in spec.normalization if name != state_name)
    result: dict[str, dict[str, npt.NDArray[np.float32]]] = {}
    for source_name, target_name in ((state_name, OBS_STATE), (action_name, ACTION)):
        stats = normalization.features[source_name]
        if stats.mode == "standard":
            result[target_name] = {
                "mean": np.array(stats.offset, copy=True),
                "std": np.array(stats.scale, copy=True),
            }
        elif stats.mode == "minmax":
            result[target_name] = {
                "min": np.array(stats.offset, copy=True),
                "max": np.array(stats.offset + stats.scale, copy=True),
            }
    return result


def _diffusion_config(config: ExperimentConfig, spec: PolicySpecV3) -> DiffusionConfig:
    parameters = config.policy["parameters"]
    state_features = {item.name: item for item in config.feature_schema.by_role("state")}
    action_feature = config.feature_schema.by_role("action")[0]
    image_features = {item.name: item for item in config.feature_schema.by_role("image")}
    inputs: dict[str, PolicyFeature] = {
        OBS_STATE: PolicyFeature(FeatureType.STATE, state_features[spec.state_order[0]].shape)
    }
    for name in spec.camera_order:
        height, width, channels = image_features[name].shape
        inputs[_upstream_camera_key(name)] = PolicyFeature(
            FeatureType.VISUAL, (channels, height, width)
        )
    state_mode = _normalization_mode(spec.normalization[spec.state_order[0]])
    action_mode = _normalization_mode(spec.normalization[action_feature.name])
    return DiffusionConfig(
        n_obs_steps=spec.history,
        horizon=spec.horizon,
        n_action_steps=int(parameters["n_action_steps"]),
        input_features=inputs,
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, action_feature.shape)},
        normalization_mapping={
            FeatureType.VISUAL: NormalizationMode.IDENTITY,
            FeatureType.STATE: state_mode,
            FeatureType.ACTION: action_mode,
        },
        vision_backbone=str(parameters["vision_backbone"]),
        crop_is_random=False,
        pretrained_backbone_weights=parameters["pretrained_backbone_weights"],
        use_group_norm=bool(parameters["use_group_norm"]),
        spatial_softmax_num_keypoints=int(parameters["spatial_softmax_num_keypoints"]),
        use_separate_rgb_encoder_per_camera=bool(
            parameters["use_separate_rgb_encoder_per_camera"]
        ),
        down_dims=tuple(int(item) for item in parameters["down_dims"]),
        kernel_size=int(parameters["kernel_size"]),
        n_groups=int(parameters["n_groups"]),
        diffusion_step_embed_dim=int(parameters["diffusion_step_embed_dim"]),
        noise_scheduler_type=str(parameters["noise_scheduler_type"]),
        num_train_timesteps=int(parameters["num_train_timesteps"]),
        beta_schedule=str(parameters["beta_schedule"]),
        prediction_type=str(parameters["prediction_type"]),
        clip_sample=bool(parameters["clip_sample"]),
        clip_sample_range=float(parameters["clip_sample_range"]),
        num_inference_steps=int(parameters["num_inference_steps"]),
        do_mask_loss_for_padding=bool(parameters["do_mask_loss_for_padding"]),
        compile_model=False,
        device=spec.device,
        use_amp=False,
        push_to_hub=False,
    )


class DiffusionPolicyV3:
    def __init__(
        self,
        policy: DiffusionPolicy,
        *,
        config: ExperimentConfig,
        spec: PolicySpecV3,
        normalization: NormalizationStatsV1,
    ) -> None:
        self.policy = policy
        self.config = config
        self.spec = spec
        self.normalization = normalization
        self.preprocessor, self.postprocessor = make_diffusion_pre_post_processors(
            policy.config, _dataset_stats(spec, normalization)
        )
        optimizer_parameters = config.training["optimizer"]["parameters"]
        self.optimizer = torch.optim.AdamW(
            self.policy.get_optim_params(),
            lr=float(config.training["learning_rate"]),
            weight_decay=float(optimizer_parameters.get("weight_decay", 1e-6)),
            eps=float(optimizer_parameters.get("eps", 1e-8)),
            betas=tuple(optimizer_parameters.get("betas", (0.95, 0.999))),
        )
        self._step = 0
        self._train_rng_state = self._seeded_cpu_state(int(config.training["seed"]))
        self._inference_generator = torch.Generator(device=spec.device)
        self._inference_generator.manual_seed(int(config.policy["parameters"]["noise_seed"]))

    @staticmethod
    def _seeded_cpu_state(seed: int) -> torch.Tensor:
        previous = torch.get_rng_state()
        torch.manual_seed(seed)
        state = torch.get_rng_state().clone()
        torch.set_rng_state(previous)
        return state

    def reset(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("policy reset seed must be a non-negative integer")
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self._inference_generator.manual_seed(seed)

    def _raw(self, sample: PolicySampleV3, *, include_action: bool) -> dict[str, Any]:
        if tuple(sample.observation.images) != self.spec.camera_order:
            raise ValueError("Diffusion observation camera order does not match PolicySpecV3")
        if sample.observation.instruction and "instruction" not in tuple(
            self.config.policy["parameters"]["unused_modalities"]
        ):
            raise ValueError("Diffusion cannot silently discard instruction")
        raw: dict[str, Any] = {
            OBS_STATE: torch.from_numpy(
                np.stack(
                    [item.state[self.spec.state_order[0]] for item in sample.observation_history]
                ).astype(np.float32)
            )
        }
        for name in self.spec.camera_order:
            images = np.stack([item.images[name] for item in sample.observation_history])
            raw[_upstream_camera_key(name)] = torch.from_numpy(
                np.transpose(images, (0, 3, 1, 2)).astype(np.float32) / 255.0
            )
        if include_action:
            if sample.action_chunk is None or sample.valid_mask is None:
                raise ValueError("Diffusion training samples require action supervision")
            raw[ACTION] = torch.from_numpy(
                np.array(sample.action_chunk, dtype=np.float32, copy=True)
            )
            raw["action_is_pad"] = torch.from_numpy(
                ~np.array(sample.valid_mask, dtype=bool, copy=True)
            )
        return raw

    def _processed(self, sample: PolicySampleV3, *, include_action: bool) -> dict[str, torch.Tensor]:
        processed = self.preprocessor(self._raw(sample, include_action=include_action))
        tensors = {
            name: value for name, value in processed.items() if isinstance(value, torch.Tensor)
        }
        return {name: value.unsqueeze(0) for name, value in tensors.items()}

    def _batch(self, batch: PolicyBatchV3) -> dict[str, torch.Tensor]:
        items = [self._processed(sample, include_action=True) for sample in batch.samples]
        keys = tuple(items[0])
        if any(tuple(item) != keys for item in items):
            raise ValueError("Diffusion processed batch keys are inconsistent")
        return {name: torch.cat([item[name] for item in items], dim=0) for name in keys}

    def train_step(
        self, batch: PolicyBatchV3, *, learning_rate: float, step: int
    ) -> TrainStepResultV3:
        if step != self._step:
            raise ValueError(f"Diffusion train step must be contiguous; expected {self._step}")
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        tensors = self._batch(batch)
        previous_rng = torch.get_rng_state()
        torch.set_rng_state(self._train_rng_state)
        started = time.perf_counter()
        try:
            loss, _ = self.policy.forward(tensors)
            if not torch.isfinite(loss):
                raise FloatingPointError("Diffusion produced a non-finite loss")
            loss.backward()
            clip = self.config.training["gradient_clip_norm"]
            if clip is None:
                gradient_norm = torch.linalg.vector_norm(
                    torch.stack(
                        [parameter.grad.detach().norm() for parameter in self.policy.parameters() if parameter.grad is not None]
                    )
                )
            else:
                gradient_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float(clip))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Diffusion produced a non-finite gradient")
            self.optimizer.step()
            self._train_rng_state = torch.get_rng_state().clone()
        finally:
            torch.set_rng_state(previous_rng)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._step += 1
        value = float(loss.detach().cpu())
        return TrainStepResultV3(
            value,
            {"diffusion_mse": value, "total": value},
            float(gradient_norm.detach().cpu()),
            learning_rate,
            step,
            True,
            {"train_step": elapsed_ms},
        )

    def predict_chunk(self, sample: PolicySampleV3) -> ActionChunk:
        self.policy.eval()
        tensors = self._processed(sample, include_action=False)
        action_dim = self.policy.config.action_feature.shape[0]
        noise = torch.randn(
            (1, self.spec.horizon, action_dim),
            generator=self._inference_generator,
            device=self.spec.device,
        )
        normalized = self.policy.predict_action_chunk(tensors, noise=noise)[0]
        values = self.postprocessor(normalized).detach().cpu().numpy().astype(np.float32)
        return ActionChunk(values, np.ones(values.shape[0], dtype=bool))

    def save_checkpoint(self, path: Path, *, metadata: Mapping[str, Any]) -> Path:
        path.mkdir(parents=True, exist_ok=False)
        model_dir = path / "model"
        processor_dir = path / "processors"
        self.policy.save_pretrained(model_dir, push_to_hub=False)
        self.preprocessor.save_pretrained(
            processor_dir, config_filename="preprocessor.json", push_to_hub=False
        )
        self.postprocessor.save_pretrained(
            processor_dir, config_filename="postprocessor.json", push_to_hub=False
        )
        training_state_path = path / "training_state.pt"
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "step": self._step,
                "train_rng_state": self._train_rng_state,
                "inference_rng_state": self._inference_generator.get_state(),
            },
            training_state_path,
        )
        marker = {
            "schema_version": 1,
            "policy_id": POLICY_ID,
            "policy_spec_sha256": self.spec.sha256(),
            "normalization_sha256": self.normalization.sha256(),
            "noise_scheduler_type": self.policy.config.noise_scheduler_type,
            "num_inference_steps": self.policy.config.num_inference_steps,
            "prediction_type": self.policy.config.prediction_type,
            "model_tree_sha256": _tree_sha256(model_dir),
            "processor_tree_sha256": _tree_sha256(processor_dir),
            "training_state_sha256": _file_sha256(training_state_path),
            "metadata": dict(metadata),
        }
        marker_path = path / "metadata.json"
        marker_path.write_text(
            json.dumps(marker, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return marker_path


def _create(
    config: ExperimentConfig, spec: PolicySpecV3, normalization: NormalizationStatsV1
) -> DiffusionPolicyV3:
    previous = torch.get_rng_state()
    torch.manual_seed(int(config.training["seed"]))
    try:
        policy = DiffusionPolicy(_diffusion_config(config, spec))
    finally:
        torch.set_rng_state(previous)
    policy.to(spec.device)
    return DiffusionPolicyV3(
        policy, config=config, spec=spec, normalization=normalization
    )


def _restore(
    checkpoint: Path,
    config: ExperimentConfig,
    spec: PolicySpecV3,
    normalization: NormalizationStatsV1,
) -> DiffusionPolicyV3:
    target = checkpoint
    if checkpoint.is_dir() and (checkpoint / "checkpoint.v3.json").exists():
        target = checkpoint / "policy" / str(config.artifacts["checkpoint_name"])
    marker = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    if set(marker) != {
        "schema_version", "policy_id", "policy_spec_sha256", "normalization_sha256",
        "noise_scheduler_type", "num_inference_steps", "prediction_type",
        "model_tree_sha256", "processor_tree_sha256", "training_state_sha256", "metadata",
    }:
        raise ValueError("Diffusion checkpoint metadata fields are invalid")
    expected_config = _diffusion_config(config, spec)
    expected = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "policy_spec_sha256": spec.sha256(),
        "normalization_sha256": normalization.sha256(),
        "noise_scheduler_type": expected_config.noise_scheduler_type,
        "num_inference_steps": expected_config.num_inference_steps,
        "prediction_type": expected_config.prediction_type,
    }
    if any(marker[name] != value for name, value in expected.items()):
        raise ValueError("Diffusion checkpoint contract does not match resolved config")
    if marker["model_tree_sha256"] != _tree_sha256(target / "model"):
        raise ValueError("Diffusion checkpoint model tree hash mismatch")
    if marker["processor_tree_sha256"] != _tree_sha256(target / "processors"):
        raise ValueError("Diffusion checkpoint processor tree hash mismatch")
    if marker["training_state_sha256"] != _file_sha256(target / "training_state.pt"):
        raise ValueError("Diffusion checkpoint training state hash mismatch")
    policy = DiffusionPolicy.from_pretrained(
        target / "model", config=expected_config, local_files_only=True, strict=True
    )
    wrapper = DiffusionPolicyV3(
        policy, config=config, spec=spec, normalization=normalization
    )
    state = torch.load(target / "training_state.pt", map_location=spec.device, weights_only=True)
    if set(state) != {"optimizer", "step", "train_rng_state", "inference_rng_state"}:
        raise ValueError("Diffusion training state fields are invalid")
    wrapper.optimizer.load_state_dict(state["optimizer"])
    wrapper._step = int(state["step"])
    wrapper._train_rng_state = state["train_rng_state"].cpu()
    wrapper._inference_generator.set_state(state["inference_rng_state"].cpu())
    return wrapper


def register_diffusion_policy(registry: PolicyRegistryV3) -> None:
    registry.register(POLICY_ID, _create, _restore)
