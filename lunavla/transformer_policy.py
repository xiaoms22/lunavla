from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Mapping, Self, Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from lunavla.contracts import Observation, PolicyBatch, normalize_device
from lunavla.temporal import TemporalEnsembler as TemporalEnsembler
from model.policy_base import ActionChunk

if TYPE_CHECKING:
    from lunavla.registry import PolicyRegistry


POLICY_ID: Final = "transformer_chunk_cvae"
CHECKPOINT_SCHEMA_VERSION: Final = 2
CHECKPOINT_FORMAT: Final = "lunavla.transformer_chunk_cvae"

_ACT_REQUIRED_CAPABILITIES: Final = frozenset(
    {
        "action_query_transformer",
        "conditional_cvae_kl",
        "valid_mask_loss",
        "temporal_ensembling",
    }
)
CAPABILITIES: Final = frozenset(_ACT_REQUIRED_CAPABILITIES)
act_alias_supported: Final = _ACT_REQUIRED_CAPABILITIES <= CAPABILITIES
ACT_ALIAS_SUPPORTED: Final = act_alias_supported

Float32Array = npt.NDArray[np.float32]
_TOKEN_PATTERN: Final = re.compile(r"[\w'-]+", flags=re.UNICODE)


def _safe_checkpoint_value(value: Any, *, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite values")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _safe_checkpoint_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} keys must be strings")
            result[key] = _safe_checkpoint_value(item, name=f"{name}.{key}")
        return result
    raise TypeError(f"{name} contains unsupported value type {type(value).__name__}")


def _cpu_checkpoint_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_checkpoint_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_checkpoint_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_checkpoint_tree(item) for item in value]
    return value


def _move_optimizer_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_optimizer_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_optimizer_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_optimizer_tree(item, device) for item in value]
    return value


@dataclass(frozen=True)
class TransformerPolicyConfig:
    """Versioned configuration for the teaching-scale chunk Transformer."""

    state_dim: int
    action_dim: int
    chunk_size: int
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 128
    latent_dim: int = 16
    instruction_dim: int = 0
    image_shape: tuple[int, ...] | None = None
    dropout: float = 0.0
    kl_weight: float = 1e-3
    max_grad_norm: float = 10.0
    sample_latent_during_training: bool = True
    seed: int = 42
    device: str = "cpu"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported policy config schema_version: {self.schema_version}")
        positive_dimensions = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "latent_dim": self.latent_dim,
        }
        for name, value in positive_dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.instruction_dim, bool)
            or not isinstance(self.instruction_dim, int)
            or self.instruction_dim < 0
        ):
            raise ValueError("instruction_dim must be a non-negative integer")
        if self.image_shape is not None:
            try:
                image_shape = tuple(int(size) for size in self.image_shape)
            except (TypeError, ValueError) as exc:
                raise ValueError("image_shape must contain integer dimensions") from exc
            if len(image_shape) not in (2, 3) or any(size <= 0 for size in image_shape):
                raise ValueError("image_shape must be positive HW or HWC dimensions")
            if len(image_shape) == 3 and image_shape[-1] not in (1, 3, 4):
                raise ValueError("image_shape HWC channels must be 1, 3, or 4")
            object.__setattr__(self, "image_shape", image_shape)
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not math.isfinite(self.kl_weight) or self.kl_weight < 0.0:
            raise ValueError("kl_weight must be finite and non-negative")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be a positive finite value")
        if not isinstance(self.sample_latent_during_training, bool):
            raise TypeError("sample_latent_during_training must be boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        object.__setattr__(self, "device", normalize_device(self.device))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Self:
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown transformer policy config fields: {unknown}")
        values = dict(raw)
        if values.get("image_shape") is not None:
            values["image_shape"] = tuple(values["image_shape"])
        return cls(**values)


class TransformerChunkCVAEPolicy(nn.Module):
    """Action-query Transformer with a conditional variational action encoder."""

    policy_id: Final = POLICY_ID
    capabilities: Final = CAPABILITIES
    act_alias_supported: Final = act_alias_supported

    def __init__(self, config: TransformerPolicyConfig | Mapping[str, Any]) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, TransformerPolicyConfig)
            else TransformerPolicyConfig.from_mapping(config)
        )
        self.action_dim = self.config.action_dim
        self.chunk_size = self.config.chunk_size
        self._condition_token_count = (
            1
            + int(self.config.instruction_dim > 0)
            + int(self.config.image_shape is not None)
        )
        requested_device = torch.device(self.config.device)
        self._check_device_available(requested_device)

        # fork_rng makes initialization repeatable without perturbing a caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.state_projection = nn.Linear(self.config.state_dim, self.config.d_model)
            self.action_projection = nn.Linear(self.action_dim, self.config.d_model)
            self.instruction_projection: nn.Linear | None
            if self.config.instruction_dim > 0:
                self.instruction_projection = nn.Linear(
                    self.config.instruction_dim,
                    self.config.d_model,
                    bias=False,
                )
            else:
                self.instruction_projection = None

            self.image_encoder: nn.Sequential | None
            if self.config.image_shape is not None:
                image_channels = (
                    1 if len(self.config.image_shape) == 2 else self.config.image_shape[-1]
                )
                hidden_channels = max(8, self.config.d_model // 4)
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(
                        image_channels,
                        hidden_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        hidden_channels,
                        self.config.d_model,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(start_dim=1),
                )
            else:
                self.image_encoder = None
            self.posterior_token = nn.Parameter(torch.empty(1, 1, self.config.d_model))
            self.action_queries = nn.Parameter(
                torch.empty(1, self.chunk_size, self.config.d_model)
            )
            self.encoder_positions = nn.Parameter(
                torch.empty(
                    1,
                    self.chunk_size + self._condition_token_count + 1,
                    self.config.d_model,
                )
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.config.d_model,
                nhead=self.config.nhead,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.posterior_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.config.num_encoder_layers,
                norm=nn.LayerNorm(self.config.d_model),
                enable_nested_tensor=False,
            )
            self.posterior_mu = nn.Linear(self.config.d_model, self.config.latent_dim)
            self.posterior_logvar = nn.Linear(self.config.d_model, self.config.latent_dim)
            self.latent_projection = nn.Linear(self.config.latent_dim, self.config.d_model)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.config.d_model,
                nhead=self.config.nhead,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.action_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=self.config.num_decoder_layers,
                norm=nn.LayerNorm(self.config.d_model),
            )
            self.action_head = nn.Linear(self.config.d_model, self.action_dim)
            self._reset_learned_tokens()

        self.to(device=requested_device, dtype=torch.float32)
        self._latent_generator = torch.Generator(device=requested_device)
        self._latent_generator.manual_seed(self.config.seed + 1)
        self._train_step = 0
        self._optimizer: torch.optim.Optimizer | None = None
        self.checkpoint_metadata: dict[str, Any] = {}

    @staticmethod
    def _check_device_available(device: torch.device) -> None:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {device} is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("requested device mps is unavailable")

    def _reset_learned_tokens(self) -> None:
        nn.init.normal_(self.posterior_token, mean=0.0, std=0.02)
        nn.init.normal_(self.action_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.encoder_positions, mean=0.0, std=0.02)

    @property
    def device(self) -> str:
        return str(self.action_queries.device)

    @property
    def dtype(self) -> torch.dtype:
        return self.action_queries.dtype

    def to(self, *args: Any, **kwargs: Any) -> Self:
        previous_device = (
            self.action_queries.device if hasattr(self, "action_queries") else None
        )
        module = super().to(*args, **kwargs)
        current_device = self.action_queries.device
        if previous_device is not None and current_device != previous_device:
            self.config = replace(self.config, device=str(current_device))
            previous_generator = getattr(self, "_latent_generator", None)
            generator = torch.Generator(device=current_device)
            if previous_generator is not None:
                try:
                    generator.set_state(previous_generator.get_state())
                except RuntimeError:
                    generator.manual_seed(self.config.seed + 1 + self._train_step)
            else:
                generator.manual_seed(self.config.seed + 1)
            self._latent_generator = generator
            optimizer = getattr(self, "_optimizer", None)
            if optimizer is not None:
                for state in optimizer.state.values():
                    for key, value in list(state.items()):
                        state[key] = _move_optimizer_tree(value, current_device)
        return module

    def _validate_states(self, states: Tensor) -> None:
        if not isinstance(states, Tensor):
            raise TypeError("states must be a torch.Tensor")
        expected_shape = (None, self.config.state_dim)
        if states.ndim != 2 or states.shape[1] != self.config.state_dim:
            raise ValueError(f"states must have shape [batch, {self.config.state_dim}]")
        if states.shape[0] <= 0:
            raise ValueError(f"states must have shape {expected_shape} with a non-empty batch")
        self._validate_float_tensor(states, name="states")

    def _validate_float_tensor(self, value: Tensor, *, name: str) -> None:
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32; got {value.dtype}")
        if value.device != self.action_queries.device:
            raise ValueError(
                f"{name} is on {value.device}, but policy parameters are on {self.device}"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains NaN or infinite values")

    @staticmethod
    def instruction_features(instruction: str, dim: int) -> Float32Array:
        """Encode text as a deterministic, normalized token-hash bag of words."""

        if dim <= 0:
            raise ValueError("instruction feature dimension must be positive")
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")
        if instruction.strip().casefold() == "[mask]":
            return np.zeros(dim, dtype=np.float32)
        tokens = _TOKEN_PATTERN.findall(instruction.strip().lower())
        if not tokens:
            raise ValueError("instruction must contain at least one token")
        features = np.zeros(dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], byteorder="little") % dim
            features[index] += 1.0
        norm = float(np.linalg.norm(features))
        return (features / norm).astype(np.float32, copy=False)

    def _image_to_chw(self, image: npt.NDArray[np.generic]) -> Float32Array:
        if self.config.image_shape is None:
            raise ValueError("image modality is disabled by image_shape=None")
        raw = np.asarray(image)
        if raw.shape != self.config.image_shape:
            raise ValueError(
                f"image must have shape {self.config.image_shape}; got {raw.shape}"
            )
        if raw.dtype == np.uint8:
            normalized = np.array(raw, dtype=np.float32, copy=True) / 255.0
        elif raw.dtype == np.float32:
            normalized = np.array(raw, dtype=np.float32, copy=True)
        else:
            raise TypeError(f"image must have dtype uint8 or float32; got {raw.dtype}")
        if not np.all(np.isfinite(normalized)):
            raise ValueError("image contains NaN or infinite values")
        if normalized.ndim == 2:
            return normalized[None, :, :]
        return np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)

    def _validate_condition_tensors(
        self,
        states: Tensor,
        instruction_features: Tensor | None,
        images: Tensor | None,
    ) -> None:
        batch_size = states.shape[0]
        if self.config.instruction_dim > 0:
            if instruction_features is None:
                raise ValueError("instruction modality is enabled but instruction_features is missing")
            expected_instruction = (batch_size, self.config.instruction_dim)
            if instruction_features.shape != expected_instruction:
                raise ValueError(
                    "instruction_features must have shape "
                    f"{expected_instruction}; got {instruction_features.shape}"
                )
            self._validate_float_tensor(
                instruction_features,
                name="instruction_features",
            )
        elif instruction_features is not None:
            raise ValueError("instruction modality is disabled by instruction_dim=0")

        if self.config.image_shape is not None:
            if images is None:
                raise ValueError("image modality is enabled but images are missing")
            height, width = self.config.image_shape[:2]
            channels = (
                1 if len(self.config.image_shape) == 2 else self.config.image_shape[-1]
            )
            expected_images = (batch_size, channels, height, width)
            if images.shape != expected_images:
                raise ValueError(f"images must have shape {expected_images}; got {images.shape}")
            self._validate_float_tensor(images, name="images")
        elif images is not None:
            raise ValueError("image modality is disabled by image_shape=None")

    def _condition_tokens(
        self,
        states: Tensor,
        instruction_features: Tensor | None,
        images: Tensor | None,
    ) -> Tensor:
        self._validate_states(states)
        self._validate_condition_tensors(states, instruction_features, images)
        tokens = [self.state_projection(states)]
        if instruction_features is not None:
            if self.instruction_projection is None:
                raise RuntimeError("instruction projection is not initialized")
            tokens.append(self.instruction_projection(instruction_features))
        if images is not None:
            if self.image_encoder is None:
                raise RuntimeError("image encoder is not initialized")
            tokens.append(self.image_encoder(images))
        return torch.stack(tokens, dim=1)

    def _observation_tensors(
        self,
        observations: Sequence[Observation],
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if not observations:
            raise ValueError("observations must not be empty")
        states_list: list[Float32Array] = []
        instruction_list: list[Float32Array] = []
        image_list: list[Float32Array] = []
        for index, observation in enumerate(observations):
            if not isinstance(observation, Observation):
                raise TypeError(f"observation {index} must be an Observation")
            state = np.asarray(observation.state)
            if state.dtype != np.float32:
                raise TypeError(
                    f"observation {index} state must have dtype float32; got {state.dtype}"
                )
            if state.shape != (self.config.state_dim,):
                raise ValueError(
                    f"observation {index} state must have shape "
                    f"{(self.config.state_dim,)}; got {state.shape}"
                )
            if not np.all(np.isfinite(state)):
                raise ValueError(f"observation {index} state contains NaN or infinite values")
            states_list.append(np.array(state, dtype=np.float32, copy=True))

            if self.config.instruction_dim > 0:
                if observation.instruction is None:
                    raise ValueError(
                        f"observation {index} is missing the enabled instruction modality"
                    )
                instruction_list.append(
                    self.instruction_features(
                        observation.instruction,
                        self.config.instruction_dim,
                    )
                )
            elif observation.instruction is not None:
                raise ValueError(
                    f"observation {index} supplies instruction but instruction_dim=0"
                )

            if self.config.image_shape is not None:
                if observation.image is None:
                    raise ValueError(
                        f"observation {index} is missing the enabled image modality"
                    )
                image_list.append(self._image_to_chw(observation.image))
            elif observation.image is not None:
                raise ValueError(
                    f"observation {index} supplies image but image_shape=None"
                )

        states = torch.from_numpy(np.stack(states_list)).to(device=self.device)
        instructions = (
            torch.from_numpy(np.stack(instruction_list)).to(device=self.device)
            if instruction_list
            else None
        )
        images = (
            torch.from_numpy(np.stack(image_list)).to(device=self.device)
            if image_list
            else None
        )
        return states, instructions, images

    def _validate_supervision(
        self,
        states: Tensor,
        actions: Tensor,
        valid_mask: Tensor,
    ) -> None:
        self._validate_states(states)
        expected_actions = (states.shape[0], self.chunk_size, self.action_dim)
        if actions.shape != expected_actions:
            raise ValueError(f"actions must have shape {expected_actions}; got {actions.shape}")
        self._validate_float_tensor(actions, name="actions")
        if valid_mask.shape != expected_actions[:2]:
            raise ValueError(
                f"valid_mask must have shape {expected_actions[:2]}; got {valid_mask.shape}"
            )
        if valid_mask.dtype != torch.bool:
            raise TypeError(f"valid_mask must have dtype torch.bool; got {valid_mask.dtype}")
        if valid_mask.device != self.action_queries.device:
            raise ValueError(
                f"valid_mask is on {valid_mask.device}, but policy parameters are on {self.device}"
            )
        if not bool(torch.all(torch.any(valid_mask, dim=1)).item()):
            raise ValueError("each sample must contain at least one valid action")

    def encode_posterior(
        self,
        states: Tensor,
        actions: Tensor,
        valid_mask: Tensor,
        instruction_features: Tensor | None = None,
        images: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode ``q(z | state, action_chunk)`` while excluding padded actions."""

        self._validate_supervision(states, actions, valid_mask)
        batch_size = states.shape[0]
        condition_tokens = self._condition_tokens(
            states,
            instruction_features,
            images,
        )
        masked_actions = actions.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        tokens = torch.cat(
            [
                self.posterior_token.expand(batch_size, -1, -1),
                condition_tokens,
                self.action_projection(masked_actions),
            ],
            dim=1,
        )
        tokens = tokens + self.encoder_positions
        padding_mask = torch.cat(
            [
                torch.zeros(
                    (batch_size, self._condition_token_count + 1),
                    dtype=torch.bool,
                    device=states.device,
                ),
                ~valid_mask,
            ],
            dim=1,
        )
        encoded = self.posterior_encoder(tokens, src_key_padding_mask=padding_mask)
        posterior = encoded[:, 0]
        mu = self.posterior_mu(posterior)
        logvar = torch.clamp(self.posterior_logvar(posterior), min=-20.0, max=20.0)
        return mu, logvar

    @staticmethod
    def reparameterize(
        mu: Tensor,
        logvar: Tensor,
        *,
        sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if mu.shape != logvar.shape:
            raise ValueError("mu and logvar must have identical shapes")
        if not sample:
            return mu
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(
            standard_deviation,
            generator=generator,
        ) * standard_deviation

    def decode_actions(
        self,
        states: Tensor,
        latent: Tensor | None = None,
        instruction_features: Tensor | None = None,
        images: Tensor | None = None,
    ) -> Tensor:
        """Decode one action per learned action query, conditioned on state and latent."""

        condition_tokens = self._condition_tokens(
            states,
            instruction_features,
            images,
        )
        batch_size = states.shape[0]
        if latent is None:
            latent = torch.zeros(
                (batch_size, self.config.latent_dim),
                dtype=torch.float32,
                device=states.device,
            )
        else:
            expected = (batch_size, self.config.latent_dim)
            if latent.shape != expected:
                raise ValueError(f"latent must have shape {expected}; got {latent.shape}")
            self._validate_float_tensor(latent, name="latent")

        memory = torch.cat(
            [condition_tokens, self.latent_projection(latent).unsqueeze(1)],
            dim=1,
        )
        queries = self.action_queries.expand(batch_size, -1, -1)
        decoded = self.action_decoder(tgt=queries, memory=memory)
        return self.action_head(decoded)

    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
        valid_mask: Tensor,
        *,
        sample_latent: bool = True,
        instruction_features: Tensor | None = None,
        images: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return masked reconstruction loss, KL divergence, and total CVAE loss."""

        mu, logvar = self.encode_posterior(
            states,
            actions,
            valid_mask,
            instruction_features,
            images,
        )
        latent = self.reparameterize(
            mu,
            logvar,
            sample=sample_latent,
            generator=self._latent_generator,
        )
        predicted_actions = self.decode_actions(
            states,
            latent,
            instruction_features,
            images,
        )
        element_mask = valid_mask.unsqueeze(-1)
        residual = torch.where(
            element_mask,
            predicted_actions - actions,
            torch.zeros_like(predicted_actions),
        )
        squared_error = residual.square()
        normalizer = element_mask.sum().to(dtype=torch.float32) * self.action_dim
        reconstruction_loss = squared_error.sum() / normalizer
        kl_per_sample = -0.5 * torch.sum(
            1.0 + logvar - mu.square() - logvar.exp(),
            dim=-1,
        )
        kl_loss = kl_per_sample.mean()
        loss = reconstruction_loss + self.config.kl_weight * kl_loss
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "predicted_actions": predicted_actions,
            "mu": mu,
            "logvar": logvar,
        }

    def forward(
        self,
        states: Tensor,
        actions: Tensor | None = None,
        valid_mask: Tensor | None = None,
        *,
        sample_latent: bool = True,
        instruction_features: Tensor | None = None,
        images: Tensor | None = None,
    ) -> Tensor | dict[str, Tensor]:
        if actions is None:
            if valid_mask is not None:
                raise ValueError("valid_mask is only valid when actions are supplied")
            return self.decode_actions(
                states,
                instruction_features=instruction_features,
                images=images,
            )
        if valid_mask is None:
            raise ValueError("valid_mask is required when actions are supplied")
        return self.compute_loss(
            states,
            actions,
            valid_mask,
            sample_latent=sample_latent,
            instruction_features=instruction_features,
            images=images,
        )

    def _batch_tensors(
        self,
        batch: PolicyBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        if not isinstance(batch, PolicyBatch):
            raise TypeError("batch must be a lunavla.contracts.PolicyBatch")
        if batch.device != self.device:
            raise ValueError(
                f"batch declares device {batch.device}, but policy parameters are on {self.device}"
            )
        states, instruction_features, images = self._observation_tensors(batch.observations)
        if batch.targets.shape != (batch.batch_size, self.chunk_size, self.action_dim):
            raise ValueError(
                "batch targets do not match the policy's chunk_size and action_dim: "
                f"got {batch.targets.shape}"
            )
        actions = torch.from_numpy(
            np.array(batch.targets, dtype=np.float32, copy=True)
        ).to(device=self.device)
        valid_mask = torch.from_numpy(
            np.array(batch.valid_mask, dtype=bool, copy=True)
        ).to(device=self.device)
        return states, actions, valid_mask, instruction_features, images

    def train_batch(self, batch: PolicyBatch, *, learning_rate: float) -> float:
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive finite value")
        states, actions, valid_mask, instruction_features, images = self._batch_tensors(batch)
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        else:
            for parameter_group in self._optimizer.param_groups:
                parameter_group["lr"] = learning_rate

        self.train()
        self._optimizer.zero_grad(set_to_none=True)
        parameter_device = self.action_queries.device
        fork_devices = (
            [parameter_device.index or 0] if parameter_device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            step_seed = self.config.seed + 10_000 + self._train_step
            if parameter_device.type == "cuda":
                torch.cuda.manual_seed(step_seed)
            else:
                torch.manual_seed(step_seed)
            losses = self.compute_loss(
                states,
                actions,
                valid_mask,
                sample_latent=self.config.sample_latent_during_training,
                instruction_features=instruction_features,
                images=images,
            )
            loss = losses["loss"]
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("training loss became NaN or infinite")
            loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.max_grad_norm)
        self._optimizer.step()
        self._train_step += 1
        return float(loss.detach().cpu().item())

    def predict_chunk(self, observation: Observation) -> ActionChunk:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be a lunavla.contracts.Observation")
        states, instruction_features, images = self._observation_tensors((observation,))
        was_training = self.training
        self.eval()
        with torch.no_grad():
            prediction = self.decode_actions(
                states,
                instruction_features=instruction_features,
                images=images,
            )[0]
        self.train(was_training)
        values = prediction.detach().cpu().numpy().astype(np.float32, copy=False)
        return ActionChunk(
            values=values,
            valid_mask=np.ones(self.chunk_size, dtype=bool),
        )

    def save_checkpoint(
        self,
        path: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        target = Path(path)
        if not target.suffix:
            target = target / "checkpoint.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "format": CHECKPOINT_FORMAT,
            "policy_id": self.policy_id,
            "config": asdict(self.config),
            "state_dict": {
                name: value.detach().cpu() for name, value in self.state_dict().items()
            },
            "optimizer_state_dict": (
                None
                if self._optimizer is None
                else _cpu_checkpoint_tree(self._optimizer.state_dict())
            ),
            "latent_rng_state": self._latent_generator.get_state().cpu(),
            "train_step": self._train_step,
            "metadata": _safe_checkpoint_value(dict(metadata or {}), name="metadata"),
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load_checkpoint(cls, path: Path, *, device: str | None = None) -> Self:
        source = Path(path)
        if source.is_dir():
            source = source / "checkpoint.pt"
        map_location = normalize_device(device or "cpu")
        payload = torch.load(source, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload must be a mapping")
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint schema_version: {payload.get('schema_version')!r}"
            )
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format: {payload.get('format')!r}")
        if payload.get("policy_id") != POLICY_ID:
            raise ValueError(f"checkpoint contains policy_id {payload.get('policy_id')!r}")
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("checkpoint config must be a mapping")
        config_data = dict(raw_config)
        config_data["device"] = map_location
        policy = cls(TransformerPolicyConfig.from_mapping(config_data))
        raw_state = payload.get("state_dict")
        if not isinstance(raw_state, Mapping):
            raise ValueError("checkpoint state_dict must be a mapping")
        policy.load_state_dict(raw_state, strict=True)
        raw_train_step = payload.get("train_step")
        if isinstance(raw_train_step, bool) or not isinstance(raw_train_step, int):
            raise ValueError("checkpoint train_step must be a non-negative integer")
        if raw_train_step < 0:
            raise ValueError("checkpoint train_step must be a non-negative integer")
        raw_rng_state = payload.get("latent_rng_state")
        if not isinstance(raw_rng_state, Tensor):
            raise ValueError("checkpoint latent_rng_state must be a tensor")
        policy._latent_generator.set_state(raw_rng_state.cpu())
        policy._train_step = raw_train_step
        raw_optimizer = payload.get("optimizer_state_dict")
        if raw_optimizer is not None:
            if not isinstance(raw_optimizer, Mapping):
                raise ValueError("checkpoint optimizer_state_dict must be a mapping or null")
            param_groups = raw_optimizer.get("param_groups")
            if not isinstance(param_groups, list) or not param_groups:
                raise ValueError("checkpoint optimizer state has no parameter groups")
            first_group = param_groups[0]
            if not isinstance(first_group, Mapping):
                raise ValueError("checkpoint optimizer parameter group must be a mapping")
            learning_rate = float(first_group.get("lr", 1e-3))
            if not math.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError("checkpoint optimizer learning rate is invalid")
            policy._optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
            policy._optimizer.load_state_dict(dict(raw_optimizer))
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("checkpoint metadata must be a mapping")
        policy.checkpoint_metadata = dict(raw_metadata)
        return policy


def create_transformer_policy(config: Mapping[str, Any]) -> TransformerChunkCVAEPolicy:
    """Registry-compatible explicit factory; importing this module requires the v2 profile."""

    return TransformerChunkCVAEPolicy(TransformerPolicyConfig.from_mapping(config))


def load_transformer_policy(
    path: Path,
    config: Mapping[str, Any],
) -> TransformerChunkCVAEPolicy:
    """Registry-compatible loader with optional device and config consistency checks."""

    device = str(config.get("device", "cpu"))
    policy = TransformerChunkCVAEPolicy.load_checkpoint(path, device=device)
    supplied = {key: value for key, value in config.items() if key != "device"}
    checkpoint_config = asdict(policy.config)
    for key, value in supplied.items():
        if key not in checkpoint_config:
            raise ValueError(f"unknown transformer policy config field: {key}")
        checkpoint_value = checkpoint_config[key]
        supplied_value = value
        if key == "image_shape" and supplied_value is not None:
            supplied_value = tuple(supplied_value)
        if checkpoint_value != supplied_value:
            raise ValueError(
                f"checkpoint config mismatch for {key}: "
                f"checkpoint={checkpoint_value!r}, requested={supplied_value!r}"
            )
    return policy


def register_transformer_policy(registry: PolicyRegistry) -> None:
    """Explicitly register the optional Torch policy and its capability-gated aliases."""

    aliases = ["transformer_chunk"]
    if act_alias_supported:
        aliases.append("act")
    registry.register(
        POLICY_ID,
        create_transformer_policy,
        loader=load_transformer_policy,
        aliases=tuple(aliases),
    )
