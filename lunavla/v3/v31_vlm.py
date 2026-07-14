from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, cast

import numpy as np
import numpy.typing as npt

from .v31_contracts import FeatureCacheIndexV1, FrozenFeatureManifestV1, VLMBackendSpecV1
from .contracts import ObservationV3
from .v31_tasks import V31TaskDataset


Float32Array = npt.NDArray[np.float32]
SMOLVLM2_REPO_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SMOLVLM2_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
QWEN3_VL_REPO_ID = "Qwen/Qwen3-VL-2B-Instruct"
QWEN3_VL_REVISION = "e2378df056d88153dc44616229fa371fcb87e236"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _contained_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("model inventory path escapes its root")
    return candidate


@dataclass(frozen=True)
class VLMPreflightResultV1:
    backend_spec_sha256: str
    local_model_root_sha256: str
    observed_files: Mapping[str, str]
    observed_bytes: int
    network_accessed: bool
    ready: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("VLMPreflightResultV1 schema_version must be integer 1")
        for name in ("backend_spec_sha256", "local_model_root_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be SHA-256")
        files = dict(sorted(self.observed_files.items()))
        if not files or any(_SHA256.fullmatch(value) is None for value in files.values()):
            raise ValueError("observed_files must contain SHA-256 values")
        object.__setattr__(self, "observed_files", MappingProxyType(files))
        if isinstance(self.observed_bytes, bool) or self.observed_bytes <= 0:
            raise ValueError("observed_bytes must be positive")
        if self.network_accessed is not False or self.ready is not True:
            raise ValueError("preflight must be local-only and ready")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_spec_sha256": self.backend_spec_sha256,
            "local_model_root_sha256": self.local_model_root_sha256,
            "observed_files": dict(self.observed_files),
            "observed_bytes": self.observed_bytes,
            "network_accessed": self.network_accessed,
            "ready": self.ready,
        }


@dataclass(frozen=True)
class VLMConnectivitySmokeV1:
    backend_spec_sha256: str
    rows: int
    task_ids: tuple[str, ...]
    strata: tuple[str, ...]
    feature_hashes: tuple[str, ...]
    finite: bool
    observational: bool
    claim_allowed: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("VLMConnectivitySmokeV1 schema_version must be integer 1")
        if _SHA256.fullmatch(self.backend_spec_sha256) is None:
            raise ValueError("backend_spec_sha256 must be SHA-256")
        if self.rows != 12 or len(self.feature_hashes) != 12:
            raise ValueError("Qwen3-VL smoke must contain exactly 12 rows")
        if len(set(self.feature_hashes)) != 12:
            raise ValueError("smoke feature hashes must be unique")
        if (
            self.finite is not True
            or self.observational is not True
            or self.claim_allowed is not False
        ):
            raise ValueError("Qwen3-VL smoke must remain finite, observational, and claim-closed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_spec_sha256": self.backend_spec_sha256,
            "rows": self.rows,
            "task_ids": list(self.task_ids),
            "strata": list(self.strata),
            "feature_hashes": list(self.feature_hashes),
            "finite": self.finite,
            "observational": self.observational,
            "claim_allowed": self.claim_allowed,
        }


class FrozenFeatureExtractor(Protocol):
    @property
    def output_dim(self) -> int: ...

    def extract(self, image: npt.NDArray[np.uint8], instruction: str) -> Float32Array: ...


class FrozenFeatureProviderV1(Protocol):
    output_dim: int

    @property
    def binding_sha256(self) -> str: ...

    def intervened_observation(
        self,
        observation: ObservationV3,
        *,
        intervention: str,
        donor_seed: int,
    ) -> tuple[Float32Array, str | None]: ...


class FrozenFeatureCacheReaderV1:
    """Read an already verified feature cache without exposing writable arrays."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.index = verify_frozen_feature_cache(self.root)
        self._features: dict[str, Float32Array] = {}
        self._manifests: dict[str, FrozenFeatureManifestV1] = {}
        manifests = sorted((self.root / "manifests").glob("*.json"))
        features = sorted((self.root / "features").glob("*.npy"))
        for manifest_path, feature_path in zip(manifests, features, strict=True):
            manifest = FrozenFeatureManifestV1.from_mapping(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            value = np.load(feature_path, allow_pickle=False)
            frozen = np.frombuffer(value.tobytes(order="C"), dtype=np.float32).reshape(value.shape)
            frozen.setflags(write=False)
            self._features[manifest.sample_id] = frozen
            self._manifests[manifest.sample_id] = manifest
        dimensions = {value.shape for value in self._features.values()}
        if len(dimensions) != 1:
            raise ValueError("frozen feature cache must use one output shape")
        shape = next(iter(dimensions))
        if len(shape) != 1:
            raise ValueError("conditioned ACT requires one-dimensional frozen features")
        self.output_dim = shape[0]

    @property
    def binding_sha256(self) -> str:
        return self.index.sha256()

    @staticmethod
    def sample_identity(*, split: str, task_id: str, episode_id: str | int, step_index: int) -> str:
        return _identity(split, task_id, episode_id, step_index)

    def get(
        self, *, split: str, task_id: str, episode_id: str | int, step_index: int
    ) -> Float32Array:
        identity = self.sample_identity(
            split=split,
            task_id=task_id,
            episode_id=episode_id,
            step_index=step_index,
        )
        try:
            return self._features[identity]
        except KeyError as exc:
            raise KeyError(f"frozen feature cache has no sample {identity}") from exc

    def intervened(
        self,
        *,
        split: str,
        task_id: str,
        episode_id: str | int,
        step_index: int,
        intervention: str,
        donor_seed: int,
    ) -> tuple[Float32Array, str | None]:
        identity = self.sample_identity(
            split=split,
            task_id=task_id,
            episode_id=episode_id,
            step_index=step_index,
        )
        if intervention == "control":
            return self._features[identity], None
        if intervention == "feature_mask":
            masked = np.zeros(self.output_dim, dtype=np.float32)
            masked.setflags(write=False)
            return masked, None
        if intervention != "feature_shuffle":
            raise ValueError(f"unsupported feature intervention {intervention!r}")
        if isinstance(donor_seed, bool) or not isinstance(donor_seed, int) or donor_seed < 0:
            raise ValueError("feature donor_seed must be a non-negative integer")
        recipient = self._features[identity]
        candidates = [
            candidate
            for candidate, value in self._features.items()
            if candidate != identity
            and json.loads(candidate)[0] == split
            and not np.array_equal(value, recipient)
        ]
        if not candidates:
            raise ValueError("feature shuffle has no non-self same-split donor")
        candidates.sort()
        digest = hashlib.sha256(f"{donor_seed}|{identity}".encode()).digest()
        donor = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
        return self._features[donor], donor

    def intervened_observation(
        self,
        observation: ObservationV3,
        *,
        intervention: str,
        donor_seed: int,
    ) -> tuple[Float32Array, str | None]:
        split = str(observation.metadata["split"])
        task_id = str(observation.metadata["task_id"])
        identity = self.sample_identity(
            split=split,
            task_id=task_id,
            episode_id=observation.episode_id,
            step_index=observation.step_index,
        )
        manifest = self._manifests.get(identity)
        if manifest is None:
            raise KeyError(f"frozen feature cache has no sample {identity}")
        image = np.asarray(observation.images["camera.primary"])
        instruction = observation.instruction
        if instruction is None:
            raise ValueError("frozen feature lookup requires an instruction")
        if (
            manifest.image_sha256 != _sha256_bytes(image.tobytes(order="C"))
            or manifest.prompt_renderer_sha256
            != _sha256_bytes(instruction.encode("utf-8"))
        ):
            raise KeyError("frozen feature identity exists with different observation content")
        return self.intervened(
            split=split,
            task_id=task_id,
            episode_id=observation.episode_id,
            step_index=observation.step_index,
            intervention=intervention,
            donor_seed=donor_seed,
        )


class ContentAddressedFrozenFeatureProviderV1:
    """Append-only runtime cache for policy-generated observations.

    The binding hash is the immutable training-cache index. Runtime entries are
    addressed by model/processor/prompt/image content and verified on every read.
    """

    def __init__(
        self,
        *,
        base_cache: FrozenFeatureCacheReaderV1,
        extractor: FrozenFeatureExtractor,
        root: str | Path,
        processor_sha256: str,
        device_environment_sha256: str,
    ) -> None:
        self.base_cache = base_cache
        self.extractor = extractor
        self.output_dim = base_cache.output_dim
        if extractor.output_dim != self.output_dim:
            raise ValueError("online extractor dimension differs from the training cache")
        for name, value in (
            ("processor_sha256", processor_sha256),
            ("device_environment_sha256", device_environment_sha256),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be SHA-256")
        self.processor_sha256 = processor_sha256
        self.device_environment_sha256 = device_environment_sha256
        self.root = Path(root).resolve()
        if not self.root.is_absolute():
            raise ValueError("online feature root must be absolute")
        self.feature_root = self.root / "features"
        self.manifest_root = self.root / "manifests"
        self.feature_root.mkdir(parents=True, exist_ok=True)
        self.manifest_root.mkdir(parents=True, exist_ok=True)

    @property
    def binding_sha256(self) -> str:
        return self.base_cache.binding_sha256

    def _content(
        self, observation: ObservationV3
    ) -> tuple[str, npt.NDArray[np.uint8], str, str, str]:
        if tuple(observation.images) != ("camera.primary",):
            raise ValueError("online frozen features require exactly camera.primary")
        image = np.asarray(observation.images["camera.primary"])
        if image.shape != (96, 96, 3) or image.dtype != np.uint8:
            raise ValueError("online frozen features require 96x96 RGB uint8")
        instruction = observation.instruction
        if instruction is None:
            raise ValueError("online frozen features require an instruction")
        image_hash = _sha256_bytes(image.tobytes(order="C"))
        prompt_hash = _sha256_bytes(instruction.encode("utf-8"))
        payload = {
            "schema_version": 1,
            "binding_sha256": self.binding_sha256,
            "processor_sha256": self.processor_sha256,
            "device_environment_sha256": self.device_environment_sha256,
            "image_sha256": image_hash,
            "prompt_sha256": prompt_hash,
        }
        key = _sha256_bytes(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        return key, cast(npt.NDArray[np.uint8], image), instruction, image_hash, prompt_hash

    def _paths(self, key: str) -> tuple[Path, Path]:
        if _SHA256.fullmatch(key) is None:
            raise ValueError("online feature key must be SHA-256")
        return self.feature_root / f"{key}.npy", self.manifest_root / f"{key}.json"

    def _read(self, key: str) -> tuple[Float32Array, Mapping[str, Any]]:
        feature_path, manifest_path = self._paths(key)
        if feature_path.exists() != manifest_path.exists():
            raise ValueError("online feature entry is incomplete")
        if not feature_path.is_file():
            raise KeyError(key)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version",
            "key",
            "binding_sha256",
            "processor_sha256",
            "device_environment_sha256",
            "image_sha256",
            "prompt_sha256",
            "feature_sha256",
            "split",
            "task_id",
            "episode_id_type",
            "episode_id",
            "step_index",
        }
        if not isinstance(manifest, dict) or set(manifest) != expected or manifest["key"] != key:
            raise ValueError("online feature manifest is invalid")
        if isinstance(manifest["schema_version"], bool) or manifest["schema_version"] != 1:
            raise ValueError("online feature manifest version is invalid")
        for name in (
            "binding_sha256",
            "processor_sha256",
            "device_environment_sha256",
            "image_sha256",
            "prompt_sha256",
            "feature_sha256",
        ):
            if not isinstance(manifest[name], str) or _SHA256.fullmatch(manifest[name]) is None:
                raise ValueError(f"online feature manifest {name} is invalid")
        expected_key = _sha256_bytes(
            json.dumps(
                {
                    "schema_version": 1,
                    "binding_sha256": manifest["binding_sha256"],
                    "processor_sha256": manifest["processor_sha256"],
                    "device_environment_sha256": manifest["device_environment_sha256"],
                    "image_sha256": manifest["image_sha256"],
                    "prompt_sha256": manifest["prompt_sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        if expected_key != key:
            raise ValueError("online feature manifest content address is invalid")
        if (
            manifest["binding_sha256"] != self.binding_sha256
            or manifest["processor_sha256"] != self.processor_sha256
            or manifest["device_environment_sha256"] != self.device_environment_sha256
            or _sha256_file(feature_path) != manifest["feature_sha256"]
        ):
            raise ValueError("online feature entry hash or contract mismatch")
        raw = np.load(feature_path, allow_pickle=False)
        if raw.shape != (self.output_dim,) or raw.dtype != np.float32 or not np.all(np.isfinite(raw)):
            raise ValueError("online feature value is invalid")
        value = np.frombuffer(raw.tobytes(order="C"), dtype=np.float32)
        value.setflags(write=False)
        return value, MappingProxyType(dict(manifest))

    def _extract(self, observation: ObservationV3) -> tuple[Float32Array, str]:
        key, image, instruction, image_hash, prompt_hash = self._content(observation)
        try:
            value, _ = self._read(key)
            return value, key
        except KeyError:
            pass
        feature = np.asarray(self.extractor.extract(image, instruction), dtype=np.float32)
        if feature.shape != (self.output_dim,) or not np.all(np.isfinite(feature)):
            raise ValueError("online extractor returned an invalid feature")
        return self._store(observation, key, image_hash, prompt_hash, feature), key

    def _store(
        self,
        observation: ObservationV3,
        key: str,
        image_hash: str,
        prompt_hash: str,
        feature: Float32Array,
    ) -> Float32Array:
        feature_path, manifest_path = self._paths(key)
        temporary_feature = feature_path.with_suffix(".npy.tmp")
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        with temporary_feature.open("wb") as stream:
            np.save(stream, feature, allow_pickle=False)
        manifest = {
            "schema_version": 1,
            "key": key,
            "binding_sha256": self.binding_sha256,
            "processor_sha256": self.processor_sha256,
            "device_environment_sha256": self.device_environment_sha256,
            "image_sha256": image_hash,
            "prompt_sha256": prompt_hash,
            "feature_sha256": _sha256_file(temporary_feature),
            "split": str(observation.metadata["split"]),
            "task_id": str(observation.metadata["task_id"]),
            "episode_id_type": type(observation.episode_id).__name__,
            "episode_id": observation.episode_id,
            "step_index": observation.step_index,
        }
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_feature.replace(feature_path)
        temporary_manifest.replace(manifest_path)
        value, _ = self._read(key)
        return value

    def ensure_observations(
        self,
        observations: tuple[ObservationV3, ...],
        *,
        batch_size: int = 4,
    ) -> tuple[str, ...]:
        """Materialize missing runtime features in deterministic content order."""

        if not observations:
            return ()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("online feature batch_size must be a positive integer")
        missing: dict[
            str,
            tuple[
                ObservationV3,
                npt.NDArray[np.uint8],
                str,
                str,
                str,
            ],
        ] = {}
        keys: list[str] = []
        for observation in observations:
            key, image, instruction, image_hash, prompt_hash = self._content(observation)
            keys.append(key)
            try:
                self.base_cache.intervened_observation(
                    observation, intervention="control", donor_seed=0
                )
                continue
            except KeyError:
                pass
            try:
                self._read(key)
                continue
            except KeyError:
                missing.setdefault(
                    key, (observation, image, instruction, image_hash, prompt_hash)
                )
        ordered = [missing[key] for key in sorted(missing)]
        batch_method = getattr(self.extractor, "extract_batch", None)
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            inputs = tuple((item[1], item[2]) for item in batch)
            features = (
                batch_method(inputs)
                if callable(batch_method)
                else tuple(self.extractor.extract(image, text) for image, text in inputs)
            )
            if len(features) != len(batch):
                raise ValueError("online extractor batch result length is invalid")
            for item, raw_feature in zip(batch, features, strict=True):
                observation, _, _, image_hash, prompt_hash = item
                key = self._content(observation)[0]
                feature = np.asarray(raw_feature, dtype=np.float32)
                if feature.shape != (self.output_dim,) or not np.all(np.isfinite(feature)):
                    raise ValueError("online extractor returned an invalid batched feature")
                self._store(observation, key, image_hash, prompt_hash, feature)
        return tuple(keys)

    def _donors(
        self, observation: ObservationV3, recipient_key: str
    ) -> list[tuple[str, Float32Array, str]]:
        split = str(observation.metadata["split"])
        step = observation.step_index
        donors: list[tuple[str, Float32Array, str]] = []
        for identity, manifest in self.base_cache._manifests.items():
            if manifest.split == split and manifest.step_index == step:
                donors.append(
                    (
                        f"base:{identity}",
                        self.base_cache._features[identity],
                        _sha256_bytes(
                            f"{manifest.image_sha256}|{manifest.prompt_renderer_sha256}".encode()
                        ),
                    )
                )
        for manifest_path in sorted(self.manifest_root.glob("*.json")):
            key = manifest_path.stem
            value, online_mapping = self._read(key)
            if (
                online_mapping["split"] == split
                and online_mapping["step_index"] == step
                and key != recipient_key
            ):
                donors.append((
                    f"online:{key}",
                    value,
                    _sha256_bytes(
                        f"{online_mapping['image_sha256']}|{online_mapping['prompt_sha256']}".encode()
                    ),
                ))
        return sorted(donors, key=lambda item: item[0])

    def intervened_observation(
        self,
        observation: ObservationV3,
        *,
        intervention: str,
        donor_seed: int,
    ) -> tuple[Float32Array, str | None]:
        if intervention == "feature_mask":
            masked = np.zeros(self.output_dim, dtype=np.float32)
            masked.setflags(write=False)
            return masked, None
        key, _, _, image_hash, prompt_hash = self._content(observation)
        if intervention == "control":
            try:
                return self.base_cache.intervened_observation(
                    observation, intervention="control", donor_seed=donor_seed
                )
            except KeyError:
                value, _ = self._extract(observation)
                return value, None
        if intervention != "feature_shuffle":
            raise ValueError(f"unsupported feature intervention {intervention!r}")
        if isinstance(donor_seed, bool) or not isinstance(donor_seed, int) or donor_seed < 0:
            raise ValueError("feature donor_seed must be a non-negative integer")
        recipient_content = _sha256_bytes(f"{image_hash}|{prompt_hash}".encode())
        candidates = [
            item for item in self._donors(observation, key) if item[2] != recipient_content
        ]
        if not candidates:
            raise ValueError("online feature shuffle has no same-split, same-step donor")
        digest = hashlib.sha256(f"{donor_seed}|{key}".encode()).digest()
        donor_id, value, _ = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
        return value, donor_id


@dataclass(frozen=True)
class DeterministicFixtureExtractor:
    output_dim: int = 16

    def __post_init__(self) -> None:
        if isinstance(self.output_dim, bool) or self.output_dim <= 0:
            raise ValueError("output_dim must be positive")

    def extract(self, image: npt.NDArray[np.uint8], instruction: str) -> Float32Array:
        raw = np.asarray(image)
        if raw.shape != (96, 96, 3) or raw.dtype != np.uint8:
            raise ValueError("fixture extractor requires 96x96 RGB uint8")
        seed = hashlib.sha256(raw.tobytes(order="C") + instruction.encode("utf-8")).digest()
        values = np.frombuffer(
            (seed * ((self.output_dim * 4 // len(seed)) + 1))[: self.output_dim * 4], dtype="<u4"
        )
        return ((values.astype(np.float64) / np.iinfo(np.uint32).max) * 2 - 1).astype(np.float32)


class TransformersFrozenExtractor:
    def __init__(self, spec: VLMBackendSpecV1, local_model_root: str | Path) -> None:
        if spec.backend_id not in {"smolvlm2_500m", "qwen3_vl_2b"}:
            raise ValueError("unsupported real VLM backend")
        preflight_local_model(spec, local_model_root)
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "real VLM extraction requires the v3-vlm dependency profile"
            ) from exc
        root = str(Path(local_model_root).resolve())
        self._torch = torch
        self._spec = spec
        self._processor = AutoProcessor.from_pretrained(root, local_files_only=True)
        model: Any = AutoModelForImageTextToText.from_pretrained(
            root, local_files_only=True, torch_dtype=getattr(torch, spec.model_dtype)
        )
        self._model = model
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._device = spec.device
        self._model.to(spec.device)
        self._output_dim = int(self._model.config.text_config.hidden_size)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def extract(self, image: npt.NDArray[np.uint8], instruction: str) -> Float32Array:
        return self.extract_batch(((image, instruction),))[0]

    def extract_batch(
        self, values: tuple[tuple[npt.NDArray[np.uint8], str], ...]
    ) -> tuple[Float32Array, ...]:
        if not values:
            raise ValueError("VLM extraction batch cannot be empty")
        texts: list[str] = []
        images: list[npt.NDArray[np.uint8]] = []
        for image, instruction in values:
            if np.asarray(image).shape != (96, 96, 3):
                raise ValueError("VLM extraction requires 96x96 RGB images")
            images.append(image)
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": instruction}],
                }
            ]
            texts.append(
                self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        processor_options: dict[str, Any] = {}
        if self._spec.image_token_layout == "single_image_no_split_512":
            processor_options = {
                "do_image_splitting": False,
                "max_image_size": {"longest_edge": 512},
            }
        inputs = self._processor(
            text=texts,
            images=[[image] for image in images],
            return_tensors="pt",
            padding=True,
            images_kwargs=processor_options,
        )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self._model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        result = pooled.detach().float().cpu().numpy().astype(np.float32, copy=True)
        if result.shape != (len(values), self.output_dim) or not np.all(np.isfinite(result)):
            raise ValueError("VLM returned an invalid pooled feature")
        return tuple(result[index] for index in range(len(values)))


def preflight_local_model(
    spec: VLMBackendSpecV1, local_model_root: str | Path
) -> VLMPreflightResultV1:
    root = Path(local_model_root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("local_model_root must be an existing absolute directory")
    expected_repo_revision = {
        "smolvlm2_500m": (SMOLVLM2_REPO_ID, SMOLVLM2_REVISION, "claim_bearing"),
        "qwen3_vl_2b": (QWEN3_VL_REPO_ID, QWEN3_VL_REVISION, "observational"),
    }
    if spec.backend_id not in expected_repo_revision:
        raise ValueError("preflight supports only the two pinned v3.1 backends")
    repo, revision, role = expected_repo_revision[spec.backend_id]
    if (spec.repo_id, spec.revision, spec.evidence_role) != (repo, revision, role):
        raise ValueError("backend repo, revision, or evidence role is not the pinned v3.1 value")
    declared = set(spec.weight_files)
    observed_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed_paths != declared:
        missing = sorted(declared - observed_paths)
        unexpected = sorted(observed_paths - declared)
        raise ValueError(f"model inventory mismatch; missing={missing}, unexpected={unexpected}")
    observed: dict[str, str] = {}
    total_bytes = 0
    for relative, expected_hash in spec.weight_files.items():
        path = _contained_file(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned model file: {relative}")
        observed_hash = _sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(f"model file hash mismatch: {relative}")
        observed[relative] = observed_hash
        total_bytes += path.stat().st_size
    if total_bytes != spec.total_weight_bytes:
        raise ValueError("model inventory byte total mismatch")
    inventory_hash = _sha256_bytes(
        json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()
    )
    return VLMPreflightResultV1(
        backend_spec_sha256=spec.sha256(),
        local_model_root_sha256=inventory_hash,
        observed_files=observed,
        observed_bytes=total_bytes,
        network_accessed=False,
        ready=True,
    )


def _identity(split: str, task_id: str, episode_id: str | int, step: int) -> str:
    return json.dumps(
        [split, task_id, type(episode_id).__name__, episode_id, step], separators=(",", ":")
    )


def build_frozen_feature_cache(
    dataset: V31TaskDataset,
    spec: VLMBackendSpecV1,
    extractor: FrozenFeatureExtractor,
    output_root: str | Path,
    *,
    processor_sha256: str,
    device_environment_sha256: str,
    overwrite: bool = False,
) -> Path:
    output = Path(output_root)
    if not output.is_absolute():
        raise ValueError("output_root must be absolute")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("feature cache exists; use overwrite explicitly")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    try:
        staging.mkdir()
        feature_dir = staging / "features"
        manifest_dir = staging / "manifests"
        feature_dir.mkdir()
        manifest_dir.mkdir()
        manifest_hashes: list[str] = []
        identities: list[str] = []
        split_counts = {"train": 0, "validation": 0, "test": 0}
        total_bytes = 0
        records: list[tuple[str, Any, str, str, Any, npt.NDArray[np.uint8], str, str]] = []
        for split in ("train", "validation", "test"):
            for episode in dataset.bundle.select(split):
                task_id = str(episode.metadata["task_id"])
                stratum = str(episode.metadata["held_out_stratum"])
                for transition in episode.transitions:
                    observation = transition.observation
                    image = cast(npt.NDArray[np.uint8], observation.images["camera.primary"])
                    instruction = observation.instruction
                    if instruction is None:
                        raise ValueError("VLM cache requires an instruction")
                    identity = _identity(split, task_id, episode.episode_id, observation.step_index)
                    records.append(
                        (
                            split,
                            episode,
                            task_id,
                            stratum,
                            observation,
                            image,
                            instruction,
                            identity,
                        )
                    )
        batch_method = getattr(extractor, "extract_batch", None)
        # Two samples stays below the 8 GiB Apple-Silicon Colima development
        # ceiling while still amortizing processor/model overhead.
        for batch_start in range(0, len(records), 2):
            batch_records = records[batch_start : batch_start + 2]
            inputs = tuple((item[5], item[6]) for item in batch_records)
            features = (
                batch_method(inputs)
                if callable(batch_method)
                else tuple(extractor.extract(image, instruction) for image, instruction in inputs)
            )
            if len(features) != len(batch_records):
                raise ValueError("extractor batch result length is invalid")
            for record, raw_feature in zip(batch_records, features, strict=True):
                split, episode, task_id, stratum, observation, image, instruction, identity = record
                feature = np.asarray(raw_feature, dtype=np.float32)
                if feature.shape != (extractor.output_dim,) or not np.all(np.isfinite(feature)):
                    raise ValueError("extractor returned invalid feature")
                if identity in identities:
                    raise ValueError("duplicate typed cache identity")
                ordinal = len(identities)
                feature_path = feature_dir / f"{ordinal:08d}.npy"
                np.save(feature_path, feature, allow_pickle=False)
                feature_hash = _sha256_file(feature_path)
                manifest = FrozenFeatureManifestV1(
                    backend_spec_sha256=spec.sha256(),
                    processor_sha256=processor_sha256,
                    prompt_renderer_sha256=_sha256_bytes(instruction.encode("utf-8")),
                    image_sha256=_sha256_bytes(image.tobytes(order="C")),
                    sample_id=identity,
                    episode_id=str(episode.episode_id),
                    step_index=observation.step_index,
                    split=split,
                    task_id=task_id,
                    held_out_stratum=stratum,
                    hidden_layer=-1,
                    pooling="attention_mask_mean",
                    dtype="float32",
                    device_environment_sha256=device_environment_sha256,
                    output_shape=(extractor.output_dim,),
                    finite=True,
                    feature_sha256=feature_hash,
                    deterministic=spec.deterministic,
                    generation_command=("lunavla-v3", "vlm-cache"),
                )
                _write_json(manifest_dir / f"{ordinal:08d}.json", manifest.to_dict())
                manifest_hashes.append(manifest.sha256())
                identities.append(identity)
                split_counts[split] += 1
                total_bytes += feature_path.stat().st_size
        index = FeatureCacheIndexV1(
            backend_spec_sha256=spec.sha256(),
            manifest_hashes=tuple(manifest_hashes),
            expected_identities=tuple(identities),
            observed_identities=tuple(identities),
            task_ids=tuple(
                dict.fromkeys(str(item.metadata["task_id"]) for item in dataset.bundle.episodes)
            ),
            held_out_strata=("train", "composition", "paraphrase"),
            split_counts=split_counts,
            total_feature_bytes=total_bytes,
        )
        _write_json(staging / "backend-spec.json", spec.to_dict())
        _write_json(staging / "cache-index.json", index.to_dict())
        verify_frozen_feature_cache(staging)
        if output.exists():
            backup = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
            output.replace(backup)
            staging.replace(output)
            shutil.rmtree(backup)
        else:
            staging.replace(output)
        return output
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_frozen_feature_cache(root: str | Path) -> FeatureCacheIndexV1:
    cache = Path(root)
    spec = VLMBackendSpecV1.from_mapping(json.loads((cache / "backend-spec.json").read_text()))
    index = FeatureCacheIndexV1.from_mapping(json.loads((cache / "cache-index.json").read_text()))
    if index.backend_spec_sha256 != spec.sha256():
        raise ValueError("cache index backend hash mismatch")
    manifests = sorted((cache / "manifests").glob("*.json"))
    features = sorted((cache / "features").glob("*.npy"))
    if len(manifests) != len(index.manifest_hashes) or len(features) != len(manifests):
        raise ValueError("cache matrix is incomplete")
    expected_names = [f"{position:08d}" for position in range(len(manifests))]
    if [path.stem for path in manifests] != expected_names or [
        path.stem for path in features
    ] != expected_names:
        raise ValueError("cache filenames are not a contiguous paired inventory")
    observed_identities: list[str] = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    total_bytes = 0
    for position, (manifest_path, feature_path) in enumerate(zip(manifests, features, strict=True)):
        manifest = FrozenFeatureManifestV1.from_mapping(json.loads(manifest_path.read_text()))
        if manifest.sha256() != index.manifest_hashes[position]:
            raise ValueError("feature manifest hash mismatch")
        if _sha256_file(feature_path) != manifest.feature_sha256:
            raise ValueError("feature content hash mismatch")
        feature = np.load(feature_path, allow_pickle=False)
        if (
            tuple(feature.shape) != manifest.output_shape
            or feature.dtype != np.float32
            or not np.all(np.isfinite(feature))
        ):
            raise ValueError("cached feature shape, dtype, or finite check failed")
        observed_identities.append(manifest.sample_id)
        split_counts[manifest.split] += 1
        total_bytes += feature_path.stat().st_size
    if tuple(observed_identities) != index.observed_identities:
        raise ValueError("cache identity order mismatch")
    if split_counts != dict(index.split_counts) or total_bytes != index.total_feature_bytes:
        raise ValueError("cache count or byte inventory mismatch")
    return index


def run_qwen_observational_smoke(
    dataset: V31TaskDataset,
    spec: VLMBackendSpecV1,
    extractor: FrozenFeatureExtractor,
) -> VLMConnectivitySmokeV1:
    if spec.backend_id != "qwen3_vl_2b" or spec.evidence_role != "observational":
        raise ValueError("Qwen smoke requires the pinned observational backend")
    hashes: list[str] = []
    tasks: list[str] = []
    strata: list[str] = []
    for task_id in ("direct_pick_place", "waypoint_sequence", "failure_recovery"):
        for stratum in ("composition", "paraphrase"):
            candidates = [
                item
                for item in dataset.bundle.select("test")
                if item.metadata["task_id"] == task_id
                and item.metadata["held_out_stratum"] == stratum
            ]
            if not candidates:
                raise ValueError("Qwen smoke dataset is incomplete")
            for episode in candidates[:2]:
                observation = episode.transitions[0].observation
                image = cast(npt.NDArray[np.uint8], observation.images["camera.primary"])
                feature = extractor.extract(image, observation.instruction or "")
                if not np.all(np.isfinite(feature)):
                    raise ValueError("Qwen smoke produced non-finite feature")
                hashes.append(_sha256_bytes(np.asarray(feature, dtype=np.float32).tobytes()))
                tasks.append(task_id)
                strata.append(stratum)
    return VLMConnectivitySmokeV1(
        backend_spec_sha256=spec.sha256(),
        rows=12,
        task_ids=tuple(tasks),
        strata=tuple(strata),
        feature_hashes=tuple(hashes),
        finite=True,
        observational=True,
        claim_allowed=False,
    )
