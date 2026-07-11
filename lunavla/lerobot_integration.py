from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Final, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from lunavla.contracts import PolicyBatch
from lunavla.lerobot_adapter import (
    LEROBOT_PUSHT_ACTION_SHAPE,
    LEROBOT_PUSHT_EPISODES,
    LEROBOT_PUSHT_IMAGE_SHAPE,
    LEROBOT_PUSHT_REPO_ID,
    LEROBOT_PUSHT_REVISION,
    LEROBOT_PUSHT_STATE_SHAPE,
    LEROBOT_PUSHT_VIDEO_BACKEND,
    LeRobotDatasetSource,
    map_lerobot_sample,
)
from lunavla.pusht_env_adapter import PUSHT_ENV_ID, PUSHT_OBS_TYPE


INTEGRATION_MANIFEST_SCHEMA_VERSION: Final = 1
INTEGRATION_ID: Final = "lerobot_pusht_v3_beta1"
OFFICIAL_REPO_ID: Final = LEROBOT_PUSHT_REPO_ID
OFFICIAL_REVISION: Final = LEROBOT_PUSHT_REVISION
OFFICIAL_EPISODE: Final = LEROBOT_PUSHT_EPISODES[0]
OFFICIAL_VIDEO_BACKEND: Final = LEROBOT_PUSHT_VIDEO_BACKEND
OFFICIAL_RETURN_UINT8: Final = True
MAX_DOWNLOAD_BYTES: Final = 12 * 1024 * 1024
EXPECTED_PLANNED_DOWNLOAD_BYTES: Final = 7_686_728
EXPECTED_FRAME_COUNT: Final = 161
EXPECTED_IMAGE_SHAPE: Final = LEROBOT_PUSHT_IMAGE_SHAPE
EXPECTED_STATE_SHAPE: Final = LEROBOT_PUSHT_STATE_SHAPE
EXPECTED_ACTION_SHAPE: Final = LEROBOT_PUSHT_ACTION_SHAPE
EXPECTED_TERMINAL_FRAME_INDICES: Final = (159, 160)
EXPECTED_NEXT_OBSERVATION_BOUNDARY: Final = (
    "next frame while active; terminal frames self-reference without crossing episodes"
)
ENV_ID: Final = PUSHT_ENV_ID
ENV_OBS_TYPE: Final = PUSHT_OBS_TYPE
ENV_SEED: Final = 202611
ENV_STEPS: Final = 3
EXPECTED_LUNAVLA_VERSION: Final = "2.0.0"
CLAIM_SCOPE: Final = (
    "This smoke verifies adapter connectivity only and does not establish "
    "PushT policy performance."
)
EXPECTED_POLICY_CONFIG: Final = MappingProxyType(
    {
        "state_dim": 2,
        "action_dim": 2,
        "chunk_size": 1,
        "d_model": 16,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "latent_dim": 4,
        "instruction_dim": 16,
        "image_shape": EXPECTED_IMAGE_SHAPE,
        "dropout": 0.0,
        "kl_weight": 0.01,
        "learning_rate": 3e-4,
    }
)
EXPECTED_DEPENDENCY_VERSIONS: Final = MappingProxyType(
    {
        "numpy": "2.2.6",
        "torch": "2.11.0+cpu",
        "torchvision": "0.26.0+cpu",
        "lerobot": "0.6.0",
        "huggingface-hub": "1.23.0",
        "av": "15.1.0",
        "gym-pusht": "0.1.6",
        "gymnasium": "1.3.0",
        "pymunk": "6.11.1",
    }
)
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")


def _exact_mapping(
    value: object,
    *,
    name: str,
    fields: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    unknown = sorted(repr(field) for field in set(result) - fields)
    missing = sorted(fields - set(result))
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")
    if missing:
        raise ValueError(f"missing {name} fields: {missing}")
    return result


def _require_expected(value: object, expected: object, *, name: str) -> None:
    """Compare a JSON value without allowing booleans to masquerade as integers."""

    if isinstance(expected, bool):
        if value is not expected:
            raise ValueError(f"{name} must equal {expected!r}")
        return
    if isinstance(expected, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value != expected:
            raise ValueError(f"{name} must equal {expected!r}")
        return
    if isinstance(expected, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) != expected:
            raise ValueError(f"{name} must equal {expected!r}")
        return
    if isinstance(expected, tuple):
        if not isinstance(value, tuple):
            raise TypeError(f"{name} must be a sequence")
        if len(value) != len(expected):
            raise ValueError(f"{name} must equal {expected!r}")
        for index, (item, expected_item) in enumerate(zip(value, expected)):
            _require_expected(item, expected_item, name=f"{name}[{index}]")
        return
    if not isinstance(value, type(expected)) or value != expected:
        raise ValueError(f"{name} must equal {expected!r}")


@dataclass(frozen=True)
class SourceFileContract:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith("/") or ".." in Path(self.path).parts:
            raise ValueError("source path must be a repository-relative safe path")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("source size must be a positive integer")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("source sha256 must be 64 lowercase hexadecimal characters")


OFFICIAL_SOURCE_FILES: Final = (
    SourceFileContract(
        path="data/chunk-000/file-000.parquet",
        size=674_393,
        sha256="9abc0431f12d10c33b5b37b1bf1e6e557c8ae1c8c29fb86658969e48fcbbbf01",
    ),
    SourceFileContract(
        path="videos/observation.image/chunk-000/file-000.mp4",
        size=6_890_970,
        sha256="f58d11857651dad5983c021b56a50a68a6ce19068834c1fb5cac099219fb3a78",
    ),
    SourceFileContract(
        path="meta/episodes/chunk-000/file-000.parquet",
        size=106_584,
        sha256="bc1226f33d3d1635ec1954f9942073709be8e106fde5f4f16b9d52edc4e0ebc4",
    ),
)


@dataclass(frozen=True)
class DownloadPreflight:
    repo_id: str
    requested_revision: str
    resolved_revision: str
    planned_download_bytes: int
    max_download_bytes: int
    source_files: tuple[SourceFileContract, ...]


@dataclass(frozen=True)
class DatasetValidation:
    frame_count: int
    image_shape: tuple[int, int, int]
    image_dtype: str
    state_shape: tuple[int, ...]
    state_dtype: str
    action_shape: tuple[int, ...]
    action_dtype: str
    episode_indices: tuple[int, ...]
    frame_index_start: int
    frame_index_end: int
    terminal_frame_indices: tuple[int, ...]
    next_observation_boundary: str


@dataclass(frozen=True)
class OptimizerStepValidation:
    policy_id: str
    device: str
    batch_size: int
    steps: int
    loss: float
    parameters_changed: bool
    policy_config: Mapping[str, object]


@dataclass(frozen=True)
class EnvironmentValidation:
    env_id: str
    obs_type: str
    seed: int
    steps: int
    pixel_shape: tuple[int, int, int]
    pixel_dtype: str
    agent_position_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    action_dtype: str
    close_completed: bool


@dataclass(frozen=True)
class IntegrationManifest:
    schema_version: int
    integration_id: str
    generated_at_utc: str
    git_sha: str
    git_dirty: bool
    source: Mapping[str, object]
    dataset_validation: Mapping[str, object]
    optimizer_step: Mapping[str, object]
    environment_smoke: Mapping[str, object]
    dependencies: Mapping[str, str]
    python: str
    device: str
    deterministic: bool
    artifact_policy: Mapping[str, object]
    claim_allowed: bool
    claim_scope: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != INTEGRATION_MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported integration manifest schema_version: "
                f"{self.schema_version!r}"
            )
        if self.integration_id != INTEGRATION_ID:
            raise ValueError(f"unexpected integration_id: {self.integration_id!r}")
        if not _GIT_SHA_PATTERN.fullmatch(self.git_sha):
            raise ValueError("git_sha must be a full 40-character lowercase commit SHA")
        if not isinstance(self.git_dirty, bool):
            raise TypeError("git_dirty must be boolean")
        if self.git_dirty:
            raise ValueError("controlled integration manifests require a clean Git checkout")
        if self.device != "cpu":
            raise ValueError("the authoritative LeRobot integration device must be cpu")
        if not isinstance(self.claim_allowed, bool):
            raise TypeError("claim_allowed must be boolean")
        if self.claim_allowed:
            raise ValueError("LeRobot integration smoke cannot authorize a performance claim")
        if self.deterministic is not True:
            raise ValueError("the bounded CPU integration must report deterministic=true")
        if not isinstance(self.generated_at_utc, str):
            raise TypeError("generated_at_utc must be a string")
        try:
            generated_at = datetime.fromisoformat(self.generated_at_utc)
        except ValueError as exc:
            raise ValueError(
                "generated_at_utc must be a valid ISO-8601 UTC timestamp"
            ) from exc
        if (
            generated_at.tzinfo is None
            or generated_at.utcoffset() != timedelta(0)
            or not self.generated_at_utc.endswith("+00:00")
            or generated_at.isoformat() != self.generated_at_utc
        ):
            raise ValueError("generated_at_utc must be a canonical ISO-8601 UTC timestamp")
        if self.claim_scope != CLAIM_SCOPE:
            raise ValueError("claim_scope does not match the fixed connectivity-only statement")

        source = _exact_mapping(
            self.source,
            name="source",
            fields={
                "repo_id",
                "revision",
                "episode",
                "video_backend",
                "return_uint8",
                "max_download_bytes",
                "planned_download_bytes",
                "sha256",
            },
        )
        source["sha256"] = _exact_mapping(
            source["sha256"],
            name="source.sha256",
            fields={contract.path for contract in OFFICIAL_SOURCE_FILES},
        )
        dataset_validation = _exact_mapping(
            self.dataset_validation,
            name="dataset_validation",
            fields={
                "frame_count",
                "image_shape",
                "image_dtype",
                "state_shape",
                "state_dtype",
                "action_shape",
                "action_dtype",
                "episode_indices",
                "frame_index_start",
                "frame_index_end",
                "terminal_frame_indices",
                "next_observation_boundary",
            },
        )
        for name in (
            "image_shape",
            "state_shape",
            "action_shape",
            "episode_indices",
            "terminal_frame_indices",
        ):
            sequence_value = dataset_validation.get(name)
            if isinstance(sequence_value, (list, tuple)):
                dataset_validation[name] = tuple(sequence_value)
        optimizer_step = _exact_mapping(
            self.optimizer_step,
            name="optimizer_step",
            fields={
                "policy_id",
                "device",
                "batch_size",
                "steps",
                "loss",
                "parameters_changed",
                "policy_config",
            },
        )
        policy_config = _exact_mapping(
            optimizer_step["policy_config"],
            name="optimizer_step.policy_config",
            fields=set(EXPECTED_POLICY_CONFIG),
        )
        if isinstance(policy_config.get("image_shape"), (list, tuple)):
            policy_config["image_shape"] = tuple(cast(Sequence[int], policy_config["image_shape"]))
        optimizer_step["policy_config"] = policy_config
        environment_smoke = _exact_mapping(
            self.environment_smoke,
            name="environment_smoke",
            fields={
                "env_id",
                "obs_type",
                "seed",
                "steps",
                "pixel_shape",
                "pixel_dtype",
                "agent_position_shape",
                "action_shape",
                "action_dtype",
                "close_completed",
            },
        )
        for name in ("pixel_shape", "agent_position_shape", "action_shape"):
            sequence_value = environment_smoke.get(name)
            if isinstance(sequence_value, (list, tuple)):
                environment_smoke[name] = tuple(sequence_value)
        dependencies = _exact_mapping(
            self.dependencies,
            name="dependencies",
            fields={"lunavla", *EXPECTED_DEPENDENCY_VERSIONS},
        )
        artifact_policy = _exact_mapping(
            self.artifact_policy,
            name="artifact_policy",
            fields={"manifest_only", "cache_uploaded", "video_uploaded"},
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "dataset_validation", dataset_validation)
        object.__setattr__(self, "optimizer_step", optimizer_step)
        object.__setattr__(self, "environment_smoke", environment_smoke)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "artifact_policy", artifact_policy)
        expected_source = {
            "repo_id": OFFICIAL_REPO_ID,
            "revision": OFFICIAL_REVISION,
            "episode": OFFICIAL_EPISODE,
            "video_backend": OFFICIAL_VIDEO_BACKEND,
            "return_uint8": OFFICIAL_RETURN_UINT8,
        }
        for name, expected in expected_source.items():
            _require_expected(source.get(name), expected, name=f"source.{name}")
        _require_expected(
            source.get("max_download_bytes"),
            MAX_DOWNLOAD_BYTES,
            name="source.max_download_bytes",
        )
        planned = source["planned_download_bytes"]
        _require_expected(
            planned,
            EXPECTED_PLANNED_DOWNLOAD_BYTES,
            name="source.planned_download_bytes",
        )
        expected_hashes = {contract.path: contract.sha256 for contract in OFFICIAL_SOURCE_FILES}
        if source.get("sha256") != expected_hashes:
            raise ValueError("source.sha256 does not match the pinned upstream files")
        validation = dict(self.dataset_validation)
        expected_dataset_values = {
            "frame_count": EXPECTED_FRAME_COUNT,
            "image_shape": EXPECTED_IMAGE_SHAPE,
            "image_dtype": "uint8",
            "state_shape": EXPECTED_STATE_SHAPE,
            "state_dtype": "float32",
            "action_shape": EXPECTED_ACTION_SHAPE,
            "action_dtype": "float32",
            "episode_indices": (OFFICIAL_EPISODE,),
            "frame_index_start": 0,
            "frame_index_end": EXPECTED_FRAME_COUNT - 1,
            "terminal_frame_indices": EXPECTED_TERMINAL_FRAME_INDICES,
            "next_observation_boundary": EXPECTED_NEXT_OBSERVATION_BOUNDARY,
        }
        for name, expected in expected_dataset_values.items():
            _require_expected(
                validation.get(name),
                expected,
                name=f"dataset_validation.{name}",
            )
        if optimizer_step["parameters_changed"] is not True:
            raise ValueError("optimizer_step must update at least one Transformer parameter")
        loss = optimizer_step["loss"]
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise ValueError("optimizer_step.loss must be finite")
        expected_optimizer = {
            "policy_id": "transformer_chunk_cvae",
            "device": "cpu",
            "batch_size": 2,
            "steps": 1,
        }
        for name, expected in expected_optimizer.items():
            _require_expected(
                optimizer_step.get(name),
                expected,
                name=f"optimizer_step.{name}",
            )
        for name, expected in EXPECTED_POLICY_CONFIG.items():
            _require_expected(
                policy_config.get(name),
                expected,
                name=f"optimizer_step.policy_config.{name}",
            )
        if environment_smoke.get("close_completed") is not True:
            raise ValueError("environment_smoke must close its resources")
        expected_environment = {
            "env_id": ENV_ID,
            "obs_type": ENV_OBS_TYPE,
            "seed": ENV_SEED,
            "steps": ENV_STEPS,
            "pixel_shape": EXPECTED_IMAGE_SHAPE,
            "pixel_dtype": "uint8",
            "agent_position_shape": EXPECTED_STATE_SHAPE,
            "action_shape": EXPECTED_ACTION_SHAPE,
            "action_dtype": "float32",
        }
        for name, expected in expected_environment.items():
            _require_expected(
                environment_smoke.get(name),
                expected,
                name=f"environment_smoke.{name}",
            )
        if not re.fullmatch(r"3\.12\.[0-9]+", self.python):
            raise ValueError("the integration Python version must be 3.12.x")
        if dependencies.get("lunavla") != EXPECTED_LUNAVLA_VERSION:
            raise ValueError(
                f"dependencies.lunavla must equal {EXPECTED_LUNAVLA_VERSION!r}"
            )
        for name, expected in EXPECTED_DEPENDENCY_VERSIONS.items():
            _require_expected(
                dependencies.get(name),
                expected,
                name=f"dependencies.{name}",
            )
        if artifact_policy.get("manifest_only") is not True:
            raise ValueError("integration artifact policy must be manifest-only")
        if artifact_policy.get("cache_uploaded") is not False:
            raise ValueError("integration caches must not be uploaded")
        if artifact_policy.get("video_uploaded") is not False:
            raise ValueError("integration videos must not be uploaded")

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], _json_safe(asdict(self), name="manifest"))
        return payload

    def write(self, path: Path) -> Path:
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite integration manifest: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: Path) -> "IntegrationManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("integration manifest must contain a JSON object")
        expected = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - expected)
        missing = sorted(expected - set(payload))
        if unknown:
            raise ValueError(f"unknown integration manifest fields: {unknown}")
        if missing:
            raise ValueError(f"missing integration manifest fields: {missing}")
        return cls(**dict(payload))


def _json_safe(value: Any, *, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite values")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} mapping keys must be strings")
            result[key] = _json_safe(item, name=f"{name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{name} contains unsupported value type {type(value).__name__}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote_size(sibling: Any) -> int:
    size = getattr(sibling, "size", None)
    if size is None:
        size = getattr(getattr(sibling, "lfs", None), "size", None)
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"Hub file {getattr(sibling, 'rfilename', None)!r} has no valid size")
    return size


def preflight_official_download(
    *,
    api: Any | None = None,
    source_files: Sequence[SourceFileContract] = OFFICIAL_SOURCE_FILES,
    max_download_bytes: int = MAX_DOWNLOAD_BYTES,
) -> DownloadPreflight:
    """Validate the pinned Hub tree before downloading any dataset payload."""

    if isinstance(max_download_bytes, bool) or not isinstance(max_download_bytes, int):
        raise TypeError("max_download_bytes must be an integer")
    if max_download_bytes <= 0 or max_download_bytes > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"max_download_bytes must be in [1, {MAX_DOWNLOAD_BYTES}]")
    if api is None:
        hub = importlib.import_module("huggingface_hub")
        api = hub.HfApi()
    info = api.dataset_info(
        OFFICIAL_REPO_ID,
        revision=OFFICIAL_REVISION,
        files_metadata=True,
    )
    resolved_revision = str(getattr(info, "sha", ""))
    if resolved_revision != OFFICIAL_REVISION:
        raise ValueError(
            f"Hub resolved {resolved_revision!r}, expected pinned revision {OFFICIAL_REVISION}"
        )
    siblings = tuple(getattr(info, "siblings", ()))
    by_path = {str(getattr(item, "rfilename", "")): item for item in siblings}
    for contract in source_files:
        if contract.path not in by_path:
            raise FileNotFoundError(f"pinned Hub revision is missing {contract.path}")
        sibling = by_path[contract.path]
        actual_size = _remote_size(sibling)
        if actual_size != contract.size:
            raise ValueError(
                f"Hub size mismatch for {contract.path}: {actual_size} != {contract.size}"
            )
        lfs_sha256 = str(getattr(getattr(sibling, "lfs", None), "sha256", ""))
        if lfs_sha256 != contract.sha256:
            raise ValueError(f"Hub SHA-256 mismatch for {contract.path}")
    planned_download_bytes = sum(_remote_size(sibling) for sibling in siblings)
    if planned_download_bytes > max_download_bytes:
        raise ValueError(
            "pinned dataset exceeds the download limit: "
            f"{planned_download_bytes} > {max_download_bytes} bytes"
        )
    if planned_download_bytes != EXPECTED_PLANNED_DOWNLOAD_BYTES:
        raise ValueError(
            "pinned Hub tree size changed: "
            f"{planned_download_bytes} != {EXPECTED_PLANNED_DOWNLOAD_BYTES} bytes"
        )
    return DownloadPreflight(
        repo_id=OFFICIAL_REPO_ID,
        requested_revision=OFFICIAL_REVISION,
        resolved_revision=resolved_revision,
        planned_download_bytes=planned_download_bytes,
        max_download_bytes=max_download_bytes,
        source_files=tuple(source_files),
    )


def verify_downloaded_source_files(
    root: Path,
    *,
    source_files: Sequence[SourceFileContract] = OFFICIAL_SOURCE_FILES,
) -> Mapping[str, str]:
    """Hash the materialized parquet, video, and episode metadata files."""

    root = Path(root)
    verified: dict[str, str] = {}
    for contract in source_files:
        path = root / contract.path
        if not path.is_file():
            raise FileNotFoundError(f"downloaded dataset is missing {contract.path}")
        actual_size = path.stat().st_size
        if actual_size != contract.size:
            raise ValueError(
                f"downloaded size mismatch for {contract.path}: "
                f"{actual_size} != {contract.size}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != contract.sha256:
            raise ValueError(f"downloaded SHA-256 mismatch for {contract.path}")
        verified[contract.path] = actual_sha256
    return verified


def load_official_dataset(
    root: Path,
    *,
    dataset_factory: Callable[..., Any] | None = None,
) -> Any:
    if dataset_factory is None:
        module = importlib.import_module("lerobot.datasets.lerobot_dataset")
        dataset_factory = module.LeRobotDataset
    return dataset_factory(
        OFFICIAL_REPO_ID,
        root=Path(root),
        episodes=[OFFICIAL_EPISODE],
        revision=OFFICIAL_REVISION,
        video_backend=OFFICIAL_VIDEO_BACKEND,
        return_uint8=OFFICIAL_RETURN_UINT8,
    )


def materialize_episode(dataset: Any) -> tuple[Mapping[str, Any], ...]:
    length = len(dataset)
    if length != EXPECTED_FRAME_COUNT:
        raise ValueError(f"episode 0 must contain {EXPECTED_FRAME_COUNT} frames; found {length}")
    samples: list[Mapping[str, Any]] = []
    for index in range(length):
        sample = dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError(f"dataset frame {index} must be a mapping")
        samples.append(sample)
    return tuple(samples)


def _numpy(value: Any, *, name: str) -> NDArray[np.generic]:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    array = np.asarray(current)
    if array.dtype == np.dtype("O"):
        raise TypeError(f"{name} must not have object dtype")
    return array


def _scalar(value: Any, *, name: str) -> Any:
    array = _numpy(value, name=name)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    scalar = array.reshape(-1)[0]
    return scalar.item() if isinstance(scalar, np.generic) else scalar


def _same_observation(left: Any, right: Any) -> bool:
    return (
        np.array_equal(left.state, right.state)
        and left.instruction == right.instruction
        and np.array_equal(left.image, right.image)
    )


def validate_official_episode(
    samples: Sequence[Mapping[str, Any]],
) -> DatasetValidation:
    if len(samples) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            f"episode 0 must contain {EXPECTED_FRAME_COUNT} frames; found {len(samples)}"
        )
    mapped = []
    terminal_indices: list[int] = []
    for index, sample in enumerate(samples):
        required = (
            "observation.image",
            "observation.state",
            "action",
            "episode_index",
            "frame_index",
            "next.done",
        )
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"dataset frame {index} is missing required fields: {missing}")
        raw_image = _numpy(sample["observation.image"], name="observation.image")
        if raw_image.dtype != np.uint8:
            raise TypeError(f"frame {index} image must have dtype uint8; got {raw_image.dtype}")
        if raw_image.shape not in {(3, 96, 96), EXPECTED_IMAGE_SHAPE}:
            raise ValueError(f"frame {index} image has unexpected shape {raw_image.shape}")
        raw_state = _numpy(sample["observation.state"], name="observation.state")
        raw_action = _numpy(sample["action"], name="action")
        if raw_state.dtype != np.float32 or raw_state.shape != EXPECTED_STATE_SHAPE:
            raise TypeError(
                f"frame {index} state must be float32{EXPECTED_STATE_SHAPE}; "
                f"got {raw_state.dtype}{raw_state.shape}"
            )
        if raw_action.dtype != np.float32 or raw_action.shape != EXPECTED_ACTION_SHAPE:
            raise TypeError(
                f"frame {index} action must be float32{EXPECTED_ACTION_SHAPE}; "
                f"got {raw_action.dtype}{raw_action.shape}"
            )
        if not np.all(np.isfinite(raw_state)) or not np.all(np.isfinite(raw_action)):
            raise ValueError(f"frame {index} state/action contains NaN or Inf")
        episode_index = _scalar(sample["episode_index"], name="episode_index")
        frame_index = _scalar(sample["frame_index"], name="frame_index")
        done = _scalar(sample["next.done"], name="next.done")
        if isinstance(episode_index, bool) or episode_index != OFFICIAL_EPISODE:
            raise ValueError(f"frame {index} has unexpected episode_index {episode_index!r}")
        if isinstance(frame_index, bool) or frame_index != index:
            raise ValueError(f"frame sequence is discontinuous at index {index}: {frame_index!r}")
        if not isinstance(done, (bool, np.bool_)):
            raise TypeError(f"frame {index} next.done must be boolean")
        if bool(done):
            terminal_indices.append(index)
        current = map_lerobot_sample(
            sample,
            require_image=True,
            repo_id=OFFICIAL_REPO_ID,
            index=index,
        )
        image = current.observation.image
        if image is None or image.shape != EXPECTED_IMAGE_SHAPE or image.dtype != np.uint8:
            raise ValueError(f"mapped frame {index} must contain a 96x96x3 uint8 image")
        if current.observation.state.dtype != np.float32:
            raise TypeError(f"mapped frame {index} state must preserve float32")
        if current.action.dtype != np.float32:
            raise TypeError(f"mapped frame {index} action must preserve float32")
        mapped.append(current)
    if tuple(terminal_indices) != EXPECTED_TERMINAL_FRAME_INDICES:
        raise ValueError(
            "episode terminal flags changed: "
            f"{tuple(terminal_indices)} != {EXPECTED_TERMINAL_FRAME_INDICES}"
        )

    materialized = tuple(samples)

    def dataset_factory(_repo_id: str, **_kwargs: object) -> Sequence[Mapping[str, Any]]:
        return materialized

    transitions = LeRobotDatasetSource(
        OFFICIAL_REPO_ID,
        episodes=[OFFICIAL_EPISODE],
        require_image=True,
        dataset_factory=dataset_factory,
    ).load()
    if len(transitions) != EXPECTED_FRAME_COUNT:
        raise ValueError("adapter changed the episode frame count")
    for index, transition in enumerate(transitions):
        expected_terminated = index in EXPECTED_TERMINAL_FRAME_INDICES
        if transition.terminated != expected_terminated:
            raise ValueError(f"adapter terminal mismatch at frame {index}")
        has_next = not expected_terminated and index + 1 < len(mapped)
        expected_next = mapped[index + 1].observation if has_next else mapped[index].observation
        if not _same_observation(transition.next_observation, expected_next):
            raise ValueError(f"adapter next_observation crosses the episode boundary at {index}")
        expected_source = "next_frame" if has_next else "terminal_self"
        if transition.info.get("next_observation_source") != expected_source:
            raise ValueError(f"adapter next_observation source mismatch at frame {index}")

    return DatasetValidation(
        frame_count=EXPECTED_FRAME_COUNT,
        image_shape=EXPECTED_IMAGE_SHAPE,
        image_dtype="uint8",
        state_shape=EXPECTED_STATE_SHAPE,
        state_dtype="float32",
        action_shape=EXPECTED_ACTION_SHAPE,
        action_dtype="float32",
        episode_indices=(OFFICIAL_EPISODE,),
        frame_index_start=0,
        frame_index_end=EXPECTED_FRAME_COUNT - 1,
        terminal_frame_indices=EXPECTED_TERMINAL_FRAME_INDICES,
        next_observation_boundary=EXPECTED_NEXT_OBSERVATION_BOUNDARY,
    )


def run_transformer_optimizer_step(
    samples: Sequence[Mapping[str, Any]],
    *,
    learning_rate: float = 3e-4,
) -> OptimizerStepValidation:
    """Run exactly one bounded CPU optimizer step on two official-format frames."""

    if len(samples) < 2:
        raise ValueError("optimizer smoke requires at least two dataset frames")
    from lunavla.transformer_policy import (
        TransformerChunkCVAEPolicy,
        TransformerPolicyConfig,
    )

    mapped = tuple(
        map_lerobot_sample(
            samples[index],
            require_image=True,
            repo_id=OFFICIAL_REPO_ID,
            index=index,
        )
        for index in range(2)
    )
    policy_config = TransformerPolicyConfig(
        state_dim=2,
        action_dim=2,
        chunk_size=1,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        latent_dim=4,
        instruction_dim=16,
        image_shape=EXPECTED_IMAGE_SHAPE,
        dropout=0.0,
        kl_weight=0.01,
        sample_latent_during_training=False,
        seed=202611,
        device="cpu",
    )
    policy = TransformerChunkCVAEPolicy(policy_config)
    batch = PolicyBatch(
        observations=tuple(item.observation for item in mapped),
        targets=np.stack([item.action for item in mapped])[:, None, :],
        valid_mask=np.ones((len(mapped), 1), dtype=bool),
        device="cpu",
    )
    before = tuple(parameter.detach().clone() for parameter in policy.parameters())
    loss = policy.train_batch(batch, learning_rate=learning_rate)
    if not math.isfinite(loss):
        raise FloatingPointError("Transformer integration loss is NaN or infinite")
    parameters_changed = any(
        not bool(np.array_equal(
            previous.detach().cpu().numpy(),
            current.detach().cpu().numpy(),
        ))
        for previous, current in zip(before, policy.parameters())
    )
    if not parameters_changed:
        raise RuntimeError("bounded Transformer optimizer step did not update any parameter")
    return OptimizerStepValidation(
        policy_id=policy.policy_id,
        device=policy.device,
        batch_size=len(mapped),
        steps=1,
        loss=loss,
        parameters_changed=True,
        policy_config={**EXPECTED_POLICY_CONFIG, "learning_rate": learning_rate},
    )


def _validate_env_observation(observation: Any, *, context: str) -> None:
    state = np.asarray(getattr(observation, "state", None))
    image = np.asarray(getattr(observation, "image", None))
    if state.shape != EXPECTED_STATE_SHAPE or state.dtype != np.float32:
        raise ValueError(f"{context}.state must be float32{EXPECTED_STATE_SHAPE}")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{context}.state contains NaN or Inf")
    if image.shape != EXPECTED_IMAGE_SHAPE or image.dtype != np.uint8:
        raise ValueError(f"{context}.image must be 96x96x3 uint8")


def run_headless_pusht_smoke(
    *,
    env_factory: Callable[..., Any] | None = None,
    seed: int = ENV_SEED,
    steps: int = ENV_STEPS,
) -> EnvironmentValidation:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("environment seed must be a non-negative integer")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0 or steps > 10:
        raise ValueError("environment steps must be an integer in [1, 10]")
    if env_factory is None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        importlib.import_module("gym_pusht")
        gym = importlib.import_module("gymnasium")
        env_factory = gym.make
    from lunavla.pusht_env_adapter import PushTEnvAdapter

    def configured_factory(env_id: str, **kwargs: object) -> Any:
        return env_factory(env_id, render_mode="rgb_array", **kwargs)

    env = PushTEnvAdapter(env_factory=configured_factory)
    close_completed = False
    try:
        observation = env.reset(seed=seed)
        _validate_env_observation(observation, context="reset observation")
        for index in range(steps):
            action = np.asarray(
                [256.0 + 8.0 * index, 256.0 - 8.0 * index],
                dtype=np.float32,
            )
            transition = env.step(action)
            _validate_env_observation(
                transition.next_observation,
                context=f"step {index} observation",
            )
            if transition.action.dtype != np.float32 or transition.action.shape != EXPECTED_ACTION_SHAPE:
                raise ValueError(f"step {index} action mapping must preserve float32(2,)")
            if not math.isfinite(transition.reward):
                raise ValueError(f"step {index} reward is not finite")
    finally:
        env.close()
        close_completed = True
    return EnvironmentValidation(
        env_id=ENV_ID,
        obs_type=ENV_OBS_TYPE,
        seed=seed,
        steps=steps,
        pixel_shape=EXPECTED_IMAGE_SHAPE,
        pixel_dtype="uint8",
        agent_position_shape=EXPECTED_STATE_SHAPE,
        action_shape=EXPECTED_ACTION_SHAPE,
        action_dtype="float32",
        close_completed=close_completed,
    )


def dependency_versions() -> Mapping[str, str]:
    names = (
        "lunavla",
        "numpy",
        "torch",
        "torchvision",
        "lerobot",
        "huggingface-hub",
        "av",
        "gym-pusht",
        "gymnasium",
        "pymunk",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"required integration dependency is missing: {name}") from exc
    return versions


def _git_state(root: Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not _GIT_SHA_PATTERN.fullmatch(sha):
        raise RuntimeError(f"Git returned an invalid commit SHA: {sha!r}")
    return sha, bool(status.strip())


def _outside_checkout(path: Path, root: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return resolved
    raise ValueError(f"{name} must be outside the Git checkout: {resolved}")


def run_official_integration(
    *,
    root: Path,
    expected_git_sha: str,
    cache_dir: Path,
    output_path: Path,
    api: Any | None = None,
    dataset_factory: Callable[..., Any] | None = None,
    env_factory: Callable[..., Any] | None = None,
) -> IntegrationManifest:
    """Execute the pinned real-data and real-environment integration contract."""

    root = Path(root).resolve()
    if not _GIT_SHA_PATTERN.fullmatch(expected_git_sha):
        raise ValueError("expected_git_sha must be a full lowercase 40-character commit SHA")
    cache_dir = _outside_checkout(Path(cache_dir), root, name="cache_dir")
    output_path = _outside_checkout(Path(output_path), root, name="output_path")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite integration manifest: {output_path}")
    git_sha, dirty = _git_state(root)
    if git_sha != expected_git_sha:
        raise ValueError(f"checkout SHA {git_sha} does not match expected {expected_git_sha}")
    if dirty:
        raise ValueError("real integration requires a clean Git checkout")

    preflight = preflight_official_download(api=api)
    dataset = load_official_dataset(cache_dir, dataset_factory=dataset_factory)
    verified_hashes = verify_downloaded_source_files(cache_dir)
    samples = materialize_episode(dataset)
    dataset_validation = validate_official_episode(samples)
    optimizer_step = run_transformer_optimizer_step(samples)
    environment_smoke = run_headless_pusht_smoke(env_factory=env_factory)
    versions = dependency_versions()

    final_sha, final_dirty = _git_state(root)
    if final_sha != git_sha or final_dirty:
        raise ValueError("Git checkout changed while the real integration was running")
    manifest = IntegrationManifest(
        schema_version=INTEGRATION_MANIFEST_SCHEMA_VERSION,
        integration_id=INTEGRATION_ID,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha,
        git_dirty=False,
        source={
            "repo_id": OFFICIAL_REPO_ID,
            "revision": OFFICIAL_REVISION,
            "episode": OFFICIAL_EPISODE,
            "video_backend": OFFICIAL_VIDEO_BACKEND,
            "return_uint8": OFFICIAL_RETURN_UINT8,
            "max_download_bytes": MAX_DOWNLOAD_BYTES,
            "planned_download_bytes": preflight.planned_download_bytes,
            "sha256": dict(verified_hashes),
        },
        dataset_validation=asdict(dataset_validation),
        optimizer_step=asdict(optimizer_step),
        environment_smoke=asdict(environment_smoke),
        dependencies=dict(versions),
        python=platform.python_version(),
        device="cpu",
        deterministic=True,
        artifact_policy={
            "manifest_only": True,
            "cache_uploaded": False,
            "video_uploaded": False,
        },
        claim_allowed=False,
        claim_scope=CLAIM_SCOPE,
    )
    manifest.write(output_path)
    return manifest
