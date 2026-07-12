from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


LEROBOT_VERSION = "0.6.0"
LEROBOT_REVISION = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
PUSHT_REPO_ID = "lerobot/pusht"
PUSHT_REVISION = "b1c3ecbae7f244acc039a3dbc255a00dad1372b9"
LIBERO_REPO_ID = "lerobot/libero"
LIBERO_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
LIBERO_SPATIAL_TASK_IDS = (0, 1, 2, 3)
PUSHT_MAX_DOWNLOAD_BYTES = 12 * 1024 * 1024
LIBERO_MAX_DOWNLOAD_BYTES = 384 * 1024 * 1024
CONNECTIVITY_STATEMENT = "Real public data and simulation adapter paths are connected."

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing {name} field(s): {', '.join(missing)}")
    return dict(value)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _integers(value: Any, name: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(_integer(item, f"{name} item") for item in value)
    if not result and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


def _hash_mapping(value: Any, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    result: dict[str, str] = {}
    for path_text, digest in value.items():
        path = Path(path_text)
        if path.is_absolute() or path_text in {"", ".", ".."} or ".." in path.parts:
            raise ValueError(f"{name} contains an unsafe repository path")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"{name}.{path_text} must be a lowercase SHA-256")
        result[path.as_posix()] = digest
    return MappingProxyType(dict(sorted(result.items())))


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExternalDatasetSpecV1:
    repo_id: str
    revision: str
    license: str
    task_ids: tuple[int, ...]
    episodes: tuple[int, ...]
    episode_selection: str
    video_backend: str
    return_uint8: bool
    max_download_bytes: int
    file_hashes: Mapping[str, str]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ExternalDatasetSpecV1 schema_version must be 1")
        repo_id = _string(self.repo_id, "repo_id")
        revision = _string(self.revision, "revision")
        if not _GIT_SHA.fullmatch(revision):
            raise ValueError("revision must be a full lowercase Git SHA")
        license_id = _string(self.license, "license")
        task_ids = _integers(self.task_ids, "task_ids", allow_empty=True)
        episodes = _integers(self.episodes, "episodes", allow_empty=True)
        if self.episode_selection not in {"explicit", "minimum_per_task"}:
            raise ValueError("episode_selection must be explicit or minimum_per_task")
        if self.episode_selection == "explicit" and not episodes:
            raise ValueError("explicit episode selection requires episodes")
        if self.episode_selection == "minimum_per_task" and (not task_ids or episodes):
            raise ValueError("minimum_per_task requires task_ids and no explicit episodes")
        if self.video_backend != "pyav":
            raise ValueError("video_backend must be pyav")
        if not isinstance(self.return_uint8, bool) or self.return_uint8 is not True:
            raise ValueError("return_uint8 must be true")
        limit = _integer(self.max_download_bytes, "max_download_bytes", positive=True)
        hashes = _hash_mapping(self.file_hashes, "file_hashes")
        object.__setattr__(self, "repo_id", repo_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "license", license_id)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "max_download_bytes", limit)
        object.__setattr__(self, "file_hashes", hashes)

    def validate_supported_source(self) -> None:
        expected: tuple[str, str, tuple[int, ...], tuple[int, ...], str, int]
        if self.repo_id == PUSHT_REPO_ID:
            expected = (PUSHT_REVISION, "apache-2.0", (), (0,), "explicit", PUSHT_MAX_DOWNLOAD_BYTES)
        elif self.repo_id == LIBERO_REPO_ID:
            expected = (
                LIBERO_REVISION,
                "apache-2.0",
                LIBERO_SPATIAL_TASK_IDS,
                (),
                "minimum_per_task",
                LIBERO_MAX_DOWNLOAD_BYTES,
            )
        else:
            raise ValueError(f"unsupported external dataset repo_id {self.repo_id!r}")
        actual = (
            self.revision,
            self.license.lower(),
            self.task_ids,
            self.episodes,
            self.episode_selection,
            self.max_download_bytes,
        )
        if actual != expected:
            raise ValueError(f"external dataset contract does not match the pinned {self.repo_id} source")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "license": self.license,
            "task_ids": list(self.task_ids),
            "episodes": list(self.episodes),
            "episode_selection": self.episode_selection,
            "video_backend": self.video_backend,
            "return_uint8": self.return_uint8,
            "max_download_bytes": self.max_download_bytes,
            "file_hashes": dict(self.file_hashes),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExternalDatasetSpecV1":
        fields = {
            "schema_version", "repo_id", "revision", "license", "task_ids", "episodes",
            "episode_selection", "video_backend", "return_uint8", "max_download_bytes",
            "file_hashes",
        }
        payload = _exact(value, fields, "ExternalDatasetSpecV1")
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class SimulationTaskSpecV1:
    environment_id: str
    suite: str
    task_ids: tuple[int, ...]
    init_state_ids: tuple[int, ...]
    control_mode: str
    camera_mapping: Mapping[str, str]
    state_mapping: Mapping[str, str]
    action_mapping: Mapping[str, str]
    max_steps: int
    headless: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("SimulationTaskSpecV1 schema_version must be 1")
        environment_id = _string(self.environment_id, "environment_id")
        suite = _string(self.suite, "suite")
        task_ids = _integers(self.task_ids, "task_ids", allow_empty=True)
        init_ids = _integers(self.init_state_ids, "init_state_ids")
        if self.control_mode not in {"absolute", "relative"}:
            raise ValueError("control_mode must be absolute or relative")
        mappings: list[Mapping[str, str]] = []
        for name, value in (
            ("camera_mapping", self.camera_mapping),
            ("state_mapping", self.state_mapping),
            ("action_mapping", self.action_mapping),
        ):
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{name} must be a non-empty mapping")
            result = {
                _string(key, f"{name} key"): _string(item, f"{name} value")
                for key, item in value.items()
            }
            mappings.append(MappingProxyType(result))
        if not isinstance(self.headless, bool):
            raise TypeError("headless must be boolean")
        object.__setattr__(self, "environment_id", environment_id)
        object.__setattr__(self, "suite", suite)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "init_state_ids", init_ids)
        object.__setattr__(self, "camera_mapping", mappings[0])
        object.__setattr__(self, "state_mapping", mappings[1])
        object.__setattr__(self, "action_mapping", mappings[2])
        object.__setattr__(self, "max_steps", _integer(self.max_steps, "max_steps", positive=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment_id": self.environment_id,
            "suite": self.suite,
            "task_ids": list(self.task_ids),
            "init_state_ids": list(self.init_state_ids),
            "control_mode": self.control_mode,
            "camera_mapping": dict(self.camera_mapping),
            "state_mapping": dict(self.state_mapping),
            "action_mapping": dict(self.action_mapping),
            "max_steps": self.max_steps,
            "headless": self.headless,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SimulationTaskSpecV1":
        fields = {
            "schema_version", "environment_id", "suite", "task_ids", "init_state_ids",
            "control_mode", "camera_mapping", "state_mapping", "action_mapping", "max_steps",
            "headless",
        }
        return cls(**_exact(value, fields, "SimulationTaskSpecV1"))

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class IntegrationManifestV1:
    git_sha: str
    git_dirty: bool
    config_sha256: str
    dependency_lock_sha256: str
    source_spec_sha256: str
    source_inventory_sha256: str
    runner_qualification_sha256: str
    metrics_sha256: str
    runner_role: str
    data_validation: Mapping[str, Any]
    environment_validation: Mapping[str, Any]
    policy_smokes: tuple[Mapping[str, Any], ...]
    downloaded_bytes: int
    claim_allowed: bool
    benchmark_claim: bool
    statement: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("IntegrationManifestV1 schema_version must be 1")
        if not _GIT_SHA.fullmatch(self.git_sha):
            raise ValueError("git_sha must be a full lowercase Git SHA")
        if not isinstance(self.git_dirty, bool):
            raise TypeError("git_dirty must be boolean")
        for name in (
            "config_sha256", "dependency_lock_sha256", "source_spec_sha256",
            "source_inventory_sha256", "runner_qualification_sha256", "metrics_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.runner_role not in {"authoritative", "secondary", "fixture"}:
            raise ValueError("runner_role must be authoritative, secondary, or fixture")
        if isinstance(self.downloaded_bytes, bool) or not isinstance(self.downloaded_bytes, int):
            raise TypeError("downloaded_bytes must be an integer")
        if self.downloaded_bytes < 0:
            raise ValueError("downloaded_bytes must be non-negative")
        if self.claim_allowed is not False or self.benchmark_claim is not False:
            raise ValueError("Beta 2 integration cannot open scientific or benchmark claims")
        if self.statement != CONNECTIVITY_STATEMENT:
            raise ValueError("statement must use the fixed connectivity-only wording")
        frozen_data = json.loads(json.dumps(self.data_validation, allow_nan=False))
        frozen_env = json.loads(json.dumps(self.environment_validation, allow_nan=False))
        frozen_smokes = tuple(json.loads(json.dumps(item, allow_nan=False)) for item in self.policy_smokes)
        object.__setattr__(self, "data_validation", MappingProxyType(frozen_data))
        object.__setattr__(self, "environment_validation", MappingProxyType(frozen_env))
        object.__setattr__(self, "policy_smokes", tuple(MappingProxyType(item) for item in frozen_smokes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "config_sha256": self.config_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "source_spec_sha256": self.source_spec_sha256,
            "source_inventory_sha256": self.source_inventory_sha256,
            "runner_qualification_sha256": self.runner_qualification_sha256,
            "metrics_sha256": self.metrics_sha256,
            "runner_role": self.runner_role,
            "data_validation": dict(self.data_validation),
            "environment_validation": dict(self.environment_validation),
            "policy_smokes": [dict(item) for item in self.policy_smokes],
            "downloaded_bytes": self.downloaded_bytes,
            "claim_allowed": self.claim_allowed,
            "benchmark_claim": self.benchmark_claim,
            "statement": self.statement,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IntegrationManifestV1":
        fields = {
            "schema_version", "git_sha", "git_dirty", "config_sha256", "dependency_lock_sha256",
            "source_spec_sha256", "source_inventory_sha256", "runner_qualification_sha256",
            "metrics_sha256", "runner_role", "data_validation", "environment_validation", "policy_smokes",
            "downloaded_bytes", "claim_allowed", "benchmark_claim", "statement",
        }
        payload = _exact(value, fields, "IntegrationManifestV1")
        smokes = payload["policy_smokes"]
        if isinstance(smokes, (str, bytes, Mapping)) or not isinstance(smokes, Sequence):
            raise TypeError("policy_smokes must be a sequence")
        payload["policy_smokes"] = tuple(smokes)
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())
