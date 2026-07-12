from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from .artifacts import ArtifactHashRecordV1


ALPHA2_TAG = "v3.0.0-alpha.2"
ALPHA2_PACKAGE_VERSION = "3.0.0a2"
SMOLVLA_REPO_ID = "lerobot/smolvla_base"
SMOLVLA_REVISION = "d06fce6e38c25c04ac5a6319eefb9fae0e257cb2"
SMOLVLA_WEIGHT_SHA256 = (
    "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
)

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SPDX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
_LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PYTHON_312 = re.compile(r"^3\.12(?:\.\d+)?$")
_DRIVER_VERSION = re.compile(r"^(\d+)\.(\d+)(?:\.\d+)?$")

_RUNNER_LABELS = {"self-hosted", "linux", "x64", "gpu", "lunavla-v3"}
_RUNNER_NETWORK_HOSTS = {
    "api.github.com",
    "download.pytorch.org",
    "github.com",
    "huggingface.co",
    "pypi.org",
}
_LICENSE_SNAPSHOT_PATHS = {
    "docs/v3/release/smolvla-file-inventory-observation.json",
    "docs/v3/release/smolvla-model-card-observation.json",
    "docs/v3/release/smolvla-repository-metadata-observation.json",
}


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown:
        raise ValueError(f"unknown field(s) in {name}: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing field(s) in {name}: {', '.join(missing)}")
    return dict(value)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX_64.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256")
    return value


def _git_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX_40.fullmatch(value):
        raise ValueError(f"{name} must be a full lowercase Git SHA")
    return value


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _records(value: Any, name: str) -> tuple[ArtifactHashRecordV1, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    records = tuple(
        item if isinstance(item, ArtifactHashRecordV1) else ArtifactHashRecordV1.from_mapping(item)
        for item in value
    )
    if not records:
        raise ValueError(f"{name} must not be empty")
    paths = [item.path for item in records]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{name} paths must be unique")
    return tuple(sorted(records, key=lambda item: item.path))


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


@dataclass(frozen=True)
class LicenseReviewV1:
    repo_id: str
    revision: str
    spdx_license: str
    evidence_url: str
    evidence_sha256: str
    scope: str
    reviewer: str
    reviewed_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "license review schema_version", minimum=1) != 1:
            raise ValueError("LicenseReviewV1 schema_version must be integer 1")
        repo_id = _string(self.repo_id, "license review repo_id")
        if repo_id != SMOLVLA_REPO_ID:
            raise ValueError(f"license review repo_id must be {SMOLVLA_REPO_ID}")
        revision = _git_sha(self.revision, "license review revision")
        license_id = _string(self.spdx_license, "license review SPDX license")
        if not _SPDX.fullmatch(license_id) or license_id == "NOASSERTION":
            raise ValueError("license review requires a concrete SPDX license identifier")
        evidence_url = _string(self.evidence_url, "license review evidence_url")
        parsed = urlparse(evidence_url)
        expected_prefix = f"/{SMOLVLA_REPO_ID}/"
        if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
            raise ValueError("weight license evidence must use an official Hugging Face HTTPS URL")
        if not parsed.path.startswith(expected_prefix):
            raise ValueError("weight license evidence must name the reviewed model repository")
        if self.scope != "model_weights":
            raise ValueError("license review scope must be model_weights")
        reviewer = _string(self.reviewer, "license review reviewer")
        if not _LOGIN.fullmatch(reviewer):
            raise ValueError("license review reviewer must be a GitHub login")
        reviewed_at = _string(self.reviewed_at, "license review reviewed_at")
        if not _DATE.fullmatch(reviewed_at):
            raise ValueError("license review reviewed_at must use YYYY-MM-DD")
        object.__setattr__(self, "repo_id", repo_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "spdx_license", license_id)
        object.__setattr__(self, "evidence_url", evidence_url)
        object.__setattr__(self, "evidence_sha256", _sha256(self.evidence_sha256, "evidence SHA-256"))
        object.__setattr__(self, "reviewer", reviewer)
        object.__setattr__(self, "reviewed_at", reviewed_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "spdx_license": self.spdx_license,
            "evidence_url": self.evidence_url,
            "evidence_sha256": self.evidence_sha256,
            "scope": self.scope,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LicenseReviewV1":
        payload = _exact(value, set(cls.__dataclass_fields__), "license review")
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())

    def save(self, path: str | Path) -> Path:
        return _write_json(path, self.to_dict())


@dataclass(frozen=True)
class WeightLicenseStatusV1:
    repo_id: str
    revision: str
    weight_sha256: str
    license_status: str
    spdx_license: str
    pretrained_enabled: bool
    conformance_only: bool
    owner_authorization_scope: str
    owner_authorization_is_license_evidence: bool
    release_eligible: bool
    source_snapshots: tuple[ArtifactHashRecordV1, ...]
    reviewer: str
    checked_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "weight license status schema_version", minimum=1) != 1:
            raise ValueError("WeightLicenseStatusV1 schema_version must be integer 1")
        if self.repo_id != SMOLVLA_REPO_ID:
            raise ValueError(f"weight license status repo_id must be {SMOLVLA_REPO_ID}")
        if self.revision != SMOLVLA_REVISION:
            raise ValueError("weight license status must bind the reviewed SmolVLA revision")
        if _sha256(self.weight_sha256, "weight SHA-256") != SMOLVLA_WEIGHT_SHA256:
            raise ValueError("weight license status must bind the reviewed SmolVLA weight")
        if self.license_status != "unverified" or self.spdx_license != "NOASSERTION":
            raise ValueError("undeclared SmolVLA weights must remain unverified/NOASSERTION")
        for name in (
            "pretrained_enabled",
            "conformance_only",
            "owner_authorization_is_license_evidence",
            "release_eligible",
        ):
            _bool(getattr(self, name), name)
        if self.pretrained_enabled or not self.conformance_only:
            raise ValueError("unverified weights must remain disabled and conformance-only")
        if self.owner_authorization_scope != "local_evaluation_only":
            raise ValueError("owner authorization scope must be local_evaluation_only")
        if self.owner_authorization_is_license_evidence or self.release_eligible:
            raise ValueError("owner authorization cannot become upstream license or release evidence")
        snapshots = _records(self.source_snapshots, "weight license source_snapshots")
        if {item.path for item in snapshots} != _LICENSE_SNAPSHOT_PATHS:
            raise ValueError("weight license status requires the exact normalized source snapshots")
        reviewer = _string(self.reviewer, "weight license status reviewer")
        if not _LOGIN.fullmatch(reviewer):
            raise ValueError("weight license status reviewer must be a GitHub login")
        checked_at = _string(self.checked_at, "weight license status checked_at")
        if not _DATE.fullmatch(checked_at):
            raise ValueError("weight license status checked_at must use YYYY-MM-DD")
        object.__setattr__(self, "source_snapshots", snapshots)
        object.__setattr__(self, "reviewer", reviewer)
        object.__setattr__(self, "checked_at", checked_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "weight_sha256": self.weight_sha256,
            "license_status": self.license_status,
            "spdx_license": self.spdx_license,
            "pretrained_enabled": self.pretrained_enabled,
            "conformance_only": self.conformance_only,
            "owner_authorization_scope": self.owner_authorization_scope,
            "owner_authorization_is_license_evidence": self.owner_authorization_is_license_evidence,
            "release_eligible": self.release_eligible,
            "source_snapshots": [item.to_dict() for item in self.source_snapshots],
            "reviewer": self.reviewer,
            "checked_at": self.checked_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WeightLicenseStatusV1":
        payload = _exact(value, set(cls.__dataclass_fields__), "weight license status")
        payload["source_snapshots"] = _records(
            payload["source_snapshots"], "weight license source_snapshots"
        )
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())

    def save(self, path: str | Path) -> Path:
        return _write_json(path, self.to_dict())


@dataclass(frozen=True)
class RunnerQualificationManifestV1:
    role: str
    git_sha: str
    dependency_lock_sha256: str
    container_image_sha256: str
    runner_name_sha256: str
    runner_labels: tuple[str, ...]
    runner_os: str
    runner_os_version: str
    runner_arch: str
    python_version: str
    cpu_count: int
    memory_bytes: int
    disk_free_bytes: int
    gpu_count: int
    cuda_visible_device_count: int
    gpu_name: str
    gpu_uuid_sha256: str
    driver_version: str
    cuda_runtime: str
    torch_version: str
    torchvision_version: str
    network_hosts: tuple[str, ...]
    workspace_clean: bool
    container_isolated: bool
    private_mounts_detected: bool
    ephemeral_declared: bool
    weight_accessed: bool
    release_eligible: bool
    claim_allowed: bool
    checked_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "runner qualification schema_version", minimum=1) != 1:
            raise ValueError("RunnerQualificationManifestV1 schema_version must be integer 1")
        if self.role not in {"authoritative", "secondary"}:
            raise ValueError("runner qualification role must be authoritative or secondary")
        object.__setattr__(self, "git_sha", _git_sha(self.git_sha, "runner qualification git_sha"))
        for name in (
            "dependency_lock_sha256",
            "container_image_sha256",
            "runner_name_sha256",
            "gpu_uuid_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if isinstance(self.runner_labels, (str, bytes)):
            raise TypeError("runner_labels must be a sequence")
        labels = tuple(_string(item, "runner label") for item in self.runner_labels)
        if len(labels) != len(set(labels)) or not _RUNNER_LABELS.issubset(labels):
            raise ValueError("runner_labels must uniquely include the LunaVLA GPU labels")
        object.__setattr__(self, "runner_labels", tuple(sorted(labels)))
        if self.runner_os != "Linux" or self.runner_arch != "X64":
            raise ValueError("runner qualification requires Linux X64")
        object.__setattr__(self, "runner_os_version", _string(self.runner_os_version, "OS version"))
        if not _PYTHON_312.fullmatch(self.python_version):
            raise ValueError("runner qualification requires Python 3.12")
        object.__setattr__(self, "cpu_count", _integer(self.cpu_count, "CPU count", minimum=1))
        object.__setattr__(
            self, "memory_bytes", _integer(self.memory_bytes, "memory bytes", minimum=16 * 1024**3)
        )
        object.__setattr__(
            self,
            "disk_free_bytes",
            _integer(self.disk_free_bytes, "disk free bytes", minimum=30 * 1024**3),
        )
        if _integer(self.gpu_count, "GPU count", minimum=1) != 1:
            raise ValueError("runner qualification requires exactly one nvidia-smi GPU")
        if _integer(self.cuda_visible_device_count, "CUDA visible count", minimum=1) != 1:
            raise ValueError("runner qualification requires exactly one PyTorch CUDA device")
        gpu_name = _string(self.gpu_name, "GPU name")
        if "NVIDIA A100" not in gpu_name:
            raise ValueError("runner qualification requires an NVIDIA A100")
        object.__setattr__(self, "gpu_name", gpu_name)
        driver = _string(self.driver_version, "driver version")
        match = _DRIVER_VERSION.fullmatch(driver)
        if match is None or (int(match.group(1)), int(match.group(2))) < (570, 26):
            raise ValueError("runner qualification requires NVIDIA driver >=570.26")
        object.__setattr__(self, "driver_version", driver)
        if self.cuda_runtime != "12.8":
            raise ValueError("runner qualification requires CUDA runtime 12.8")
        if self.torch_version != "2.11.0+cu128":
            raise ValueError("runner qualification requires Torch 2.11.0+cu128")
        if self.torchvision_version != "0.26.0+cu128":
            raise ValueError("runner qualification requires torchvision 0.26.0+cu128")
        if isinstance(self.network_hosts, (str, bytes)):
            raise TypeError("network_hosts must be a sequence")
        hosts = tuple(_string(item, "network host") for item in self.network_hosts)
        if len(hosts) != len(set(hosts)) or set(hosts) != _RUNNER_NETWORK_HOSTS:
            raise ValueError("runner qualification must verify the exact outbound host set")
        object.__setattr__(self, "network_hosts", tuple(sorted(hosts)))
        for name in (
            "workspace_clean",
            "container_isolated",
            "private_mounts_detected",
            "ephemeral_declared",
            "weight_accessed",
            "release_eligible",
            "claim_allowed",
        ):
            _bool(getattr(self, name), name)
        if not self.workspace_clean or not self.container_isolated or not self.ephemeral_declared:
            raise ValueError("runner qualification requires a clean isolated ephemeral runner")
        if self.private_mounts_detected or self.weight_accessed:
            raise ValueError("runner qualification forbids private mounts and weight access")
        if self.release_eligible or self.claim_allowed:
            raise ValueError("runner preflight cannot open release or scientific-claim gates")
        checked_at = _string(self.checked_at, "runner qualification checked_at")
        if not _DATE.fullmatch(checked_at):
            raise ValueError("runner qualification checked_at must use YYYY-MM-DD")
        object.__setattr__(self, "checked_at", checked_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "git_sha": self.git_sha,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "container_image_sha256": self.container_image_sha256,
            "runner_name_sha256": self.runner_name_sha256,
            "runner_labels": list(self.runner_labels),
            "runner_os": self.runner_os,
            "runner_os_version": self.runner_os_version,
            "runner_arch": self.runner_arch,
            "python_version": self.python_version,
            "cpu_count": self.cpu_count,
            "memory_bytes": self.memory_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "gpu_count": self.gpu_count,
            "cuda_visible_device_count": self.cuda_visible_device_count,
            "gpu_name": self.gpu_name,
            "gpu_uuid_sha256": self.gpu_uuid_sha256,
            "driver_version": self.driver_version,
            "cuda_runtime": self.cuda_runtime,
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "network_hosts": list(self.network_hosts),
            "workspace_clean": self.workspace_clean,
            "container_isolated": self.container_isolated,
            "private_mounts_detected": self.private_mounts_detected,
            "ephemeral_declared": self.ephemeral_declared,
            "weight_accessed": self.weight_accessed,
            "release_eligible": self.release_eligible,
            "claim_allowed": self.claim_allowed,
            "checked_at": self.checked_at,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunnerQualificationManifestV1":
        payload = _exact(value, set(cls.__dataclass_fields__), "runner qualification manifest")
        payload["runner_labels"] = tuple(payload["runner_labels"])
        payload["network_hosts"] = tuple(payload["network_hosts"])
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())

    def save(self, path: str | Path) -> Path:
        return _write_json(path, self.to_dict())


@dataclass(frozen=True)
class GpuValidationManifestV1:
    git_sha: str
    package_version: str
    license_review_sha256: str
    model_source_sha256: str
    dependency_lock_sha256: str
    dispatcher_sha256: str
    runner_labels: tuple[str, ...]
    runner_os: str
    runner_arch: str
    gpu_count: int
    gpu_name: str
    gpu_uuid_sha256: str
    driver_version: str
    cuda_runtime: str
    torch_version: str
    torchvision_version: str
    downloaded_files: tuple[ArtifactHashRecordV1, ...]
    model_bytes: int
    train_seed: int
    loss_before: float
    loss_after: float
    gradient_norm: float
    checkpoint_sha256: str
    restored_action_sha256: str
    resume_rtol: float
    resume_atol: float
    optimizer_step_verified: bool
    resume_verified: bool
    inference_verified: bool
    claim_allowed: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "GPU manifest schema_version", minimum=1) != 1:
            raise ValueError("GpuValidationManifestV1 schema_version must be integer 1")
        object.__setattr__(self, "git_sha", _git_sha(self.git_sha, "GPU manifest git_sha"))
        if self.package_version != ALPHA2_PACKAGE_VERSION:
            raise ValueError(f"GPU manifest package_version must be {ALPHA2_PACKAGE_VERSION}")
        for name in (
            "license_review_sha256",
            "model_source_sha256",
            "dependency_lock_sha256",
            "dispatcher_sha256",
            "gpu_uuid_sha256",
            "checkpoint_sha256",
            "restored_action_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if isinstance(self.runner_labels, (str, bytes)):
            raise TypeError("runner_labels must be a sequence")
        labels = tuple(_string(item, "runner label") for item in self.runner_labels)
        required = {"self-hosted", "linux", "x64", "gpu", "lunavla-v3"}
        if len(labels) != len(set(labels)) or not required.issubset(labels):
            raise ValueError("runner_labels must uniquely include the LunaVLA GPU labels")
        object.__setattr__(self, "runner_labels", tuple(sorted(labels)))
        if self.runner_os != "Linux" or self.runner_arch != "X64":
            raise ValueError("Alpha 2 GPU evidence requires Linux X64")
        if _integer(self.gpu_count, "GPU count", minimum=1) != 1:
            raise ValueError("Alpha 2 GPU evidence requires exactly one GPU")
        for name in ("gpu_name", "driver_version"):
            object.__setattr__(self, name, _string(getattr(self, name), name))
        expected_versions = {
            "cuda_runtime": "12.8",
            "torch_version": "2.11.0+cu128",
            "torchvision_version": "0.26.0+cu128",
        }
        for name, expected in expected_versions.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected}")
        files = _records(self.downloaded_files, "downloaded_files")
        model = [item for item in files if item.path == "model.safetensors"]
        if len(model) != 1 or model[0].sha256 != SMOLVLA_WEIGHT_SHA256:
            raise ValueError("GPU manifest must bind the reviewed SmolVLA weight hash")
        object.__setattr__(self, "downloaded_files", files)
        object.__setattr__(self, "model_bytes", _integer(self.model_bytes, "model_bytes", minimum=1))
        object.__setattr__(self, "train_seed", _integer(self.train_seed, "train_seed"))
        for name in ("loss_before", "loss_after", "gradient_norm"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0.0))
        for name in ("resume_rtol", "resume_atol"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0.0))
        if self.resume_rtol != 1e-5 or self.resume_atol != 1e-6:
            raise ValueError("GPU resume tolerances must be rtol=1e-5 and atol=1e-6")
        for name in (
            "optimizer_step_verified",
            "resume_verified",
            "inference_verified",
            "claim_allowed",
        ):
            _bool(getattr(self, name), name)
        if not all((self.optimizer_step_verified, self.resume_verified, self.inference_verified)):
            raise ValueError("GPU manifest requires optimizer, resume, and inference verification")
        if self.claim_allowed:
            raise ValueError("Alpha 2 GPU smoke cannot allow scientific claims")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "git_sha": self.git_sha,
            "package_version": self.package_version,
            "license_review_sha256": self.license_review_sha256,
            "model_source_sha256": self.model_source_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "dispatcher_sha256": self.dispatcher_sha256,
            "runner_labels": list(self.runner_labels),
            "runner_os": self.runner_os,
            "runner_arch": self.runner_arch,
            "gpu_count": self.gpu_count,
            "gpu_name": self.gpu_name,
            "gpu_uuid_sha256": self.gpu_uuid_sha256,
            "driver_version": self.driver_version,
            "cuda_runtime": self.cuda_runtime,
            "torch_version": self.torch_version,
            "torchvision_version": self.torchvision_version,
            "downloaded_files": [item.to_dict() for item in self.downloaded_files],
            "model_bytes": self.model_bytes,
            "train_seed": self.train_seed,
            "loss_before": self.loss_before,
            "loss_after": self.loss_after,
            "gradient_norm": self.gradient_norm,
            "checkpoint_sha256": self.checkpoint_sha256,
            "restored_action_sha256": self.restored_action_sha256,
            "resume_rtol": self.resume_rtol,
            "resume_atol": self.resume_atol,
            "optimizer_step_verified": self.optimizer_step_verified,
            "resume_verified": self.resume_verified,
            "inference_verified": self.inference_verified,
            "claim_allowed": self.claim_allowed,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GpuValidationManifestV1":
        payload = _exact(value, set(cls.__dataclass_fields__), "GPU validation manifest")
        payload["downloaded_files"] = _records(payload["downloaded_files"], "downloaded_files")
        payload["runner_labels"] = tuple(payload["runner_labels"])
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())

    def save(self, path: str | Path) -> Path:
        return _write_json(path, self.to_dict())


@dataclass(frozen=True)
class Alpha2ReleaseCandidateV1:
    expected_tag: str
    git_sha: str
    package_version: str
    gpu_manifest_sha256: str
    gpu_attestation_sha256: str
    required_checks_sha256: str
    dispatcher_sha256: str
    assets: tuple[ArtifactHashRecordV1, ...]
    claim_allowed: bool = False
    pypi_published: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "release candidate schema_version", minimum=1) != 1:
            raise ValueError("Alpha2ReleaseCandidateV1 schema_version must be integer 1")
        if self.expected_tag != ALPHA2_TAG:
            raise ValueError(f"release candidate expected_tag must be {ALPHA2_TAG}")
        object.__setattr__(self, "git_sha", _git_sha(self.git_sha, "release candidate git_sha"))
        if self.package_version != ALPHA2_PACKAGE_VERSION:
            raise ValueError(f"release candidate package_version must be {ALPHA2_PACKAGE_VERSION}")
        for name in (
            "gpu_manifest_sha256",
            "gpu_attestation_sha256",
            "required_checks_sha256",
            "dispatcher_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        assets = _records(self.assets, "release candidate assets")
        required_names = {
            "dist/lunavla-3.0.0a2-py3-none-any.whl",
            "dist/lunavla-3.0.0a2.tar.gz",
            "environment-requirements.txt",
            "sbom.json",
            "gpu-validation-manifest.json",
            "gpu-attestation-bundle.jsonl",
            "lunavla-v3-alpha2-evidence.tar.gz",
        }
        if not required_names.issubset(item.path for item in assets):
            raise ValueError("release candidate is missing required Alpha 2 assets")
        object.__setattr__(self, "assets", assets)
        _bool(self.claim_allowed, "claim_allowed")
        _bool(self.pypi_published, "pypi_published")
        if self.claim_allowed or self.pypi_published:
            raise ValueError("Alpha 2 candidate cannot allow claims or publish to PyPI")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "expected_tag": self.expected_tag,
            "git_sha": self.git_sha,
            "package_version": self.package_version,
            "gpu_manifest_sha256": self.gpu_manifest_sha256,
            "gpu_attestation_sha256": self.gpu_attestation_sha256,
            "required_checks_sha256": self.required_checks_sha256,
            "dispatcher_sha256": self.dispatcher_sha256,
            "assets": [item.to_dict() for item in self.assets],
            "claim_allowed": self.claim_allowed,
            "pypi_published": self.pypi_published,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Alpha2ReleaseCandidateV1":
        payload = _exact(value, set(cls.__dataclass_fields__), "Alpha 2 release candidate")
        payload["assets"] = _records(payload["assets"], "release candidate assets")
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())

    def save(self, path: str | Path) -> Path:
        return _write_json(path, self.to_dict())
