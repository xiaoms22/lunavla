from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Union, cast
from urllib.parse import unquote, urlparse

import numpy as np

from .artifacts import ArtifactHashRecordV1
from .artifacts import sha256_file


RC_TAG = "v3.0.0-rc.1"
RC_PACKAGE_VERSION = "3.0.0rc1"
STABLE_TAG = "v3.0.0"
STABLE_PACKAGE_VERSION = "3.0.0"
STABLE_TRAIN_SEEDS = (11, 22, 33, 44, 55)
STABLE_DATA_SEED = 42
STABLE_ANALYSIS_SEED = 202701
STABLE_BOOTSTRAP_SAMPLES = 10_000
PUSHT_POLICIES = ("act_v3", "diffusion_v3")
FIXTURE_TASK_IDS = (0, 1, 2)
LIBERO_ROUTES = ("none", "expert_only", "prompt_only", "dual")
PROMPT_INTERVENTIONS = ("control", "mask", "shuffle", "counterfactual", "layout_drift")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_STUDIES = {
    "fixture_policy_ladder",
    "fixture_state_routes",
    "fixture_prompt_interventions",
}
_EVIDENCE_PATHS = {
    "evidence/fixture-policy-ladder.json",
    "evidence/fixture-state-routes.json",
    "evidence/fixture-prompt-interventions.json",
}
_COMMON_RELEASE_ASSETS = {
    "contract-descriptor.json",
    "evidence.tar.gz",
    "migration-report.json",
    "portfolio.tar.gz",
    "privacy-scan.json",
    "provenance.jsonl",
    "public-api.json",
    "required-checks.json",
    "sbom.json",
}
_RC_RELEASE_ASSETS = _COMMON_RELEASE_ASSETS | {
    "dist/lunavla-3.0.0rc1-py3-none-any.whl",
    "dist/lunavla-3.0.0rc1.tar.gz",
}
_STABLE_RELEASE_ASSETS = _COMMON_RELEASE_ASSETS | {
    "dist/lunavla-3.0.0-py3-none-any.whl",
    "dist/lunavla-3.0.0.tar.gz",
}
_STABLE_DIST_FILENAMES = {
    "lunavla-3.0.0-py3-none-any.whl": "dist/lunavla-3.0.0-py3-none-any.whl",
    "lunavla-3.0.0.tar.gz": "dist/lunavla-3.0.0.tar.gz",
}


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
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _strings(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} items must be non-empty strings")
        result.append(item.strip())
    if not result and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(result)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256")
    return value


def _git_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _GIT_SHA.fullmatch(value):
        raise ValueError(f"{name} must be a full lowercase Git SHA")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class TrustedPublisherIdentityV1:
    owner: str
    repository: str
    workflow: str
    environment: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "trusted publisher schema_version", positive=True) != 1:
            raise ValueError("TrustedPublisherIdentityV1 schema_version must be integer 1")
        for field in ("owner", "repository", "workflow", "environment"):
            value = _nonempty(getattr(self, field), f"trusted publisher {field}")
            if "/" in value or "\\" in value or value in {".", ".."}:
                raise ValueError(f"trusted publisher {field} must be a basename")
            object.__setattr__(self, field, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner": self.owner,
            "repository": self.repository,
            "workflow": self.workflow,
            "environment": self.environment,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrustedPublisherIdentityV1":
        return cls(**_exact(value, set(cls.__dataclass_fields__), "trusted publisher identity"))

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class PyPIFileRecordV1:
    filename: str
    sha256: str
    size_bytes: int
    file_url: str
    provenance_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "PyPI file schema_version", positive=True) != 1:
            raise ValueError("PyPIFileRecordV1 schema_version must be integer 1")
        filename = _nonempty(self.filename, "PyPI filename")
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("PyPI filename must be a basename")
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "sha256", _sha256(self.sha256, "PyPI file sha256"))
        object.__setattr__(self, "size_bytes", _integer(self.size_bytes, "PyPI file size_bytes", positive=True))
        file_url = _nonempty(self.file_url, "PyPI file_url")
        parsed = urlparse(file_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "files.pythonhosted.org"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
            or Path(unquote(parsed.path)).name != filename
        ):
            raise ValueError(
                "PyPI file_url must be an HTTPS files.pythonhosted.org URL whose basename matches filename"
            )
        object.__setattr__(self, "file_url", file_url)
        object.__setattr__(
            self,
            "provenance_sha256",
            _sha256(self.provenance_sha256, "PyPI file provenance_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "file_url": self.file_url,
            "provenance_sha256": self.provenance_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PyPIFileRecordV1":
        return cls(**_exact(value, set(cls.__dataclass_fields__), "PyPI file record"))


@dataclass(frozen=True)
class PyPIPublishRecordV1:
    project: str
    version: str
    files: tuple[PyPIFileRecordV1, ...]
    publisher: TrustedPublisherIdentityV1
    publish_receipt_sha256: str
    attestation_sha256: str | None
    project_url: str
    published: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "PyPI publish record schema_version", positive=True) != 1:
            raise ValueError("PyPIPublishRecordV1 schema_version must be integer 1")
        object.__setattr__(self, "project", _nonempty(self.project, "PyPI project"))
        object.__setattr__(self, "version", _nonempty(self.version, "PyPI version"))
        files = tuple(
            item if isinstance(item, PyPIFileRecordV1) else PyPIFileRecordV1.from_mapping(item)
            for item in self.files
        )
        if len(files) != 2 or len({item.filename for item in files}) != 2:
            raise ValueError("PyPI publish record must contain exactly two unique distribution files")
        object.__setattr__(self, "files", tuple(sorted(files, key=lambda item: item.filename)))
        if not isinstance(self.publisher, TrustedPublisherIdentityV1):
            object.__setattr__(self, "publisher", TrustedPublisherIdentityV1.from_mapping(self.publisher))
        if self.publisher != TrustedPublisherIdentityV1(
            owner="xiaoms22",
            repository="lunavla",
            workflow="v3-pypi-release.yml",
            environment="pypi",
        ):
            raise ValueError("PyPI publisher identity must match the frozen LunaVLA Trusted Publisher")
        object.__setattr__(self, "publish_receipt_sha256", _sha256(self.publish_receipt_sha256, "PyPI receipt sha256"))
        if self.attestation_sha256 is not None:
            object.__setattr__(
                self,
                "attestation_sha256",
                _sha256(self.attestation_sha256, "PyPI attestation sha256"),
            )
        url = _nonempty(self.project_url, "PyPI project_url")
        if url != "https://pypi.org/project/lunavla/3.0.0/":
            raise ValueError("PyPI project_url must identify the immutable lunavla 3.0.0 page")
        object.__setattr__(self, "project_url", url)
        _boolean(self.published, "PyPI published")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project": self.project,
            "version": self.version,
            "files": [item.to_dict() for item in self.files],
            "publisher": self.publisher.to_dict(),
            "publish_receipt_sha256": self.publish_receipt_sha256,
            "attestation_sha256": self.attestation_sha256,
            "project_url": self.project_url,
            "published": self.published,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PyPIPublishRecordV1":
        payload = _exact(value, set(cls.__dataclass_fields__), "PyPI publish record")
        payload["files"] = tuple(PyPIFileRecordV1.from_mapping(item) for item in payload["files"])
        payload["publisher"] = TrustedPublisherIdentityV1.from_mapping(payload["publisher"])
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


def _records(value: Any, name: str) -> tuple[ArtifactHashRecordV1, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    records = tuple(
        item if isinstance(item, ArtifactHashRecordV1) else ArtifactHashRecordV1.from_mapping(item)
        for item in value
    )
    if not records:
        raise ValueError(f"{name} must not be empty")
    paths = tuple(record.path for record in records)
    if len(paths) != len(set(paths)):
        raise ValueError(f"{name} paths must be unique")
    return tuple(sorted(records, key=lambda record: record.path))


@dataclass(frozen=True)
class StableEvidenceDesignV1:
    study_id: str
    train_seeds: tuple[int, ...]
    data_seed: int
    analysis_seed: int
    bootstrap_samples: int
    policies: tuple[str, ...]
    task_ids: tuple[int, ...]
    evaluation_ids: tuple[int, ...]
    routes: tuple[str, ...]
    interventions: tuple[str, ...]
    repeat_train_seed: int
    reduced_design: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "stable design schema_version", positive=True) != 1:
            raise ValueError("StableEvidenceDesignV1 schema_version must be integer 1")
        if self.study_id not in _STUDIES:
            raise ValueError(f"unsupported stable evidence study_id {self.study_id!r}")
        object.__setattr__(self, "train_seeds", _integers(self.train_seeds, "train_seeds"))
        object.__setattr__(self, "data_seed", _integer(self.data_seed, "data_seed"))
        object.__setattr__(self, "analysis_seed", _integer(self.analysis_seed, "analysis_seed"))
        object.__setattr__(self, "bootstrap_samples", _integer(self.bootstrap_samples, "bootstrap_samples", positive=True))
        object.__setattr__(self, "policies", _strings(self.policies, "policies"))
        object.__setattr__(self, "task_ids", _integers(self.task_ids, "task_ids", allow_empty=True))
        object.__setattr__(self, "evaluation_ids", _integers(self.evaluation_ids, "evaluation_ids"))
        object.__setattr__(self, "routes", _strings(self.routes, "routes", allow_empty=True))
        object.__setattr__(self, "interventions", _strings(self.interventions, "interventions", allow_empty=True))
        object.__setattr__(self, "repeat_train_seed", _integer(self.repeat_train_seed, "repeat_train_seed"))
        _boolean(self.reduced_design, "reduced_design")
        if self.repeat_train_seed not in self.train_seeds:
            raise ValueError("repeat_train_seed must be present in train_seeds")

    def validate_stable_matrix(self) -> None:
        common = (
            self.train_seeds == STABLE_TRAIN_SEEDS
            and self.data_seed == STABLE_DATA_SEED
            and self.analysis_seed == STABLE_ANALYSIS_SEED
            and self.bootstrap_samples == STABLE_BOOTSTRAP_SAMPLES
            and self.repeat_train_seed == 11
            and not self.reduced_design
        )
        if not common:
            raise ValueError("stable evidence design does not match the frozen seed/statistics plan")
        expected: tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...], tuple[str, ...], tuple[str, ...]]
        if self.study_id == "fixture_policy_ladder":
            expected = (PUSHT_POLICIES, (), tuple(range(20)), (), ())
        elif self.study_id == "fixture_state_routes":
            expected = (("act_v3",), FIXTURE_TASK_IDS, tuple(range(10)), LIBERO_ROUTES, ())
        else:
            expected = (
                ("act_v3",),
                FIXTURE_TASK_IDS,
                tuple(range(10)),
                ("dual",),
                PROMPT_INTERVENTIONS,
            )
        actual = (
            self.policies,
            self.task_ids,
            self.evaluation_ids,
            self.routes,
            self.interventions,
        )
        if actual != expected:
            raise ValueError(f"{self.study_id} does not match the frozen stable matrix")

    @property
    def expected_rows(self) -> int:
        task_factor = max(1, len(self.task_ids))
        route_factor = max(1, len(self.routes))
        intervention_factor = max(1, len(self.interventions))
        return (
            len(self.train_seeds)
            * len(self.policies)
            * task_factor
            * len(self.evaluation_ids)
            * route_factor
            * intervention_factor
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "train_seeds": list(self.train_seeds),
            "data_seed": self.data_seed,
            "analysis_seed": self.analysis_seed,
            "bootstrap_samples": self.bootstrap_samples,
            "policies": list(self.policies),
            "task_ids": list(self.task_ids),
            "evaluation_ids": list(self.evaluation_ids),
            "routes": list(self.routes),
            "interventions": list(self.interventions),
            "repeat_train_seed": self.repeat_train_seed,
            "reduced_design": self.reduced_design,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StableEvidenceDesignV1":
        payload = _exact(value, set(cls.__dataclass_fields__), "stable evidence design")
        for field in ("train_seeds", "policies", "task_ids", "evaluation_ids", "routes", "interventions"):
            payload[field] = tuple(payload[field])
        return cls(**payload)

    @classmethod
    def load(cls, path: str | Path) -> "StableEvidenceDesignV1":
        import yaml

        value = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(value, Mapping):
            raise TypeError("stable evidence design YAML must contain a mapping")
        return cls.from_mapping(value)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class StableEvidenceSummaryV1:
    study_id: str
    design_sha256: str
    git_sha: str
    dependency_lock_sha256: str
    upstream_identity_sha256: str
    row_inventory_sha256: str
    statistics_sha256: str
    sentinel_sha256: str
    expected_rows: int
    observed_rows: int
    matrix_complete: bool
    homogeneous_source: bool
    sentinel_verified: bool
    claim_gate_sha256: str
    release_eligible: bool
    gate_reasons: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "stable evidence summary schema_version", positive=True) != 1:
            raise ValueError("StableEvidenceSummaryV1 schema_version must be integer 1")
        if self.study_id not in _STUDIES:
            raise ValueError("stable evidence summary has an unknown study_id")
        for name in (
            "design_sha256",
            "dependency_lock_sha256",
            "upstream_identity_sha256",
            "row_inventory_sha256",
            "statistics_sha256",
            "sentinel_sha256",
            "claim_gate_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "git_sha", _git_sha(self.git_sha, "stable evidence git_sha"))
        object.__setattr__(self, "expected_rows", _integer(self.expected_rows, "expected_rows", positive=True))
        object.__setattr__(self, "observed_rows", _integer(self.observed_rows, "observed_rows"))
        for name in ("matrix_complete", "homogeneous_source", "sentinel_verified", "release_eligible"):
            _boolean(getattr(self, name), name)
        reasons = _strings(self.gate_reasons, "gate_reasons", allow_empty=True)
        allowed = {"incomplete_matrix", "mixed_source", "sentinel_failure"}
        if not set(reasons).issubset(allowed):
            raise ValueError("stable evidence summary contains an unknown gate reason")
        expected_reasons: list[str] = []
        if not self.matrix_complete or self.observed_rows != self.expected_rows:
            expected_reasons.append("incomplete_matrix")
        if not self.homogeneous_source:
            expected_reasons.append("mixed_source")
        if not self.sentinel_verified:
            expected_reasons.append("sentinel_failure")
        if tuple(expected_reasons) != reasons:
            raise ValueError("stable evidence gate reasons do not match the recorded state")
        eligible = not expected_reasons
        if self.release_eligible != eligible:
            raise ValueError("stable evidence release_eligible must be recomputed fail-closed")
        object.__setattr__(self, "gate_reasons", reasons)

    def verify_design(self, design: StableEvidenceDesignV1) -> None:
        design.validate_stable_matrix()
        if self.study_id != design.study_id or self.design_sha256 != design.sha256():
            raise ValueError("stable evidence summary does not bind the frozen design")
        if self.expected_rows != design.expected_rows:
            raise ValueError("stable evidence expected_rows does not match the design")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "design_sha256": self.design_sha256,
            "git_sha": self.git_sha,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "upstream_identity_sha256": self.upstream_identity_sha256,
            "row_inventory_sha256": self.row_inventory_sha256,
            "statistics_sha256": self.statistics_sha256,
            "sentinel_sha256": self.sentinel_sha256,
            "expected_rows": self.expected_rows,
            "observed_rows": self.observed_rows,
            "matrix_complete": self.matrix_complete,
            "homogeneous_source": self.homogeneous_source,
            "sentinel_verified": self.sentinel_verified,
            "claim_gate_sha256": self.claim_gate_sha256,
            "release_eligible": self.release_eligible,
            "gate_reasons": list(self.gate_reasons),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StableEvidenceSummaryV1":
        payload = _exact(value, set(cls.__dataclass_fields__), "stable evidence summary")
        payload["gate_reasons"] = tuple(payload["gate_reasons"])
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class StableEvidenceRowV1:
    study_id: str
    policy: str
    train_seed: int
    task_id: int | None
    evaluation_id: int
    route: str | None
    intervention: str | None
    git_sha: str
    dependency_lock_sha256: str
    upstream_identity_sha256: str
    run_manifest_sha256: str
    metrics_sha256: str
    success: bool
    final_metric: float
    smoothness: float
    first_action_mse: float | None
    failure_count: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "stable evidence row schema_version", positive=True) != 1:
            raise ValueError("StableEvidenceRowV1 schema_version must be integer 1")
        if self.study_id not in _STUDIES:
            raise ValueError("stable evidence row has an unknown study_id")
        if not isinstance(self.policy, str) or not self.policy.strip():
            raise ValueError("stable evidence row policy must be a non-empty string")
        object.__setattr__(self, "policy", self.policy.strip())
        object.__setattr__(self, "train_seed", _integer(self.train_seed, "row train_seed"))
        if self.task_id is not None:
            object.__setattr__(self, "task_id", _integer(self.task_id, "row task_id"))
        object.__setattr__(self, "evaluation_id", _integer(self.evaluation_id, "row evaluation_id"))
        for name in ("route", "intervention"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"row {name} must be null or a non-empty string")
            if value is not None:
                object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "git_sha", _git_sha(self.git_sha, "row git_sha"))
        for name in (
            "dependency_lock_sha256",
            "upstream_identity_sha256",
            "run_manifest_sha256",
            "metrics_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        _boolean(self.success, "row success")
        for name in ("final_metric", "smoothness"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.first_action_mse is not None:
            object.__setattr__(self, "first_action_mse", _finite(self.first_action_mse, "first_action_mse"))
        object.__setattr__(self, "failure_count", _integer(self.failure_count, "failure_count"))

    @property
    def matrix_key(self) -> tuple[Any, ...]:
        return (
            self.policy,
            self.train_seed,
            self.task_id,
            self.evaluation_id,
            self.route,
            self.intervention,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "policy": self.policy,
            "train_seed": self.train_seed,
            "task_id": self.task_id,
            "evaluation_id": self.evaluation_id,
            "route": self.route,
            "intervention": self.intervention,
            "git_sha": self.git_sha,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "upstream_identity_sha256": self.upstream_identity_sha256,
            "run_manifest_sha256": self.run_manifest_sha256,
            "metrics_sha256": self.metrics_sha256,
            "success": self.success,
            "final_metric": self.final_metric,
            "smoothness": self.smoothness,
            "first_action_mse": self.first_action_mse,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StableEvidenceRowV1":
        return cls(**_exact(value, set(cls.__dataclass_fields__), "stable evidence row"))


@dataclass(frozen=True)
class StableRepeatSentinelV1:
    study_id: str
    train_seed: int
    source_row_inventory_sha256: str
    repeat_row_inventory_sha256: str
    source_checkpoint_sha256: str
    repeat_checkpoint_sha256: str
    source_metrics_sha256: str
    repeat_metrics_sha256: str
    verified: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "repeat sentinel schema_version", positive=True) != 1:
            raise ValueError("StableRepeatSentinelV1 schema_version must be integer 1")
        if self.study_id not in _STUDIES:
            raise ValueError("repeat sentinel has an unknown study_id")
        if _integer(self.train_seed, "repeat sentinel train_seed") != 11:
            raise ValueError("stable repeat sentinel must use train seed 11")
        for name in (
            "source_row_inventory_sha256",
            "repeat_row_inventory_sha256",
            "source_checkpoint_sha256",
            "repeat_checkpoint_sha256",
            "source_metrics_sha256",
            "repeat_metrics_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        _boolean(self.verified, "repeat sentinel verified")
        equal = (
            self.source_row_inventory_sha256 == self.repeat_row_inventory_sha256
            and self.source_checkpoint_sha256 == self.repeat_checkpoint_sha256
            and self.source_metrics_sha256 == self.repeat_metrics_sha256
        )
        if self.verified != equal:
            raise ValueError("repeat sentinel verified must reflect exact row/checkpoint/metrics hashes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "train_seed": self.train_seed,
            "source_row_inventory_sha256": self.source_row_inventory_sha256,
            "repeat_row_inventory_sha256": self.repeat_row_inventory_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "repeat_checkpoint_sha256": self.repeat_checkpoint_sha256,
            "source_metrics_sha256": self.source_metrics_sha256,
            "repeat_metrics_sha256": self.repeat_metrics_sha256,
            "verified": self.verified,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StableRepeatSentinelV1":
        return cls(**_exact(value, set(cls.__dataclass_fields__), "stable repeat sentinel"))

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class _ReleaseCandidateV1:
    expected_tag: str
    package_version: str
    git_sha: str
    merge_sha: str
    public_api_sha256: str
    migration_report_sha256: str
    contract_descriptor_sha256: str
    portfolio_bundle_sha256: str
    evidence_manifests: tuple[ArtifactHashRecordV1, ...]
    required_checks_sha256: str
    sbom_sha256: str
    provenance_sha256: str
    checksums_sha256: str
    assets: tuple[ArtifactHashRecordV1, ...]
    signed_tag_verified: bool
    post_merge_evidence: bool
    privacy_scan_verified: bool
    pypi_published: bool
    schema_version: int = 1

    def _validate(
        self, *, tag: str, version: str, assets: set[str], stage: str, require_pypi: bool
    ) -> None:
        if _integer(self.schema_version, f"{stage} release candidate schema_version", positive=True) != 1:
            raise ValueError(f"{stage} release candidate schema_version must be integer 1")
        if self.expected_tag != tag or self.package_version != version:
            raise ValueError(f"{stage} release candidate tag/package mapping is invalid")
        git_sha = _git_sha(self.git_sha, f"{stage} release git_sha")
        merge_sha = _git_sha(self.merge_sha, f"{stage} release merge_sha")
        if git_sha != merge_sha:
            raise ValueError(f"{stage} release evidence must be rerun on the actual merge SHA")
        object.__setattr__(self, "git_sha", git_sha)
        object.__setattr__(self, "merge_sha", merge_sha)
        for name in (
            "public_api_sha256",
            "migration_report_sha256",
            "contract_descriptor_sha256",
            "portfolio_bundle_sha256",
            "required_checks_sha256",
            "sbom_sha256",
            "provenance_sha256",
            "checksums_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        evidence = _records(self.evidence_manifests, "evidence_manifests")
        asset_records = _records(self.assets, f"{stage} release assets")
        if {item.path for item in evidence} != _EVIDENCE_PATHS:
            raise ValueError(f"{stage} release requires all three frozen evidence manifests")
        if {item.path for item in asset_records} != assets:
            raise ValueError(f"{stage} release candidate asset inventory must be exact")
        by_path = {item.path: item.sha256 for item in asset_records}
        bindings = {
            "public_api_sha256": "public-api.json",
            "migration_report_sha256": "migration-report.json",
            "contract_descriptor_sha256": "contract-descriptor.json",
            "portfolio_bundle_sha256": "portfolio.tar.gz",
            "required_checks_sha256": "required-checks.json",
            "sbom_sha256": "sbom.json",
            "provenance_sha256": "provenance.jsonl",
        }
        for field, path in bindings.items():
            if getattr(self, field) != by_path[path]:
                raise ValueError(f"{stage} release candidate {field} does not bind {path}")
        for name in (
            "signed_tag_verified",
            "post_merge_evidence",
            "privacy_scan_verified",
            "pypi_published",
        ):
            _boolean(getattr(self, name), name)
        if not self.signed_tag_verified or not self.post_merge_evidence or not self.privacy_scan_verified:
            raise ValueError(f"{stage} release requires signed tag, post-merge evidence and privacy scan")
        if self.pypi_published != require_pypi:
            state = "published" if require_pypi else "unpublished"
            raise ValueError(f"{stage} release candidate must be {state} on PyPI")
        object.__setattr__(self, "evidence_manifests", evidence)
        object.__setattr__(self, "assets", asset_records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "expected_tag": self.expected_tag,
            "package_version": self.package_version,
            "git_sha": self.git_sha,
            "merge_sha": self.merge_sha,
            "public_api_sha256": self.public_api_sha256,
            "migration_report_sha256": self.migration_report_sha256,
            "contract_descriptor_sha256": self.contract_descriptor_sha256,
            "portfolio_bundle_sha256": self.portfolio_bundle_sha256,
            "evidence_manifests": [item.to_dict() for item in self.evidence_manifests],
            "required_checks_sha256": self.required_checks_sha256,
            "sbom_sha256": self.sbom_sha256,
            "provenance_sha256": self.provenance_sha256,
            "checksums_sha256": self.checksums_sha256,
            "assets": [item.to_dict() for item in self.assets],
            "signed_tag_verified": self.signed_tag_verified,
            "post_merge_evidence": self.post_merge_evidence,
            "privacy_scan_verified": self.privacy_scan_verified,
            "pypi_published": self.pypi_published,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_ReleaseCandidateV1":
        payload = _exact(value, set(cls.__dataclass_fields__), "release candidate")
        for field in ("evidence_manifests", "assets"):
            payload[field] = _records(payload[field], field)
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class RcReleaseCandidateV1(_ReleaseCandidateV1):
    def __post_init__(self) -> None:
        self._validate(
            tag=RC_TAG,
            version=RC_PACKAGE_VERSION,
            assets=_RC_RELEASE_ASSETS,
            stage="RC",
            require_pypi=False,
        )


@dataclass(frozen=True)
class StablePrePublishCandidateV1(_ReleaseCandidateV1):
    def __post_init__(self) -> None:
        self._validate(
            tag=STABLE_TAG,
            version=STABLE_PACKAGE_VERSION,
            assets=_STABLE_RELEASE_ASSETS,
            stage="stable pre-publish",
            require_pypi=False,
        )


@dataclass(frozen=True)
class StableReleaseCandidateV1(_ReleaseCandidateV1):
    pypi_publish_record: PyPIPublishRecordV1 | None = None

    def __post_init__(self) -> None:
        self._validate(
            tag=STABLE_TAG,
            version=STABLE_PACKAGE_VERSION,
            assets=_STABLE_RELEASE_ASSETS,
            stage="stable",
            require_pypi=True,
        )
        record = self.pypi_publish_record
        if not isinstance(record, PyPIPublishRecordV1):
            raise ValueError("stable release candidate requires a PyPI publish receipt")
        if record.project != "lunavla" or record.version != STABLE_PACKAGE_VERSION or not record.published:
            raise ValueError("stable release candidate requires the published lunavla 3.0.0 PyPI record")
        expected = {
            filename: next(item.sha256 for item in self.assets if item.path == path)
            for filename, path in _STABLE_DIST_FILENAMES.items()
        }
        actual = {item.filename: item.sha256 for item in record.files}
        if actual != expected:
            raise ValueError("PyPI wheel/sdist hashes must match the stable release assets")

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["pypi_publish_record"] = (
            self.pypi_publish_record.to_dict() if self.pypi_publish_record is not None else None
        )
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StableReleaseCandidateV1":
        payload = _exact(value, set(cls.__dataclass_fields__), "stable release candidate")
        for field in ("evidence_manifests", "assets"):
            payload[field] = _records(payload[field], field)
        record = payload["pypi_publish_record"]
        payload["pypi_publish_record"] = (
            PyPIPublishRecordV1.from_mapping(record) if isinstance(record, Mapping) else record
        )
        return cls(**payload)


ReleaseCandidateV1 = Union[
    RcReleaseCandidateV1, StablePrePublishCandidateV1, StableReleaseCandidateV1
]


def release_candidate_from_mapping(value: Mapping[str, Any]) -> ReleaseCandidateV1:
    tag = value.get("expected_tag") if isinstance(value, Mapping) else None
    if tag == RC_TAG:
        return cast(RcReleaseCandidateV1, RcReleaseCandidateV1.from_mapping(value))
    if tag == STABLE_TAG:
        if "pypi_publish_record" in value:
            return StableReleaseCandidateV1.from_mapping(value)
        return cast(StablePrePublishCandidateV1, StablePrePublishCandidateV1.from_mapping(value))
    raise ValueError("release candidate expected_tag is unknown")


def verify_release_candidate_assets(candidate: ReleaseCandidateV1, root: str | Path) -> None:
    resolved_root = Path(root).resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"release asset root does not exist: {resolved_root}")
    if any(path.is_symlink() for path in resolved_root.rglob("*")):
        raise ValueError("release asset root must not contain symbolic links")
    actual = {
        path.relative_to(resolved_root).as_posix()
        for path in resolved_root.rglob("*")
        if path.is_file()
    }
    expected_records = {record.path: record.sha256 for record in candidate.assets}
    expected = set(expected_records) | {"SHA256SUMS"}
    if actual != expected:
        raise ValueError("release asset directory inventory does not match the candidate")
    for relative, expected_sha256 in expected_records.items():
        path = (resolved_root / relative).resolve()
        if not path.is_relative_to(resolved_root) or sha256_file(path) != expected_sha256:
            raise ValueError(f"release asset hash mismatch: {relative}")
    checksums_path = resolved_root / "SHA256SUMS"
    if sha256_file(checksums_path) != candidate.checksums_sha256:
        raise ValueError("SHA256SUMS hash does not match the release candidate")
    parsed: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\\]+)", line)
        if match is None:
            raise ValueError("SHA256SUMS contains a malformed line")
        digest, relative = match.groups()
        if relative in parsed:
            raise ValueError("SHA256SUMS contains a duplicate path")
        parsed[relative] = digest
    if parsed != expected_records:
        raise ValueError("SHA256SUMS must cover the exact candidate asset inventory")


def validate_stable_design_set(designs: Sequence[StableEvidenceDesignV1]) -> Mapping[str, int]:
    by_id = {design.study_id: design for design in designs}
    if len(by_id) != len(designs) or set(by_id) != _STUDIES:
        raise ValueError("stable design set must contain each frozen study exactly once")
    rows: dict[str, int] = {}
    for study_id in sorted(by_id):
        design = by_id[study_id]
        design.validate_stable_matrix()
        rows[study_id] = design.expected_rows
    if sum(rows.values()) != 1_550:
        raise ValueError("stable evidence design set must contain exactly 1,550 evaluation rows")
    return rows


def expected_stable_matrix_keys(design: StableEvidenceDesignV1) -> tuple[tuple[Any, ...], ...]:
    design.validate_stable_matrix()
    task_ids: tuple[int | None, ...] = design.task_ids or (None,)
    routes: tuple[str | None, ...] = design.routes or (None,)
    interventions: tuple[str | None, ...] = design.interventions or (None,)
    return tuple(
        (
            policy,
            train_seed,
            task_id,
            evaluation_id,
            route,
            intervention,
        )
        for policy in design.policies
        for train_seed in design.train_seeds
        for task_id in task_ids
        for evaluation_id in design.evaluation_ids
        for route in routes
        for intervention in interventions
    )


def stable_row_inventory_sha256(rows: Sequence[StableEvidenceRowV1]) -> str:
    actual_rows = tuple(rows)
    if any(not isinstance(row, StableEvidenceRowV1) for row in actual_rows):
        raise TypeError("stable evidence rows must contain StableEvidenceRowV1 values")
    return _stable_hash(
        {
            "rows": [
                row.to_dict()
                for row in sorted(
                    actual_rows,
                    key=lambda item: json.dumps(item.matrix_key, separators=(",", ":")),
                )
            ]
        }
    )


def build_stable_evidence_summary(
    design: StableEvidenceDesignV1,
    rows: Sequence[StableEvidenceRowV1],
    sentinel: StableRepeatSentinelV1,
    *,
    expected_git_sha: str,
    statistics_sha256: str,
    claim_gate_sha256: str,
) -> StableEvidenceSummaryV1:
    design.validate_stable_matrix()
    expected_git_sha = _git_sha(expected_git_sha, "expected evidence git_sha")
    statistics_sha256 = _sha256(statistics_sha256, "statistics_sha256")
    claim_gate_sha256 = _sha256(claim_gate_sha256, "claim_gate_sha256")
    actual_rows = tuple(rows)
    if any(not isinstance(row, StableEvidenceRowV1) for row in actual_rows):
        raise TypeError("stable evidence rows must contain StableEvidenceRowV1 values")
    expected = set(expected_stable_matrix_keys(design))
    actual_keys = tuple(row.matrix_key for row in actual_rows)
    actual = set(actual_keys)
    matrix_complete = (
        len(actual_keys) == len(actual)
        and actual == expected
        and all(row.study_id == design.study_id for row in actual_rows)
    )
    inventory = stable_row_inventory_sha256(actual_rows)
    repeat_inventory = stable_row_inventory_sha256(
        tuple(row for row in actual_rows if row.train_seed == design.repeat_train_seed)
    )
    dependencies: dict[str, set[str]] = {}
    upstreams: dict[str, set[str]] = {}
    for row in actual_rows:
        dependencies.setdefault(row.policy, set()).add(row.dependency_lock_sha256)
        upstreams.setdefault(row.policy, set()).add(row.upstream_identity_sha256)
    homogeneous = (
        all(row.git_sha == expected_git_sha for row in actual_rows)
        and all(len(values) == 1 for values in dependencies.values())
        and all(len(values) == 1 for values in upstreams.values())
    )
    dependency_hash = _stable_hash(
        {policy: sorted(values) for policy, values in sorted(dependencies.items())}
    )
    upstream_hash = _stable_hash(
        {policy: sorted(values) for policy, values in sorted(upstreams.items())}
    )
    sentinel_verified = (
        sentinel.study_id == design.study_id
        and sentinel.train_seed == design.repeat_train_seed
        and sentinel.source_row_inventory_sha256 == repeat_inventory
        and sentinel.verified
    )
    reasons: list[str] = []
    if not matrix_complete:
        reasons.append("incomplete_matrix")
    if not homogeneous:
        reasons.append("mixed_source")
    if not sentinel_verified:
        reasons.append("sentinel_failure")
    return StableEvidenceSummaryV1(
        study_id=design.study_id,
        design_sha256=design.sha256(),
        git_sha=expected_git_sha,
        dependency_lock_sha256=dependency_hash,
        upstream_identity_sha256=upstream_hash,
        row_inventory_sha256=inventory,
        statistics_sha256=statistics_sha256,
        sentinel_sha256=sentinel.sha256(),
        expected_rows=design.expected_rows,
        observed_rows=len(actual_rows),
        matrix_complete=matrix_complete,
        homogeneous_source=homogeneous,
        sentinel_verified=sentinel_verified,
        claim_gate_sha256=claim_gate_sha256,
        release_eligible=not reasons,
        gate_reasons=tuple(reasons),
    )


def verify_stable_evidence_bundle(
    design: StableEvidenceDesignV1,
    rows: Sequence[StableEvidenceRowV1],
    sentinel: StableRepeatSentinelV1,
    summary: StableEvidenceSummaryV1,
) -> None:
    recomputed = build_stable_evidence_summary(
        design,
        rows,
        sentinel,
        expected_git_sha=summary.git_sha,
        statistics_sha256=summary.statistics_sha256,
        claim_gate_sha256=summary.claim_gate_sha256,
    )
    if recomputed != summary:
        raise ValueError("stable evidence summary does not match independently recomputed rows")


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    successes = _integer(successes, "successes")
    trials = _integer(trials, "trials", positive=True)
    if successes > trials:
        raise ValueError("successes cannot exceed trials")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return (center - radius, center + radius)


def clustered_paired_bootstrap(
    differences_by_seed: Mapping[int, Sequence[float]],
    *,
    samples: int = STABLE_BOOTSTRAP_SAMPLES,
    seed: int = STABLE_ANALYSIS_SEED,
) -> tuple[float, float]:
    samples = _integer(samples, "bootstrap samples", positive=True)
    seed = _integer(seed, "bootstrap seed")
    if not differences_by_seed:
        raise ValueError("clustered bootstrap requires at least one training seed")
    clusters: dict[int, np.ndarray[Any, np.dtype[np.float64]]] = {}
    for train_seed, cluster_values in differences_by_seed.items():
        key = _integer(train_seed, "cluster train seed")
        array = np.asarray(tuple(cluster_values), dtype=np.float64)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError("every bootstrap cluster must be a non-empty finite vector")
        clusters[key] = array
    keys = tuple(sorted(clusters))
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_keys = rng.choice(keys, size=len(keys), replace=True)
        sampled_values: list[float] = []
        for key in sampled_keys:
            cluster = clusters[int(key)]
            selected = rng.choice(cluster, size=cluster.size, replace=True)
            sampled_values.extend(float(item) for item in selected)
        estimates[index] = float(np.mean(sampled_values))
    low, high = np.quantile(estimates, (0.025, 0.975))
    return (float(low), float(high))
