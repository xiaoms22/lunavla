from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import ArtifactHashRecordV1


RC_TAG = "v3.0.0-rc.1"
RC_PACKAGE_VERSION = "3.0.0rc1"
STABLE_TAG = "v3.0.0"
STABLE_PACKAGE_VERSION = "3.0.0"
STABLE_TRAIN_SEEDS = (11, 22, 33, 44, 55)
STABLE_DATA_SEED = 42
STABLE_ANALYSIS_SEED = 202701
STABLE_BOOTSTRAP_SAMPLES = 10_000
PUSHT_POLICIES = ("act_v3", "diffusion_v3", "lerobot_smolvla")
LIBERO_TASK_IDS = (0, 1, 2, 3)
LIBERO_ROUTES = ("none", "expert_only", "prompt_only", "dual")
PROMPT_INTERVENTIONS = ("control", "mask", "shuffle", "counterfactual", "layout_drift")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_STUDIES = {
    "pusht_policy_ladder",
    "libero_state_routes",
    "libero_prompt_interventions",
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
        if self.study_id == "pusht_policy_ladder":
            expected = (PUSHT_POLICIES, (), tuple(range(20)), (), ())
        elif self.study_id == "libero_state_routes":
            expected = (("lerobot_smolvla",), LIBERO_TASK_IDS, tuple(range(10)), LIBERO_ROUTES, ())
        else:
            expected = (
                ("lerobot_smolvla",),
                LIBERO_TASK_IDS,
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
class StableReleaseCandidateV1:
    expected_tag: str
    package_version: str
    git_sha: str
    merge_sha: str
    public_api_sha256: str
    migration_report_sha256: str
    alpha2_gpu_manifest_sha256: str
    license_review_sha256: str
    authoritative_runner_sha256: str
    secondary_runner_sha256: str
    integration_manifests: tuple[ArtifactHashRecordV1, ...]
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

    def __post_init__(self) -> None:
        if _integer(self.schema_version, "stable release candidate schema_version", positive=True) != 1:
            raise ValueError("StableReleaseCandidateV1 schema_version must be integer 1")
        if self.expected_tag != STABLE_TAG or self.package_version != STABLE_PACKAGE_VERSION:
            raise ValueError("stable release candidate tag/package mapping is invalid")
        git_sha = _git_sha(self.git_sha, "stable release git_sha")
        merge_sha = _git_sha(self.merge_sha, "stable release merge_sha")
        if git_sha != merge_sha:
            raise ValueError("stable release evidence must be rerun on the actual merge SHA")
        object.__setattr__(self, "git_sha", git_sha)
        object.__setattr__(self, "merge_sha", merge_sha)
        for name in (
            "public_api_sha256",
            "migration_report_sha256",
            "alpha2_gpu_manifest_sha256",
            "license_review_sha256",
            "authoritative_runner_sha256",
            "secondary_runner_sha256",
            "required_checks_sha256",
            "sbom_sha256",
            "provenance_sha256",
            "checksums_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        integrations = _records(self.integration_manifests, "integration_manifests")
        evidence = _records(self.evidence_manifests, "evidence_manifests")
        assets = _records(self.assets, "stable release assets")
        if {item.path for item in integrations} != {
            "integration/authoritative.json",
            "integration/secondary.json",
        }:
            raise ValueError("stable release requires authoritative and secondary integration manifests")
        if {item.path for item in evidence} != {
            "evidence/pusht-policy-ladder.json",
            "evidence/libero-state-routes.json",
            "evidence/libero-prompt-interventions.json",
        }:
            raise ValueError("stable release requires all three frozen evidence manifests")
        required_assets = {
            "dist/lunavla-3.0.0-py3-none-any.whl",
            "dist/lunavla-3.0.0.tar.gz",
            "sbom.json",
            "provenance.jsonl",
            "evidence.tar.gz",
        }
        if not required_assets.issubset(item.path for item in assets):
            raise ValueError("stable release candidate is missing required assets")
        for name in (
            "signed_tag_verified",
            "post_merge_evidence",
            "privacy_scan_verified",
            "pypi_published",
        ):
            _boolean(getattr(self, name), name)
        if not self.signed_tag_verified or not self.post_merge_evidence or not self.privacy_scan_verified:
            raise ValueError("stable release requires signed tag, post-merge evidence and privacy scan")
        if self.pypi_published:
            raise ValueError("LunaVLA v3 stable does not publish to PyPI")
        object.__setattr__(self, "integration_manifests", integrations)
        object.__setattr__(self, "evidence_manifests", evidence)
        object.__setattr__(self, "assets", assets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "expected_tag": self.expected_tag,
            "package_version": self.package_version,
            "git_sha": self.git_sha,
            "merge_sha": self.merge_sha,
            "public_api_sha256": self.public_api_sha256,
            "migration_report_sha256": self.migration_report_sha256,
            "alpha2_gpu_manifest_sha256": self.alpha2_gpu_manifest_sha256,
            "license_review_sha256": self.license_review_sha256,
            "authoritative_runner_sha256": self.authoritative_runner_sha256,
            "secondary_runner_sha256": self.secondary_runner_sha256,
            "integration_manifests": [item.to_dict() for item in self.integration_manifests],
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
    def from_mapping(cls, value: Mapping[str, Any]) -> "StableReleaseCandidateV1":
        payload = _exact(value, set(cls.__dataclass_fields__), "stable release candidate")
        for field in ("integration_manifests", "evidence_manifests", "assets"):
            payload[field] = _records(payload[field], field)
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


def validate_stable_design_set(designs: Sequence[StableEvidenceDesignV1]) -> Mapping[str, int]:
    by_id = {design.study_id: design for design in designs}
    if len(by_id) != len(designs) or set(by_id) != _STUDIES:
        raise ValueError("stable design set must contain each frozen study exactly once")
    rows: dict[str, int] = {}
    for study_id in sorted(by_id):
        design = by_id[study_id]
        design.validate_stable_matrix()
        rows[study_id] = design.expected_rows
    if sum(rows.values()) != 2_100:
        raise ValueError("stable evidence design set must contain exactly 2,100 evaluation rows")
    return rows

