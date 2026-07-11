from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def _exact_mapping(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown:
        raise ValueError(f"unknown field(s) in {name}: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing field(s) in {name}: {', '.join(missing)}")
    return dict(value)


def _integer(value: Any, name: str, expected: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if expected is not None and value != expected:
        raise ValueError(f"{name} must be {expected}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX_64.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256")
    return value


def _basename(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if value in {".", ".."} or Path(value).is_absolute() or Path(value).name != value or "\\" in value:
        raise ValueError(f"{name} must be a relative basename")
    return value


def _contained_file(root: Path, name: str) -> Path:
    filename = _basename(name, "artifact filename")
    resolved_root = root.resolve()
    candidate = (resolved_root / filename).resolve()
    if candidate.parent != resolved_root:
        raise ValueError(f"artifact path escapes run directory: {name}")
    if not candidate.is_file():
        raise FileNotFoundError(f"artifact file does not exist: {filename}")
    return candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


@dataclass(frozen=True)
class CheckpointEnvelopeV4:
    policy_id: str
    checkpoint_file: str
    checkpoint_sha256: str
    config_sha256: str
    feature_schema_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 4,
            "contract_revision": 1,
            "format": "lunavla_v3_checkpoint_envelope",
            "policy_id": self.policy_id,
            "checkpoint_file": self.checkpoint_file,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckpointEnvelopeV4":
        payload = _exact_mapping(
            value,
            {
                "schema_version", "contract_revision", "format", "policy_id",
                "checkpoint_file", "checkpoint_sha256", "config_sha256",
                "feature_schema_sha256",
            },
            "checkpoint envelope",
        )
        _integer(payload["schema_version"], "checkpoint schema_version", 4)
        _integer(payload["contract_revision"], "checkpoint contract_revision", 1)
        if payload["format"] != "lunavla_v3_checkpoint_envelope":
            raise ValueError("unsupported checkpoint envelope format")
        policy_id = payload["policy_id"]
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("checkpoint policy_id must be a non-empty string")
        return cls(
            policy_id=policy_id,
            checkpoint_file=_basename(payload["checkpoint_file"], "checkpoint_file"),
            checkpoint_sha256=_sha256(payload["checkpoint_sha256"], "checkpoint_sha256"),
            config_sha256=_sha256(payload["config_sha256"], "config_sha256"),
            feature_schema_sha256=_sha256(
                payload["feature_schema_sha256"], "feature_schema_sha256"
            ),
        )

    def save(self, path: str | Path) -> Path:
        return _write_json(Path(path), self.to_dict())


@dataclass(frozen=True)
class RunManifestV4:
    git_sha: str
    git_dirty: bool
    config_sha256: str
    feature_schema_sha256: str
    data_audit_sha256: str
    checkpoint_envelope_sha256: str
    metrics_sha256: str
    policy_id: str
    task_id: str
    train_seed: int
    evaluation_seeds: tuple[int, ...]
    deterministic: bool
    claim_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 4,
            "contract_revision": 1,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "config_sha256": self.config_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "data_audit_sha256": self.data_audit_sha256,
            "checkpoint_envelope_sha256": self.checkpoint_envelope_sha256,
            "metrics_sha256": self.metrics_sha256,
            "policy_id": self.policy_id,
            "task_id": self.task_id,
            "train_seed": self.train_seed,
            "evaluation_seeds": list(self.evaluation_seeds),
            "deterministic": self.deterministic,
            "claim_allowed": self.claim_allowed,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunManifestV4":
        fields = {
            "schema_version", "contract_revision", "git_sha", "git_dirty",
            "config_sha256", "feature_schema_sha256", "data_audit_sha256",
            "checkpoint_envelope_sha256", "metrics_sha256", "policy_id", "task_id",
            "train_seed", "evaluation_seeds", "deterministic", "claim_allowed",
        }
        payload = _exact_mapping(value, fields, "run manifest")
        _integer(payload["schema_version"], "manifest schema_version", 4)
        _integer(payload["contract_revision"], "manifest contract_revision", 1)
        if not isinstance(payload["git_sha"], str) or not _HEX_40.fullmatch(payload["git_sha"]):
            raise ValueError("manifest git_sha must be a full lowercase commit SHA")
        for name in ("git_dirty", "deterministic", "claim_allowed"):
            if not isinstance(payload[name], bool):
                raise TypeError(f"manifest {name} must be boolean")
        for name in (
            "config_sha256", "feature_schema_sha256", "data_audit_sha256",
            "checkpoint_envelope_sha256", "metrics_sha256",
        ):
            payload[name] = _sha256(payload[name], name)
        for name in ("policy_id", "task_id"):
            if not isinstance(payload[name], str) or not payload[name]:
                raise ValueError(f"manifest {name} must be a non-empty string")
        train_seed = _integer(payload["train_seed"], "manifest train_seed")
        seeds = payload["evaluation_seeds"]
        if isinstance(seeds, (str, bytes, Mapping)) or not isinstance(seeds, list):
            raise TypeError("manifest evaluation_seeds must be a list")
        evaluation_seeds = tuple(
            _integer(item, "manifest evaluation seed") for item in seeds
        )
        if not evaluation_seeds or len(evaluation_seeds) != len(set(evaluation_seeds)):
            raise ValueError("manifest evaluation_seeds must be non-empty and unique")
        return cls(
            git_sha=payload["git_sha"],
            git_dirty=payload["git_dirty"],
            config_sha256=payload["config_sha256"],
            feature_schema_sha256=payload["feature_schema_sha256"],
            data_audit_sha256=payload["data_audit_sha256"],
            checkpoint_envelope_sha256=payload["checkpoint_envelope_sha256"],
            metrics_sha256=payload["metrics_sha256"],
            policy_id=payload["policy_id"],
            task_id=payload["task_id"],
            train_seed=train_seed,
            evaluation_seeds=evaluation_seeds,
            deterministic=payload["deterministic"],
            claim_allowed=payload["claim_allowed"],
        )

    def save(self, path: str | Path) -> Path:
        return _write_json(Path(path), self.to_dict())


def verify_run_directory(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    manifest_path = _contained_file(root, "manifest.json")
    manifest = RunManifestV4.from_mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
    payload = manifest.to_dict()
    checks = {
        "resolved_config.json": manifest.config_sha256,
        "data_audit.json": manifest.data_audit_sha256,
        "checkpoint.v3.json": manifest.checkpoint_envelope_sha256,
        "metrics.json": manifest.metrics_sha256,
    }
    for name, expected in checks.items():
        actual = sha256_file(_contained_file(root, name))
        if actual != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    envelope_path = _contained_file(root, "checkpoint.v3.json")
    envelope = CheckpointEnvelopeV4.from_mapping(
        json.loads(envelope_path.read_text(encoding="utf-8"))
    )
    checkpoint = _contained_file(root, envelope.checkpoint_file)
    if sha256_file(checkpoint) != envelope.checkpoint_sha256:
        raise ValueError("artifact hash mismatch: policy checkpoint")
    return payload
