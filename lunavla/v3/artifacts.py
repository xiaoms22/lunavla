from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


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
            "format": "lunavla_v3_checkpoint_envelope",
            "policy_id": self.policy_id,
            "checkpoint_file": self.checkpoint_file,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
        }

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

    def save(self, path: str | Path) -> Path:
        return _write_json(Path(path), self.to_dict())


def verify_run_directory(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("manifest.json does not exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 4:
        raise ValueError("manifest schema_version must be 4")
    checks = {
        "resolved_config.json": manifest["config_sha256"],
        "data_audit.json": manifest["data_audit_sha256"],
        "checkpoint.v3.json": manifest["checkpoint_envelope_sha256"],
        "metrics.json": manifest["metrics_sha256"],
    }
    for name, expected in checks.items():
        actual = sha256_file(root / name)
        if actual != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    envelope = json.loads((root / "checkpoint.v3.json").read_text(encoding="utf-8"))
    checkpoint = root / envelope["checkpoint_file"]
    if sha256_file(checkpoint) != envelope["checkpoint_sha256"]:
        raise ValueError("artifact hash mismatch: policy checkpoint")
    return manifest
