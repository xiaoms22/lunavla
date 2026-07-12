from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_v3_code_release import (
    REQUIRED_CHECKS,
    _asset_records,
    verify_test_manifest,
    write_evidence_bundle,
)


GIT_SHA = "a" * 40


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "git_sha": GIT_SHA,
        "checks": {name: "success" for name in sorted(REQUIRED_CHECKS)},
        "pretrained_enabled": False,
        "weight_network_accessed": False,
        "claim_allowed": False,
    }


def test_code_release_manifest_is_exact_and_weight_free(tmp_path: Path) -> None:
    path = tmp_path / "test-manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert verify_test_manifest(path, expected_git_sha=GIT_SHA)["claim_allowed"] is False
    payload = _manifest()
    payload["weight_network_accessed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="weight network access"):
        verify_test_manifest(path, expected_git_sha=GIT_SHA)
    payload = _manifest()
    payload["unknown"] = "forbidden"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        verify_test_manifest(path, expected_git_sha=GIT_SHA)


def test_code_evidence_archive_is_byte_reproducible(tmp_path: Path) -> None:
    for name in (
        "environment-requirements.txt",
        "provenance.jsonl",
        "sbom.json",
        "smolvla-conformance-status.json",
        "test-manifest.json",
    ):
        (tmp_path / name).write_text(f"{name}\n", encoding="utf-8")
    first = write_evidence_bundle(tmp_path).read_bytes()
    (tmp_path / "lunavla-v3-alpha2-code-evidence.tar.gz").unlink()
    second = write_evidence_bundle(tmp_path).read_bytes()
    assert first == second


def test_code_asset_inventory_rejects_weights_and_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "safe.json").write_text("{}\n", encoding="utf-8")
    assert len(_asset_records(tmp_path)) == 1
    (tmp_path / "model.safetensors").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="forbidden model artifact"):
        _asset_records(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "checkpoint-summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden model artifact"):
        _asset_records(tmp_path)
