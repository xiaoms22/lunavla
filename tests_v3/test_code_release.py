from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.run_v3_code_release import (
    REQUIRED_CHECKS,
    _asset_records,
    normalize_sdist,
    verify_test_manifest,
    write_evidence_bundle,
)


GIT_SHA = "a" * 40


def _write_sdist(path: Path, *, mtime: int, member_name: str = "pkg/data.txt") -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        payload = b"stable payload\n"
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        info.mtime = mtime
        info.uid = mtime
        info.gid = mtime
        info.uname = "builder"
        info.gname = "builder"
        archive.addfile(info, io.BytesIO(payload))


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
    (tmp_path / "lunavla-v3-alpha3-code-evidence.tar.gz").unlink()
    second = write_evidence_bundle(tmp_path).read_bytes()
    assert first == second


def test_sdist_normalization_is_byte_reproducible_and_safe(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_sdist(first, mtime=10)
    _write_sdist(second, mtime=20)
    normalize_sdist(first, source_date_epoch=123)
    normalize_sdist(second, source_date_epoch=123)
    assert first.read_bytes() == second.read_bytes()

    escaped = tmp_path / "escaped.tar.gz"
    _write_sdist(escaped, mtime=10, member_name="../outside.txt")
    with pytest.raises(ValueError, match="escapes"):
        normalize_sdist(escaped, source_date_epoch=123)


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
