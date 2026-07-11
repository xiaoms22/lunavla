from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lunavla.published_evidence import (
    INSTRUCTION_FOLLOWING_DENIED,
    PUBLISHED_LANGUAGE_EVIDENCE_SHA256,
    PUBLISHED_LANGUAGE_GIT_SHA,
    PUBLISHED_LANGUAGE_SNAPSHOT_SHA256,
    PUBLISHED_VISUAL_EVIDENCE_SHA256,
    PUBLISHED_VISUAL_GIT_SHA,
    PUBLISHED_VISUAL_SNAPSHOT_SHA256,
    VISUAL_CONTROL_CONTRIBUTION_DENIED,
    verify_language_snapshot,
    verify_visual_snapshot,
)
from scripts.render_readme_results import render_v2_language, render_v2_visual


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "results" / "v2" / "language-alpha2"
VISUAL_SNAPSHOT = ROOT / "results" / "v2" / "visual-beta1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _copy_snapshot(tmp_path: Path) -> Path:
    destination = tmp_path / "language-alpha2"
    shutil.copytree(SNAPSHOT, destination)
    return destination


def _rehash_evidence(snapshot: Path) -> None:
    evidence_path = snapshot / "evidence_manifest.json"
    digest = _sha256(evidence_path)
    manifest_path = snapshot / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["evidence_manifest.json"] = digest
    manifest["source_evidence_manifest_sha256"] = digest
    _write_json(manifest_path, manifest)


def _rehash_snapshot_file(snapshot: Path, relative: str) -> None:
    manifest_path = snapshot / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][relative] = _sha256(snapshot / relative)
    _write_json(manifest_path, manifest)


def _trust_modified_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: Path,
    *,
    evidence_changed: bool = False,
) -> None:
    from lunavla import published_evidence

    if evidence_changed:
        monkeypatch.setattr(
            published_evidence,
            "PUBLISHED_LANGUAGE_EVIDENCE_SHA256",
            _sha256(snapshot / "evidence_manifest.json"),
        )
    monkeypatch.setattr(
        published_evidence,
        "PUBLISHED_LANGUAGE_SNAPSHOT_SHA256",
        _sha256(snapshot / "snapshot_manifest.json"),
    )


def test_tracked_language_snapshot_is_read_only_verified_and_claim_closed() -> None:
    before = {path: _sha256(path) for path in SNAPSHOT.rglob("*") if path.is_file()}
    published = verify_language_snapshot(SNAPSHOT)
    after = {path: _sha256(path) for path in SNAPSHOT.rglob("*") if path.is_file()}

    assert after == before
    assert published.train_seed_count == 5
    assert published.control_trials == 120
    assert [item.scope for item in published.arm_wilson] == [
        "control",
        "mask",
        "shuffle",
        "counterfactual",
    ]
    assert all(item.sample_n == 120 for item in published.arm_wilson)
    assert published.counterfactual_final_distance.sample_n == 120
    assert published.counterfactual_success.sample_n == 120
    assert published.statement == INSTRUCTION_FOLLOWING_DENIED
    assert published.git_sha == PUBLISHED_LANGUAGE_GIT_SHA
    assert published.evidence_manifest_sha256 == PUBLISHED_LANGUAGE_EVIDENCE_SHA256
    assert _sha256(SNAPSHOT / "snapshot_manifest.json") == PUBLISHED_LANGUAGE_SNAPSHOT_SHA256


def test_v2_renderer_reports_every_wilson_arm_and_both_paired_intervals() -> None:
    rendered = render_v2_language(SNAPSHOT, ROOT / "README.md")

    assert "5 training seeds and 120 paired control trials" in rendered
    assert "| `control` | 5 | 120 | 3.3% (1.3%–8.3%) |" in rendered
    assert "| `mask` | 5 | 120 | 0.0% (0.0%–3.1%) |" in rendered
    assert "| `shuffle` | 5 | 120 | 13.3% (8.4%–20.6%) |" in rendered
    assert "| `counterfactual` | 5 | 120 | 0.0% (0.0%–3.1%) |" in rendered
    assert "| Final distance | 120 | +0.0544 | [+0.0144, +0.0910] |" in rendered
    assert "| Success-rate difference | 120 | -3.3 pp | [-7.5, +0.0] pp |" in rendered
    assert f"**Claim gate: {INSTRUCTION_FOLLOWING_DENIED}**" in rendered
    assert "does not establish that the policy follows instructions" in rendered
    assert "verified failed check is `control_success_advantage`" in rendered
    assert "must not be described as successful instruction-following" in rendered
    assert "`a546695`" in rendered
    assert "workflow run 29106885353" in rendered
    assert PUBLISHED_LANGUAGE_EVIDENCE_SHA256 in rendered


def test_tracked_visual_snapshot_is_read_only_verified_and_claim_closed() -> None:
    before = {path: _sha256(path) for path in VISUAL_SNAPSHOT.rglob("*") if path.is_file()}
    published = verify_visual_snapshot(VISUAL_SNAPSHOT)
    after = {path: _sha256(path) for path in VISUAL_SNAPSHOT.rglob("*") if path.is_file()}

    assert after == before
    assert published.train_seed_count == 5
    assert published.control_trials == 120
    assert [item.scope for item in published.arm_wilson] == [
        "control",
        "occlusion",
        "shuffle",
        "state_only",
    ]
    assert all(item.sample_n == 120 for item in published.arm_wilson)
    assert [item.sample_n for item in published.paired_final_distance] == [
        120,
        60,
        60,
        120,
        60,
        60,
    ]
    assert published.statement == VISUAL_CONTROL_CONTRIBUTION_DENIED
    assert published.git_sha == PUBLISHED_VISUAL_GIT_SHA
    assert published.evidence_manifest_sha256 == PUBLISHED_VISUAL_EVIDENCE_SHA256
    assert _sha256(VISUAL_SNAPSHOT / "snapshot_manifest.json") == PUBLISHED_VISUAL_SNAPSHOT_SHA256


def test_v2_visual_renderer_reports_all_claim_critical_intervals() -> None:
    rendered = render_v2_visual(VISUAL_SNAPSHOT, ROOT / "README.md")

    assert "5 image-policy and 5 state-only training runs" in rendered
    assert "| `control` | 5 | 120 | 1.7% (0.5%–5.9%) |" in rendered
    assert "| `occlusion` | 5 | 120 | 5.8% (2.9%–11.6%) |" in rendered
    assert "| `state_only` | 5 | 120 | 1.7% (0.5%–5.9%) |" in rendered
    assert "| `occlusion:all` | 120 | -0.0106 | [-0.0907, +0.0667] |" in rendered
    assert "| `state_only:all` | 120 | +0.0121 | [-0.0193, +0.0453] |" in rendered
    assert f"**Claim gate: {VISUAL_CONTROL_CONTRIBUTION_DENIED}**" in rendered
    assert "must not be described as evidence that images improve control" in rendered
    assert "`bf0e550`" in rendered
    assert "workflow run 29110701437" in rendered
    assert PUBLISHED_VISUAL_EVIDENCE_SHA256 in rendered


def test_snapshot_verifier_rejects_a_changed_listed_file(tmp_path: Path) -> None:
    snapshot = _copy_snapshot(tmp_path)
    with (snapshot / "per_pair.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")

    with pytest.raises(ValueError, match="snapshot file hash mismatch: per_pair.csv"):
        verify_language_snapshot(snapshot)


def _tamper_statistic(payload: dict[str, Any]) -> None:
    statistic = next(
        item for item in payload["statistics"] if item["statistic_id"] == "control-success"
    )
    statistic["estimate"] = 0.05


def _tamper_claim(payload: dict[str, Any]) -> None:
    claim = payload["claims"][0]
    claim["checks"]["control_success_advantage"] = True
    claim["allowed"] = True
    claim["failed_checks"] = []
    claim["statement"] = claim["allowed_statement"]


@pytest.mark.parametrize("tamper", [_tamper_statistic, _tamper_claim])
def test_snapshot_verifier_recomputes_statistics_and_claims_after_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: Callable[[dict[str, Any]], None],
) -> None:
    snapshot = _copy_snapshot(tmp_path)
    evidence_path = snapshot / "evidence_manifest.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    tamper(evidence)
    _write_json(evidence_path, evidence)
    _rehash_evidence(snapshot)
    _trust_modified_snapshot(monkeypatch, snapshot, evidence_changed=True)

    with pytest.raises(ValueError, match="statistics or claim gates differ"):
        verify_language_snapshot(snapshot)


def test_snapshot_verifier_rejects_source_evidence_hash_tampering(tmp_path: Path) -> None:
    snapshot = _copy_snapshot(tmp_path)
    manifest_path = snapshot / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_evidence_manifest_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="registered official workflow artifact"):
        verify_language_snapshot(snapshot)


def test_publication_registry_rejects_coordinated_snapshot_rehash(
    tmp_path: Path,
) -> None:
    snapshot = _copy_snapshot(tmp_path)
    manifest_path = snapshot / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verification"]["git_sha"] = "b" * 40
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="registered official workflow artifact"):
        verify_language_snapshot(snapshot)


def test_snapshot_verifier_binds_primary_and_repeat_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _copy_snapshot(tmp_path)
    primary = "runs/language-alpha2-seed-11/metrics.json"
    metrics = json.loads((snapshot / primary).read_text(encoding="utf-8"))
    metrics["final_loss"] = 999.0
    _write_json(snapshot / primary, metrics)
    _rehash_snapshot_file(snapshot, primary)
    _trust_modified_snapshot(monkeypatch, snapshot)
    with pytest.raises(ValueError, match="not bound by RunManifest"):
        verify_language_snapshot(snapshot)

    snapshot = _copy_snapshot(tmp_path / "repeat")
    repeat = "reproducibility/language-alpha2-seed-11-repeat/metrics.json"
    metrics = json.loads((snapshot / repeat).read_text(encoding="utf-8"))
    metrics["final_loss"] = 999.0
    _write_json(snapshot / repeat, metrics)
    _rehash_snapshot_file(snapshot, repeat)
    _trust_modified_snapshot(monkeypatch, snapshot)
    with pytest.raises(ValueError, match="repeat metrics differ"):
        verify_language_snapshot(snapshot)


def test_snapshot_verifier_binds_per_pair_family_to_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _copy_snapshot(tmp_path)
    path = snapshot / "per_pair.csv"
    rows = path.read_text(encoding="utf-8").splitlines()
    fields = rows[1].split(",")
    fields[7] = "fabricated_family"
    rows[1] = ",".join(fields)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _rehash_snapshot_file(snapshot, "per_pair.csv")
    _trust_modified_snapshot(monkeypatch, snapshot)

    with pytest.raises(ValueError, match="task family differs from eval fixture"):
        verify_language_snapshot(snapshot)
