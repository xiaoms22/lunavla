from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_v2_release_profile import (
    EVIDENCE_ARCHIVE_NAMES,
    SHA256_PATTERN,
    canonical_origin,
    evidence_profile_design,
    installed_requirements,
    sbom_command,
    sha256_file,
    validated_project_version,
    write_evidence_candidate,
)


def test_release_sha_contract_is_exact() -> None:
    assert SHA256_PATTERN.fullmatch("a" * 40)
    assert not SHA256_PATTERN.fullmatch("A" * 40)
    assert not SHA256_PATTERN.fullmatch("a" * 39)
    assert not SHA256_PATTERN.fullmatch("../" + "a" * 40)


def test_release_sha256_file(tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"lunavla-v2\n")
    assert sha256_file(source) == hashlib.sha256(b"lunavla-v2\n").hexdigest()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "https://github.com/xiaoms22/lunavla.git",
            "https://github.com/xiaoms22/lunavla",
        ),
        (
            "https://github.com/xiaoms22/lunavla/",
            "https://github.com/xiaoms22/lunavla",
        ),
        ("git@github.com:xiaoms22/lunavla.git", "git@github.com:xiaoms22/lunavla"),
    ],
)
def test_release_origin_normalization(raw: str, expected: str) -> None:
    assert canonical_origin(raw) == expected


def test_installed_requirements_is_sorted_and_contains_project() -> None:
    rows = installed_requirements().splitlines()
    canonical = [row.split("==", maxsplit=1)[0].lower().replace("_", "-") for row in rows]
    assert canonical == sorted(canonical)
    assert any(row.lower().startswith("lunavla==") for row in rows)


def test_release_version_contract_matches_all_sources() -> None:
    assert validated_project_version() == "2.0.0b1"


def test_release_version_contract_fails_closed_on_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    monkeypatch.setattr(release.importlib.metadata, "version", lambda _: "2.0.0a0")
    with pytest.raises(RuntimeError, match="inconsistent LunaVLA version contract"):
        release.validated_project_version()


def test_sbom_command_uses_cyclonedx_7_output_contract(tmp_path: Path) -> None:
    command = sbom_command(tmp_path / "requirements.txt", tmp_path / "sbom.json")
    assert command[:2] == ("cyclonedx-py", "requirements")
    assert "--output-reproducible" in command
    assert "--output-file" in command
    assert "--outfile" not in command
    assert command[-1].endswith("sbom.json")


@pytest.mark.parametrize(
    ("profile", "suite", "design_id", "archive_name", "run_count"),
    [
        (
            "language",
            "language",
            "language-alpha2",
            "lunavla-v2-language-evidence.tar.gz",
            5,
        ),
        (
            "vision",
            "visual",
            "visual-beta1",
            "lunavla-v2-vision-evidence.tar.gz",
            10,
        ),
    ],
)
def test_controlled_profiles_select_exact_full_designs(
    profile: str,
    suite: str,
    design_id: str,
    archive_name: str,
    run_count: int,
) -> None:
    from lunavla.evidence_runner import derive_plan, is_full_design

    _, design = evidence_profile_design(profile)
    plan = derive_plan(design)
    assert design.suite == suite
    assert design.design_id == design_id
    assert is_full_design(design)
    assert plan.expected_training_runs == run_count
    assert plan.expected_arm_episodes == 480
    assert EVIDENCE_ARCHIVE_NAMES[profile] == archive_name


def test_language_and_vision_main_route_without_running_full_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(release, "verify_source", lambda expected_sha: None)
    monkeypatch.setattr(
        release,
        "run_controlled_profile",
        lambda profile, expected_sha: calls.append((profile, expected_sha)),
    )
    for profile in ("language", "vision"):
        assert release.main(["--profile", profile, "--expected-sha", "a" * 40]) == 0
    assert calls == [("language", "a" * 40), ("vision", "a" * 40)]


def test_evidence_candidate_preserves_fail_closed_claim_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    release_root = tmp_path / "release-assets"
    release_root.mkdir()
    monkeypatch.setattr(release, "RELEASE_ROOT", release_root)
    monkeypatch.setattr(release, "validated_project_version", lambda: "2.0.0b1")
    claim_payload = {
        "claim_id": "instruction_following",
        "allowed": False,
        "statement": "Instruction-following has not yet been established.",
    }
    claim = SimpleNamespace(allowed=False, to_dict=lambda: claim_payload)
    evidence = SimpleNamespace(
        profile="language",
        design=SimpleNamespace(
            design_id="language-alpha2",
            sha256=lambda: "b" * 64,
        ),
        manifest=SimpleNamespace(
            claims=(claim,),
            matrix_complete=True,
            reduced_design=False,
        ),
        verification=SimpleNamespace(source_count=5, arm_episode_count=480),
        output_root=release.ROOT / "outputs/evidence/language-alpha2",
        snapshot_root=release.ROOT / "results/v2/language-alpha2",
        runs=(),
    )
    metadata = {
        name: release_root / filename
        for name, filename in {
            "design": "language-evidence-design.yaml",
            "manifest": "language-evidence-manifest.json",
            "verification": "language-evidence-verification.json",
            "claims": "language-claim-summary.json",
            "file_checksums": "language-evidence-files.SHA256SUMS",
        }.items()
    }
    for path in metadata.values():
        path.write_text("{}\n", encoding="utf-8")
    distribution = release_root / "dist" / "lunavla.whl"
    distribution.parent.mkdir()
    distribution.write_bytes(b"wheel")
    candidate = write_evidence_candidate(
        expected_sha="a" * 40,
        evidence=evidence,
        metadata=metadata,
        distributions=(distribution,),
    )
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    assert payload["package_version"] == "2.0.0b1"
    assert payload["modality_effect_claims"] is False
    assert payload["claims"] == [claim_payload]


def test_distribution_names_must_match_beta1_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    release_root = tmp_path / "release-assets"
    monkeypatch.setattr(release, "RELEASE_ROOT", release_root)
    monkeypatch.setattr(release, "validated_project_version", lambda: "2.0.0b1")

    def fake_run(command: tuple[str, ...], *, capture: bool = False) -> str:
        del capture
        if "build" in command:
            destination = release_root / "dist"
            destination.mkdir(parents=True)
            (destination / "lunavla-2.0.0b1-py3-none-any.whl").write_bytes(b"wheel")
            (destination / "lunavla-2.0.0b1.tar.gz").write_bytes(b"sdist")
        return ""

    monkeypatch.setattr(release, "run", fake_run)
    assert {path.name for path in release.build_distributions()} == {
        "lunavla-2.0.0b1-py3-none-any.whl",
        "lunavla-2.0.0b1.tar.gz",
    }


@pytest.mark.parametrize(
    ("field", "replacement", "failure"),
    [
        ("git_sha", "b" * 40, "git_sha"),
        ("training_device", "mps", "training_device"),
        ("recorded_device", "cuda", "recorded_device"),
    ],
)
def test_controlled_source_checks_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
    failure: str,
) -> None:
    from scripts import run_v2_release_profile as release

    output_root = tmp_path / "outputs/evidence/language-alpha2"
    manifest_path = output_root / "runs/run-11/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    values = {
        "git_sha": "a" * 40,
        "training_device": "cpu",
        "recorded_device": "cpu",
    }
    values[field] = replacement
    manifest = SimpleNamespace(
        schema_version=3,
        git_sha=values["git_sha"],
        git_dirty=False,
        source_diff_sha256=None,
        config={
            "training": {"device": values["training_device"]},
            "policy": {"device": "cpu"},
        },
        runtime_determinism={"device": values["recorded_device"]},
        checkpoint_sha256="c" * 64,
        policy_id="transformer_chunk_cvae",
        task_id="language_conditioned_point_reach",
    )
    monkeypatch.setattr(
        release.RunManifest,
        "verify_run_dir",
        lambda run_dir: manifest,
    )
    monkeypatch.setattr(release, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match=failure):
        release._release_run_row(
            output_root=output_root,
            source_run_id="run-11",
            expected_sha="a" * 40,
        )


def test_controlled_archives_have_distinct_names_and_both_evidence_trees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    release_root = tmp_path / "release-assets"
    output_root = tmp_path / "outputs/evidence/language-alpha2"
    snapshot_root = tmp_path / "results/v2/language-alpha2"
    (release_root / "dist").mkdir(parents=True)
    output_root.mkdir(parents=True)
    snapshot_root.mkdir(parents=True)
    (output_root / "checkpoint.pt").write_bytes(b"checkpoint")
    (snapshot_root / "snapshot_manifest.json").write_text("{}\n", encoding="utf-8")
    for name in (
        "release-candidate.json",
        "environment-requirements.txt",
        "sbom.json",
    ):
        (release_root / name).write_text("{}\n", encoding="utf-8")
    distribution = release_root / "dist/lunavla.whl"
    distribution.write_bytes(b"wheel")
    metadata = {
        name: release_root / filename
        for name, filename in {
            "design": "language-evidence-design.yaml",
            "manifest": "language-evidence-manifest.json",
            "verification": "language-evidence-verification.json",
            "claims": "language-claim-summary.json",
        }.items()
    }
    for path in metadata.values():
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "RELEASE_ROOT", release_root)
    evidence = SimpleNamespace(
        profile="language",
        output_root=output_root,
        snapshot_root=snapshot_root,
    )
    metadata["file_checksums"] = release.write_evidence_file_checksums(
        evidence,
        metadata,
        (distribution,),
    )
    archive = release.write_controlled_evidence_archive(evidence, metadata)
    assert archive.name == "lunavla-v2-language-evidence.tar.gz"
    with tarfile.open(archive, "r:gz") as stream:
        names = set(stream.getnames())
    assert "outputs/evidence/language-alpha2/checkpoint.pt" in names
    assert "results/v2/language-alpha2/snapshot_manifest.json" in names
    assert "release-assets/language-evidence-manifest.json" in names
    assert "release-assets/language-evidence-files.SHA256SUMS" in names
    assert "release-assets/dist/lunavla.whl" in names
    checksum_text = metadata["file_checksums"].read_text(encoding="utf-8")
    assert "outputs/evidence/language-alpha2/checkpoint.pt" in checksum_text
    assert "results/v2/language-alpha2/snapshot_manifest.json" in checksum_text


def test_stable_profile_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_v2_release_profile as release

    monkeypatch.setattr(release, "verify_source", lambda expected_sha: None)
    with pytest.raises(RuntimeError, match="stable release contract"):
        release.main(["--profile", "stable", "--expected-sha", "a" * 40])
