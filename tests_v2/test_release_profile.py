from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from lunavla.evidence import ClaimDecision, EvidenceManifest, EvidenceSource
from scripts.run_v2_release_profile import (
    EVIDENCE_ARCHIVE_NAMES,
    RC_ARCHIVE_NAME,
    RC_CONTRACT_SOURCES,
    RC_INTEGRITY_NAME,
    SHA256_PATTERN,
    STABLE_ARCHIVE_NAME,
    STABLE_OUTPUTS,
    STABLE_PACKAGE_VERSION,
    STABLE_SIGNER_WORKFLOW,
    STABLE_SOURCE_REF,
    STABLE_TAG,
    canonical_origin,
    evidence_profile_design,
    installed_requirements,
    sbom_command,
    sha256_file,
    validated_project_version,
    write_evidence_candidate,
)


def test_rc_contract_bundle_includes_evidence_semantics() -> None:
    assert Path("docs/v2/evidence_contract.md") in RC_CONTRACT_SOURCES


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
    assert validated_project_version() == "3.0.0rc1"


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


def test_rc_main_routes_to_contract_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_v2_release_profile as release

    calls: list[str] = []
    monkeypatch.setattr(release, "verify_source", lambda expected_sha: None)
    monkeypatch.setattr(release, "run_rc", lambda expected_sha: calls.append(expected_sha))

    assert release.main(["--profile", "rc", "--expected-sha", "a" * 40]) == 0
    assert calls == ["a" * 40]


def test_rc_candidate_and_archive_bind_contracts_evidence_and_distributions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    release_root = tmp_path / "release-assets"
    (release_root / "dist").mkdir(parents=True)
    (release_root / "contracts").mkdir()
    monkeypatch.setattr(release, "RELEASE_ROOT", release_root)
    monkeypatch.setattr(release, "validated_project_version", lambda: "2.0.0rc1")
    distribution = release_root / "dist/lunavla-2.0.0rc1-py3-none-any.whl"
    distribution.write_bytes(b"wheel")
    contract = release_root / "contracts/public_api_contract.json"
    contract.write_text("{}\n", encoding="utf-8")
    for name in ("environment-requirements.txt", "sbom.json"):
        (release_root / name).write_text("{}\n", encoding="utf-8")
    contracts = [
        {
            "source_path": "docs/v2/public_api_contract.json",
            "release_path": "contracts/public_api_contract.json",
            "sha256": sha256_file(contract),
        }
    ]
    evidence = [
        {
            "suite": "visual",
            "source_git_sha": "b" * 40,
            "evidence_manifest_sha256": "c" * 64,
            "snapshot_manifest_sha256": "d" * 64,
            "workflow_url": "https://github.com/xiaoms22/lunavla/actions/runs/1",
            "claim_allowed": False,
            "statement": "Visual-control contribution has not yet been established.",
        }
    ]

    candidate = release.write_rc_candidate(
        expected_sha="a" * 40,
        published_evidence=evidence,
        contracts=contracts,
        distributions=(distribution,),
    )
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    assert payload["profile"] == "rc"
    assert payload["package_version"] == "2.0.0rc1"
    assert payload["contract_freeze"] is True
    assert payload["modality_effect_claims"] is False
    assert payload["contracts"] == contracts
    assert payload["published_evidence"] == evidence
    assert payload["release_assets"]["environment_requirements"] == {
        "path": "environment-requirements.txt",
        "sha256": sha256_file(release_root / "environment-requirements.txt"),
    }
    assert payload["release_assets"]["sbom"] == {
        "path": "sbom.json",
        "sha256": sha256_file(release_root / "sbom.json"),
    }
    assert payload["release_assets"]["rc_evidence_archive"] == {
        "path": RC_ARCHIVE_NAME,
        "sha256_record": RC_INTEGRITY_NAME,
    }

    archive = release.write_rc_archive()
    assert archive.name == RC_ARCHIVE_NAME
    with tarfile.open(archive, "r:gz") as stream:
        names = set(stream.getnames())
    assert "release-assets/release-candidate.json" in names
    assert "release-assets/contracts/public_api_contract.json" in names
    assert "release-assets/dist/lunavla-2.0.0rc1-py3-none-any.whl" in names
    assert f"release-assets/{RC_INTEGRITY_NAME}" not in names

    integrity = release.write_rc_integrity("a" * 40, archive)
    integrity_payload = json.loads(integrity.read_text(encoding="utf-8"))
    assert integrity.name == RC_INTEGRITY_NAME
    assert integrity_payload["hash_layer"] == "post-archive"
    integrity_assets = {item["path"]: item["sha256"] for item in integrity_payload["assets"]}
    assert integrity_assets[RC_ARCHIVE_NAME] == sha256_file(archive)
    assert integrity_assets["release-candidate.json"] == sha256_file(candidate)
    assert RC_INTEGRITY_NAME not in integrity_assets

    checksums = release.write_checksums()
    checksum_text = checksums.read_text(encoding="utf-8")
    assert RC_INTEGRITY_NAME in checksum_text
    assert RC_ARCHIVE_NAME in checksum_text


def test_evidence_candidate_preserves_fail_closed_claim_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    release_root = tmp_path / "release-assets"
    release_root.mkdir()
    monkeypatch.setattr(release, "RELEASE_ROOT", release_root)
    monkeypatch.setattr(release, "validated_project_version", lambda: "2.0.0rc1")
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
    assert payload["package_version"] == "2.0.0rc1"
    assert payload["modality_effect_claims"] is False
    assert payload["claims"] == [claim_payload]


def test_distribution_names_must_match_rc1_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    release_root = tmp_path / "release-assets"
    monkeypatch.setattr(release, "RELEASE_ROOT", release_root)
    monkeypatch.setattr(release, "validated_project_version", lambda: "2.0.0rc1")

    def fake_run(command: tuple[str, ...], *, capture: bool = False) -> str:
        del capture
        if "build" in command:
            destination = release_root / "dist"
            destination.mkdir(parents=True)
            (destination / "lunavla-2.0.0rc1-py3-none-any.whl").write_bytes(b"wheel")
            (destination / "lunavla-2.0.0rc1.tar.gz").write_bytes(b"sdist")
        return ""

    monkeypatch.setattr(release, "run", fake_run)
    assert {path.name for path in release.build_distributions()} == {
        "lunavla-2.0.0rc1-py3-none-any.whl",
        "lunavla-2.0.0rc1.tar.gz",
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


def test_stable_profile_requires_real_integration_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    monkeypatch.setattr(release, "verify_source", lambda expected_sha: None)
    with pytest.raises(ValueError, match="requires --integration-manifest"):
        release.main(["--profile", "stable", "--expected-sha", "a" * 40])


def test_stable_main_routes_all_same_sha_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    calls: list[tuple[str, Path, Path]] = []
    manifest = tmp_path / "integration_manifest.json"
    bundle = tmp_path / "bundle.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    bundle.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(release, "verify_source", lambda expected_sha: None)
    monkeypatch.setattr(
        release,
        "run_stable",
        lambda expected_sha, *, integration_manifest, integration_attestation_bundle: calls.append(
            (expected_sha, integration_manifest, integration_attestation_bundle)
        ),
    )
    assert (
        release.main(
            [
                "--profile",
                "stable",
                "--expected-sha",
                "a" * 40,
                "--integration-manifest",
                str(manifest),
                "--integration-attestation-bundle",
                str(bundle),
            ]
        )
        == 0
    )
    assert calls == [("a" * 40, manifest, bundle)]


def test_stable_inputs_are_rejected_for_nonstable_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    monkeypatch.setattr(release, "verify_source", lambda expected_sha: None)
    with pytest.raises(ValueError, match="only by the stable profile"):
        release.main(
            [
                "--profile",
                "rc",
                "--expected-sha",
                "a" * 40,
                "--integration-manifest",
                str(tmp_path / "manifest.json"),
            ]
        )


def test_stable_source_must_equal_origin_main_tip(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_v2_release_profile as release

    def fake_run(command: tuple[str, ...], *, capture: bool = False) -> str:
        assert capture
        assert "refs/remotes/origin/main^{commit}" in command
        return "b" * 40

    monkeypatch.setattr(release, "run", fake_run)
    monkeypatch.setattr(release, "validated_project_version", lambda: STABLE_PACKAGE_VERSION)
    with pytest.raises(RuntimeError, match="origin/main tip"):
        release.verify_stable_main_tip("a" * 40)


def test_stable_source_requires_exact_stable_version(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_v2_release_profile as release

    monkeypatch.setattr(release, "run", lambda *args, **kwargs: "a" * 40)
    monkeypatch.setattr(release, "validated_project_version", lambda: "2.0.0rc1")
    with pytest.raises(RuntimeError, match="package version 2.0.0"):
        release.verify_stable_main_tip("a" * 40)


def test_stable_execution_design_is_full_but_never_targets_tracked_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release
    from lunavla.evidence_runner import is_full_design

    canonical_path, canonical = evidence_profile_design("language")
    release_root = tmp_path / "release-assets"
    release_root.mkdir()
    monkeypatch.setattr(release, "RELEASE_ROOT", release_root)
    monkeypatch.setattr(
        release,
        "evidence_profile_design",
        lambda profile: (canonical_path, canonical),
    )
    monkeypatch.setattr(release, "_require_ignored_destination", lambda path: None)
    _, generated_path, generated, output_root, snapshot_root = release._stable_execution_design(
        "language"
    )
    assert is_full_design(generated)
    assert generated.output.run_root == STABLE_OUTPUTS["language"][0].as_posix()
    assert generated.output.snapshot_root == STABLE_OUTPUTS["language"][1].as_posix()
    assert snapshot_root not in {
        release.ROOT / "results/v2/language-alpha2",
        release.ROOT / "results/v2/visual-beta1",
    }
    assert output_root != release.ROOT / canonical.output.run_root
    assert canonical.output.snapshot_root == "results/v2/language-alpha2"
    assert generated_path.is_file()


def test_stable_integration_requires_same_sha_and_verified_subject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    release_root = tmp_path / "release-assets"
    release_root.mkdir()
    manifest = tmp_path / "integration_manifest.json"
    bundle = tmp_path / "integration.jsonl"
    manifest.write_bytes(b'{"manifest":true}\n')
    bundle.write_bytes(b'{"bundle":true}\n')
    manifest_digest = sha256_file(manifest)
    commands: list[tuple[str, ...]] = []

    monkeypatch.setattr(release, "RELEASE_ROOT", release_root)
    monkeypatch.setattr(
        release.IntegrationManifest,
        "load",
        lambda path: SimpleNamespace(git_sha="a" * 40),
    )

    def fake_run(command: tuple[str, ...], *, capture: bool = False) -> str:
        assert capture
        commands.append(command)
        return json.dumps(
            [
                {
                    "verificationResult": {
                        "statement": {
                            "subject": [
                                {
                                    "name": "integration_manifest.json",
                                    "digest": {"sha256": manifest_digest},
                                }
                            ]
                        }
                    }
                }
            ]
        )

    monkeypatch.setattr(release, "run", fake_run)
    result = release.verify_stable_integration(
        expected_sha="a" * 40,
        manifest_source=manifest,
        bundle_source=bundle,
    )
    assert result.manifest_sha256 == manifest_digest
    assert result.verified_attestation_count == 1
    command = commands[0]
    assert command[:3] == ("gh", "attestation", "verify")
    for flag, expected in (
        ("--bundle", str(bundle.resolve())),
        ("--repo", "xiaoms22/lunavla"),
        ("--signer-workflow", STABLE_SIGNER_WORKFLOW),
        ("--source-digest", "a" * 40),
        ("--source-ref", STABLE_SOURCE_REF),
        ("--format", "json"),
    ):
        assert command[command.index(flag) + 1] == expected
    assert "--deny-self-hosted-runners" in command

    monkeypatch.setattr(
        release,
        "run",
        lambda command, *, capture=False: json.dumps(
            [{"verificationResult": {"statement": {"subject": [{"digest": {"sha256": "0" * 64}}]}}}]
        ),
    )
    with pytest.raises(RuntimeError, match="does not bind the integration manifest"):
        release.verify_stable_integration(
            expected_sha="a" * 40,
            manifest_source=manifest,
            bundle_source=bundle,
        )

    monkeypatch.setattr(
        release.IntegrationManifest,
        "load",
        lambda path: SimpleNamespace(git_sha="b" * 40),
    )
    with pytest.raises(RuntimeError, match="differs from the stable candidate"):
        release.verify_stable_integration(
            expected_sha="a" * 40,
            manifest_source=manifest,
            bundle_source=bundle,
        )


def _fake_stable_study(
    *,
    release_root: Path,
    root: Path,
    profile: str,
    source_count: int,
    episodes: int,
    claim_id: str,
) -> SimpleNamespace:
    metadata: dict[str, Path] = {}
    for key in ("canonical_design", "design", "manifest", "verification", "claims"):
        path = release_root / f"{profile}-{key}.json"
        path.write_text("{}\n", encoding="utf-8")
        metadata[key] = path
    output_root = root / f"outputs/stable-release/{profile}"
    snapshot_root = root / f"results/v2/stable-release/{profile}"
    output_root.mkdir(parents=True)
    snapshot_root.mkdir(parents=True)
    claim_payload = {
        "claim_id": claim_id,
        "allowed": False,
        "statement": f"{claim_id} has not yet been established.",
    }
    claim = SimpleNamespace(allowed=False, to_dict=lambda: claim_payload)
    controlled = SimpleNamespace(
        profile=profile,
        design=SimpleNamespace(design_id=f"{profile}-design", sha256=lambda: "d" * 64),
        manifest=SimpleNamespace(
            claims=(claim,),
            matrix_complete=True,
            reduced_design=False,
        ),
        verification=SimpleNamespace(
            source_count=source_count,
            arm_episode_count=episodes,
        ),
        output_root=output_root,
        snapshot_root=snapshot_root,
        runs=(),
    )
    return SimpleNamespace(controlled=controlled, metadata=metadata)


def _write_stable_test_manifest(
    path: Path,
    *,
    profile: str,
    claim_id: str,
) -> EvidenceManifest:
    claim = ClaimDecision.from_checks(
        claim_id=claim_id,
        checks={"effect_established": False},
        allowed_statement=f"Controlled evidence establishes {claim_id}.",
        denied_statement=f"{claim_id} has not yet been established.",
    )
    manifest = EvidenceManifest(
        schema_version=1,
        design_id=f"{profile}-design",
        design_sha256="d" * 64,
        reduced_design=False,
        matrix_complete=True,
        integrity_checks=(("clean_source", True),),
        sources=(EvidenceSource(f"{profile}-seed-11", "b" * 64),),
        statistics=(),
        claims=(claim,),
    )
    manifest.write(path)
    return manifest


def test_stable_candidate_binds_960_matrix_tag_and_current_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    release_root = tmp_path / "release-assets"
    (release_root / "dist").mkdir(parents=True)
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "RELEASE_ROOT", release_root)
    monkeypatch.setattr(release, "validated_project_version", lambda: STABLE_PACKAGE_VERSION)
    studies = (
        _fake_stable_study(
            release_root=release_root,
            root=tmp_path,
            profile="language",
            source_count=5,
            episodes=480,
            claim_id="instruction_following",
        ),
        _fake_stable_study(
            release_root=release_root,
            root=tmp_path,
            profile="vision",
            source_count=10,
            episodes=480,
            claim_id="visual_control_contribution",
        ),
    )
    integration_dir = release_root / "integration"
    integration_dir.mkdir()
    integration_files = [
        integration_dir / "integration_manifest.json",
        integration_dir / "bundle.jsonl",
        integration_dir / "verification.json",
    ]
    for path in integration_files:
        path.write_text("{}\n", encoding="utf-8")
    integration = SimpleNamespace(
        manifest_path=integration_files[0],
        manifest_sha256=sha256_file(integration_files[0]),
        bundle_path=integration_files[1],
        bundle_sha256=sha256_file(integration_files[1]),
        verification_path=integration_files[2],
        verification_sha256=sha256_file(integration_files[2]),
        verified_attestation_count=1,
    )
    distribution = release_root / "dist/lunavla-2.0.0-py3-none-any.whl"
    distribution.write_bytes(b"wheel")
    archive = release_root / STABLE_ARCHIVE_NAME
    archive.write_bytes(b"archive")
    sbom = release_root / "sbom.json"
    sbom.write_text("{}\n", encoding="utf-8")
    environment = release_root / "environment-requirements.txt"
    environment.write_text("lunavla==2.0.0\n", encoding="utf-8")
    checksums = release_root / "stable-evidence-files.SHA256SUMS"
    checksums.write_text("", encoding="utf-8")
    candidate = release.write_stable_candidate(
        expected_sha="a" * 40,
        studies=studies,
        integration=integration,
        contracts=(),
        distributions=(distribution,),
        archive=archive,
        evidence_checksums=checksums,
    )
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    assert payload["package_version"] == STABLE_PACKAGE_VERSION
    assert payload["expected_tag"] == STABLE_TAG
    assert payload["default_branch"] == "main"
    assert payload["evidence"]["source_count"] == 15
    assert payload["evidence"]["arm_episode_count"] == 960
    language_record = payload["evidence"]["studies"][0]
    assert language_record["design_sha256"] == "d" * 64
    assert language_record["execution_design_sha256"] == sha256_file(studies[0].metadata["design"])
    assert payload["modality_effect_claims"] is False
    assert [claim["claim_id"] for claim in payload["claims"]] == [
        "instruction_following",
        "visual_control_contribution",
    ]
    assert payload["publish_pypi"] is False
    assert payload["gpu_required"] is False
    assert payload["assets"]["environment_requirements"] == {
        "path": "environment-requirements.txt",
        "sha256": sha256_file(environment),
    }

    integration.verified_attestation_count = 0
    with pytest.raises(RuntimeError, match="verified GitHub attestation"):
        release.write_stable_candidate(
            expected_sha="a" * 40,
            studies=studies,
            integration=integration,
            contracts=(),
            distributions=(distribution,),
            archive=archive,
            evidence_checksums=checksums,
        )
    integration.verified_attestation_count = 1
    studies[1].controlled.verification.arm_episode_count = 479
    with pytest.raises(RuntimeError, match="stable vision evidence"):
        release.write_stable_candidate(
            expected_sha="a" * 40,
            studies=studies,
            integration=integration,
            contracts=(),
            distributions=(distribution,),
            archive=archive,
            evidence_checksums=checksums,
        )


def test_stable_archive_combines_both_studies_integration_and_release_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    release_root = tmp_path / "release-assets"
    (release_root / "contracts").mkdir(parents=True)
    (release_root / "dist").mkdir()
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "RELEASE_ROOT", release_root)
    studies = (
        _fake_stable_study(
            release_root=release_root,
            root=tmp_path,
            profile="language",
            source_count=5,
            episodes=480,
            claim_id="instruction_following",
        ),
        _fake_stable_study(
            release_root=release_root,
            root=tmp_path,
            profile="vision",
            source_count=10,
            episodes=480,
            claim_id="visual_control_contribution",
        ),
    )
    for study in studies:
        (study.controlled.output_root / "evidence_manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (study.controlled.snapshot_root / "snapshot_manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )
    (release_root / "contracts/public_api_contract.json").write_text("{}\n", encoding="utf-8")
    (release_root / "dist/lunavla-2.0.0-py3-none-any.whl").write_bytes(b"wheel")
    for name in (
        "environment-requirements.txt",
        "sbom.json",
        "stable-evidence-files.SHA256SUMS",
    ):
        (release_root / name).write_text("{}\n", encoding="utf-8")
    integration_root = release_root / "integration"
    integration_root.mkdir()
    manifest = integration_root / "integration_manifest.json"
    bundle = integration_root / "integration-attestation-bundle.jsonl"
    verification = integration_root / "integration-attestation-verification.json"
    for path in (manifest, bundle, verification):
        path.write_text("{}\n", encoding="utf-8")
    integration = SimpleNamespace(
        manifest_path=manifest,
        bundle_path=bundle,
        verification_path=verification,
    )

    archive = release.write_stable_archive(studies=studies, integration=integration)
    with tarfile.open(archive, "r:gz") as stream:
        names = set(stream.getnames())
    assert {
        "outputs/stable-release/language/evidence_manifest.json",
        "results/v2/stable-release/language/snapshot_manifest.json",
        "outputs/stable-release/vision/evidence_manifest.json",
        "results/v2/stable-release/vision/snapshot_manifest.json",
        "release-assets/contracts/public_api_contract.json",
        "release-assets/dist/lunavla-2.0.0-py3-none-any.whl",
        "release-assets/integration/integration_manifest.json",
        "release-assets/integration/integration-attestation-bundle.jsonl",
        "release-assets/integration/integration-attestation-verification.json",
        "release-assets/environment-requirements.txt",
        "release-assets/sbom.json",
        "release-assets/stable-evidence-files.SHA256SUMS",
    } <= names


def test_release_checksums_cover_nested_names_and_reject_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    release_root = tmp_path / "release-assets"
    nested = release_root / "nested"
    nested.mkdir(parents=True)
    (release_root / "asset.txt").write_text("asset\n", encoding="utf-8")
    (nested / "SHA256SUMS").write_text("nested\n", encoding="utf-8")
    monkeypatch.setattr(release, "RELEASE_ROOT", release_root)
    checksums = release.write_checksums()
    rows = release._checksum_rows(checksums)
    assert set(rows) == {"asset.txt", "nested/SHA256SUMS"}

    (release_root / "linked.txt").symlink_to(release_root / "asset.txt")
    with pytest.raises(RuntimeError, match="cannot contain symlinks"):
        release.write_checksums()


def test_stable_asset_consistency_detects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_v2_release_profile as release

    release_root = tmp_path / "release-assets"
    release_root.mkdir()
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "RELEASE_ROOT", release_root)
    paths = {
        "dist/lunavla-2.0.0-py3-none-any.whl": b"wheel",
        STABLE_ARCHIVE_NAME: b"archive",
        "sbom.json": b"{}\n",
        "environment-requirements.txt": b"lunavla==2.0.0\n",
        "integration/integration_manifest.json": b"manifest\n",
        "integration/bundle.jsonl": b"bundle\n",
        "integration/verification.json": b"verification\n",
    }
    for relative, content in paths.items():
        path = release_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    study_records: list[dict[str, object]] = []
    for profile, source_count, episodes, claim_id in (
        ("language", 5, 480, "instruction_following"),
        ("vision", 10, 480, "visual_control_contribution"),
    ):
        metadata: dict[str, Path] = {}
        for name in ("canonical", "execution", "manifest", "verification", "claims"):
            path = release_root / f"{profile}-{name}.json"
            path.write_text("{}\n", encoding="utf-8")
            metadata[name] = path
        manifest = _write_stable_test_manifest(
            metadata["manifest"],
            profile=profile,
            claim_id=claim_id,
        )
        manifest_claims = [claim.to_dict() for claim in manifest.claims]
        metadata["claims"].write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "profile": profile,
                    "design_id": manifest.design_id,
                    "design_sha256": manifest.design_sha256,
                    "claim_allowed": False,
                    "claims": manifest_claims,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        study_records.append(
            {
                "profile": profile,
                "design_id": manifest.design_id,
                "design_sha256": manifest.design_sha256,
                "source_count": source_count,
                "arm_episode_count": episodes,
                "matrix_complete": True,
                "reduced_design": False,
                "canonical_design_path": metadata["canonical"].name,
                "canonical_design_sha256": sha256_file(metadata["canonical"]),
                "execution_design_path": metadata["execution"].name,
                "execution_design_sha256": sha256_file(metadata["execution"]),
                "evidence_manifest_path": metadata["manifest"].name,
                "evidence_manifest_sha256": sha256_file(metadata["manifest"]),
                "verification_path": metadata["verification"].name,
                "verification_sha256": sha256_file(metadata["verification"]),
                "claim_summary_path": metadata["claims"].name,
                "claim_summary_sha256": sha256_file(metadata["claims"]),
                "claims": manifest_claims,
            }
        )
    evidence_checksums = release_root / "stable-evidence-files.SHA256SUMS"
    environment = release_root / "environment-requirements.txt"
    evidence_checksums.write_text(
        f"{sha256_file(environment)}  release-assets/environment-requirements.txt\n",
        encoding="utf-8",
    )
    claims = [claim for study in study_records for claim in study["claims"]]
    payload = {
        "profile": "stable",
        "expected_sha": "a" * 40,
        "package_version": STABLE_PACKAGE_VERSION,
        "expected_tag": STABLE_TAG,
        "default_branch": "main",
        "authoritative_device": "cpu",
        "contract_freeze": True,
        "publish_pypi": False,
        "gpu_required": False,
        "modality_effect_claims": False,
        "claims": claims,
        "evidence": {
            "post_merge_rerun": True,
            "source_count": 15,
            "arm_episode_count": 960,
            "studies": study_records,
        },
        "contracts": [],
        "distributions": [
            {
                "path": "dist/lunavla-2.0.0-py3-none-any.whl",
                "sha256": sha256_file(release_root / "dist/lunavla-2.0.0-py3-none-any.whl"),
            }
        ],
        "assets": {
            "combined_evidence_archive": {
                "path": STABLE_ARCHIVE_NAME,
                "sha256": sha256_file(release_root / STABLE_ARCHIVE_NAME),
            },
            "sbom": {"path": "sbom.json", "sha256": sha256_file(release_root / "sbom.json")},
            "environment_requirements": {
                "path": "environment-requirements.txt",
                "sha256": sha256_file(environment),
            },
            "evidence_files_sha256": {
                "path": "stable-evidence-files.SHA256SUMS",
                "sha256": sha256_file(evidence_checksums),
            },
        },
        "integration": {
            "manifest_path": "integration/integration_manifest.json",
            "manifest_sha256": sha256_file(release_root / "integration/integration_manifest.json"),
            "attestation_bundle_path": "integration/bundle.jsonl",
            "attestation_bundle_sha256": sha256_file(release_root / "integration/bundle.jsonl"),
            "attestation_verification_path": "integration/verification.json",
            "attestation_verification_sha256": sha256_file(
                release_root / "integration/verification.json"
            ),
            "verified_attestation_count": 1,
            "signer_workflow": STABLE_SIGNER_WORKFLOW,
            "source_ref": STABLE_SOURCE_REF,
            "source_digest": "a" * 40,
            "deny_self_hosted_runners": True,
            "claim_allowed": False,
        },
    }
    candidate = release_root / "release-candidate.json"
    candidate.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    release.write_checksums()
    release.verify_stable_asset_set("a" * 40)

    original_candidate = candidate.read_text(encoding="utf-8")
    tampered = json.loads(original_candidate)
    tampered["claims"][0]["allowed"] = True
    tampered["claims"][0]["statement"] = "Fabricated modality-effect success."
    tampered["modality_effect_claims"] = True
    candidate.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    release.write_checksums()
    with pytest.raises(RuntimeError, match="claims differ from verified EvidenceManifest"):
        release.verify_stable_asset_set("a" * 40)

    candidate.write_text(original_candidate, encoding="utf-8")
    release.write_checksums()
    release.verify_stable_asset_set("a" * 40)
    (release_root / "sbom.json").write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="asset hash mismatch"):
        release.verify_stable_asset_set("a" * 40)
