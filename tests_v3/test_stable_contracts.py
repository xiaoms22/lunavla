from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lunavla.v3 import (
    ArtifactHashRecordV1,
    RC_PACKAGE_VERSION,
    RC_TAG,
    RcReleaseCandidateV1,
    STABLE_PACKAGE_VERSION,
    STABLE_TAG,
    StableEvidenceDesignV1,
    StableEvidenceRowV1,
    StableEvidenceSummaryV1,
    StableRepeatSentinelV1,
    StableReleaseCandidateV1,
    build_stable_evidence_summary,
    clustered_paired_bootstrap,
    expected_stable_matrix_keys,
    stable_row_inventory_sha256,
    validate_stable_design_set,
    verify_release_candidate_assets,
    verify_stable_evidence_bundle,
    wilson_interval,
)
from lunavla.v3.cli import main as cli_main


SHA = "a" * 64
GIT_SHA = "b" * 40
DESIGN_PATHS = (
    Path("configs/v3/stable_pusht_policy_design.yaml"),
    Path("configs/v3/stable_libero_route_design.yaml"),
    Path("configs/v3/stable_libero_prompt_design.yaml"),
)


def _designs() -> tuple[StableEvidenceDesignV1, ...]:
    return tuple(StableEvidenceDesignV1.load(path) for path in DESIGN_PATHS)


def _rows(design: StableEvidenceDesignV1) -> tuple[StableEvidenceRowV1, ...]:
    return tuple(
        StableEvidenceRowV1(
            study_id=design.study_id,
            policy=policy,
            train_seed=train_seed,
            task_id=task_id,
            evaluation_id=evaluation_id,
            route=route,
            intervention=intervention,
            git_sha=GIT_SHA,
            dependency_lock_sha256=SHA,
            upstream_identity_sha256=SHA,
            run_manifest_sha256=SHA,
            metrics_sha256=SHA,
            success=True,
            final_metric=0.1,
            smoothness=0.2,
            first_action_mse=0.3,
            failure_count=0,
        )
        for policy, train_seed, task_id, evaluation_id, route, intervention
        in expected_stable_matrix_keys(design)
    )


def _summary_bundle(
    design: StableEvidenceDesignV1, *, complete: bool = True
) -> tuple[tuple[StableEvidenceRowV1, ...], StableRepeatSentinelV1, StableEvidenceSummaryV1]:
    rows = _rows(design)
    if not complete:
        rows = rows[:-1]
    repeat_inventory = stable_row_inventory_sha256(
        tuple(row for row in rows if row.train_seed == design.repeat_train_seed)
    )
    sentinel = StableRepeatSentinelV1(
        study_id=design.study_id,
        train_seed=11,
        source_row_inventory_sha256=repeat_inventory,
        repeat_row_inventory_sha256=repeat_inventory,
        source_checkpoint_sha256=SHA,
        repeat_checkpoint_sha256=SHA,
        source_metrics_sha256=SHA,
        repeat_metrics_sha256=SHA,
        verified=True,
    )
    summary = build_stable_evidence_summary(
        design,
        rows,
        sentinel,
        expected_git_sha=GIT_SHA,
        statistics_sha256=SHA,
        claim_gate_sha256=SHA,
    )
    return rows, sentinel, summary


def _records(paths: set[str]) -> tuple[ArtifactHashRecordV1, ...]:
    return tuple(ArtifactHashRecordV1(path, SHA) for path in sorted(paths))


def _candidate() -> StableReleaseCandidateV1:
    return StableReleaseCandidateV1(
        expected_tag=STABLE_TAG,
        package_version=STABLE_PACKAGE_VERSION,
        git_sha=GIT_SHA,
        merge_sha=GIT_SHA,
        public_api_sha256=SHA,
        migration_report_sha256=SHA,
        contract_descriptor_sha256=SHA,
        portfolio_bundle_sha256=SHA,
        evidence_manifests=_records(
            {
                "evidence/fixture-policy-ladder.json",
                "evidence/fixture-state-routes.json",
                "evidence/fixture-prompt-interventions.json",
            }
        ),
        required_checks_sha256=SHA,
        sbom_sha256=SHA,
        provenance_sha256=SHA,
        checksums_sha256=SHA,
        assets=_records(
            {
                "contract-descriptor.json",
                "dist/lunavla-3.0.0-py3-none-any.whl",
                "dist/lunavla-3.0.0.tar.gz",
                "evidence.tar.gz",
                "migration-report.json",
                "portfolio.tar.gz",
                "privacy-scan.json",
                "provenance.jsonl",
                "public-api.json",
                "required-checks.json",
                "sbom.json",
            }
        ),
        signed_tag_verified=True,
        post_merge_evidence=True,
        privacy_scan_verified=True,
        pypi_published=False,
    )


def test_stable_design_set_is_exact_and_has_1550_rows() -> None:
    designs = _designs()
    rows = validate_stable_design_set(designs)
    assert rows == {
        "fixture_prompt_interventions": 750,
        "fixture_state_routes": 600,
        "fixture_policy_ladder": 200,
    }
    assert StableEvidenceDesignV1.from_mapping(designs[0].to_dict()) == designs[0]


def test_stable_design_rejects_boolean_version_and_matrix_drift() -> None:
    design = _designs()[0]
    payload = design.to_dict()
    payload["schema_version"] = True
    with pytest.raises((TypeError, ValueError), match="schema_version"):
        StableEvidenceDesignV1.from_mapping(payload)
    payload = design.to_dict()
    payload["evaluation_ids"] = payload["evaluation_ids"][:-1]
    drifted = StableEvidenceDesignV1.from_mapping(payload)
    with pytest.raises(ValueError, match="frozen stable matrix"):
        drifted.validate_stable_matrix()
    payload = design.to_dict()
    payload["unknown"] = "forbidden"
    with pytest.raises(ValueError, match="unknown"):
        StableEvidenceDesignV1.from_mapping(payload)


def test_stable_evidence_summary_round_trip_and_fail_closed_gates() -> None:
    design = _designs()[1]
    rows, sentinel, summary = _summary_bundle(design)
    verify_stable_evidence_bundle(design, rows, sentinel, summary)
    assert StableEvidenceSummaryV1.from_mapping(summary.to_dict()) == summary
    assert summary.row_inventory_sha256 != sentinel.source_row_inventory_sha256
    incomplete_rows, incomplete_sentinel, incomplete = _summary_bundle(design, complete=False)
    verify_stable_evidence_bundle(design, incomplete_rows, incomplete_sentinel, incomplete)
    incomplete.verify_design(design)
    assert incomplete.release_eligible is False
    payload = incomplete.to_dict()
    payload["release_eligible"] = True
    with pytest.raises(ValueError, match="release_eligible"):
        StableEvidenceSummaryV1.from_mapping(payload)


def test_row_matrix_tamper_mixed_source_and_sentinel_are_recomputed() -> None:
    design = _designs()[0]
    rows, sentinel, summary = _summary_bundle(design)
    duplicated = rows + (rows[0],)
    recomputed = build_stable_evidence_summary(
        design,
        duplicated,
        sentinel,
        expected_git_sha=GIT_SHA,
        statistics_sha256=SHA,
        claim_gate_sha256=SHA,
    )
    assert recomputed.release_eligible is False
    assert "incomplete_matrix" in recomputed.gate_reasons
    with pytest.raises(ValueError, match="independently recomputed"):
        verify_stable_evidence_bundle(design, duplicated, sentinel, summary)
    payload = rows[0].to_dict()
    payload["git_sha"] = "c" * 40
    mixed = (StableEvidenceRowV1.from_mapping(payload), *rows[1:])
    mixed_summary = build_stable_evidence_summary(
        design,
        mixed,
        sentinel,
        expected_git_sha=GIT_SHA,
        statistics_sha256=SHA,
        claim_gate_sha256=SHA,
    )
    assert "mixed_source" in mixed_summary.gate_reasons
    sentinel_payload = sentinel.to_dict()
    sentinel_payload["repeat_metrics_sha256"] = "c" * 64
    sentinel_payload["verified"] = False
    failed_sentinel = StableRepeatSentinelV1.from_mapping(sentinel_payload)
    failed = build_stable_evidence_summary(
        design,
        rows,
        failed_sentinel,
        expected_git_sha=GIT_SHA,
        statistics_sha256=SHA,
        claim_gate_sha256=SHA,
    )
    assert "sentinel_failure" in failed.gate_reasons


def test_stable_statistics_are_deterministic_and_finite() -> None:
    low, high = wilson_interval(8, 10)
    assert 0.0 < low < 0.8 < high < 1.0
    first = clustered_paired_bootstrap(
        {11: (1.0, 2.0), 22: (2.0, 3.0)}, samples=200, seed=202701
    )
    second = clustered_paired_bootstrap(
        {11: (1.0, 2.0), 22: (2.0, 3.0)}, samples=200, seed=202701
    )
    assert first == second
    with pytest.raises(ValueError, match="finite"):
        clustered_paired_bootstrap({11: (float("nan"),)}, samples=10)


def test_stable_candidate_binds_release_sha_assets_and_cpu_evidence_gates() -> None:
    candidate = _candidate()
    assert StableReleaseCandidateV1.from_mapping(candidate.to_dict()) == candidate
    payload = candidate.to_dict()
    payload["merge_sha"] = "c" * 40
    with pytest.raises(ValueError, match="actual merge SHA"):
        StableReleaseCandidateV1.from_mapping(payload)
    payload = candidate.to_dict()
    payload["signed_tag_verified"] = False
    with pytest.raises(ValueError, match="signed tag"):
        StableReleaseCandidateV1.from_mapping(payload)
    payload = candidate.to_dict()
    payload["pypi_published"] = True
    with pytest.raises(ValueError, match="PyPI"):
        StableReleaseCandidateV1.from_mapping(payload)


def test_rc_candidate_verifies_exact_asset_directory_and_checksums(tmp_path: Path) -> None:
    paths = {
        "contract-descriptor.json",
        "dist/lunavla-3.0.0rc1-py3-none-any.whl",
        "dist/lunavla-3.0.0rc1.tar.gz",
        "evidence.tar.gz",
        "migration-report.json",
        "portfolio.tar.gz",
        "privacy-scan.json",
        "provenance.jsonl",
        "public-api.json",
        "required-checks.json",
        "sbom.json",
    }
    records: list[ArtifactHashRecordV1] = []
    for relative in sorted(paths):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
        records.append(ArtifactHashRecordV1(relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    checksums = "".join(f"{record.sha256}  {record.path}\n" for record in records)
    (tmp_path / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    hashes = {record.path: record.sha256 for record in records}
    candidate = RcReleaseCandidateV1(
        expected_tag=RC_TAG,
        package_version=RC_PACKAGE_VERSION,
        git_sha=GIT_SHA,
        merge_sha=GIT_SHA,
        public_api_sha256=hashes["public-api.json"],
        migration_report_sha256=hashes["migration-report.json"],
        contract_descriptor_sha256=hashes["contract-descriptor.json"],
        portfolio_bundle_sha256=hashes["portfolio.tar.gz"],
        evidence_manifests=_records(
            {
                "evidence/fixture-policy-ladder.json",
                "evidence/fixture-state-routes.json",
                "evidence/fixture-prompt-interventions.json",
            }
        ),
        required_checks_sha256=hashes["required-checks.json"],
        sbom_sha256=hashes["sbom.json"],
        provenance_sha256=hashes["provenance.jsonl"],
        checksums_sha256=hashlib.sha256(checksums.encode("utf-8")).hexdigest(),
        assets=tuple(records),
        signed_tag_verified=True,
        post_merge_evidence=True,
        privacy_scan_verified=True,
        pypi_published=False,
    )
    assert RcReleaseCandidateV1.from_mapping(candidate.to_dict()) == candidate
    verify_release_candidate_assets(candidate, tmp_path)
    (tmp_path / "sbom.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_release_candidate_assets(candidate, tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("hosted_integration_manifest_sha256", SHA),
        (
            "integration_manifests",
            [ArtifactHashRecordV1("integration/hosted-cpu-pusht-libero.json", SHA).to_dict()],
        ),
    ),
)
def test_stable_candidate_rejects_retired_real_integration_gates(
    field: str, value: object
) -> None:
    payload = _candidate().to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match="unknown"):
        StableReleaseCandidateV1.from_mapping(payload)


def test_stable_cli_validates_designs_and_evidence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["validate-stable-designs", *(str(path) for path in DESIGN_PATHS)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["total_rows"] == 1550
    design = _designs()[2]
    rows, sentinel, summary = _summary_bundle(design)
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps([row.to_dict() for row in rows]), encoding="utf-8")
    sentinel_path = tmp_path / "sentinel.json"
    sentinel_path.write_text(json.dumps(sentinel.to_dict()), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary.to_dict()), encoding="utf-8")
    assert cli_main(
        [
            "verify-stable-evidence",
            str(DESIGN_PATHS[2]),
            str(rows_path),
            str(sentinel_path),
            str(summary_path),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["release_eligible"] is True
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(_candidate().to_dict()), encoding="utf-8")
    assert cli_main(["verify-stable-candidate", str(candidate_path)]) == 0
    assert json.loads(capsys.readouterr().out)["tag"] == STABLE_TAG
