from __future__ import annotations

import json
from pathlib import Path

import pytest

from lunavla.v3 import (
    ArtifactHashRecordV1,
    STABLE_PACKAGE_VERSION,
    STABLE_TAG,
    StableEvidenceDesignV1,
    StableEvidenceSummaryV1,
    StableReleaseCandidateV1,
    validate_stable_design_set,
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


def _summary(design: StableEvidenceDesignV1, *, complete: bool = True) -> StableEvidenceSummaryV1:
    observed = design.expected_rows if complete else design.expected_rows - 1
    return StableEvidenceSummaryV1(
        study_id=design.study_id,
        design_sha256=design.sha256(),
        git_sha=GIT_SHA,
        dependency_lock_sha256=SHA,
        upstream_identity_sha256=SHA,
        row_inventory_sha256=SHA,
        statistics_sha256=SHA,
        sentinel_sha256=SHA,
        expected_rows=design.expected_rows,
        observed_rows=observed,
        matrix_complete=complete,
        homogeneous_source=True,
        sentinel_verified=True,
        claim_gate_sha256=SHA,
        release_eligible=complete,
        gate_reasons=() if complete else ("incomplete_matrix",),
    )


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
        hosted_integration_manifest_sha256=SHA,
        portfolio_bundle_sha256=SHA,
        integration_manifests=_records(
            {"integration/hosted-cpu-pusht-libero.json"}
        ),
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
                "dist/lunavla-3.0.0-py3-none-any.whl",
                "dist/lunavla-3.0.0.tar.gz",
                "sbom.json",
                "provenance.jsonl",
                "evidence.tar.gz",
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
    summary = _summary(design)
    summary.verify_design(design)
    assert StableEvidenceSummaryV1.from_mapping(summary.to_dict()) == summary
    incomplete = _summary(design, complete=False)
    incomplete.verify_design(design)
    assert incomplete.release_eligible is False
    payload = incomplete.to_dict()
    payload["release_eligible"] = True
    with pytest.raises(ValueError, match="release_eligible"):
        StableEvidenceSummaryV1.from_mapping(payload)


def test_stable_candidate_binds_release_sha_assets_and_external_gates() -> None:
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


def test_stable_cli_validates_designs_and_evidence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["validate-stable-designs", *(str(path) for path in DESIGN_PATHS)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["total_rows"] == 1550
    design = _designs()[2]
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary(design).to_dict()), encoding="utf-8")
    assert cli_main(["verify-stable-evidence", str(DESIGN_PATHS[2]), str(summary_path)]) == 0
    assert json.loads(capsys.readouterr().out)["release_eligible"] is True
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(_candidate().to_dict()), encoding="utf-8")
    assert cli_main(["verify-stable-candidate", str(candidate_path)]) == 0
    assert json.loads(capsys.readouterr().out)["tag"] == STABLE_TAG
