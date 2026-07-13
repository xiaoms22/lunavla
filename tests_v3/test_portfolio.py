from __future__ import annotations

import json
from pathlib import Path

import pytest

from lunavla.v3 import (
    PORTFOLIO_LIMITATIONS,
    PORTFOLIO_STATEMENT,
    PortfolioBundleV1,
    StableEvidenceDesignV1,
    StableEvidenceRowV1,
    StableRepeatSentinelV1,
    build_portfolio,
    build_stable_evidence_summary,
    expected_stable_matrix_keys,
    stable_row_inventory_sha256,
    verify_portfolio,
)
from lunavla.v3.cli import main as cli_main


SHA = "a" * 64
GIT_SHA = "b" * 40
DESIGN_PATHS = (
    Path("configs/v3/stable_pusht_policy_design.yaml"),
    Path("configs/v3/stable_libero_route_design.yaml"),
    Path("configs/v3/stable_libero_prompt_design.yaml"),
)


def _rows(
    design: StableEvidenceDesignV1, *, git_sha: str = GIT_SHA
) -> tuple[StableEvidenceRowV1, ...]:
    return tuple(
        StableEvidenceRowV1(
            study_id=design.study_id,
            policy=policy,
            train_seed=train_seed,
            task_id=task_id,
            evaluation_id=evaluation_id,
            route=route,
            intervention=intervention,
            git_sha=git_sha,
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
        for policy, train_seed, task_id, evaluation_id, route, intervention in
        expected_stable_matrix_keys(design)
    )


def _write_study(
    root: Path,
    design: StableEvidenceDesignV1,
    *,
    complete: bool = True,
    git_sha: str = GIT_SHA,
) -> None:
    rows = _rows(design, git_sha=git_sha)
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
        expected_git_sha=git_sha,
        statistics_sha256=SHA,
        claim_gate_sha256=SHA,
    )
    study = root / design.study_id
    study.mkdir(parents=True)
    (study / "design.json").write_text(
        json.dumps(design.to_dict()), encoding="utf-8"
    )
    (study / "rows.json").write_text(
        json.dumps([row.to_dict() for row in rows]), encoding="utf-8"
    )
    (study / "sentinel.json").write_text(
        json.dumps(sentinel.to_dict()), encoding="utf-8"
    )
    (study / "summary.json").write_text(
        json.dumps(summary.to_dict()), encoding="utf-8"
    )


def _evidence_root(tmp_path: Path) -> Path:
    root = tmp_path / "evidence"
    for path in DESIGN_PATHS:
        _write_study(root, StableEvidenceDesignV1.load(path))
    return root


def test_portfolio_build_verify_and_cli_are_fail_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence = _evidence_root(tmp_path)
    output = tmp_path / "portfolio"
    assert cli_main(["portfolio-build", str(evidence), "--out", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "output_dir": str(output.resolve()),
        "release_eligible": True,
        "total_rows": 1550,
    }
    assert cli_main(["portfolio-verify", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    manifest = verify_portfolio(output)
    assert PortfolioBundleV1.from_mapping(manifest.to_dict()) == manifest
    portfolio = json.loads((output / "portfolio.json").read_text(encoding="utf-8"))
    assert portfolio["capability_statements"] == [PORTFOLIO_STATEMENT]
    assert portfolio["resume_bullet_templates"] == [PORTFOLIO_STATEMENT]
    assert portfolio["limitations"] == list(PORTFOLIO_LIMITATIONS)


def test_portfolio_rejects_claim_closed_mixed_and_private_sources(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    for index, path in enumerate(DESIGN_PATHS):
        _write_study(
            incomplete,
            StableEvidenceDesignV1.load(path),
            complete=index != 0,
        )
    with pytest.raises(ValueError, match="claim-closed"):
        build_portfolio(incomplete, tmp_path / "incomplete-out")

    mixed = tmp_path / "mixed"
    for index, path in enumerate(DESIGN_PATHS):
        _write_study(
            mixed,
            StableEvidenceDesignV1.load(path),
            git_sha=("c" * 40 if index == 0 else GIT_SHA),
        )
    with pytest.raises(ValueError, match="mixed source provenance"):
        build_portfolio(mixed, tmp_path / "mixed-out")

    private = _evidence_root(tmp_path / "private-case")
    target = private / "fixture_policy_ladder" / "summary.json"
    private_marker = "/" + "Users/private/repo"
    target.write_text(target.read_text(encoding="utf-8") + f"\n{private_marker}\n")
    with pytest.raises(ValueError, match="source evidence privacy scan"):
        build_portfolio(private, tmp_path / "private-out")


def test_portfolio_rejects_output_tamper_unknown_fields_and_overlap(tmp_path: Path) -> None:
    evidence = _evidence_root(tmp_path)
    output = build_portfolio(evidence, tmp_path / "portfolio")
    target = output / "portfolio.md"
    target.write_text(target.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_portfolio(output)

    manifest_path = output / "portfolio_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["unknown"] = True
    with pytest.raises(ValueError, match="exact fields"):
        PortfolioBundleV1.from_mapping(payload)
    with pytest.raises(ValueError, match="overlap"):
        build_portfolio(evidence, evidence / "export")
