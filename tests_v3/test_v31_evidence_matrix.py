from __future__ import annotations

import json
from pathlib import Path

import pytest

from lunavla.v3 import (
    DeterministicV31FixtureExecutor,
    V31EvidenceDesignV1,
    V31EvidenceRowV1,
    V31RepeatSentinelV1,
    run_v31_evidence,
    v31_expected_matrix_keys,
    verify_v31_evidence,
)


DESIGN = Path("configs/v3/v31_frozen_vlm_evidence.yaml")
GIT_SHA = "a" * 40
H0 = "0" * 64


def test_exact_2400_matrix_repeat_sentinel_statistics_and_fixture_gate(
    tmp_path: Path,
) -> None:
    design = V31EvidenceDesignV1.load(DESIGN)
    assert len(v31_expected_matrix_keys(design)) == 2_400
    output = run_v31_evidence(
        DESIGN,
        tmp_path / "evidence",
        DeterministicV31FixtureExecutor(git_sha=GIT_SHA),
    )
    manifest = verify_v31_evidence(output)
    assert manifest.observed_rows == manifest.expected_rows == 2_400
    assert manifest.matrix_complete is True
    assert manifest.homogeneous is True
    assert manifest.sentinel_verified is True
    assert manifest.release_eligible is False
    assert manifest.claim_allowed is False
    assert "fixture_source" in manifest.gate_reasons
    rows = json.loads((output / "rows.json").read_text())
    assert len(rows) == 2_400
    assert (
        len(
            {
                tuple(
                    row[key]
                    for key in ("train_seed", "arm", "task_id", "held_out_stratum", "episode_index")
                )
                for row in rows
            }
        )
        == 2_400
    )
    statistics = json.loads((output / "statistics.json").read_text())
    assert statistics["bootstrap_samples"] == 10_000
    assert len(statistics["strata_noninferiority"]) == 6


def test_evidence_verifier_rejects_rows_statistics_and_tree_tampering(
    tmp_path: Path,
) -> None:
    output = run_v31_evidence(
        DESIGN,
        tmp_path / "evidence",
        DeterministicV31FixtureExecutor(git_sha=GIT_SHA),
    )
    rows = output / "rows.json"
    rows.write_bytes(rows.read_bytes() + b" ")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_v31_evidence(output)

    output = run_v31_evidence(
        DESIGN,
        tmp_path / "evidence",
        DeterministicV31FixtureExecutor(git_sha=GIT_SHA),
        overwrite=True,
    )
    source = output / "source" / "rows.json"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="execution tree hash mismatch"):
        verify_v31_evidence(output)


def test_row_and_sentinel_contracts_reject_invalid_values() -> None:
    row = V31EvidenceRowV1(
        train_seed=11,
        arm="baseline",
        task_id="direct_pick_place",
        held_out_stratum="composition",
        episode_index=0,
        pair_id="direct_pick_place:composition:0",
        git_sha=GIT_SHA,
        dependency_lock_sha256=H0,
        feature_source_sha256=H0,
        checkpoint_sha256=H0,
        success=False,
        final_distance=0.2,
        first_action_mse=0.1,
    )
    payload = row.to_dict()
    payload["success"] = 1
    with pytest.raises(TypeError, match="boolean"):
        V31EvidenceRowV1.from_mapping(payload)
    with pytest.raises(ValueError, match="reflect exact"):
        V31RepeatSentinelV1(11, H0, H0, H0, H0, H0, "1" * 64, True)


def test_failed_generation_preserves_previous_evidence(tmp_path: Path) -> None:
    output = run_v31_evidence(
        DESIGN,
        tmp_path / "evidence",
        DeterministicV31FixtureExecutor(git_sha=GIT_SHA),
    )
    previous = (output / "evidence-manifest.json").read_bytes()

    class FailingExecutor:
        def execute(self, design, *, only_train_seed, output_dir):
            raise RuntimeError("injected matrix failure")

    with pytest.raises(RuntimeError, match="injected"):
        run_v31_evidence(DESIGN, output, FailingExecutor(), overwrite=True)
    assert (output / "evidence-manifest.json").read_bytes() == previous
    verify_v31_evidence(output)
