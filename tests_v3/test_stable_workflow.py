from __future__ import annotations

import json
from pathlib import Path

import pytest

from lunavla.v3 import (
    StableEvidenceDesignV1,
    StableEvidenceRowV1,
    StableExecutionBatchV1,
    build_portfolio,
    expected_stable_matrix_keys,
    run_stable_study,
    verify_stable_study,
)
from lunavla.v3.stable_executor import TeachingFixtureStableExecutor
from lunavla.v3.cli import main as cli_main


SHA = "a" * 64
GIT_SHA = "b" * 40
DESIGN = Path("configs/v3/stable_pusht_policy_design.yaml")


class _DeterministicExecutor:
    def __init__(
        self,
        *,
        corrupt_repeat: bool = False,
        fail: bool = False,
        binary_artifact: bool = False,
    ) -> None:
        self.corrupt_repeat = corrupt_repeat
        self.fail = fail
        self.binary_artifact = binary_artifact

    def execute(
        self,
        design: StableEvidenceDesignV1,
        *,
        only_train_seed: int | None,
        output_dir: Path,
    ) -> StableExecutionBatchV1:
        if self.fail:
            raise RuntimeError("injected stable executor failure")
        output_dir.mkdir(parents=True)
        rows: list[StableEvidenceRowV1] = []
        for policy, train_seed, task_id, evaluation_id, route, intervention in (
            expected_stable_matrix_keys(design)
        ):
            if only_train_seed is not None and train_seed != only_train_seed:
                continue
            final_metric = float(evaluation_id) / 100.0
            if self.corrupt_repeat and only_train_seed is not None and evaluation_id == 0:
                final_metric += 1.0
            rows.append(
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
                    success=evaluation_id % 2 == 0,
                    final_metric=final_metric,
                    smoothness=0.0,
                    first_action_mse=0.0,
                    failure_count=0,
                )
            )
        (output_dir / "execution.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "only_train_seed": only_train_seed,
                    "rows": len(rows),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if self.binary_artifact:
            (output_dir / "checkpoint.pt").write_bytes(b"\x80LunaVLA-test-checkpoint")
        return StableExecutionBatchV1(tuple(rows), SHA, SHA)


def test_stable_workflow_runs_atomically_and_verifies_tampering(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = run_stable_study(DESIGN, tmp_path, _DeterministicExecutor())
    summary = verify_stable_study(output)
    assert summary.observed_rows == 200
    assert summary.sentinel_verified is True
    assert cli_main(["stable-verify", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["release_eligible"] is True
    rows = output / "rows.json"
    rows.write_text(rows.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_stable_study(output)


def test_stable_workflow_preserves_previous_generation_on_failure(tmp_path: Path) -> None:
    output = run_stable_study(DESIGN, tmp_path, _DeterministicExecutor())
    before = (output / "summary.json").read_bytes()
    with pytest.raises(RuntimeError, match="injected"):
        run_stable_study(
            DESIGN,
            tmp_path,
            _DeterministicExecutor(fail=True),
            overwrite=True,
        )
    assert (output / "summary.json").read_bytes() == before
    verify_stable_study(output)


def test_stable_workflow_records_repeat_failure_without_opening_release(
    tmp_path: Path,
) -> None:
    output = run_stable_study(
        DESIGN, tmp_path, _DeterministicExecutor(corrupt_repeat=True)
    )
    summary = verify_stable_study(output)
    assert summary.sentinel_verified is False
    assert summary.release_eligible is False
    assert summary.gate_reasons == ("sentinel_failure",)


def test_teaching_executor_runs_real_act_engine_and_repeats_exactly(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    payload = StableEvidenceDesignV1.load(DESIGN).to_dict()
    payload["train_seeds"] = [11]
    payload["policies"] = ["act_v3"]
    payload["evaluation_ids"] = [0]
    payload["reduced_design"] = True
    design = StableEvidenceDesignV1.from_mapping(payload)
    executor = TeachingFixtureStableExecutor(git_sha=GIT_SHA)
    first = executor.execute(design, only_train_seed=11, output_dir=tmp_path / "first")
    second = executor.execute(design, only_train_seed=11, output_dir=tmp_path / "second")
    assert first.rows == second.rows
    assert first.checkpoint_inventory_sha256 == second.checkpoint_inventory_sha256
    assert first.metrics_inventory_sha256 == second.metrics_inventory_sha256
    assert len(first.rows) == 1
    assert first.rows[0].first_action_mse is not None
    assert not any(path.name == "checkpoint" for path in (tmp_path / "first").rglob("*"))


def test_portfolio_reads_verified_text_contracts_not_binary_execution_artifacts(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    for design in (
        "configs/v3/stable_pusht_policy_design.yaml",
        "configs/v3/stable_libero_route_design.yaml",
        "configs/v3/stable_libero_prompt_design.yaml",
    ):
        run_stable_study(
            design,
            evidence,
            _DeterministicExecutor(binary_artifact=True),
        )
    portfolio = build_portfolio(evidence, tmp_path / "portfolio")
    assert (portfolio / "portfolio_manifest.json").is_file()
