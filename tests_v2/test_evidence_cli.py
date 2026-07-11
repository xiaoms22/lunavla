from __future__ import annotations

import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from lunavla.config import ExperimentConfig
from lunavla.evidence import EvidenceSource
from lunavla.evidence_design import EvidenceDesign
from lunavla.evidence_runner import (
    _PAIR_FIELDS,
    _aggregate,
    _evaluation_fixture,
    _execute_job,
    _read_pair_rows,
    _resolved_job_config,
    derive_plan,
    is_full_design,
    run_evidence_design,
    snapshot_evidence,
    verify_evidence,
)
from lunavla.manifest import RunManifest, sha256_file


def _design_payload(*, suite: str = "language", reduced: bool = False) -> dict[str, Any]:
    evaluation = list(range(1000, 1024))
    train = [11, 22, 33, 44, 55]
    if reduced:
        evaluation = evaluation[:6]
        train = train[:2]
    arms = [
        {"id": "control", "role": "control", "mode": "none"},
        {"id": "mask", "role": "intervention", "mode": "mask"},
        {"id": "shuffle", "role": "intervention", "mode": "shuffle"},
        {
            "id": "counterfactual",
            "role": "intervention",
            "mode": "counterfactual",
        },
    ]
    base_config = "configs/v2/transformer_chunk_cpu.yaml"
    dataset_episodes = 96
    training_steps = 1_000
    if suite == "visual":
        arms = [
            {"id": "control", "role": "control", "mode": "none"},
            {"id": "occlusion", "role": "intervention", "mode": "occlusion"},
            {"id": "shuffle", "role": "intervention", "mode": "shuffle"},
            {"id": "state_only", "role": "baseline", "mode": "state_only"},
        ]
        base_config = "configs/v2/transformer_visual_cpu.yaml"
        dataset_episodes = 64
        training_steps = 1_500
    return {
        "schema_version": 1,
        "design_id": "language-alpha2" if suite == "language" else "visual-beta1",
        "suite": suite,
        "base_config": base_config,
        "seeds": {
            "train": train,
            "data": 42,
            "split": 42,
            "evaluation": evaluation,
            "bootstrap": 202611,
        },
        "arms": arms,
        "metrics": [
            {"name": "success_rate", "kind": "binary", "direction": "negative"},
            {
                "name": "final_distance",
                "kind": "continuous",
                "direction": "positive",
            },
            {
                "name": "first_action_mse",
                "kind": "continuous",
                "direction": "positive",
            },
        ],
        "budget": {
            "dataset_episodes": dataset_episodes,
            "batch_size": 32,
            "training_steps": training_steps,
            "learning_rate": 0.0003,
            "evaluation_episodes": len(evaluation),
            "bootstrap_samples": 10_000,
        },
        "output": {
            "run_root": f"outputs/evidence/{suite}-study",
            "snapshot_root": f"results/v2/{suite}-study",
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _evidence_fixture(tmp_path: Path) -> Path:
    payload = _design_payload(reduced=True)
    design = EvidenceDesign.from_mapping(payload)
    plan = derive_plan(design, allow_reduced_design=True)
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "design.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    base = ExperimentConfig.load(design.base_config)
    (root / "base_config.json").write_text(
        json.dumps(base.to_dict()), encoding="utf-8"
    )
    (root / "plan.json").write_text(
        json.dumps({**plan.to_dict(), "base_config_sha256": base.sha256()}),
        encoding="utf-8",
    )
    all_rows: list[dict[str, Any]] = []
    sources: list[EvidenceSource] = []
    manifests: list[RunManifest] = []
    for job in plan.jobs:
        run_dir = root / "runs" / job.run_id
        run_dir.mkdir(parents=True)
        config = _resolved_job_config(design, base, job)
        checkpoint = run_dir / str(config.artifacts["checkpoint_name"])
        checkpoint.write_bytes(b"test checkpoint\n")
        resolved = run_dir / "resolved_config.json"
        resolved.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        metrics = run_dir / "metrics.json"
        metrics.write_text('{"claim_allowed": false}\n', encoding="utf-8")
        donor = run_dir / "donor_bank.json"
        donor.write_text("{}\n", encoding="utf-8")
        evidence_fixture = _evaluation_fixture(config, design)
        fixture_by_seed = {
            int(item["eval_seed"]): item for item in evidence_fixture["episodes"]
        }
        rows: list[dict[str, Any]] = []
        rollouts: list[dict[str, Any]] = []
        for arm_id in job.arm_ids:
            arm = next(item for item in design.arms if item.id == arm_id)
            for eval_seed in design.seeds.evaluation:
                pair_id = f"{job.train_seed}:{eval_seed}:{arm_id}"
                shared_pair_id = f"{job.train_seed}:{eval_seed}"
                rows.append(
                    {
                        "pair_id": shared_pair_id,
                        "cell_id": pair_id,
                        "run_id": job.run_id,
                        "train_seed": job.train_seed,
                        "eval_seed": eval_seed,
                        "arm_id": arm_id,
                        "arm_mode": arm.mode,
                        "family": fixture_by_seed[eval_seed]["task_id"],
                        "success": 0,
                        "final_distance": 0.5,
                        "first_action_mse": 0.25,
                        "total_reward": -1.0,
                        "steps": 2,
                    }
                )
                rollouts.append(
                    {
                        "pair_id": shared_pair_id,
                        "cell_id": pair_id,
                        "arm_id": arm_id,
                        "success": False,
                        "final_distance": 0.5,
                        "first_action_mse": 0.25,
                        "total_reward": -1.0,
                        "steps": 2,
                    }
                )
        pairs = run_dir / "per_pair.csv"
        _write_csv(pairs, rows)
        rollout_path = run_dir / "rollouts.json"
        rollout_path.write_text(json.dumps(rollouts), encoding="utf-8")
        created = RunManifest.create(
            root=Path.cwd(),
            config=config,
            data_sha256="a" * 64,
            checkpoint_path=checkpoint,
            dataset_split={"train": [0], "validation": [], "test": []},
            command=["lunavla-v2", "evidence-run"],
            metrics={"claim_allowed": False},
            artifact_paths={
                "resolved_config.json": resolved,
                "metrics.json": metrics,
                "donor_bank.json": donor,
                "per_pair.csv": pairs,
                "rollouts.json": rollout_path,
            },
            design_id=design.design_id,
            design_sha256=design.sha256(),
            condition={
                "variant": job.variant,
                "train_seed": job.train_seed,
                "arm_ids": list(job.arm_ids),
            },
            eval_fixture={
                "evidence_fixture_sha256": evidence_fixture["sha256"],
                "evidence_episodes": evidence_fixture["episodes"],
            },
            paired_data={
                "donor_bank_sha256": sha256_file(donor),
                "non_image_pairing_sha256": "c" * 64,
            },
            arms=[
                next(item for item in design.arms if item.id == arm_id).to_dict()
                for arm_id in job.arm_ids
            ],
            pairs=[
                {
                    "pair_id": f"{job.train_seed}:{eval_seed}",
                    "train_seed": job.train_seed,
                    "eval_seed": eval_seed,
                    "arms": list(job.arm_ids),
                }
                for eval_seed in design.seeds.evaluation
            ],
            runtime_determinism={"seeded": True},
        )
        manifest = replace(
            created,
            git_sha="0" * 40,
            git_dirty=False,
            source_diff_sha256=None,
        )
        manifest_path = manifest.write(run_dir / "manifest.json")
        sources.append(EvidenceSource(job.run_id, sha256_file(manifest_path)))
        manifests.append(manifest)
        all_rows.extend(rows)
    _write_csv(root / "per_pair.csv", all_rows)
    (root / "reproducibility.json").write_text(
        json.dumps({"required": False, "verified": True}), encoding="utf-8"
    )
    evidence = _aggregate(
        design,
        plan,
        manifests,
        all_rows,
        sources,
        reproducibility_verified=True,
    )
    evidence.write(root / "evidence_manifest.json")
    (root / "aggregate.json").write_text(
        json.dumps(
            {
                "statistics": [item.to_dict() for item in evidence.statistics],
                "claims": [claim.to_dict() for claim in evidence.claims],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_full_matrix_derivation_matches_fixed_training_and_episode_counts() -> None:
    language = EvidenceDesign.from_mapping(_design_payload())
    language_plan = derive_plan(language)
    assert is_full_design(language)
    assert language_plan.expected_training_runs == 5
    assert language_plan.expected_arm_episodes == 480

    visual = EvidenceDesign.from_mapping(_design_payload(suite="visual"))
    visual_plan = derive_plan(visual)
    assert is_full_design(visual)
    assert visual_plan.expected_training_runs == 10
    assert visual_plan.expected_arm_episodes == 480
    assert sum(job.variant == "image" for job in visual_plan.jobs) == 5
    assert sum(job.variant == "state_only" for job in visual_plan.jobs) == 5
    assert (
        language_plan.expected_training_runs + visual_plan.expected_training_runs
    ) == 15
    assert (
        language_plan.expected_arm_episodes + visual_plan.expected_arm_episodes
    ) == 960


def test_reduced_design_is_explicit_and_always_observational() -> None:
    design = EvidenceDesign.from_mapping(_design_payload(reduced=True))
    with pytest.raises(ValueError, match="--allow-reduced-design"):
        derive_plan(design)
    plan = derive_plan(design, allow_reduced_design=True)
    assert plan.reduced_design
    assert plan.to_dict()["observational"] is True
    assert plan.to_dict()["claim_allowed"] is False

    invalid = _design_payload(reduced=True)
    invalid["base_config"] = "configs/v2/numpy_baseline.yaml"
    with pytest.raises(ValueError, match="smaller canonical matrix"):
        derive_plan(
            EvidenceDesign.from_mapping(invalid),
            allow_reduced_design=True,
        )


def test_verify_and_snapshot_accept_complete_reduced_matrix(tmp_path: Path) -> None:
    root = _evidence_fixture(tmp_path)
    report = verify_evidence(root)
    assert report.reduced_design
    assert report.source_count == 2
    assert report.arm_episode_count == 48

    snapshot = snapshot_evidence(root, tmp_path / "results" / "v2" / "snapshot")
    assert (snapshot / "snapshot_manifest.json").is_file()
    assert list(snapshot.rglob("rollouts.sample.json"))
    assert not list(snapshot.rglob("checkpoint*"))
    assert not list(snapshot.rglob("*.pt"))


def test_verify_rejects_hashed_source_artifact_tampering(tmp_path: Path) -> None:
    root = _evidence_fixture(tmp_path)
    donor = next((root / "runs").glob("*/donor_bank.json"))
    donor.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_evidence(root)


def test_verify_rejects_tampered_aggregate_claim_text(tmp_path: Path) -> None:
    root = _evidence_fixture(tmp_path)
    aggregate_path = root / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["claims"][0]["statement"] = "Unsupported rewritten claim."
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate.json"):
        verify_evidence(root)


@pytest.mark.torch
def test_reduced_runner_trains_once_and_evaluates_every_arm(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    payload = _design_payload(reduced=True)
    payload["base_config"] = "configs/v2/transformer_chunk_cpu.yaml"
    payload["budget"].update(
        {
            "dataset_episodes": 6,
            "batch_size": 2,
            "training_steps": 1,
            "bootstrap_samples": 100,
        }
    )
    design = EvidenceDesign.from_mapping(payload)
    job = derive_plan(design, allow_reduced_design=True).jobs[0]
    base = ExperimentConfig.load(design.base_config)
    config = _resolved_job_config(design, base, job)
    run_dir = tmp_path / job.run_id

    manifest = _execute_job(
        root=Path.cwd(),
        run_dir=run_dir,
        design=design,
        job=job,
        config=config,
        command=["lunavla-v2", "evidence-run", "design.yaml"],
    )

    rows = _read_pair_rows(run_dir / "per_pair.csv")
    assert len(rows) == len(job.arm_ids) * len(design.seeds.evaluation) == 24
    assert {row["arm_id"] for row in rows} == set(job.arm_ids)
    assert len(manifest.pairs) == 6
    assert manifest.design_sha256 == design.sha256()


@pytest.mark.integration
@pytest.mark.torch
def test_reduced_study_runs_verifies_and_snapshots_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    payload = _design_payload(reduced=True)
    payload["budget"].update(
        {
            "dataset_episodes": 6,
            "batch_size": 2,
            "training_steps": 1,
            "bootstrap_samples": 100,
        }
    )
    relative_output = f"outputs/evidence/test-{tmp_path.name}"
    payload["output"]["run_root"] = relative_output
    payload["output"]["snapshot_root"] = f"results/v2/test-{tmp_path.name}"
    design_path = tmp_path / "design.yaml"
    design_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    repository = Path.cwd().resolve()
    output = repository / relative_output
    shutil.rmtree(output, ignore_errors=True)

    import lunavla.evidence_runner as runner_module
    import lunavla.manifest as manifest_module

    monkeypatch.setattr(runner_module, "_repository_root", lambda _: repository)
    monkeypatch.setattr(runner_module, "git_source_state", lambda _: (False, None))
    monkeypatch.setattr(manifest_module, "git_source_state", lambda _: (False, None))
    try:
        evidence = run_evidence_design(
            design_path,
            allow_reduced_design=True,
            command=["lunavla-v2", "evidence-run", str(design_path)],
        )
        assert evidence.reduced_design
        assert not any(claim.allowed for claim in evidence.claims)
        report = verify_evidence(output)
        assert report.source_count == 2
        assert report.arm_episode_count == 48
        snapshot = snapshot_evidence(output, tmp_path / "results" / "v2" / "e2e")
        assert (snapshot / "evidence_manifest.json").is_file()
        assert not list(snapshot.rglob("checkpoint*"))
    finally:
        shutil.rmtree(output, ignore_errors=True)
