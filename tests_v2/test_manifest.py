from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from lunavla.config import ExperimentConfig
from lunavla.manifest import MANIFEST_SCHEMA_VERSION, RunManifest, sha256_transitions
from lunavla.memory_data import make_point_reach_demonstrations


def _config(output_dir: str) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "schema_version": 2,
            "project_name": "manifest-test",
            "engine": "lunavla_v2",
            "policy": {
                "type": "numpy_linear_chunk",
                "state_dim": 4,
                "action_dim": 2,
                "chunk_size": 1,
                "device": "cpu",
            },
            "task": {"id": "pusht_style_point_reach"},
            "dataset": {"type": "memory", "split": "train", "seed": 4},
            "training": {"device": "cpu", "seed": 5},
            "evaluation": {"episodes": 2, "seed": 90},
            "artifacts": {"output_dir": output_dir},
        }
    )


def test_transition_hash_is_deterministic_and_content_sensitive() -> None:
    first = make_point_reach_demonstrations(episodes=2, steps_per_episode=3, seed=7).load()
    second = make_point_reach_demonstrations(episodes=2, steps_per_episode=3, seed=7).load()
    different = make_point_reach_demonstrations(episodes=2, steps_per_episode=3, seed=8).load()
    assert sha256_transitions(first) == sha256_transitions(second)
    assert sha256_transitions(first) != sha256_transitions(different)


def test_manifest_round_trip_and_config_hash_validation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text('{"schema_version": 1}', encoding="utf-8")
    config = _config("outputs/test-run")
    transitions = make_point_reach_demonstrations(episodes=2, steps_per_episode=3).load()
    manifest = RunManifest.create(
        root=Path.cwd(),
        config=config,
        data_sha256=sha256_transitions(transitions),
        checkpoint_path=checkpoint,
        dataset_split={"train": [0, 1], "validation": [], "test": []},
        command=["python", "-m", "lunavla.cli"],
        metrics={"success_rate": 0.5},
        design_id="language-v2-alpha2",
        design_sha256="a" * 64,
        condition={"arm": "control", "metadata": {"enabled": True}},
        eval_fixture={"seeds": [90, 91], "sha256": "b" * 64},
        paired_data={"donor_bank_sha256": "c" * 64},
        arms=[{"id": "control"}, {"id": "mask"}],
        pairs=[{"pair_id": "seed-5:episode-90", "arms": ["control", "mask"]}],
        runtime_determinism={"repeatability_check": "passed"},
    )
    path = manifest.write(tmp_path / "manifest.json")
    assert RunManifest.load(path) == manifest
    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION == 3
    assert manifest.eval_seeds == [90, 91]
    assert len(manifest.eval_fixture_sha256 or "") == 64
    assert len(manifest.paired_data_sha256 or "") == 64
    assert manifest.condition == {
        "arm": "control",
        "image_ablation": "none",
        "language_ablation": "none",
        "metadata": {"enabled": True},
    }
    assert manifest.eval_fixture["task_id"] == "pusht_style_point_reach"
    assert manifest.eval_fixture["family"] == "point_reach"
    assert manifest.eval_fixture["execution_mode"] == "receding_horizon"
    assert manifest.eval_fixture["eval_seeds"] == [90, 91]
    assert manifest.eval_fixture["max_steps"] == 40
    assert manifest.paired_data["pair_ids"] == []
    assert manifest.runtime_determinism["device"] == "cpu"
    assert manifest.runtime_determinism["status"] == "unverified"
    assert isinstance(manifest.runtime_determinism["deterministic_flags_satisfied"], bool)
    assert manifest.runtime_determinism["numpy_bit_generator"] == "PCG64"
    assert "PYTHONHASHSEED" in manifest.runtime_determinism
    assert "torch_deterministic_algorithms" in manifest.runtime_determinism
    assert "cudnn_deterministic" in manifest.runtime_determinism
    assert "cudnn_benchmark" in manifest.runtime_determinism
    assert manifest.runtime_determinism["repeatability_check"] == "passed"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["project_name"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="config_sha256"):
        RunManifest.load(path)


def test_schema2_is_read_only_and_unknown_schemas_are_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text('{"schema_version": 1}', encoding="utf-8")
    config = _config("outputs/legacy-run")
    transitions = make_point_reach_demonstrations(episodes=1, steps_per_episode=2).load()
    current = RunManifest.create(
        root=Path.cwd(),
        config=config,
        data_sha256=sha256_transitions(transitions),
        checkpoint_path=checkpoint,
        dataset_split={"train": [0], "validation": [], "test": []},
        command=["lunavla-v2", "train"],
        metrics={"success_rate": 0.0},
    )
    payload = current.to_dict()
    payload["schema_version"] = 2
    for field_name in (
        "design_id",
        "design_sha256",
        "condition",
        "eval_fixture",
        "eval_fixture_sha256",
        "paired_data",
        "paired_data_sha256",
        "arms",
        "pairs",
        "runtime_determinism",
    ):
        payload.pop(field_name)
    path = tmp_path / "legacy-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    legacy = RunManifest.load(path)
    assert legacy.schema_version == 2
    assert legacy.design_id is None
    assert legacy.runtime_determinism == {"status": "unverified"}
    with pytest.raises(ValueError, match="read-only"):
        legacy.write(tmp_path / "rewritten.json")

    for invalid in (True, 1, 4):
        payload["schema_version"] = invalid
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="unsupported manifest schema_version"):
            RunManifest.load(path)


def test_schema3_requires_strict_json_safe_evidence_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text('{"schema_version": 1}', encoding="utf-8")
    config = _config("outputs/evidence-contract")
    transitions = make_point_reach_demonstrations(episodes=1, steps_per_episode=2).load()

    def create(**evidence: Any) -> RunManifest:
        return RunManifest.create(
            root=Path.cwd(),
            config=config,
            data_sha256=sha256_transitions(transitions),
            checkpoint_path=checkpoint,
            dataset_split={"train": [0], "validation": [], "test": []},
            command=["lunavla-v2", "train"],
            metrics={"success_rate": 0.0},
            **evidence,
        )

    with pytest.raises(ValueError, match="provided together"):
        create(design_id="orphan")
    with pytest.raises(ValueError, match="design_sha256"):
        create(
            design_id="bad-hash",
            design_sha256="ABC",
        )
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        create(condition={1: "bad"})
    with pytest.raises(ValueError, match="finite"):
        create(eval_fixture={"distance": float("nan")})
    with pytest.raises(TypeError, match=r"arms\[0\] must be a mapping"):
        create(arms=["not-a-mapping"])
    with pytest.raises(ValueError, match="conflicts with the recorded run value"):
        create(condition={"language_ablation": "mask"})

    paired = create(
        ablation={
            "ablation_mode": "mask",
            "pair_ids": ["pair-1", "pair-2"],
            "interval": {"low": -0.1, "high": 0.2},
        }
    )
    assert paired.pairs == [{"pair_id": "pair-1"}, {"pair_id": "pair-2"}]
    assert paired.paired_data == {
        "pair_ids": ["pair-1", "pair-2"],
        "paired_intervals": [{"low": -0.1, "high": 0.2}],
    }

    manifest = create()
    path = manifest.write(tmp_path / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown manifest field"):
        RunManifest.load(path)

    payload.pop("unexpected")
    payload.pop("runtime_determinism")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing manifest field"):
        RunManifest.load(path)


def test_schema3_detects_evidence_object_tampering(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text('{"schema_version": 1}', encoding="utf-8")
    config = _config("outputs/evidence-hash-contract")
    transitions = make_point_reach_demonstrations(
        episodes=1, steps_per_episode=2
    ).load()
    manifest = RunManifest.create(
        root=Path.cwd(),
        config=config,
        data_sha256=sha256_transitions(transitions),
        checkpoint_path=checkpoint,
        dataset_split={"train": [0], "validation": [], "test": []},
        command=["lunavla-v2", "train"],
        metrics={"success_rate": 0.0},
        paired_data={"donor_bank_sha256": "a" * 64},
    )
    path = manifest.write(tmp_path / "manifest.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["eval_fixture"]["max_steps"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="eval_fixture_sha256"):
        RunManifest.load(path)

    payload = manifest.to_dict()
    payload["paired_data"]["donor_bank_sha256"] = "b" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="paired_data_sha256"):
        RunManifest.load(path)

    payload = manifest.to_dict()
    payload["eval_fixture"]["max_steps"] = 999
    payload["eval_fixture_sha256"] = hashlib.sha256(
        json.dumps(
            payload["eval_fixture"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="eval_fixture.max_steps conflicts"):
        RunManifest.load(path)

    payload = manifest.to_dict()
    payload["runtime_determinism"]["status"] = "verified"
    payload["runtime_determinism"]["deterministic_flags_satisfied"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="requires deterministic flags"):
        RunManifest.load(path)


def test_run_directory_verification_detects_artifact_tampering(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint.json"
    checkpoint.write_text('{"schema_version": 1}', encoding="utf-8")
    metrics = run_dir / "metrics.json"
    metrics.write_text('{"success_rate": 0.5}', encoding="utf-8")
    config = _config("outputs/test-run")
    transitions = make_point_reach_demonstrations(episodes=1, steps_per_episode=2).load()
    manifest = RunManifest.create(
        root=Path.cwd(),
        config=config,
        data_sha256=sha256_transitions(transitions),
        checkpoint_path=checkpoint,
        dataset_split={"train": [0], "validation": [], "test": []},
        command=["lunavla-v2", "train"],
        metrics={"success_rate": 0.5},
        artifact_paths={"metrics.json": metrics},
    )
    manifest.write(run_dir / "manifest.json")
    assert RunManifest.verify_run_dir(run_dir) == manifest
    metrics.write_text('{"success_rate": 1.0}', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        RunManifest.verify_run_dir(run_dir)
