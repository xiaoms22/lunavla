from __future__ import annotations

import json
from pathlib import Path

import pytest

from lunavla.config import ExperimentConfig
from lunavla.manifest import RunManifest, sha256_transitions
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
    )
    path = manifest.write(tmp_path / "manifest.json")
    assert RunManifest.load(path) == manifest
    assert manifest.eval_seeds == [90, 91]

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["project_name"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="config_sha256"):
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
