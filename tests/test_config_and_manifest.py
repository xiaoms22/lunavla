from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from dataset import generate_mock_pusht_records, split_records_by_episode
from trainer.artifacts import RunManifest, sha256_file
from trainer.config import ExperimentConfig


def config_mapping(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "project_name": "test-run",
        "framework": "lunavla",
        "task": "pusht_style_point_reach",
        "policy": {
            "type": "numpy_linear_chunk",
            "chunk_size": 2,
            "observation_dim": 4,
            "instruction_dim": 8,
            "action_dim": 2,
        },
        "dataset": {
            "source": "mock_pusht",
            "num_episodes": 5,
            "steps_per_episode": 3,
            "seed": 7,
            "split_seed": 8,
        },
        "training": {
            "device": "cpu",
            "batch_size": 4,
            "num_steps": 2,
            "learning_rate": 0.01,
            "seed": 9,
            "log_interval": 1,
        },
        "eval": {"episodes": 2, "seed": 100},
        "artifacts": {"output_dir": "outputs/test-run"},
    }
    payload.update(overrides)
    return payload


def test_canonical_config_defaults_and_stable_hash() -> None:
    first = ExperimentConfig.from_mapping(config_mapping())
    second = ExperimentConfig.from_mapping(config_mapping())

    assert first.schema_version == 1
    assert first.policy["type"] == "numpy_linear_chunk"
    assert first.eval["execution_mode"] == "open_loop_chunk"
    assert first.artifacts["checkpoint_name"] == "checkpoint.json"
    assert first.sha256() == second.sha256()


def test_chunk_size_one_defaults_to_receding_horizon() -> None:
    mapping = config_mapping()
    mapping["policy"]["chunk_size"] = 1
    config = ExperimentConfig.from_mapping(mapping)
    assert config.eval["execution_mode"] == "receding_horizon"


def test_legacy_config_is_migrated_with_deprecation_warning() -> None:
    legacy = config_mapping()
    legacy.pop("schema_version")
    legacy["task"] = "pusht_mock"
    legacy["model"] = {
        "name": "mini_vla",
        "policy_type": "act",
        "observation_dim": 4,
        "instruction_dim": 8,
        "action_dim": 2,
        "chunk_size": 2,
    }
    legacy["policy"] = {"name": "act", "chunk_size": 2}
    legacy["artifacts"]["checkpoint_name"] = "checkpoint.pt"

    with pytest.warns(DeprecationWarning):
        migrated = ExperimentConfig.from_mapping(legacy)

    assert migrated.policy["type"] == "numpy_linear_chunk"
    assert migrated.task == "pusht_style_point_reach"
    assert migrated.artifacts["checkpoint_name"] == "checkpoint.json"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"surprise": True}), "unknown field"),
        (lambda value: value["policy"].update({"surprise": True}), "unknown field"),
        (lambda value: value.update({"schema_version": 999}), "schema_version"),
        (lambda value: value["training"].update({"device": "cuda"}), "NumPy-only"),
        (lambda value: value["policy"].update({"type": "unknown"}), "unsupported policy"),
        (
            lambda value: value["eval"].update({"execution_mode": "not-a-mode"}),
            "execution_mode",
        ),
        (
            lambda value: value["dataset"].update(
                {"train_fraction": 0.5, "validation_fraction": 0.5, "test_fraction": 0.5}
            ),
            "sum to 1",
        ),
    ],
)
def test_config_rejects_unknown_or_unsupported_values(mutate: Any, message: str) -> None:
    mapping = config_mapping()
    mutate(mapping)
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.from_mapping(mapping)


def test_config_loads_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_mapping(), sort_keys=False), encoding="utf-8")
    assert ExperimentConfig.load(path).project_name == "test-run"


def test_manifest_round_trip_records_hashes_splits_and_seeds(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(config_mapping())
    data_path = tmp_path / "dataset.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    data_path.write_text("deterministic dataset bytes\n", encoding="utf-8")
    checkpoint_path.write_text("deterministic checkpoint bytes\n", encoding="utf-8")
    splits = split_records_by_episode(generate_mock_pusht_records(5, 3, seed=7), seed=8)

    manifest = RunManifest.create(
        root=tmp_path,
        config=config,
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        splits=splits,
        command=[
            str(tmp_path / ".venv" / "bin" / "python"),
            str(tmp_path / "trainer" / "train_act_pusht.py"),
        ],
        metrics={"success_rate": 0.5},
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    loaded = RunManifest.load(path)

    assert loaded.schema_version == 1
    assert loaded.run_id == "test-run"
    assert loaded.config_sha256 == config.sha256()
    assert loaded.data_sha256 == sha256_file(data_path)
    assert loaded.checkpoint_sha256 == sha256_file(checkpoint_path)
    assert loaded.train_seeds == [7, 9]
    assert loaded.eval_seeds == [100, 101]
    assert loaded.policy_id == "numpy_linear_chunk"
    assert loaded.command == ["python", "trainer/train_act_pusht.py"]
    assert loaded.metrics == {"success_rate": 0.5}
    split_ids = {name: set(ids) for name, ids in loaded.dataset_split.items()}
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])


def test_manifest_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        RunManifest.load(path)


def test_schema_one_manifest_without_run_id_uses_project_name(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(config_mapping())
    data_path = tmp_path / "dataset.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    data_path.write_text("data\n", encoding="utf-8")
    checkpoint_path.write_text("checkpoint\n", encoding="utf-8")
    manifest = RunManifest.create(
        root=tmp_path,
        config=config,
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        splits=split_records_by_episode(generate_mock_pusht_records(5, 3, seed=7), seed=8),
        command=["python", "train.py"],
        metrics={},
    ).to_dict()
    manifest.pop("run_id")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="config.project_name"):
        loaded = RunManifest.load(path)
    assert loaded.run_id == "test-run"
