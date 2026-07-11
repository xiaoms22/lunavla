from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from lunavla.config import ExperimentConfig as ExperimentConfigV2
from lunavla.contracts import Observation
from lunavla.v3 import (
    ExperimentConfig,
    migrate_v2_mapping,
    observation_from_v2,
    observation_to_v2,
)


def _v2() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "project_name": "v3-migration",
        "engine": "lunavla_v2",
        "policy": {"type": "numpy_linear_chunk", "state_dim": 4, "action_dim": 2, "chunk_size": 2, "device": "cpu"},
        "task": {"id": "pusht_style_point_reach", "max_steps": 8},
        "dataset": {"type": "memory", "split": "train", "seed": 7},
        "training": {"device": "cpu", "seed": 7, "steps": 3, "batch_size": 2, "learning_rate": 0.01},
        "evaluation": {"execution_mode": "open_loop_chunk", "episodes": 2, "seed": 10, "seeds": [10, 11]},
        "artifacts": {"output_dir": "outputs/v3-migration", "checkpoint_name": "checkpoint.json"},
    }


def test_v2_migration_preserves_semantics_and_is_stable() -> None:
    source = _v2()
    v2 = ExperimentConfigV2.from_mapping(source)
    migrated = migrate_v2_mapping(source)
    v3 = ExperimentConfig.from_mapping(migrated)
    assert migrated == migrate_v2_mapping(source)
    assert v3.policy["type"] == v2.policy["type"]
    assert v3.task["id"] == v2.task["id"]
    assert v3.training["seed"] == v2.training["seed"]
    assert v3.artifacts["output_dir"] == v2.artifacts["output_dir"]
    assert v3.feature_schema.by_role("state")[0].shape == (4,)
    assert v3.feature_schema.by_role("action")[0].shape == (2,)
    assert v3.embodiment["control_rate_hz"] is None
    assert v3.policy["parameters"]["compat_read_only"] is True


@pytest.mark.parametrize("version", [True, 2.0, "2"])
def test_migration_rejects_schema_coercion(version: object) -> None:
    source = _v2()
    source["schema_version"] = version
    with pytest.raises(TypeError, match="schema_version must be an integer"):
        migrate_v2_mapping(source)


def test_v3_config_is_strict_immutable_and_hash_stable() -> None:
    payload = migrate_v2_mapping(_v2())
    config = ExperimentConfig.from_mapping(payload)
    duplicate = ExperimentConfig.from_mapping(copy.deepcopy(payload))
    assert config.sha256() == duplicate.sha256()
    payload["policy"]["typo"] = 1
    with pytest.raises(ValueError, match="unknown field.*policy"):
        ExperimentConfig.from_mapping(payload)
    with pytest.raises(TypeError):
        config.policy["type"] = "other"  # type: ignore[index]


@pytest.mark.parametrize("version", [True, 3.0, "3"])
def test_v3_rejects_boolean_float_and_string_schema_versions(version: object) -> None:
    payload = migrate_v2_mapping(_v2())
    payload["schema_version"] = version
    with pytest.raises(TypeError, match="schema_version must be an integer"):
        ExperimentConfig.from_mapping(payload)


def test_v3_rejects_shape_unit_and_cross_section_errors() -> None:
    payload = migrate_v2_mapping(_v2())
    payload["features"]["items"][0]["shape"] = [True]
    with pytest.raises(TypeError, match="shape.*integer"):
        ExperimentConfig.from_mapping(payload)
    payload = migrate_v2_mapping(_v2())
    payload["features"]["items"][0]["unit"] = ""
    with pytest.raises(ValueError, match="unit must be a non-empty"):
        ExperimentConfig.from_mapping(payload)
    payload = migrate_v2_mapping(_v2())
    payload["embodiment"]["task_id"] = "fake_libero"
    with pytest.raises(ValueError, match="must match task.id"):
        ExperimentConfig.from_mapping(payload)


def test_all_tracked_v2_configs_migrate() -> None:
    for path in sorted(Path("configs/v2").glob("*.yaml")):
        ExperimentConfig.from_mapping(migrate_v2_mapping(yaml.safe_load(path.read_text())))


def test_v2_observation_round_trip_is_explicit() -> None:
    original = Observation(np.asarray([1.0, 2.0]), instruction="move")
    migrated = observation_from_v2(original, episode_id="ep", step_index=0, timestamp_s=0.0)
    assert observation_to_v2(migrated) == original


def test_unknown_nested_parameters_fail() -> None:
    payload = migrate_v2_mapping(_v2())
    payload["dataset"] = {
        "type": "fake_pusht",
        "split": "train",
        "seed": 1,
        "parameters": {"typo": 1},
    }
    with pytest.raises(ValueError, match="dataset.parameters.*typo"):
        ExperimentConfig.from_mapping(payload)


def test_runnable_task_dataset_and_policy_combinations_fail_fast() -> None:
    payload = ExperimentConfig.load("configs/v3/fake_pusht_alpha.yaml").to_dict()
    payload["dataset"]["type"] = "fake_libero"
    with pytest.raises(ValueError, match="task.id and dataset.type must match"):
        ExperimentConfig.from_mapping(payload)
    payload = ExperimentConfig.load("configs/v3/fake_pusht_alpha.yaml").to_dict()
    payload["policy"] = {
        "type": "transformer_chunk_cvae",
        "parameters": {
            "state_dim": 4,
            "instruction_dim": 0,
            "action_dim": 2,
            "chunk_size": 4,
        },
    }
    with pytest.raises(ValueError, match="migration-only"):
        ExperimentConfig.from_mapping(payload)


@pytest.mark.parametrize(
    "checkpoint_name",
    ["../escape.json", "/tmp/escape.json", "nested/policy.json", "checkpoint.v3.json", "manifest.json"],
)
def test_checkpoint_name_is_a_contained_non_reserved_basename(checkpoint_name: str) -> None:
    payload = ExperimentConfig.load("configs/v3/fake_pusht_alpha.yaml").to_dict()
    payload["artifacts"]["checkpoint_name"] = checkpoint_name
    with pytest.raises(ValueError, match="checkpoint_name"):
        ExperimentConfig.from_mapping(payload)
