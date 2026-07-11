from __future__ import annotations

import copy
from typing import Any

import pytest

from lunavla.config import ExperimentConfig as ExperimentConfigV2
from lunavla.v3 import ExperimentConfig, migrate_v2_mapping


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
