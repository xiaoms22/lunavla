from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lunavla.cli import main
from lunavla.config import ExperimentConfig
from lunavla.migration import migrate_v11_mapping
from lunavla.run import _dataset_source, _split_transitions
from scripts.validate_configs import validate


def _v2_mapping() -> dict[str, object]:
    return {
        "schema_version": 2,
        "project_name": "v2-test",
        "engine": "lunavla_v2",
        "policy": {
            "type": "numpy_linear_chunk",
            "state_dim": 4,
            "action_dim": 2,
            "chunk_size": 2,
            "device": "cpu",
        },
        "task": {"id": "pusht_style_point_reach", "max_steps": 8},
        "dataset": {"type": "memory", "split": "train", "seed": 7},
        "training": {"device": "cpu", "steps": 3, "batch_size": 2},
        "evaluation": {"episodes": 2, "seed": 10},
        "artifacts": {"output_dir": "outputs/v2-test"},
    }


def _visual_mapping(*, state_only: bool = False) -> dict[str, object]:
    payload = _v2_mapping()
    payload["policy"] = {
        "type": "transformer_chunk",
        "state_dim": 7,
        "instruction_dim": 8,
        "image_shape": None if state_only else [32, 32, 3],
        "action_dim": 2,
        "chunk_size": 2,
        "d_model": 16,
        "nhead": 4,
        "device": "cpu",
    }
    payload["task"] = {
        "id": "rendered_visual_point_reach",
        "family": "all",
        "max_steps": 8,
        "render_size": 32,
    }
    payload["dataset"] = {
        "type": "memory",
        "split": "train",
        "seed": 7,
        "parameters": {"state_only": state_only},
    }
    payload["evaluation"] = {
        "episodes": 2,
        "seed": 10,
        "image_ablation": "state_only" if state_only else "occlusion",
    }
    payload["artifacts"] = {
        "output_dir": "outputs/v2-test",
        "checkpoint_name": "checkpoint.pt",
    }
    return payload


def test_v2_config_is_strict_and_hash_is_stable() -> None:
    first = ExperimentConfig.from_mapping(_v2_mapping())
    second = ExperimentConfig.from_mapping(_v2_mapping())
    assert first.sha256() == second.sha256()
    invalid = _v2_mapping()
    assert isinstance(invalid["policy"], dict)
    invalid["policy"]["typo"] = True
    with pytest.raises(ValueError, match="unknown field.*typo"):
        ExperimentConfig.from_mapping(invalid)


def test_v2_schema_version_rejects_boolean() -> None:
    invalid = _v2_mapping()
    invalid["schema_version"] = True
    with pytest.raises(TypeError, match="schema_version must be an integer"):
        ExperimentConfig.from_mapping(invalid)


def test_config_loader_and_validator_report_malformed_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text(
        "schema_version: 2\npolicy: [unterminated\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid YAML.*line"):
        ExperimentConfig.load(config_path)
    errors = validate(config_path)
    assert len(errors) == 1
    assert errors[0].startswith("invalid YAML:")


def test_standalone_validator_rejects_boolean_schema_version(tmp_path: Path) -> None:
    config_path = tmp_path / "boolean-schema.yaml"
    payload = _v2_mapping()
    payload["schema_version"] = True
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    assert validate(config_path) == ["schema_version must be an integer"]


def test_numpy_device_must_be_cpu() -> None:
    invalid = _v2_mapping()
    assert isinstance(invalid["policy"], dict)
    assert isinstance(invalid["training"], dict)
    invalid["policy"]["device"] = "cuda"
    invalid["training"]["device"] = "cuda"
    with pytest.raises(ValueError, match="NumPy-only"):
        ExperimentConfig.from_mapping(invalid)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("policy", "state_dim", 1.9, "must be an integer"),
        ("training", "steps", True, "must be an integer"),
        ("dataset", "type", "mystery", "unsupported dataset.type"),
        ("task", "id", "nonexistent_robot_task", "unsupported task.id"),
    ],
)
def test_v2_config_rejects_coercions_and_unknown_registrations(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    invalid = _v2_mapping()
    assert isinstance(invalid[section], dict)
    invalid[section][field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        ExperimentConfig.from_mapping(invalid)


def test_transformer_rejects_unknown_device_and_invalid_temporal_mode() -> None:
    invalid = _v2_mapping()
    assert isinstance(invalid["policy"], dict)
    assert isinstance(invalid["training"], dict)
    invalid["policy"]["type"] = "transformer_chunk"
    invalid["policy"]["device"] = "gpu"
    invalid["training"]["device"] = "gpu"
    invalid["artifacts"] = {
        "output_dir": "outputs/v2-test",
        "checkpoint_name": "checkpoint.pt",
    }
    with pytest.raises(ValueError, match="device must"):
        ExperimentConfig.from_mapping(invalid)


def test_config_rejects_cross_modality_ablation_combinations() -> None:
    invalid = _v2_mapping()
    assert isinstance(invalid["evaluation"], dict)
    invalid["evaluation"]["language_ablation"] = "drop"
    with pytest.raises(ValueError, match="must be none, mask, shuffle"):
        ExperimentConfig.from_mapping(invalid)

    invalid = _visual_mapping()
    assert isinstance(invalid["evaluation"], dict)
    invalid["evaluation"]["language_ablation"] = "mask"
    with pytest.raises(ValueError, match="one language or image ablation"):
        ExperimentConfig.from_mapping(invalid)

    invalid = _v2_mapping()
    assert isinstance(invalid["evaluation"], dict)
    invalid["evaluation"]["language_ablation"] = "shuffle"
    with pytest.raises(ValueError, match="language_ablation requires"):
        ExperimentConfig.from_mapping(invalid)

    invalid = _v2_mapping()
    assert isinstance(invalid["evaluation"], dict)
    invalid["evaluation"]["image_ablation"] = "occlusion"
    with pytest.raises(ValueError, match="image_ablation requires"):
        ExperimentConfig.from_mapping(invalid)


def test_visual_config_rejects_inconsistent_image_and_state_only_modes() -> None:
    invalid = _visual_mapping()
    assert isinstance(invalid["policy"], dict)
    invalid["policy"]["image_shape"] = [32, 24, 3]
    with pytest.raises(ValueError, match="must match the rendered RGB"):
        ExperimentConfig.from_mapping(invalid)

    invalid = _visual_mapping()
    assert isinstance(invalid["task"], dict)
    invalid["task"]["render_size"] = 16
    with pytest.raises(ValueError, match="at least 24"):
        ExperimentConfig.from_mapping(invalid)

    invalid = _visual_mapping(state_only=True)
    assert isinstance(invalid["policy"], dict)
    invalid["policy"]["image_shape"] = [32, 32, 3]
    with pytest.raises(ValueError, match="state-only mode requires"):
        ExperimentConfig.from_mapping(invalid)

    invalid = _visual_mapping()
    assert isinstance(invalid["evaluation"], dict)
    invalid["evaluation"]["image_ablation"] = "state_only"
    with pytest.raises(ValueError, match="state_only requires"):
        ExperimentConfig.from_mapping(invalid)

    invalid = _visual_mapping()
    assert isinstance(invalid["dataset"], dict)
    assert isinstance(invalid["dataset"]["parameters"], dict)
    invalid["dataset"]["parameters"]["state_only"] = "false"
    with pytest.raises(TypeError, match="state_only must be boolean"):
        ExperimentConfig.from_mapping(invalid)


def test_task_specific_dimensions_and_transformer_heads_fail_during_parse() -> None:
    invalid = _v2_mapping()
    assert isinstance(invalid["policy"], dict)
    invalid["policy"]["state_dim"] = 3
    with pytest.raises(ValueError, match="requires policy.state_dim=4"):
        ExperimentConfig.from_mapping(invalid)

    invalid = _visual_mapping()
    assert isinstance(invalid["policy"], dict)
    invalid["policy"]["d_model"] = 15
    with pytest.raises(ValueError, match="divisible"):
        ExperimentConfig.from_mapping(invalid)

    invalid = _visual_mapping()
    assert isinstance(invalid["task"], dict)
    invalid["task"]["family"] = "unknown"
    with pytest.raises(ValueError, match="task.family must be"):
        ExperimentConfig.from_mapping(invalid)

    invalid["policy"]["device"] = "cpu"
    invalid["training"]["device"] = "cpu"
    invalid["policy"]["temporal_ensemble_decay"] = 0.1
    with pytest.raises(ValueError, match="requires receding_horizon"):
        ExperimentConfig.from_mapping(invalid)


def test_v11_mapping_migrates_without_changing_policy_contract() -> None:
    with Path("configs/act_pusht_cpu_smoke.yaml").open(encoding="utf-8") as stream:
        v11 = yaml.safe_load(stream)
    migrated = ExperimentConfig.from_mapping(migrate_v11_mapping(v11))
    assert migrated.schema_version == 2
    assert migrated.policy["type"] == "numpy_linear_chunk"
    assert migrated.policy["chunk_size"] == v11["policy"]["chunk_size"]
    assert migrated.task["id"] == "pusht_style_point_reach"


def test_v11_act_alias_does_not_become_the_v2_transformer() -> None:
    with Path("configs/act_pusht_cpu_smoke.yaml").open(encoding="utf-8") as stream:
        v11 = yaml.safe_load(stream)
    v11["policy"]["type"] = "act"
    with pytest.warns(DeprecationWarning, match="policy alias"):
        migrated = ExperimentConfig.from_mapping(migrate_v11_mapping(v11))
    assert migrated.policy["type"] == "numpy_linear_chunk"


def test_migrated_v11_jsonl_config_has_an_executable_v2_data_path() -> None:
    with Path("configs/act_pusht_jsonl_smoke.yaml").open(encoding="utf-8") as stream:
        v11 = yaml.safe_load(stream)
    migrated = ExperimentConfig.from_mapping(migrate_v11_mapping(v11))
    source = _dataset_source(migrated, Path.cwd())
    transitions = tuple(source.load())
    splits = _split_transitions(transitions, migrated)
    assert transitions
    assert splits[migrated.dataset["split"]]
    ids = [
        {item.info["episode_id"] for item in splits[name]}
        for name in ("train", "validation", "test")
    ]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])


def test_cli_validates_and_refuses_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_v2_mapping()), encoding="utf-8")
    assert main(["validate-config", str(config_path)]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == 2

    source = Path("configs/act_pusht_cpu_smoke.yaml")
    destination = tmp_path / "migrated.yaml"
    assert main(["migrate-config", str(source), str(destination)]) == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(["migrate-config", str(source), str(destination)])
