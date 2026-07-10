from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from lunavla.config import (
    ExperimentConfig,
    experiment_config_schema_descriptor,
)
from lunavla.evidence_design import (
    EvidenceDesign,
    evidence_design_schema_descriptor,
)
from lunavla.migration import migrate_v11_mapping


def _point_mapping() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "project_name": "rc-config-contract",
        "engine": "lunavla_v2",
        "policy": {
            "type": "numpy_linear_chunk",
            "state_dim": 4,
            "instruction_dim": 0,
            "action_dim": 2,
            "chunk_size": 2,
            "device": "cpu",
        },
        "task": {
            "id": "pusht_style_point_reach",
            "family": "point_reach",
            "max_steps": 8,
            "goal": [0.8, 0.2],
            "parameters": {},
        },
        "dataset": {
            "type": "memory",
            "split": "train",
            "seed": 7,
            "episode_count": 4,
            "parameters": {},
        },
        "training": {
            "device": "cpu",
            "seed": 7,
            "batch_size": 2,
            "steps": 3,
            "learning_rate": 0.01,
            "kl_weight": 0.0,
        },
        "evaluation": {
            "execution_mode": "open_loop_chunk",
            "episodes": 2,
            "seed": 1000,
            "seeds": [1000, 1001],
            "language_ablation": "none",
            "image_ablation": "none",
            "parameters": {},
        },
        "artifacts": {
            "output_dir": "outputs/rc-config-contract",
            "checkpoint_name": "checkpoint.json",
        },
    }


def test_experiment_config_is_deeply_immutable_and_thaws_cleanly() -> None:
    source = _point_mapping()
    config = ExperimentConfig.from_mapping(source)

    source["policy"]["chunk_size"] = 99
    source["task"]["goal"][0] = 0.0
    assert config.policy["chunk_size"] == 2
    assert config.task["goal"] == (0.8, 0.2)
    assert config.evaluation["seeds"] == (1000, 1001)

    with pytest.raises(TypeError):
        config.policy["chunk_size"] = 4  # type: ignore[index]
    with pytest.raises(TypeError):
        config.dataset["parameters"]["split_seed"] = 9  # type: ignore[index]
    with pytest.raises(AttributeError):
        config.evaluation["seeds"].append(1002)

    thawed = config.to_dict()
    assert isinstance(thawed["policy"], dict)
    assert isinstance(thawed["task"]["goal"], list)
    assert isinstance(thawed["evaluation"]["seeds"], list)
    json.dumps(thawed, allow_nan=False)
    assert ExperimentConfig.from_mapping(thawed).to_dict() == thawed
    assert ExperimentConfig.from_mapping(thawed).sha256() == config.sha256()
    thawed["task"]["goal"][0] = 0.1
    assert config.task["goal"] == (0.8, 0.2)


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ("task", "unknown field.*task.parameters.*typo"),
        ("dataset", "unknown field.*dataset.parameters.*typo"),
        ("evaluation", "unknown field.*evaluation.parameters.*typo"),
    ],
)
def test_parameter_typos_fail_during_config_parse(section: str, message: str) -> None:
    payload = _point_mapping()
    payload[section]["parameters"]["typo"] = 1
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("mutate", "error_type", "message"),
    [
        (
            lambda value: value["task"]["parameters"].update({"start_low": True}),
            TypeError,
            "task.parameters.start_low must be a number",
        ),
        (
            lambda value: value["task"]["parameters"].update({"start_low": 0.9, "start_high": 0.2}),
            ValueError,
            "start range",
        ),
        (
            lambda value: value["task"]["parameters"].update({"action_clip": 0}),
            ValueError,
            "action_clip must be positive",
        ),
        (
            lambda value: value["dataset"]["parameters"].update({"split_seed": True}),
            TypeError,
            "split_seed must be an integer",
        ),
        (
            lambda value: value["dataset"]["parameters"].update({"train_fraction": 0.5}),
            ValueError,
            "split fractions must sum to one",
        ),
        (
            lambda value: value["dataset"]["parameters"].update({"steps_per_episode": 2.5}),
            TypeError,
            "steps_per_episode must be an integer",
        ),
        (
            lambda value: value["dataset"]["parameters"].update({"action_gain": "0.2"}),
            TypeError,
            "action_gain must be a number",
        ),
        (
            lambda value: value["evaluation"]["parameters"].update({"bootstrap_samples": 0}),
            ValueError,
            "bootstrap_samples must be a positive integer",
        ),
    ],
)
def test_parameter_types_and_ranges_are_resolved_at_parse_time(
    mutate: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    payload = _point_mapping()
    mutate(payload)
    with pytest.raises(error_type, match=message):
        ExperimentConfig.from_mapping(payload)


def test_task_specific_required_parameters_are_explicit() -> None:
    language = _point_mapping()
    language["policy"].update(
        {
            "type": "transformer_chunk",
            "state_dim": 2,
            "instruction_dim": 16,
            "d_model": 16,
            "nhead": 4,
            "device": "cpu",
        }
    )
    language["task"] = {
        "id": "language_conditioned_point_reach",
        "family": "point_reach",
        "max_steps": 8,
        "parameters": {},
    }
    language["evaluation"]["execution_mode"] = "receding_horizon"
    language["artifacts"]["checkpoint_name"] = "checkpoint.pt"
    with pytest.raises(ValueError, match="requires task.parameters.language_split"):
        ExperimentConfig.from_mapping(language)

    language["task"]["parameters"] = {"language_split": True}
    with pytest.raises(TypeError, match="language_split must be a string"):
        ExperimentConfig.from_mapping(language)

    visual = copy.deepcopy(language)
    visual["policy"].update({"state_dim": 3, "instruction_dim": 8, "image_shape": [32, 32, 3]})
    visual["task"] = {
        "id": "rendered_visual_point_reach",
        "family": "all",
        "max_steps": 8,
        "render_size": 32,
        "parameters": {},
    }
    visual["dataset"]["parameters"] = {}
    with pytest.raises(ValueError, match="requires dataset.parameters.observation_mode"):
        ExperimentConfig.from_mapping(visual)


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_v11_migration_rejects_schema_version_coercions(version: object) -> None:
    with pytest.raises(TypeError, match="schema_version must be an integer"):
        migrate_v11_mapping({"schema_version": version})


@pytest.mark.parametrize("version", [True, 2.0, "2"])
def test_v2_config_rejects_schema_version_coercions(version: object) -> None:
    payload = _point_mapping()
    payload["schema_version"] = version
    with pytest.raises(TypeError, match="schema_version must be an integer"):
        ExperimentConfig.from_mapping(payload)


def test_tracked_configs_and_evidence_designs_still_parse() -> None:
    for path in sorted(Path("configs/v2").glob("*.yaml")):
        ExperimentConfig.load(path)
    for path in sorted(Path("configs/v2/evidence").glob("*.yaml")):
        EvidenceDesign.load(path)
    for path in sorted(Path("configs").glob("*.yaml")):
        migrated = migrate_v11_mapping(yaml.safe_load(path.read_text(encoding="utf-8-sig")))
        ExperimentConfig.from_mapping(migrated)


def test_machine_readable_contract_descriptor_is_golden() -> None:
    expected = json.loads(
        Path("docs/v2/contracts/config-design-schema.json").read_text(encoding="utf-8")
    )
    actual = {
        "descriptor_version": 1,
        "contracts": {
            "ExperimentConfig": experiment_config_schema_descriptor(),
            "EvidenceDesign": evidence_design_schema_descriptor(),
        },
    }
    assert actual == expected
