from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lunavla.config import ExperimentConfig
from lunavla.lerobot_adapter import (
    LEROBOT_PUSHT_REPO_ID,
    LEROBOT_PUSHT_REVISION,
    LeRobotDatasetSource,
)
from lunavla import pusht_env_adapter
from lunavla.pusht_env_adapter import PushTEnvAdapter, load_pusht_env_factory
from lunavla.run import _dataset_source, _task_env


def _lerobot_mapping() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "project_name": "v2-lerobot-pusht-test",
        "engine": "lunavla_v2",
        "policy": {
            "type": "transformer_chunk",
            "state_dim": 2,
            "instruction_dim": 8,
            "image_shape": [96, 96, 3],
            "action_dim": 2,
            "chunk_size": 4,
            "d_model": 16,
            "nhead": 4,
            "device": "cpu",
        },
        "task": {
            "id": "lerobot_pusht",
            "family": "pusht",
            "max_steps": 8,
        },
        "dataset": {
            "type": "lerobot",
            "split": "train",
            "seed": 42,
            "repo_id": LEROBOT_PUSHT_REPO_ID,
            "revision": LEROBOT_PUSHT_REVISION,
            "episodes": [0],
            "video_backend": "pyav",
            "return_uint8": True,
        },
        "training": {
            "device": "cpu",
            "seed": 11,
            "batch_size": 2,
            "steps": 1,
            "learning_rate": 3e-4,
            "kl_weight": 0.01,
        },
        "evaluation": {
            "execution_mode": "receding_horizon",
            "episodes": 1,
            "seed": 1000,
        },
        "artifacts": {
            "output_dir": "outputs/v2-lerobot-test",
            "checkpoint_name": "checkpoint.pt",
        },
    }


def test_lerobot_dataset_contract_is_explicit_and_wired_without_network() -> None:
    config = ExperimentConfig.from_mapping(_lerobot_mapping())
    assert config.dataset["repo_id"] == LEROBOT_PUSHT_REPO_ID
    assert config.dataset["revision"] == LEROBOT_PUSHT_REVISION
    assert config.dataset["episodes"] == [0]
    assert config.dataset["video_backend"] == "pyav"
    assert config.dataset["return_uint8"] is True

    source = _dataset_source(config, Path.cwd())
    assert isinstance(source, LeRobotDatasetSource)
    assert source.repo_id == LEROBOT_PUSHT_REPO_ID
    assert source.revision == LEROBOT_PUSHT_REVISION
    assert source.episodes == (0,)
    assert source.video_backend == "pyav"
    assert source.return_uint8 is True

    env = _task_env(config)
    assert isinstance(env, PushTEnvAdapter)
    assert env._env is None


def test_pusht_factory_lazily_imports_environment_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def factory(*_args: object, **_kwargs: object) -> object:
        return object()

    def import_module(name: str) -> object:
        imports.append(name)
        if name == "gymnasium":
            return SimpleNamespace(make=factory)
        if name == "gym_pusht":
            return object()
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(pusht_env_adapter.importlib, "import_module", import_module)

    assert load_pusht_env_factory() is factory
    assert imports == ["gymnasium", "gym_pusht"]


@pytest.mark.parametrize(
    ("episodes", "error", "message"),
    [
        ([True], TypeError, "must be an integer"),
        ([-1], ValueError, "must be non-negative"),
        ([0, 0], ValueError, "duplicate"),
        ([], ValueError, "cannot be empty"),
        ("0", TypeError, "sequence of integers"),
    ],
)
def test_lerobot_episode_ids_are_strict(
    episodes: object,
    error: type[Exception],
    message: str,
) -> None:
    payload = _lerobot_mapping()
    payload["dataset"]["episodes"] = episodes
    with pytest.raises(error, match=message):
        ExperimentConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("repo_id", True, ValueError, "requires dataset.repo_id"),
        ("revision", True, ValueError, "requires dataset.revision"),
        ("video_backend", "torchcodec", ValueError, "must be pyav"),
        ("video_backend", True, TypeError, "must be a string"),
        ("return_uint8", False, ValueError, "must be true"),
        ("return_uint8", 1, TypeError, "must be boolean"),
    ],
)
def test_lerobot_provider_fields_fail_closed(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    payload = _lerobot_mapping()
    payload["dataset"][field] = value
    with pytest.raises(error, match=message):
        ExperimentConfig.from_mapping(payload)


@pytest.mark.parametrize(
    "field",
    ["repo_id", "revision", "episodes", "video_backend", "return_uint8"],
)
def test_lerobot_explicit_provider_fields_are_required(field: str) -> None:
    payload = _lerobot_mapping()
    payload["dataset"].pop(field)
    with pytest.raises((TypeError, ValueError), match=f"requires dataset.{field}"):
        ExperimentConfig.from_mapping(payload)


def test_lerobot_unknown_parameters_and_non_lerobot_provider_fields_fail() -> None:
    payload = _lerobot_mapping()
    payload["dataset"]["parameters"] = {"max_samples": 10}
    with pytest.raises(ValueError, match="unknown field.*max_samples"):
        ExperimentConfig.from_mapping(payload)

    payload = _lerobot_mapping()
    payload["dataset"]["type"] = "memory"
    with pytest.raises(ValueError, match="LeRobot dataset field.*repo_id"):
        ExperimentConfig.from_mapping(payload)


def test_legacy_lerobot_path_warns_and_resolves_to_repo_id() -> None:
    payload = _lerobot_mapping()
    payload["dataset"]["path"] = payload["dataset"].pop("repo_id")
    with pytest.warns(DeprecationWarning, match="dataset.path"):
        config = ExperimentConfig.from_mapping(payload)
    assert config.dataset["repo_id"] == LEROBOT_PUSHT_REPO_ID
    assert "path" not in config.dataset


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("policy", "state_dim", 4, "policy.state_dim=2"),
        ("policy", "image_shape", [64, 64, 3], "image_shape=\\[96, 96, 3\\]"),
        ("task", "family", "point_reach", "task.family=pusht"),
        ("dataset", "repo_id", "someone/pusht", "dataset.repo_id=lerobot/pusht"),
        ("dataset", "revision", "main", "pinned official dataset revision"),
    ],
)
def test_lerobot_pusht_cross_section_shape_contract(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _lerobot_mapping()
    payload[section][field] = value
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.from_mapping(payload)
