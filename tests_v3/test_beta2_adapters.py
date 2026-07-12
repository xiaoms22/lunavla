from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lunavla.v3 import (
    ExperimentConfig,
    LeRobotDatasetSourceV3,
    LiberoSpatialEnvV3,
    PushTEnvV3,
    select_minimum_episode_per_task,
)


def _pusht_rows() -> list[dict[str, Any]]:
    return [
        {
            "episode_index": 0,
            "frame_index": step,
            "observation.image": np.full((96, 96, 3), step, dtype=np.uint8),
            "observation.state": np.asarray([step, step + 1], dtype=np.float32),
            "action": np.asarray([0.1, -0.1], dtype=np.float32),
            "private": {"must_not_leak": True},
        }
        for step in range(3)
    ]


def test_lerobot_source_maps_frames_without_raw_metadata_leakage() -> None:
    config = ExperimentConfig.load("configs/v3/beta2_pusht_integration.yaml")
    received: dict[str, Any] = {}

    def factory(**kwargs: Any) -> list[dict[str, Any]]:
        received.update(kwargs)
        return _pusht_rows()

    source = LeRobotDatasetSourceV3(
        config.external_dataset_spec,  # type: ignore[arg-type]
        config.feature_schema,
        dataset_factory=factory,
    )
    episodes = source.load()
    assert received == {
        "repo_id": "lerobot/pusht",
        "root": None,
        "revision": "b1c3ecbae7f244acc039a3dbc255a00dad1372b9",
        "episodes": [0],
        "video_backend": "pyav",
        "download_videos": True,
        "return_uint8": True,
    }
    assert len(episodes) == 1 and len(episodes[0].transitions) == 3
    assert episodes[0].transitions[-1].terminated
    assert episodes[0].transitions[-1].next_observation.step_index == 3
    assert episodes[0].transitions[0].observation.metadata == {}
    assert tuple(episodes[0].transitions[0].observation.images) == ("camera.primary",)


def test_lerobot_source_rejects_shape_and_terminal_sequence_drift() -> None:
    config = ExperimentConfig.load("configs/v3/beta2_pusht_integration.yaml")
    rows = _pusht_rows()
    rows[1]["frame_index"] = 2
    source = LeRobotDatasetSourceV3(
        config.external_dataset_spec,  # type: ignore[arg-type]
        config.feature_schema,
        dataset_factory=lambda **_: rows,
    )
    with pytest.raises(ValueError, match="frame_index must be contiguous"):
        source.load()
    rows = _pusht_rows()
    rows[0]["observation.image"] = np.zeros((95, 96, 3), dtype=np.uint8)
    source = LeRobotDatasetSourceV3(
        config.external_dataset_spec,  # type: ignore[arg-type]
        config.feature_schema,
        dataset_factory=lambda **_: rows,
    )
    with pytest.raises(ValueError, match="shape"):
        source.load()


def test_libero_selects_minimum_distinct_episode_and_validates_language() -> None:
    metadata = [
        {"task_index": task, "episode_index": 10 + task, "task": f"task language {task}"}
        for task in range(4)
    ] + [{"task_index": 0, "episode_index": 99, "task": "task language 0"}]
    expected = {task: f"task language {task}" for task in range(4)}
    selection = select_minimum_episode_per_task(
        metadata, task_ids=(0, 1, 2, 3), expected_languages=expected
    )
    assert selection.episodes == (10, 11, 12, 13)
    bad = list(metadata)
    bad[0] = dict(bad[0], task="changed")
    with pytest.raises(ValueError, match="task-language mapping drift"):
        select_minimum_episode_per_task(bad, task_ids=(0, 1, 2, 3), expected_languages=expected)
    duplicate = list(metadata)
    duplicate[1] = dict(duplicate[1], episode_index=10)
    with pytest.raises(ValueError, match="duplicate episode_index"):
        select_minimum_episode_per_task(duplicate, task_ids=(0, 1, 2, 3))


class _FakeEnv:
    def __init__(self, observation: Mapping[str, Any]) -> None:
        self.observation = observation
        self.closed = 0
        self.steps = 0

    def reset(self, **_: Any) -> tuple[Mapping[str, Any], dict[str, Any]]:
        return self.observation, {"private": True}

    def step(self, _action: np.ndarray) -> tuple[Mapping[str, Any], float, bool, bool, dict[str, Any]]:
        self.steps += 1
        return self.observation, 0.0, self.steps == 3, False, {"private": True}

    def close(self) -> None:
        self.closed += 1


def test_pusht_env_maps_steps_and_closes_exactly_once() -> None:
    config = ExperimentConfig.load("configs/v3/beta2_pusht_integration.yaml")
    upstream = _FakeEnv(
        {
            "pixels": np.zeros((96, 96, 3), dtype=np.uint8),
            "agent_pos": np.zeros(2, dtype=np.float32),
        }
    )
    env = PushTEnvV3(
        config.feature_schema,
        config.simulation_task_spec,  # type: ignore[arg-type]
        env_factory=lambda *_args, **_kwargs: upstream,
    )
    try:
        observation = env.reset(seed=1000)
        assert observation.metadata == {}
        for _ in range(3):
            transition = env.step(np.zeros(2, dtype=np.float32))
        assert transition.terminated
    finally:
        env.close()
        env.close()
    assert upstream.closed == 1


def test_pusht_env_registers_namespace_before_default_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig.load("configs/v3/beta2_pusht_integration.yaml")
    upstream = _FakeEnv(
        {
            "pixels": np.zeros((96, 96, 3), dtype=np.uint8),
            "agent_pos": np.zeros(2, dtype=np.float32),
        }
    )
    imports: list[str] = []

    def import_module(name: str) -> Any:
        imports.append(name)
        if name == "gym_pusht":
            return SimpleNamespace()
        if name == "gymnasium":
            return SimpleNamespace(make=lambda *_args, **_kwargs: upstream)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("lunavla.v3.real_adapters.importlib.import_module", import_module)
    env = PushTEnvV3(
        config.feature_schema,
        config.simulation_task_spec,  # type: ignore[arg-type]
    )
    env.close()
    assert imports == ["gym_pusht", "gymnasium"]
    assert upstream.closed == 1


def test_libero_env_dual_camera_mapping_action_range_and_failure_close() -> None:
    config = ExperimentConfig.load("configs/v3/beta2_libero_integration.yaml")
    raw = {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((256, 256, 3), dtype=np.uint8),
        "observation.state": np.zeros(8, dtype=np.float32),
    }
    upstream = _FakeEnv(raw)
    env = LiberoSpatialEnvV3(
        config.feature_schema,
        config.simulation_task_spec,  # type: ignore[arg-type]
        task_id=0,
        init_state_id=0,
        task_language="pick the black bowl",
        env_factory=lambda **_: upstream,
        observation_processor=lambda value: {
            "observation.images.image": np.moveaxis(
                value["agentview_image"][None, ...], -1, 1
            ),
            "observation.images.image2": np.moveaxis(
                value["robot0_eye_in_hand_image"][None, ...], -1, 1
            ),
            "observation.state": value["observation.state"][None, ...],
        },
    )
    try:
        observation = env.reset(seed=1000)
        assert tuple(observation.images) == ("camera.agentview", "camera.wrist")
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            env.step(np.full(7, 2.0, dtype=np.float32))
    finally:
        env.close()
        env.close()
    assert upstream.closed == 1
