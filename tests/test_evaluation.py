from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eval_vla import (
    classify_failure,
    count_frame_subtasks,
    final_task_context,
    make_input,
    mean_action_norm,
    rollout_episode,
    main as evaluate_main,
)
from model import ActionChunk, NumpyLinearChunkPolicy


class FixedChunkPolicy:
    def __init__(self, second_action: float = 0.04) -> None:
        self.second_action = second_action

    def predict_chunk(self, sample: np.ndarray) -> ActionChunk:
        assert sample.shape == (12,)
        return ActionChunk(
            np.array([[0.01, 0.0], [self.second_action, 0.0]], dtype=np.float32),
            np.array([True, True]),
        )


def rollout(policy: Any, execution_mode: str) -> dict[str, Any]:
    return rollout_episode(
        policy,
        seed=17,
        rollout_steps=4,
        success_distance=1e-4,
        instruction="move right",
        execution_mode=execution_mode,
        goal=(0.9, 0.9),
        start_low=0.25,
        start_high=0.26,
        action_clip=0.1,
    )


def test_second_chunk_action_changes_only_open_loop_trajectory() -> None:
    small = FixedChunkPolicy(second_action=0.02)
    large = FixedChunkPolicy(second_action=0.08)

    receding_small = rollout(small, "receding_horizon")
    receding_large = rollout(large, "receding_horizon")
    open_small = rollout(small, "open_loop_chunk")
    open_large = rollout(large, "open_loop_chunk")

    assert [frame["position"] for frame in receding_small["frames"]] == [
        frame["position"] for frame in receding_large["frames"]
    ]
    assert open_small["frames"][0]["position"] == open_large["frames"][0]["position"]
    assert open_small["frames"][1]["position"] != open_large["frames"][1]["position"]
    assert [frame["chunk_index"] for frame in open_small["frames"]] == [0, 1, 0, 1]
    assert {frame["chunk_index"] for frame in receding_small["frames"]} == {0}


def test_rollout_uses_configured_goal_clip_and_instruction_dimension() -> None:
    policy = FixedChunkPolicy(second_action=5.0)
    result = rollout_episode(
        policy,
        seed=3,
        rollout_steps=2,
        success_distance=1e-4,
        instruction="right",
        execution_mode="open_loop_chunk",
        goal=(0.75, 0.65),
        start_low=0.2,
        start_high=0.21,
        action_clip=0.03,
    )
    assert result["goal"] == pytest.approx([0.75, 0.65])
    assert result["action_clip"] == 0.03
    assert result["frames"][1]["action"] == pytest.approx([0.03, 0.0])
    assert result["execution_mode"] == "open_loop_chunk"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"execution_mode": "invalid"}, "execution_mode"),
        ({"rollout_steps": 0}, "must be positive"),
        ({"success_distance": 0.0}, "must be positive"),
        ({"action_clip": -1.0}, "must be positive"),
        ({"start_low": 0.8, "start_high": 0.2}, "start range"),
        ({"goal": (0.5,)}, "goal"),
        ({"goal": (np.nan, 0.5)}, "goal"),
    ],
)
def test_rollout_rejects_invalid_configuration(
    kwargs: dict[str, Any], message: str
) -> None:
    parameters: dict[str, Any] = {
        "seed": 1,
        "rollout_steps": 2,
        "success_distance": 0.1,
        "instruction": None,
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError, match=message):
        rollout_episode(FixedChunkPolicy(), **parameters)


def test_rollout_rejects_invalid_policy_chunk() -> None:
    class WrongTypePolicy:
        def predict_chunk(self, sample: np.ndarray) -> np.ndarray:
            return np.zeros((1, 2))

    class WrongActionDimPolicy:
        def predict_chunk(self, sample: np.ndarray) -> ActionChunk:
            return ActionChunk(np.zeros((1, 3)), np.ones(1))

    with pytest.raises(TypeError, match="ActionChunk"):
        rollout(WrongTypePolicy(), "receding_horizon")
    with pytest.raises(ValueError, match="action dimension"):
        rollout(WrongActionDimPolicy(), "receding_horizon")


def test_legacy_single_action_policy_remains_diagnostic_compatible() -> None:
    class LegacyPolicy:
        def predict_action(self, sample: np.ndarray) -> np.ndarray:
            return np.array([0.01, -0.01], dtype=np.float32)

    result = rollout(LegacyPolicy(), "receding_horizon")
    assert result["steps"] == 4
    assert all(frame["chunk_index"] == 0 for frame in result["frames"])


def test_evaluation_summary_helpers_and_failure_categories() -> None:
    assert mean_action_norm([]) == 0.0
    assert final_task_context({}) == {}
    assert count_frame_subtasks([{"frames": []}]) == {}
    assert classify_failure({"frames": []}, 0.1)["category"] == "did_not_reach_goal"

    wrong_direction = {
        "initial_distance": 0.3,
        "final_distance": 0.5,
        "action_smoothness": 0.0,
        "frames": [
            {"distance_to_goal": 0.4, "action": [0.02, 0.0]},
            {"distance_to_goal": 0.5, "action": [0.02, 0.0]},
        ],
    }
    assert classify_failure(wrong_direction, 0.1)["category"] == "wrong_direction"


def test_make_input_respects_instruction_dimension() -> None:
    value = make_input(
        np.array([0.1, 0.2]),
        np.array([0.8, 0.9]),
        "move to target",
        instruction_dim=3,
    )
    assert value.shape == (7,)
    np.testing.assert_allclose(value[:4], [0.1, 0.2, 0.8, 0.9])


def test_evaluation_cli_writes_clean_summary_and_rollout_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = NumpyLinearChunkPolicy(input_dim=12, action_dim=2, chunk_size=1, seed=8)
    policy.weights.fill(0.0)
    policy.bias.fill(0.0)
    checkpoint = policy.save_pretrained(
        tmp_path / "run",
        {
            "eval": {
                "episodes": 1,
                "seed": 31,
                "rollout_steps": 2,
                "success_distance": 0.001,
                "execution_mode": "receding_horizon",
                "goal": [0.8, 0.2],
            },
            "dataset": {"language_instruction": "move to the goal"},
            "policy": {"instruction_dim": 8},
            "action_stats": {"source": "unit-test", "mean": [0.0, 0.0]},
        },
    )
    output_dir = tmp_path / "evaluation"
    stale_rollouts = output_dir / "rollouts"
    stale_rollouts.mkdir(parents=True)
    (stale_rollouts / "old.json").write_text("{}", encoding="utf-8")
    (output_dir / "failure_cases.jsonl").write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_vla.py",
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            "1",
            "--save-rollouts",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert evaluate_main() == 0
    summary = json.loads((output_dir / "eval_summary.json").read_text(encoding="utf-8"))
    assert summary["episodes"] == 1
    assert summary["checkpoint"] == "checkpoint.json"
    assert summary["eval_seeds"] == [31]
    assert summary["execution_mode"] == "receding_horizon"
    assert not (stale_rollouts / "old.json").exists()
    assert (stale_rollouts / "episode_000.json").exists()
    assert "stale" not in (output_dir / "failure_cases.jsonl").read_text(encoding="utf-8")
