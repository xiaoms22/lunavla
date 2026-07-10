from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset import (
    VLARecord,
    build_training_arrays,
    generate_mock_pusht_records,
    load_jsonl,
    save_jsonl,
    split_records_by_episode,
    validate_records,
)


def make_record(
    episode_id: int,
    timestep: int,
    *,
    observation: list[float] | None = None,
    action: list[float] | None = None,
) -> VLARecord:
    obs = observation or [float(timestep), 0.0, 1.0, 1.0]
    act = action or [float(timestep + 1), -float(timestep + 1)]
    return VLARecord(
        observation=obs,
        next_observation=list(obs),
        action=act,
        episode_id=episode_id,
        timestep=timestep,
        success=False,
        terminated=timestep == 2,
        language_instruction="move to the goal",
        metadata={},
    )


def test_generated_transition_context_is_aligned_with_observation() -> None:
    records = generate_mock_pusht_records(
        num_episodes=3,
        steps_per_episode=5,
        seed=11,
        action_noise_std=0.0,
        success_distance=0.001,
    )

    for record in records:
        position = np.asarray(record.observation[:2])
        next_position = np.asarray(record.next_observation[:2])
        goal = np.asarray(record.observation[2:])
        np.testing.assert_allclose(next_position, position + record.action, atol=1e-7)
        assert record.metadata["distance_to_goal"] == pytest.approx(
            np.linalg.norm(goal - position)
        )
        assert record.metadata["next_distance_to_goal"] == pytest.approx(
            np.linalg.norm(goal - next_position)
        )
        context = record.metadata["task_context"]
        assert context["metadata"]["distance_to_goal"] == pytest.approx(
            record.metadata["distance_to_goal"]
        )
        assert record.task_id == "pusht_style_point_reach"
        assert record.phase == context["phase"]


def test_build_training_arrays_uses_real_padding_mask() -> None:
    records = [make_record(0, timestep) for timestep in range(3)]
    arrays = build_training_arrays(records, chunk_size=3, instruction_dim=2)

    assert arrays.inputs.shape == (3, 6)
    assert arrays.targets.shape == (3, 3, 2)
    assert arrays.valid_mask.tolist() == [
        [True, True, True],
        [True, True, False],
        [True, False, False],
    ]
    np.testing.assert_array_equal(arrays.targets[2, 1:], np.zeros((2, 2)))
    np.testing.assert_array_equal(arrays.targets[0, 1], records[1].action)


def test_episode_split_is_deterministic_and_has_no_leakage() -> None:
    records = [make_record(episode, timestep) for episode in range(10) for timestep in range(3)]
    first = split_records_by_episode(records, seed=123)
    second = split_records_by_episode(records, seed=123)
    ids = {
        name: {record.episode_id for record in split_records}
        for name, split_records in first.items()
    }

    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])
    assert set.union(*ids.values()) == set(range(10))
    assert {
        name: [record.episode_id for record in split_records]
        for name, split_records in first.items()
    } == {
        name: [record.episode_id for record in split_records]
        for name, split_records in second.items()
    }


def test_jsonl_round_trip_is_deterministic(tmp_path: Path) -> None:
    records = generate_mock_pusht_records(3, 4, seed=7)
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    save_jsonl(records, first_path)
    loaded = load_jsonl(first_path)
    save_jsonl(loaded, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert [record.to_dict() for record in loaded] == [record.to_dict() for record in records]


@pytest.mark.parametrize(
    ("records", "error_type", "message"),
    [
        ([], ValueError, "no records"),
        ([make_record(0, 0), make_record(0, 0)], ValueError, "duplicate"),
        ([make_record(0, 0), make_record(0, 2)], ValueError, "contiguous"),
        (
            [make_record(0, 0), make_record(0, 1, observation=[0.0, 1.0])],
            ValueError,
            "observation",
        ),
        (
            [make_record(0, 0, action=[float("nan"), 0.0])],
            ValueError,
            "NaN",
        ),
        (
            [make_record(0, 0, action=["bad", "action"])],  # type: ignore[list-item]
            TypeError,
            "numeric",
        ),
    ],
)
def test_record_validation_rejects_invalid_data(
    records: list[VLARecord],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        validate_records(records)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{not-json}\n", "invalid JSON"),
        ("[]\n", "not a JSON object"),
        (json.dumps({"episode_id": 0}) + "\n", "invalid record"),
    ],
)
def test_jsonl_loader_reports_invalid_line(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_jsonl(path)


def test_split_rejects_invalid_fractions() -> None:
    records = [make_record(episode, timestep) for episode in range(3) for timestep in range(3)]
    with pytest.raises(ValueError, match="sum to 1"):
        split_records_by_episode(
            records,
            train_fraction=0.5,
            validation_fraction=0.5,
            test_fraction=0.5,
        )
