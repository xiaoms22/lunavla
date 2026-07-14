from __future__ import annotations

import numpy as np
import pytest

from lunavla.v3 import (
    V31_HELD_OUT_STRATA,
    V31_TASK_IDS,
    V31RolloutEnvV1,
    make_v31_episode,
    make_v31_task_dataset,
    task_dataset_sha256,
    task_suite_spec_v1,
    v31_feature_schema,
)
from lunavla.v3.data import episode_sha256


def test_real_rollout_env_is_dynamic_deterministic_and_hides_oracle() -> None:
    env = V31RolloutEnvV1(
        task_id="direct_pick_place", stratum="composition", episode_index=3
    )
    first = env.reset(seed=1003)
    assert "target_xy" not in first.metadata
    transition = env.step(np.asarray([0.14, 0.14, 0.0], dtype=np.float32))
    assert transition.next_observation.step_index == 1
    assert not np.array_equal(
        first.state["state.proprioception"],
        transition.next_observation.state["state.proprioception"],
    )
    assert np.isfinite(transition.info["final_distance"])
    env.close()
    assert env.closed
    with pytest.raises(RuntimeError, match="closed"):
        env.reset(seed=1003)


@pytest.mark.parametrize("task_id", V31_TASK_IDS)
@pytest.mark.parametrize("stratum", V31_HELD_OUT_STRATA)
def test_real_rollout_env_replays_registered_expert_to_success(
    task_id: str, stratum: str
) -> None:
    episode = make_v31_episode(
        task_id=task_id, split="test", stratum=stratum, data_seed=42, index=2
    )
    env = V31RolloutEnvV1(task_id=task_id, stratum=stratum, episode_index=2)
    env.reset(seed=1002)
    try:
        result = None
        for expert in episode.transitions:
            result = env.step(expert.action)
            if result.terminated or result.truncated:
                break
        assert result is not None
        assert result.terminated
        assert result.info["success"] is True
        assert result.info["final_distance"] <= 0.045
    finally:
        env.close()


HELD_OUT_COMBINATIONS = {"red:diamond", "blue:square", "green:circle"}
FORBIDDEN = {"goal", "goal_coordinates", "waypoints", "oracle_action", "answer_key", "target_xy"}


def test_registered_suite_contract_matches_fixed_teaching_boundary() -> None:
    spec = task_suite_spec_v1()
    assert spec.task_ids == V31_TASK_IDS
    assert spec.held_out_strata == V31_HELD_OUT_STRATA
    assert spec.image_shape == (96, 96, 3)
    assert spec.control_rate_hz == 10
    assert spec.max_steps == 64
    schema = v31_feature_schema()
    assert [item.shape for item in schema.by_role("image")] == [(96, 96, 3)]
    assert [item.shape for item in schema.by_role("state")] == [(4,)]
    assert [item.shape for item in schema.by_role("action")] == [(3,)]


@pytest.mark.parametrize("task_id", V31_TASK_IDS)
@pytest.mark.parametrize("stratum", V31_HELD_OUT_STRATA)
def test_episode_generation_is_deterministic_contiguous_and_finite(task_id: str, stratum: str) -> None:
    first = make_v31_episode(
        task_id=task_id, split="test", stratum=stratum, data_seed=42, index=0
    )
    repeat = make_v31_episode(
        task_id=task_id, split="test", stratum=stratum, data_seed=42, index=0
    )
    different = make_v31_episode(
        task_id=task_id, split="test", stratum=stratum, data_seed=42, index=1
    )
    assert episode_sha256(first) == episode_sha256(repeat)
    assert episode_sha256(first) != episode_sha256(different)
    assert len(first.transitions) <= 64
    assert first.transitions[-1].terminated or first.transitions[-1].truncated
    for index, transition in enumerate(first.transitions):
        assert transition.observation.step_index == index
        assert transition.next_observation.step_index == index + 1
        assert transition.observation.images["camera.primary"].shape == (96, 96, 3)
        assert transition.observation.images["camera.primary"].dtype == np.uint8
        assert transition.observation.state["state.proprioception"].shape == (4,)
        assert np.all(np.isfinite(transition.action))
        assert np.all(transition.action >= -1) and np.all(transition.action <= 1)


def test_composition_and_paraphrase_holdouts_are_disjoint_from_training() -> None:
    dataset = make_v31_task_dataset(data_seed=42, train_per_task=8, held_out_per_cell=3)
    train = dataset.bundle.select("train")
    validation = dataset.bundle.select("validation")
    test = dataset.bundle.select("test")
    assert len(train) == 24
    assert len(validation) == len(test) == 18
    assert not (set(dataset.bundle.split["train"]) & set(dataset.bundle.split["validation"]))
    assert not (set(dataset.bundle.split["train"]) & set(dataset.bundle.split["test"]))
    assert not (set(dataset.bundle.split["validation"]) & set(dataset.bundle.split["test"]))
    assert all(str(item.metadata["combination"]) not in HELD_OUT_COMBINATIONS for item in train)
    composition = [
        item for item in (*validation, *test) if item.metadata["held_out_stratum"] == "composition"
    ]
    paraphrase = [
        item for item in (*validation, *test) if item.metadata["held_out_stratum"] == "paraphrase"
    ]
    assert composition and paraphrase
    assert all(str(item.metadata["combination"]) in HELD_OUT_COMBINATIONS for item in composition)
    train_instructions = {
        transition.observation.instruction for item in train for transition in item.transitions
    }
    paraphrase_instructions = {
        transition.observation.instruction for item in paraphrase for transition in item.transitions
    }
    assert train_instructions.isdisjoint(paraphrase_instructions)


def test_policy_observations_do_not_expose_oracle_geometry() -> None:
    dataset = make_v31_task_dataset(data_seed=42, train_per_task=2, held_out_per_cell=1)
    for episode in dataset.bundle.episodes:
        for transition in episode.transitions:
            observation = transition.observation
            assert not (set(observation.metadata) & FORBIDDEN)
            assert set(observation.state) == {"state.proprioception"}
            assert observation.state["state.proprioception"].shape == (4,)
            assert "[" not in (observation.instruction or "")
            assert "0." not in (observation.instruction or "")


def test_dataset_audit_and_hash_are_reproducible_and_seed_bound() -> None:
    first = make_v31_task_dataset(data_seed=42, train_per_task=2, held_out_per_cell=1)
    repeat = make_v31_task_dataset(data_seed=42, train_per_task=2, held_out_per_cell=1)
    other = make_v31_task_dataset(data_seed=43, train_per_task=2, held_out_per_cell=1)
    assert first.audit.episode_count == 18
    assert first.audit.sha256() == repeat.audit.sha256()
    assert task_dataset_sha256(first) == task_dataset_sha256(repeat)
    assert task_dataset_sha256(first) != task_dataset_sha256(other)
    assert first.audit.feature_schema_sha256 == v31_feature_schema().sha256()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task_id": "unknown", "split": "train", "stratum": "train"}, "task_id"),
        ({"task_id": "direct_pick_place", "split": "train", "stratum": "composition"}, "stratum=train"),
        ({"task_id": "direct_pick_place", "split": "test", "stratum": "train"}, "held-out"),
    ],
)
def test_invalid_task_and_stratum_combinations_fail_early(
    kwargs: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_v31_episode(data_seed=42, index=0, **kwargs)
