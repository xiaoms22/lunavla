from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from lunavla.contracts import Observation, PolicyBatch, Transition, VLAPolicy
from lunavla.engine import Engine, EngineConfig
from lunavla.memory_data import (
    InMemoryDatasetSource,
    PointReachTaskEnv,
    make_point_reach_demonstrations,
)
from lunavla.numpy_policy import register_numpy_policies
from lunavla.registry import PolicyRegistry
from model.policy_base import ActionChunk


def registry() -> PolicyRegistry:
    value = PolicyRegistry()
    register_numpy_policies(value)
    return value


def test_contracts_and_registry_reject_invalid_values() -> None:
    from lunavla.contracts import normalize_device

    assert normalize_device("cuda") == "cuda:0"
    assert normalize_device("mps") == "mps:0"
    with pytest.raises(ValueError, match="rank-1"):
        Observation(np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="uint8 or floating"):
        Observation(np.zeros(2), image=np.zeros((4, 4, 3), dtype=np.int16))
    with pytest.raises(TypeError, match="boolean"):
        PolicyBatch(
            observations=(Observation(np.zeros(2)),),
            targets=np.zeros((1, 1, 1), dtype=np.float32),
            valid_mask=np.ones((1, 1), dtype=np.int64),
        )
    observation = Observation(np.zeros(1, dtype=np.float32))
    with pytest.raises(TypeError, match="unsupported value type"):
        Transition(observation, np.zeros(1), 0.0, observation, True, {"bad": object()})
    transition = Transition(
        observation,
        np.zeros(1),
        0.0,
        observation,
        True,
        {"nested": {"value": 1}},
    )
    with pytest.raises(TypeError):
        transition.info["new"] = 2  # type: ignore[index]

    policies = registry()
    assert policies.available() == ("numpy_bc_mlp", "numpy_linear_chunk")
    assert policies.resolve("linear_chunk") == "numpy_linear_chunk"
    with pytest.raises(KeyError, match="unknown policy"):
        policies.create("missing", {})
    with pytest.raises(KeyError, match="already registered"):
        register_numpy_policies(policies)


def test_registry_replace_does_not_leave_aliases_to_removed_entries() -> None:
    policies = PolicyRegistry()
    policies.register("first", lambda config: _ChunkPolicy(), aliases=("old_alias",))
    policies.register(
        "replacement",
        lambda config: _ChunkPolicy(),
        aliases=("first",),
        replace=True,
    )
    assert policies.resolve("first") == "replacement"
    with pytest.raises(KeyError, match="unknown policy"):
        policies.resolve("old_alias")


@pytest.mark.parametrize(
    ("policy_id", "policy_config"),
    [
        (
            "numpy_linear_chunk",
            {"state_dim": 4, "instruction_dim": 4, "action_dim": 2, "chunk_size": 2},
        ),
        (
            "numpy_bc_mlp",
            {
                "state_dim": 4,
                "instruction_dim": 4,
                "action_dim": 2,
                "chunk_size": 1,
                "hidden_dim": 16,
            },
        ),
    ],
)
def test_two_numpy_policies_train_and_evaluate_in_one_engine(
    policy_id: str,
    policy_config: dict[str, Any],
) -> None:
    engine = Engine(
        EngineConfig(
            seed=17,
            batch_size=16,
            train_steps=30,
            learning_rate=0.04,
            eval_episodes=2,
            max_steps=8,
            execution_mode="receding",
        ),
        registry=registry(),
    )
    data = make_point_reach_demonstrations(episodes=8, steps_per_episode=10, seed=5)
    assert all("episode_id" in item.info and "timestep" in item.info for item in data.load())
    trained = engine.train(policy_id, data, policy_config=policy_config)

    assert isinstance(trained.policy, VLAPolicy)
    assert trained.policy.policy_id == policy_id
    assert trained.samples == len(data.load())
    assert len(trained.losses) == 30
    assert np.isfinite(trained.final_loss)

    evaluation = engine.evaluate(trained.policy, PointReachTaskEnv())
    assert len(evaluation.episodes) == 2
    assert 0.0 <= evaluation.success_rate <= 1.0
    assert all(0 < episode.steps <= 8 for episode in evaluation.episodes)


def test_training_is_deterministic_for_seed() -> None:
    config = EngineConfig(seed=91, batch_size=12, train_steps=20, learning_rate=0.03)
    data = make_point_reach_demonstrations(episodes=6, steps_per_episode=8, seed=12)
    policy_config = {
        "state_dim": 4,
        "instruction_dim": 2,
        "action_dim": 2,
        "chunk_size": 3,
    }
    first = Engine(config, registry=registry()).train(
        "numpy_linear_chunk", data, policy_config=policy_config
    )
    second = Engine(config, registry=registry()).train(
        "numpy_linear_chunk", data, policy_config=policy_config
    )
    sample = data.load()[0].observation

    assert first.losses == second.losses
    assert np.array_equal(
        first.policy.predict_chunk(sample).values,
        second.policy.predict_chunk(sample).values,
    )


def test_supervision_chunks_stop_at_terminated_transition() -> None:
    first = Observation(np.asarray([0.0], dtype=np.float32))
    second = Observation(np.asarray([1.0], dtype=np.float32))
    source = InMemoryDatasetSource(
        (
            Transition(first, np.asarray([1.0]), 0.0, first, True, {}),
            Transition(second, np.asarray([9.0]), 0.0, second, True, {}),
        )
    )
    engine = Engine(EngineConfig(train_steps=1), registry=registry())
    policy = engine.create_policy(
        "numpy_linear_chunk",
        {"state_dim": 1, "action_dim": 1, "chunk_size": 3},
    )
    batch = engine._supervision_batch(tuple(source.load()), policy)

    assert np.array_equal(batch.valid_mask, [[True, False, False], [True, False, False]])
    assert np.array_equal(batch.targets[:, 0, 0], [1.0, 9.0])
    assert np.count_nonzero(batch.targets[:, 1:]) == 0


@pytest.mark.parametrize(
    ("policy_id", "extra"),
    [
        ("numpy_linear_chunk", {}),
        ("numpy_bc_mlp", {"hidden_dim": 8}),
    ],
)
def test_numpy_adapter_checkpoint_round_trip(
    tmp_path: Path,
    policy_id: str,
    extra: dict[str, Any],
) -> None:
    engine = Engine(
        EngineConfig(seed=4, batch_size=8, train_steps=10), registry=registry()
    )
    data = make_point_reach_demonstrations(episodes=4, steps_per_episode=6, seed=3)
    policy_config = {
        "state_dim": 4,
        "instruction_dim": 3,
        "action_dim": 2,
        "chunk_size": 1 if policy_id == "numpy_bc_mlp" else 2,
        **extra,
    }
    policy = engine.train(policy_id, data, policy_config=policy_config).policy
    observation = data.load()[0].observation
    before = policy.predict_chunk(observation)

    checkpoint = engine.save_checkpoint(
        policy, tmp_path / policy_id, metadata={"purpose": "round-trip-test"}
    )
    restored = engine.load_checkpoint(checkpoint)
    after = restored.predict_chunk(observation)

    assert checkpoint.name == "checkpoint.json"
    assert restored.policy_id == policy_id
    assert np.array_equal(after.values, before.values)
    assert np.array_equal(after.valid_mask, before.valid_mask)


class _ChunkPolicy:
    policy_id = "test_chunk"
    device = "cpu"
    action_dim = 1
    chunk_size = 2

    def train_batch(self, batch: PolicyBatch, *, learning_rate: float) -> float:
        return 0.0

    def predict_chunk(self, observation: Observation) -> ActionChunk:
        return ActionChunk(
            np.asarray([[1.0], [2.0]], dtype=np.float32),
            np.asarray([True, True]),
        )

    def save_checkpoint(
        self,
        path: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return path


class _CountingEnv:
    def __init__(self) -> None:
        self.observation: Observation | None = None

    def reset(self, *, seed: int | None = None) -> Observation:
        self.observation = Observation(np.asarray([float(seed or 0)], dtype=np.float32))
        return self.observation

    def step(self, action: np.ndarray[Any, Any]) -> Transition:
        assert self.observation is not None
        next_observation = Observation(self.observation.state + action)
        transition = Transition(
            self.observation,
            action,
            0.0,
            next_observation,
            False,
            {"success": False},
        )
        self.observation = next_observation
        return transition


def test_open_loop_and_receding_execution_are_distinct() -> None:
    policy = _ChunkPolicy()
    receding = Engine(
        EngineConfig(eval_episodes=1, max_steps=3, execution_mode="receding")
    ).evaluate(policy, _CountingEnv())
    open_loop = Engine(
        EngineConfig(eval_episodes=1, max_steps=3, execution_mode="open_loop")
    ).evaluate(policy, _CountingEnv())

    assert receding.episodes[0].actions == ((1.0,), (1.0,), (1.0,))
    assert open_loop.episodes[0].actions == ((1.0,), (2.0,), (1.0,))


def test_receding_temporal_ensemble_uses_aligned_chunk_history() -> None:
    policy = _ChunkPolicy()
    result = Engine(
        EngineConfig(
            eval_episodes=1,
            max_steps=2,
            execution_mode="receding",
            temporal_ensemble_decay=np.log(2.0),
        )
    ).evaluate(policy, _CountingEnv())
    assert result.episodes[0].actions[0] == (1.0,)
    assert result.episodes[0].actions[1] == pytest.approx((4.0 / 3.0,))

    with pytest.raises(ValueError, match="requires receding"):
        EngineConfig(
            execution_mode="open_loop",
            temporal_ensemble_decay=0.1,
        )


def test_eval_seed_is_independent_and_defaults_to_training_seed() -> None:
    policy = _ChunkPolicy()
    inherited = Engine(
        EngineConfig(seed=23, eval_episodes=2, max_steps=1)
    ).evaluate(policy, _CountingEnv())
    explicit = Engine(
        EngineConfig(seed=23, eval_seed=700, eval_episodes=2, max_steps=1)
    ).evaluate(policy, _CountingEnv())
    noncontiguous = Engine(
        EngineConfig(
            seed=23,
            eval_seeds=(8, 99),
            eval_episodes=2,
            max_steps=1,
        )
    ).evaluate(policy, _CountingEnv())

    assert [item.seed for item in inherited.episodes] == [23, 24]
    assert [item.seed for item in explicit.episodes] == [700, 701]
    assert [item.seed for item in noncontiguous.episodes] == [8, 99]
    assert inherited.episodes[0].final_state == (24.0,)
    assert explicit.episodes[0].final_state == (701.0,)
    with pytest.raises(ValueError, match="exactly eval_episodes"):
        EngineConfig(eval_seeds=(1,), eval_episodes=2)


def test_numpy_device_and_image_fail_clearly() -> None:
    with pytest.raises(ValueError, match="only device='cpu'"):
        registry().create(
            "numpy_linear_chunk",
            {"state_dim": 2, "device": "cuda", "action_dim": 1},
        )
    policy = registry().create(
        "numpy_linear_chunk", {"state_dim": 2, "action_dim": 1}
    )
    observation = Observation(
        np.zeros(2, dtype=np.float32), image=np.zeros((4, 4, 3), dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="state-only"):
        policy.predict_chunk(observation)
