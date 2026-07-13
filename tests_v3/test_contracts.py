from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lunavla.v3 import (
    EmbodimentSpec,
    EpisodeRecordV3,
    ExperimentConfig,
    FeatureSchema,
    FeatureSpec,
    ModelSourceContractV1,
    NormalizationStatsV1,
    ObservationV3,
    PolicyBatchV3,
    PolicyRegistryV3,
    PolicySampleV3,
    PolicySpecV3,
    TrainStepResultV3,
    TransitionV3,
    fit_normalization_stats,
)
from lunavla.v3.fake_tasks import fake_feature_schema, make_fake_episodes
from model.policy_base import ActionChunk


def _schema() -> FeatureSchema:
    return FeatureSchema(
        (
            FeatureSpec("camera.primary", "image", "uint8", (4, 4, 3), "pixel", "camera", 10, "none", "image"),
            FeatureSpec("state.proprioception", "state", "float32", (2,), "unitless", "robot", 10, "standard", "state"),
            FeatureSpec("action.primary", "action", "float32", (2,), "unitless", "robot", 10, "standard", "action"),
        )
    )


def _observation(step: int, *, episode: str = "ep-1") -> ObservationV3:
    return ObservationV3(
        images={"camera.primary": np.zeros((4, 4, 3), dtype=np.uint8)},
        state={"state.proprioception": np.asarray([0.1, 0.2])},
        instruction="move",
        timestamp_s=step / 10,
        episode_id=episode,
        step_index=step,
        metadata={"nested": [1, 2]},
    )


def test_feature_schema_is_stable_strict_and_validates_values() -> None:
    schema = _schema()
    assert FeatureSchema.from_mapping(schema.to_dict()) == schema
    assert schema.sha256() == FeatureSchema.from_mapping(schema.to_dict()).sha256()
    schema.validate_observation(_observation(0))
    schema.validate_action(np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="unique"):
        FeatureSchema((schema.features[0], schema.features[0]))
    payload = schema.to_dict()
    payload["items"][0]["typo"] = True
    with pytest.raises(ValueError, match="unknown FeatureSpec"):
        FeatureSchema.from_mapping(payload)


def test_observation_owns_arrays_and_metadata() -> None:
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    state = np.asarray([1.0, 2.0], dtype=np.float64)
    nested = [1, 2]
    observation = ObservationV3(
        {"camera.primary": image}, {"state.proprioception": state}, "move", 0.0, "ep", 0,
        {"nested": nested},
    )
    equivalent = ObservationV3(
        {"camera.primary": image.copy()}, {"state.proprioception": state.copy()}, "move", 0.0, "ep", 0,
        {"nested": [1, 2]},
    )
    image[:] = 0
    state[:] = 9
    nested.append(3)
    assert observation == equivalent
    assert observation.state["state.proprioception"].dtype == np.float32
    with pytest.raises(ValueError, match="read-only"):
        observation.state["state.proprioception"][0] = 2
    with pytest.raises(TypeError):
        observation.metadata["x"] = 1  # type: ignore[index]


def test_embodiment_mappings_match_feature_roles() -> None:
    spec = EmbodimentSpec(
        "fake-arm", "fake_libero", 10, {"image": "camera.primary"},
        {"state": "state.proprioception"}, {"action": "action.primary"},
    )
    spec.validate_schema(_schema())
    with pytest.raises(ValueError, match="not a declared image"):
        EmbodimentSpec(
            "fake-arm", "fake_libero", 10, {"image": "state.proprioception"},
            {"state": "state.proprioception"}, {"action": "action.primary"},
        ).validate_schema(_schema())


def test_episode_requires_contiguous_steps_and_final_boundary() -> None:
    first = TransitionV3(_observation(0), np.zeros(2), 0.0, _observation(1), False, False)
    final = TransitionV3(_observation(1), np.zeros(2), 1.0, _observation(2), True, False)
    record = EpisodeRecordV3("ep-1", (first, final))
    assert record.to_dict()["steps"] == 2
    with pytest.raises(ValueError, match="final transition"):
        EpisodeRecordV3("ep-1", (first,))
    bad = TransitionV3(_observation(2), np.zeros(2), 1.0, _observation(3), True, False)
    with pytest.raises(ValueError, match="contiguous"):
        EpisodeRecordV3("ep-1", (first, bad))


@pytest.mark.parametrize(
    "value",
    [np.asarray([np.nan, 0.0]), np.asarray([np.inf, 0.0])],
)
def test_contracts_reject_non_finite_values(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        ObservationV3({}, {"state.proprioception": value}, None, 0.0, "ep", 0)


def _policy_spec() -> PolicySpecV3:
    return PolicySpecV3(
        policy_id="numpy_linear_chunk",
        backend="numpy_v2_compat",
        model_source=ModelSourceContractV1(
            "lunavla/native", "v3-alpha2-contracts", {}, "not_required", False
        ),
        required_modalities=("state",),
        camera_order=(),
        state_order=("state.proprioception",),
        history=2,
        chunk_size=2,
        horizon=2,
        execution_steps=1,
        normalization={"state.proprioception": "standard"},
        device="cpu",
        deterministic=True,
    )


def test_policy_and_model_source_contracts_are_strict_and_hash_stable() -> None:
    spec = _policy_spec()
    assert PolicySpecV3.from_mapping(spec.to_dict()) == spec
    assert PolicySpecV3.from_mapping(spec.to_dict()).sha256() == spec.sha256()
    payload = spec.to_dict()
    payload["typo"] = True
    with pytest.raises(ValueError, match="unknown field.*policy spec"):
        PolicySpecV3.from_mapping(payload)
    with pytest.raises(ValueError, match="license_status=verified"):
        ModelSourceContractV1(
            "lerobot/smolvla_base", "revision", {}, "unverified", True
        )
    with pytest.raises(ValueError, match="camera_order"):
        PolicySpecV3(
            policy_id="image_policy",
            backend="torch",
            model_source=spec.model_source,
            required_modalities=("image", "state"),
            camera_order=(),
            state_order=("state.proprioception",),
            history=1,
            chunk_size=1,
            horizon=1,
            execution_steps=1,
            normalization={},
            device="cpu",
            deterministic=True,
        )


def test_policy_sample_and_batch_own_history_actions_and_masks() -> None:
    action = np.ones((2, 2), dtype=np.float64)
    history_mask = np.asarray([False, True])
    valid_mask = np.asarray([True, False])
    sample = PolicySampleV3(
        (_observation(0), _observation(0)),
        history_mask,
        action,
        valid_mask,
        "ep-1",
        0,
    )
    action[:] = 9
    history_mask[:] = True
    valid_mask[:] = True
    assert sample.action_chunk is not None
    assert sample.action_chunk.dtype == np.float32
    assert np.all(sample.action_chunk == 1)
    assert sample.history_mask.tolist() == [False, True]
    assert sample.valid_mask is not None and sample.valid_mask.tolist() == [True, False]
    assert PolicyBatchV3((sample,), device="cpu").batch_size == 1
    inference = PolicySampleV3(
        (_observation(0),), np.asarray([True]), None, None, "ep-1", 0
    )
    with pytest.raises(ValueError, match="training batches require"):
        PolicyBatchV3((inference,))


def test_train_only_normalization_is_stable_round_trip() -> None:
    episodes = make_fake_episodes(
        task_id="fake_pusht", seed=42, episode_count=3, steps=3
    )
    schema = fake_feature_schema("fake_pusht")
    stats = fit_normalization_stats(episodes[:2], schema)
    assert stats.source_split == "train"
    assert stats.sha256() == NormalizationStatsV1.from_mapping(stats.to_dict()).sha256()
    state_stats = stats.features["state.proprioception"]
    assert state_stats.sample_count == 6
    value = episodes[0].transitions[0].observation.state["state.proprioception"]
    assert np.allclose(state_stats.denormalize(state_stats.normalize(value)), value)
    with pytest.raises(ValueError, match="train split"):
        NormalizationStatsV1(schema.sha256(), "validation", {})


def test_policy_registry_v3_dispatches_create_and_restore_strictly(tmp_path: Path) -> None:
    spec = _policy_spec()
    config = ExperimentConfig.load("configs/v3/fake_pusht_alpha.yaml")
    stats = fit_normalization_stats(
        make_fake_episodes(task_id="fake_pusht", seed=42, episode_count=3, steps=2),
        config.feature_schema,
    )

    class StubPolicy:
        def __init__(self, value: PolicySpecV3) -> None:
            self.spec = value

        def reset(self, seed: int) -> None:
            del seed

        def train_step(
            self, batch: PolicyBatchV3, *, learning_rate: float, step: int
        ) -> TrainStepResultV3:
            del batch
            return TrainStepResultV3(
                1.0, {"loss": 1.0}, None, learning_rate, step, True, {}
            )

        def predict_chunk(self, sample: PolicySampleV3) -> ActionChunk:
            del sample
            return ActionChunk(np.zeros((2, 2)), np.ones(2, dtype=bool))

        def save_checkpoint(self, path: Path, *, metadata: object) -> Path:
            del metadata
            return path

    registry = PolicyRegistryV3()
    registry.register(
        spec.policy_id,
        lambda config, registered, normalization: StubPolicy(registered),
        lambda path, config, registered, normalization: StubPolicy(registered),
    )
    assert registry.create(config, spec, stats).spec == spec
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    assert registry.restore(checkpoint, config, spec, stats).spec == spec
    with pytest.raises(KeyError, match="unknown v3 policy"):
        PolicyRegistryV3().create(config, spec, stats)
