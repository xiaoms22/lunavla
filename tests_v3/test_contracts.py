from __future__ import annotations

import numpy as np
import pytest

from lunavla.v3 import (
    EmbodimentSpec,
    EpisodeRecordV3,
    FeatureSchema,
    FeatureSpec,
    ObservationV3,
    TransitionV3,
)


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
