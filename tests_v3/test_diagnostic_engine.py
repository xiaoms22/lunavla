from __future__ import annotations

import numpy as np
import pytest

from lunavla.v3 import (
    DiagnosticRouterV1,
    ExperimentConfig,
    FeatureNormalizationV1,
    NormalizationStatsV1,
    StateRouteSpecV1,
    typed_episode_key,
)
from lunavla.v3.engine import EngineV3, dataset_for_config
from lunavla.v3.fake_tasks import FakePointEnvV3


def _stats(config: ExperimentConfig) -> NormalizationStatsV1:
    feature = FeatureNormalizationV1(
        "state.proprioception",
        "standard",
        4,
        np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32),
        np.ones(4, dtype=np.float32),
    )
    return NormalizationStatsV1(
        config.feature_schema.sha256(), "train", {feature.feature_name: feature}
    )


def test_route_views_mask_after_normalization_and_strip_hidden_inputs() -> None:
    config = ExperimentConfig.load("configs/v3/diagnostic_fake_libero.yaml")
    canonical = dataset_for_config(config).episodes[0].transitions[0].observation
    expert = DiagnosticRouterV1(config, _stats(config)).route_observation(canonical)
    assert expert.expert_state_keys == ("state.proprioception",)
    assert expert.prompt_state_keys == ()
    assert expert.observation.images == {}
    assert set(expert.observation.metadata) == {"diagnostic"}
    assert "task_id" not in expert.observation.metadata

    route = StateRouteSpecV1("prompt_only", ("state.proprioception",))
    prompt_only = DiagnosticRouterV1(
        config, _stats(config), route=route
    ).route_observation(canonical)
    masked = prompt_only.observation.state["state.proprioception"]
    normalized = _stats(config).features["state.proprioception"].normalize(masked)
    assert np.array_equal(normalized, np.zeros(4, dtype=np.float32))
    assert prompt_only.expert_state_keys == ()
    assert prompt_only.prompt_state_keys == ("state.proprioception",)
    assert tuple(prompt_only.prompt_spec.state_values) == ("state.proprioception",)


def test_typed_episode_identity_does_not_collide() -> None:
    assert typed_episode_key(1) != typed_episode_key("1")


def test_shuffle_requires_deranged_same_split_donor() -> None:
    from lunavla.v3 import InterventionSpecV1

    config = ExperimentConfig.load("configs/v3/diagnostic_fake_libero.yaml")
    canonical = dataset_for_config(config).episodes[0].transitions[0].observation
    arm = InterventionSpecV1("shuffle", "prompt", "shuffle", "rollout", {})
    with pytest.raises(ValueError, match="missing.*donor"):
        DiagnosticRouterV1(config, _stats(config), intervention=arm).route_observation(
            canonical
        )
    recipient = typed_episode_key(canonical.episode_id)
    with pytest.raises(ValueError, match="cannot reference"):
        DiagnosticRouterV1(
            config,
            _stats(config),
            intervention=arm,
            donor_instructions={recipient: (recipient, "donor instruction")},
        ).route_observation(canonical)


def test_diagnostic_engine_trains_and_evaluates_sanitized_views() -> None:
    config = ExperimentConfig.load("configs/v3/diagnostic_fake_libero.yaml")
    bundle = dataset_for_config(config)
    engine = EngineV3(config)
    policy, losses = engine.train(bundle.source("train"))
    assert len(losses) == config.training["steps"]
    assert engine.diagnostic_router is not None
    metrics = engine.evaluate(
        policy, FakePointEnvV3("fake_libero", config.evaluation["max_steps"])
    )
    assert metrics["episodes"] == 2
