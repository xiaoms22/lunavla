from __future__ import annotations

import numpy as np
import pytest

from lunavla.v3 import (
    DiagnosticRouterV1,
    DiagnosticExecutionError,
    ExperimentConfig,
    FeatureNormalizationV1,
    NormalizationStatsV1,
    StateRouteSpecV1,
    typed_episode_key,
)
from lunavla.v3.engine import EngineV3, dataset_for_config
from lunavla.v3.fake_tasks import FakePointEnvV3
from model.policy_base import ActionChunk


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


def test_image_shuffle_requires_a_matching_donor_step() -> None:
    from lunavla.v3 import InterventionSpecV1

    config = ExperimentConfig.load("configs/v3/diagnostic_act_image.yaml")
    canonical = dataset_for_config(config).episodes[0].transitions[0].observation
    arm = InterventionSpecV1(
        "image_shuffle", "image", "shuffle", "rollout", {}
    )
    with pytest.raises(ValueError, match="missing.*donor step"):
        DiagnosticRouterV1(
            config, _stats(config), intervention=arm
        ).route_observation(canonical)


def test_diagnostic_engine_trains_and_evaluates_sanitized_views() -> None:
    config = ExperimentConfig.load("configs/v3/diagnostic_fake_libero.yaml")
    bundle = dataset_for_config(config)
    engine = EngineV3(config)
    policy, losses = engine.train(bundle.source("train"))
    assert len(losses) == config.training["steps"]
    assert engine.diagnostic_router is not None
    metrics = engine.evaluate(
        policy,
        FakePointEnvV3(
            "fake_libero", config.evaluation["max_steps"], "region_instruction_v1"
        ),
    )
    assert metrics["episodes"] == 2


@pytest.mark.torch
def test_act_v3_uses_the_same_diagnostic_route_and_renderer() -> None:
    payload = ExperimentConfig.load("configs/v3/act_fake_libero_cpu.yaml").to_dict()
    payload["contract_revision"] = 2
    payload["training"]["steps"] = 1
    payload["evaluation"]["episodes"] = 1
    payload["evaluation"]["seeds"] = [1000]
    payload["diagnostics"]["enabled"] = True
    payload["dataset"]["parameters"]["instruction_variant"] = "region_instruction_v1"
    payload["prompt"] = {
        "enabled": True,
        "renderer_id": "lunavla.canonical_json",
        "renderer_version": 1,
        "assistant_target": "action_chunk",
        "neutral_token": "[MASKED]",
        "camera_order": ["camera.primary"],
        "public_slots": {"task_family": "fake_libero"},
    }
    payload["routing"] = {
        "mode": "expert_only",
        "state_features": ["state.proprioception"],
    }
    config = ExperimentConfig.from_mapping(payload)
    bundle = dataset_for_config(config)
    engine = EngineV3(config)
    policy, losses = engine.train(bundle.source("train"))
    assert len(losses) == 1 and np.isfinite(losses[0])
    assert engine.diagnostic_router is not None
    routed = engine.diagnostic_router.route_observation(
        bundle.select("test")[0].transitions[0].observation
    )
    assert tuple(routed.observation.images) == ("camera.primary",)
    assert routed.observation.instruction == routed.prompt_spec.rendered_text
    metrics = engine.evaluate(
        policy, FakePointEnvV3("fake_libero", 2, "region_instruction_v1")
    )
    assert metrics["episodes"] == 1


def test_diagnostic_prediction_failure_records_stage_and_closes_once() -> None:
    config = ExperimentConfig.load("configs/v3/diagnostic_fake_libero.yaml")
    engine = EngineV3(config)
    engine.diagnostic_router = DiagnosticRouterV1(config, _stats(config))
    env = FakePointEnvV3("fake_libero", 2, "region_instruction_v1")

    class BrokenPolicy:
        spec = engine._policy_spec()

        def reset(self, seed: int) -> None:
            del seed

        def predict_chunk(self, sample: object) -> object:
            del sample
            raise RuntimeError("prediction failed")

    with pytest.raises(DiagnosticExecutionError, match="prediction failed") as captured:
        engine.evaluate(BrokenPolicy(), env)  # type: ignore[arg-type]
    assert captured.value.stage == "predict"
    assert captured.value.origin == "policy"
    assert env.closed


class _ConstantPolicy:
    def __init__(self, engine: EngineV3) -> None:
        self.spec = engine._policy_spec()

    def reset(self, seed: int) -> None:
        del seed

    def predict_chunk(self, sample: object) -> ActionChunk:
        del sample
        return ActionChunk(np.zeros((1, 2), dtype=np.float32), np.asarray([True]))


class _CountingEnv(FakePointEnvV3):
    close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def _diagnostic_engine() -> tuple[ExperimentConfig, EngineV3]:
    config = ExperimentConfig.load("configs/v3/diagnostic_fake_libero.yaml")
    engine = EngineV3(config)
    engine.diagnostic_router = DiagnosticRouterV1(config, _stats(config))
    return config, engine


def test_render_failure_is_classified_and_environment_closes_once() -> None:
    config, engine = _diagnostic_engine()

    class BrokenRouter:
        def route_observation(self, observation: object, *, phase: str) -> object:
            del observation, phase
            raise RuntimeError("render failed")

    engine.diagnostic_router = BrokenRouter()  # type: ignore[assignment]
    env = _CountingEnv("fake_libero", 2, "region_instruction_v1")
    with pytest.raises(DiagnosticExecutionError) as captured:
        engine.evaluate(_ConstantPolicy(engine), env)  # type: ignore[arg-type]
    assert (captured.value.stage, captured.value.origin) == ("render", "adapter")
    assert env.close_calls == 1


def test_environment_and_policy_reset_failures_close_once() -> None:
    _, engine = _diagnostic_engine()

    class BrokenResetEnv(_CountingEnv):
        def reset(self, *, seed: int | None = None) -> object:
            del seed
            raise RuntimeError("environment reset failed")

    env = BrokenResetEnv("fake_libero", 2, "region_instruction_v1")
    with pytest.raises(DiagnosticExecutionError) as captured:
        engine.evaluate(_ConstantPolicy(engine), env)  # type: ignore[arg-type]
    assert (captured.value.stage, captured.value.origin) == (
        "reset",
        "environment",
    )
    assert env.close_calls == 1

    class BrokenResetPolicy(_ConstantPolicy):
        def reset(self, seed: int) -> None:
            del seed
            raise RuntimeError("policy reset failed")

    policy_env = _CountingEnv("fake_libero", 2, "region_instruction_v1")
    with pytest.raises(DiagnosticExecutionError) as captured:
        engine.evaluate(BrokenResetPolicy(engine), policy_env)  # type: ignore[arg-type]
    assert (captured.value.stage, captured.value.origin) == ("reset", "policy")
    assert policy_env.close_calls == 1


def test_preprocess_failure_is_classified_and_environment_closes_once() -> None:
    _, engine = _diagnostic_engine()

    class BrokenSpec:
        @property
        def history(self) -> int:
            raise RuntimeError("preprocess failed")

    policy = _ConstantPolicy(engine)
    policy.spec = BrokenSpec()  # type: ignore[assignment]
    env = _CountingEnv("fake_libero", 2, "region_instruction_v1")
    with pytest.raises(DiagnosticExecutionError) as captured:
        engine.evaluate(policy, env)  # type: ignore[arg-type]
    assert (captured.value.stage, captured.value.origin) == (
        "preprocess",
        "engine",
    )
    assert env.close_calls == 1


def test_step_and_trace_failures_are_classified_and_close_once() -> None:
    config, engine = _diagnostic_engine()

    class BrokenStepEnv(_CountingEnv):
        def step(self, action: np.ndarray) -> object:
            del action
            raise RuntimeError("step failed")

    step_env = BrokenStepEnv("fake_libero", 2, "region_instruction_v1")
    with pytest.raises(DiagnosticExecutionError) as captured:
        engine.evaluate(_ConstantPolicy(engine), step_env)  # type: ignore[arg-type]
    assert (captured.value.stage, captured.value.origin) == ("step", "environment")
    assert step_env.close_calls == 1

    trace_env = _CountingEnv("fake_libero", 2, "region_instruction_v1")

    def broken_trace(*values: object) -> None:
        del values
        raise RuntimeError("trace failed")

    with pytest.raises(DiagnosticExecutionError) as captured:
        engine.evaluate(
            _ConstantPolicy(engine), trace_env, trace_callback=broken_trace  # type: ignore[arg-type]
        )
    assert (captured.value.stage, captured.value.origin) == ("trace", "engine")
    assert trace_env.close_calls == 1


def test_postprocess_failure_is_classified_and_closes_once() -> None:
    _, engine = _diagnostic_engine()

    class BrokenChunk:
        @property
        def values(self) -> object:
            raise RuntimeError("postprocess failed")

        valid_mask = np.asarray([True])

    class BrokenPostprocessPolicy(_ConstantPolicy):
        def predict_chunk(self, sample: object) -> object:
            del sample
            return BrokenChunk()

    env = _CountingEnv("fake_libero", 2, "region_instruction_v1")
    with pytest.raises(DiagnosticExecutionError) as captured:
        engine.evaluate(BrokenPostprocessPolicy(engine), env)  # type: ignore[arg-type]
    assert (captured.value.stage, captured.value.origin) == ("postprocess", "engine")
    assert env.close_calls == 1
