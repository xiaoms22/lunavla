from __future__ import annotations

import copy

import pytest

from lunavla.v3 import (
    DiagnosticDesignV1,
    DonorBankV1,
    DonorRecordV1,
    ExperimentConfig,
    FailureRecordV1,
    InterventionSpecV1,
    PromptSpecV1,
    StateRouteSpecV1,
)


def _arm(operator: str) -> InterventionSpecV1:
    return InterventionSpecV1(
        arm_id=operator,
        kind="prompt",
        operator=operator,
        phase="rollout",
        parameters={},
    )


def test_prompt_render_is_byte_stable_unicode_and_tamper_evident() -> None:
    first = PromptSpecV1(
        raw_instruction="把方块移到左边",
        public_slots={"target": "左", "count": 1},
        state_values={"state.proprioception": (0.25, -0.5)},
        renderer_id="lunavla.canonical_json",
        renderer_version=1,
        camera_order=("camera.primary",),
        assistant_target="action_chunk",
    )
    second = PromptSpecV1(
        raw_instruction="把方块移到左边",
        public_slots={"count": 1, "target": "左"},
        state_values={"state.proprioception": (0.25, -0.5)},
        renderer_id="lunavla.canonical_json",
        renderer_version=1,
        camera_order=("camera.primary",),
        assistant_target="action_chunk",
    )
    assert first.rendered_text == second.rendered_text
    assert first.rendered_text.endswith("\n")
    assert not first.rendered_text.endswith("\n\n")
    assert first.rendered_sha256 == second.rendered_sha256
    assert PromptSpecV1.from_mapping(first.to_dict()).sha256() == first.sha256()
    tampered = first.to_dict()
    tampered["rendered_text"] = tampered["rendered_text"].replace("左", "右")
    with pytest.raises(ValueError, match="tampered"):
        PromptSpecV1.from_mapping(tampered)


def test_nested_prompt_and_intervention_values_are_deeply_immutable() -> None:
    slots = {"nested": {"values": [1, 2]}}
    prompt = PromptSpecV1(
        "move", slots, {}, "lunavla.canonical_json", 1, (), "action_chunk"
    )
    before = prompt.sha256()
    slots["nested"]["values"].append(3)
    assert prompt.sha256() == before
    with pytest.raises(AttributeError):
        prompt.public_slots["nested"]["values"].append(4)
    parameters = {"nested": {"values": [1]}}
    arm = InterventionSpecV1("control", "prompt", "control", "rollout", parameters)
    parameters["nested"]["values"].append(2)
    assert arm.to_dict()["parameters"] == {"nested": {"values": [1]}}


def test_observation_preserves_prompt_bytes_including_terminal_lf() -> None:
    from lunavla.v3 import ObservationV3

    observation = ObservationV3({}, {"state.proprioception": [0.0]}, "prompt\n", 0, "ep", 0)
    assert observation.instruction == "prompt\n"


def test_layout_drift_changes_only_render_layout_identity() -> None:
    arguments = {
        "raw_instruction": "move",
        "public_slots": {},
        "state_values": {},
        "renderer_id": "lunavla.canonical_json",
        "renderer_version": 1,
        "camera_order": ("camera.primary",),
        "assistant_target": "action_chunk",
    }
    control = PromptSpecV1(**arguments)
    drift = PromptSpecV1(**arguments, layout_variant="assistant_before_cameras_v1")
    assert control.rendered_sha256 != drift.rendered_sha256
    assert control.raw_instruction == drift.raw_instruction
    assert control.camera_order == drift.camera_order


def test_route_intervention_and_design_contracts_are_strict() -> None:
    routes = (
        StateRouteSpecV1("expert_only", ("state.proprioception",)),
        StateRouteSpecV1("prompt_only", ("state.proprioception",)),
    )
    design = DiagnosticDesignV1(
        design_id="fake-pusht-diagnostic-ci",
        base_config="configs/v3/diagnostic_fake_pusht.yaml",
        output_dir="outputs/v3/diagnostic-ci",
        train_seeds=(11,),
        evaluation_seeds=(1000, 1001),
        routes=routes,
        interventions=(_arm("control"), _arm("mask"), _arm("shuffle")),
        donor_seed=42,
        analysis_seed=202701,
        bootstrap_samples=100,
        counterfactual_transform_id=None,
        reduced_design=True,
    )
    assert DiagnosticDesignV1.from_mapping(design.to_dict()).sha256() == design.sha256()
    assert routes[0].expert_enabled and not routes[0].prompt_enabled
    assert routes[1].prompt_enabled and not routes[1].expert_enabled
    with pytest.raises(ValueError, match="unsupported image"):
        InterventionSpecV1("bad", "image", "occlude", "rollout", {})
    with pytest.raises(ValueError, match="contained relative"):
        DiagnosticDesignV1(
            **{**design.__dict__, "base_config": "../secret.yaml"}
        )


def test_failure_taxonomy_requires_evidence_provenance() -> None:
    record = FailureRecordV1(
        "execution", "environment_timeout", "timeout_v1", "automatic", True
    )
    assert FailureRecordV1.from_mapping(record.to_dict()) == record
    with pytest.raises(ValueError, match="failure layer"):
        FailureRecordV1("model", "unknown", "rule", "automatic")
    with pytest.raises(ValueError, match="provenance"):
        FailureRecordV1("state", "leak", "sentinel_v1", "guessed")


def test_donor_contract_rejects_self_cross_split_duplicate_and_equal_content() -> None:
    one = "1" * 64
    two = "2" * 64
    record = DonorRecordV1("string:a", "string:b", "evaluation", None, one, two)
    bank = DonorBankV1("instruction", "evaluation", 42, (record,))
    assert DonorBankV1.from_mapping(bank.to_dict()).sha256() == bank.sha256()
    with pytest.raises(ValueError, match="recipient"):
        DonorRecordV1("same", "same", "evaluation", None, one, two)
    with pytest.raises(ValueError, match="differ"):
        DonorRecordV1("a", "b", "evaluation", None, one, one)
    cross = DonorRecordV1("a", "b", "test", None, one, two)
    with pytest.raises(ValueError, match="cross"):
        DonorBankV1("instruction", "evaluation", 42, (cross,))
    with pytest.raises(ValueError, match="duplicate"):
        DonorBankV1("instruction", "evaluation", 42, (record, record))


def test_config_revision_one_hash_is_stable_and_revision_two_is_strict() -> None:
    legacy = ExperimentConfig.load("configs/v3/fake_pusht_alpha.yaml")
    legacy_hash = legacy.sha256()
    assert legacy_hash == "e18c22b5984ac80b26b321c66694fea6ba8e5eb4016d3b0859b9e1cd14c7c5bb"
    assert legacy.contract_revision == 1
    assert ExperimentConfig.from_mapping(legacy.to_dict()).sha256() == legacy_hash

    payload = legacy.to_dict()
    payload["contract_revision"] = 2
    payload["policy"]["parameters"]["instruction_dim"] = 8
    payload["diagnostics"]["enabled"] = True
    payload["dataset"]["type"] = "fake_libero"
    payload["task"]["id"] = "fake_libero"
    payload["embodiment"]["task_id"] = "fake_libero"
    payload["dataset"]["parameters"]["instruction_variant"] = "region_instruction_v1"
    payload["prompt"] = {
        "enabled": True,
        "renderer_id": "lunavla.canonical_json",
        "renderer_version": 1,
        "assistant_target": "action_chunk",
        "neutral_token": "[MASKED]",
        "camera_order": [],
        "public_slots": {"task_family": "fake_pusht"},
    }
    payload["routing"] = {
        "mode": "expert_only",
        "state_features": ["state.proprioception"],
    }
    config = ExperimentConfig.from_mapping(payload)
    assert config.contract_revision == 2
    assert ExperimentConfig.from_mapping(config.to_dict()).sha256() == config.sha256()
    boolean_revision = copy.deepcopy(payload)
    boolean_revision["contract_revision"] = True
    with pytest.raises(TypeError, match="contract_revision"):
        ExperimentConfig.from_mapping(boolean_revision)
    wrong_route = copy.deepcopy(payload)
    wrong_route["routing"]["mode"] = "prompt_only"
    wrong_route["policy"]["parameters"]["instruction_dim"] = 0
    with pytest.raises(ValueError, match="instruction-consuming"):
        ExperimentConfig.from_mapping(wrong_route)


def test_diagnostic_config_rejects_diffusion_and_open_loop() -> None:
    payload = ExperimentConfig.load("configs/v3/diffusion_fake_libero_cpu.yaml").to_dict()
    payload["contract_revision"] = 2
    payload["diagnostics"]["enabled"] = True
    payload["prompt"] = {
        "enabled": True,
        "renderer_id": "lunavla.canonical_json",
        "renderer_version": 1,
        "assistant_target": "action_chunk",
        "neutral_token": "[MASKED]",
        "camera_order": payload["policy"]["parameters"]["camera_features"],
        "public_slots": {},
    }
    payload["routing"] = {
        "mode": "expert_only",
        "state_features": [payload["policy"]["parameters"]["state_feature"]],
    }
    with pytest.raises(ValueError, match="diffusion_v3"):
        ExperimentConfig.from_mapping(payload)
