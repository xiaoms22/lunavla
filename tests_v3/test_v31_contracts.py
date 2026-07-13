from __future__ import annotations

import copy

import pytest

from lunavla.v3 import (
    ExperimentConfig,
    FeatureCacheIndexV1,
    FrozenFeatureManifestV1,
    TaskSuiteSpecV1,
    TraceBundleManifestV1,
    V31_CONFIG_CONTRACT_REVISION,
    VLMBackendSpecV1,
)


H0 = "0" * 64
H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def _backend() -> VLMBackendSpecV1:
    return VLMBackendSpecV1(
        backend_id="smolvlm2_500m",
        repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        revision=REVISION,
        spdx_license="Apache-2.0",
        license_scope="model_weights",
        license_evidence_sha256=H0,
        processor_class="AutoProcessor",
        processor_config_sha256=H1,
        model_config_sha256=H2,
        hidden_layer=-1,
        pooling="attention_mask_mean",
        image_token_layout="processor_native",
        camera_order=("camera.primary",),
        model_dtype="float32",
        device="mps",
        offload_plan="none",
        deterministic=False,
        evidence_role="claim_bearing",
        weight_files={"model.safetensors": H3},
        total_weight_bytes=1024,
    )


def _suite() -> TaskSuiteSpecV1:
    return TaskSuiteSpecV1(
        suite_id="synthetic_vlm_v1",
        task_ids=("direct_pick_place", "waypoint_sequence", "failure_recovery"),
        geometry_generator="seeded_geometry_v1",
        visible_modalities=("camera.primary", "instruction", "state.proprioception"),
        instruction_generator="compositional_instruction_v1",
        held_out_strata=("composition", "paraphrase"),
        success_conditions={"distance_threshold": 0.05, "ordered": True},
        image_shape=(96, 96, 3),
        state_fields=("x", "y", "gripper", "phase"),
        action_fields=("dx", "dy", "gripper"),
        action_min=-1,
        action_max=1,
        control_rate_hz=10,
        max_steps=64,
        oracle_excluded_fields=("goal_coordinates", "oracle_action", "answer_key"),
    )


def test_vlm_backend_is_strict_immutable_and_hash_stable() -> None:
    files = {"model.safetensors": H3}
    backend = _backend()
    files["unexpected.bin"] = H0
    assert dict(backend.weight_files) == {"model.safetensors": H3}
    assert VLMBackendSpecV1.from_mapping(backend.to_dict()) == backend
    assert VLMBackendSpecV1.from_mapping(backend.to_dict()).sha256() == backend.sha256()
    payload = backend.to_dict()
    payload["typo"] = True
    with pytest.raises(ValueError, match="unknown VLMBackendSpecV1"):
        VLMBackendSpecV1.from_mapping(payload)
    payload = backend.to_dict()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="integer 1"):
        VLMBackendSpecV1.from_mapping(payload)
    payload = backend.to_dict()
    payload["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        VLMBackendSpecV1.from_mapping(payload)


def test_feature_manifest_and_index_reject_drift_and_duplicate_identity() -> None:
    manifest = FrozenFeatureManifestV1(
        backend_spec_sha256=_backend().sha256(),
        processor_sha256=H0,
        prompt_renderer_sha256=H1,
        image_sha256=H2,
        sample_id="sample-1",
        episode_id="episode-1",
        step_index=0,
        split="train",
        task_id="direct_pick_place",
        held_out_stratum="composition",
        hidden_layer=-1,
        pooling="attention_mask_mean",
        dtype="float32",
        device_environment_sha256=H3,
        output_shape=(768,),
        finite=True,
        feature_sha256=H0,
        deterministic=False,
        generation_command=("lunavla-v3", "vlm-cache"),
    )
    assert FrozenFeatureManifestV1.from_mapping(manifest.to_dict()) == manifest
    identities = ("train:task:ep-1:0", "validation:task:ep-2:0", "test:task:ep-3:0")
    index = FeatureCacheIndexV1(
        backend_spec_sha256=_backend().sha256(),
        manifest_hashes=(H0, H1, H2),
        expected_identities=identities,
        observed_identities=identities,
        task_ids=("direct_pick_place",),
        held_out_strata=("composition",),
        split_counts={"train": 1, "validation": 1, "test": 1},
        total_feature_bytes=9216,
    )
    assert FeatureCacheIndexV1.from_mapping(index.to_dict()).sha256() == index.sha256()
    with pytest.raises(ValueError, match="match exactly"):
        FeatureCacheIndexV1(
            backend_spec_sha256=index.backend_spec_sha256,
            manifest_hashes=index.manifest_hashes,
            expected_identities=index.expected_identities,
            observed_identities=index.observed_identities[:-1] + ("test:task:other:0",),
            task_ids=index.task_ids,
            held_out_strata=index.held_out_strata,
            split_counts=index.split_counts,
            total_feature_bytes=index.total_feature_bytes,
        )


def test_task_and_trace_contracts_freeze_nested_values_and_paths() -> None:
    conditions = {"nested": {"thresholds": [0.1, 0.2]}}
    suite = _suite()
    conditions["nested"]["thresholds"].append(9.0)
    assert TaskSuiteSpecV1.from_mapping(suite.to_dict()).sha256() == suite.sha256()
    trace = TraceBundleManifestV1(
        evidence_manifest_sha256=H0,
        run_manifest_hashes=(H1,),
        paired_identity_hash=H2,
        static_files={"index.html": H3, "data/trace.json": H0},
        privacy_report_sha256=H1,
        languages=("en", "zh-CN"),
        offline=True,
        csp_sha256=H2,
    )
    assert TraceBundleManifestV1.from_mapping(trace.to_dict()) == trace
    payload = trace.to_dict()
    payload["static_files"] = {"../private.json": H0}
    with pytest.raises(ValueError, match="contained"):
        TraceBundleManifestV1.from_mapping(payload)


def test_revision4_config_round_trip_and_cross_contract_hash() -> None:
    from lunavla.v3.config import ExperimentConfig as Config

    base = Config.load("configs/v3/act_fake_libero_cpu.yaml").to_dict()
    base["contract_revision"] = V31_CONFIG_CONTRACT_REVISION
    base["prompt"] = {
        "enabled": False,
        "renderer_id": "lunavla.canonical_json",
        "renderer_version": 1,
        "assistant_target": "action_chunk",
        "neutral_token": "[MASKED]",
        "camera_order": ["camera.primary"],
        "public_slots": {},
    }
    base["routing"] = {
        "mode": "expert_only",
        "state_features": ["state.proprioception"],
    }
    base["vlm"] = _backend().to_dict()
    base["feature_cache"] = {
        "enabled": True,
        "root": "outputs/v31-feature-cache",
        "backend_spec_sha256": _backend().sha256(),
        "read_only": True,
    }
    base["task_suite"] = _suite().to_dict()
    base["trace"] = {
        "enabled": True,
        "output_dir": "outputs/v31-trace",
        "languages": ["en", "zh-CN"],
        "offline": True,
    }
    config = ExperimentConfig.from_mapping(base)
    assert config.contract_revision == 4
    assert ExperimentConfig.from_mapping(config.to_dict()).sha256() == config.sha256()
    mutated = copy.deepcopy(config.to_dict())
    mutated["feature_cache"]["backend_spec_sha256"] = H0
    with pytest.raises(ValueError, match="does not match vlm"):
        ExperimentConfig.from_mapping(mutated)
    mutated = copy.deepcopy(config.to_dict())
    mutated["trace"]["output_dir"] = "../private"
    with pytest.raises(ValueError, match="contained"):
        ExperimentConfig.from_mapping(mutated)
