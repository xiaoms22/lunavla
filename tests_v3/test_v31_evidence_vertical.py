from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from lunavla.v3 import (
    DeterministicFixtureExtractor,
    ExperimentConfig,
    FrozenFeatureCacheReaderV1,
    V31EvidenceDesignV1,
    VLMBackendSpecV1,
    build_frozen_feature_cache,
    make_v31_task_dataset,
)
from lunavla.v3.engine import dataset_for_config


H0 = "0" * 64
H1 = "1" * 64


def _backend() -> VLMBackendSpecV1:
    return VLMBackendSpecV1(
        backend_id="smolvlm2_500m",
        repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        spdx_license="Apache-2.0",
        license_scope="model_weights",
        license_evidence_sha256=H0,
        processor_class="AutoProcessor",
        processor_config_sha256=H0,
        model_config_sha256=H1,
        hidden_layer=-1,
        pooling="attention_mask_mean",
        image_token_layout="processor_native",
        camera_order=("camera.primary",),
        model_dtype="float32",
        device="cpu",
        offload_plan="none",
        deterministic=True,
        evidence_role="claim_bearing",
        weight_files={"model.safetensors": hashlib.sha256(b"fixture").hexdigest()},
        total_weight_bytes=7,
    )


def test_fixed_evidence_design_is_exactly_2400_and_fail_closed() -> None:
    design = V31EvidenceDesignV1.load("configs/v3/v31_frozen_vlm_evidence.yaml")
    assert design.expected_rows == 2_400
    assert design.claim_allowed is False
    assert len(design.sha256()) == 64
    payload = design.to_dict()
    payload["episodes_per_cell"] = 19
    with pytest.raises(ValueError, match="20 paired episodes"):
        V31EvidenceDesignV1.from_mapping(payload)
    payload = design.to_dict()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="integer 1"):
        V31EvidenceDesignV1.from_mapping(payload)


def test_v31_config_selects_all_three_tasks_and_two_held_out_strata() -> None:
    config = ExperimentConfig.load("configs/v3/v31_fixture_frozen_cpu.yaml")
    bundle = dataset_for_config(config)
    test_episodes = bundle.select("test")
    assert {item.metadata["task_id"] for item in test_episodes} == {
        "direct_pick_place",
        "waypoint_sequence",
        "failure_recovery",
    }
    assert {item.metadata["held_out_stratum"] for item in test_episodes} == {
        "composition",
        "paraphrase",
    }
    payload = config.to_dict()
    payload["dataset"]["parameters"]["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        ExperimentConfig.from_mapping(payload)
    payload = config.to_dict()
    payload["task"]["id"] = "fake_libero"
    with pytest.raises(ValueError, match="must match|must be used together"):
        ExperimentConfig.from_mapping(payload)


def test_feature_mask_and_shuffle_apply_per_step_without_crossing_split(
    tmp_path: Path,
) -> None:
    dataset = make_v31_task_dataset(data_seed=42, train_per_task=1, held_out_per_cell=1)
    cache = tmp_path / "cache"
    build_frozen_feature_cache(
        dataset,
        _backend(),
        DeterministicFixtureExtractor(16),
        cache,
        processor_sha256=H0,
        device_environment_sha256=H1,
    )
    reader = FrozenFeatureCacheReaderV1(cache)
    seen_donors: dict[str, str] = {}
    for episode in dataset.bundle.select("test"):
        for transition in episode.transitions:
            observation = transition.observation
            kwargs = {
                "split": "test",
                "task_id": str(episode.metadata["task_id"]),
                "episode_id": episode.episode_id,
                "step_index": observation.step_index,
                "donor_seed": 202701,
            }
            control, control_donor = reader.intervened(**kwargs, intervention="control")
            masked, mask_donor = reader.intervened(**kwargs, intervention="feature_mask")
            shuffled, donor = reader.intervened(**kwargs, intervention="feature_shuffle")
            repeat, repeat_donor = reader.intervened(**kwargs, intervention="feature_shuffle")
            assert control_donor is mask_donor is None
            assert np.count_nonzero(masked) == 0
            assert donor is not None and donor == repeat_donor
            assert donor != reader.sample_identity(
                split="test",
                task_id=str(episode.metadata["task_id"]),
                episode_id=episode.episode_id,
                step_index=observation.step_index,
            )
            assert donor.startswith('["test",')
            assert np.array_equal(shuffled, repeat)
            assert not np.array_equal(control, shuffled)
            seen_donors[donor] = donor
    assert seen_donors


def test_feature_interventions_are_bound_to_conditioned_config() -> None:
    base = ExperimentConfig.load("configs/v3/v31_fixture_frozen_cpu.yaml")
    for arm in ("feature_mask", "feature_shuffle"):
        payload = copy.deepcopy(base.to_dict())
        payload["policy"]["parameters"]["feature_intervention"] = arm
        assert (
            ExperimentConfig.from_mapping(payload).policy["parameters"]["feature_intervention"]
            == arm
        )
    payload = copy.deepcopy(base.to_dict())
    payload["policy"]["parameters"]["condition_mode"] = "learned_null"
    payload["policy"]["parameters"]["feature_intervention"] = "feature_mask"
    with pytest.raises(ValueError, match="require condition_mode=frozen_feature"):
        ExperimentConfig.from_mapping(payload)
