from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import lunavla.v3.act_policy as act_module  # noqa: E402
from lunavla.v3 import (  # noqa: E402
    DeterministicFixtureExtractor,
    ExperimentConfig,
    FrozenFeatureCacheReaderV1,
    PolicyBatchV3,
    V31_CONFIG_CONTRACT_REVISION,
    VLMBackendSpecV1,
    build_frozen_feature_cache,
    make_v31_task_dataset,
    task_suite_spec_v1,
    v31_feature_schema,
)
from lunavla.v3.act_policy import (  # noqa: E402
    ActPolicyV3,
    _restore,
    _transformer_config,
    act_policy_spec,
)
from lunavla.v3.engine import EngineV3  # noqa: E402
from lunavla.v3.normalization import fit_normalization_stats  # noqa: E402
from lunavla.transformer_policy import TransformerChunkCVAEPolicy  # noqa: E402


H0 = "0" * 64
H1 = "1" * 64


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        weight_files={"model.safetensors": _sha(b"fixture")},
        total_weight_bytes=7,
    )


def _config(mode: str, backend: VLMBackendSpecV1) -> ExperimentConfig:
    payload = ExperimentConfig.load("configs/v3/act_fake_libero_cpu.yaml").to_dict()
    payload["contract_revision"] = V31_CONFIG_CONTRACT_REVISION
    payload["policy"]["parameters"].update(
        {
            "instruction_dim": 16,
            "condition_mode": mode,
            "condition_input_dim": 16,
            "sample_latent_during_training": False,
        }
    )
    payload["features"] = v31_feature_schema().to_dict()
    payload["prompt"] = {
        "enabled": False,
        "renderer_id": "lunavla.canonical_json",
        "renderer_version": 1,
        "assistant_target": "action_chunk",
        "neutral_token": "[MASKED]",
        "camera_order": ["camera.primary"],
        "public_slots": {},
    }
    payload["routing"] = {
        "mode": "expert_only",
        "state_features": ["state.proprioception"],
    }
    payload["vlm"] = backend.to_dict()
    payload["feature_cache"] = {
        "enabled": True,
        "root": "cache",
        "backend_spec_sha256": backend.sha256(),
        "read_only": True,
    }
    payload["task_suite"] = task_suite_spec_v1().to_dict()
    payload["trace"] = {
        "enabled": False,
        "output_dir": "outputs/v31-trace",
        "languages": ["en", "zh-CN"],
        "offline": True,
    }
    return ExperimentConfig.from_mapping(payload)


def _policies(tmp_path: Path):
    dataset = make_v31_task_dataset(data_seed=42, train_per_task=1, held_out_per_cell=1)
    backend = _backend()
    cache = tmp_path / "cache"
    build_frozen_feature_cache(
        dataset,
        backend,
        DeterministicFixtureExtractor(16),
        cache,
        processor_sha256=H0,
        device_environment_sha256=H1,
    )
    reader = FrozenFeatureCacheReaderV1(cache)
    frozen_config = _config("frozen_feature", backend)
    null_config = _config("learned_null", backend)
    episodes = dataset.bundle.select("train")
    normalization = fit_normalization_stats(episodes, frozen_config.feature_schema)
    frozen_spec = act_policy_spec(frozen_config)
    null_spec = act_policy_spec(null_config)
    frozen = ActPolicyV3(
        TransformerChunkCVAEPolicy(_transformer_config(frozen_config, frozen_spec)),
        spec=frozen_spec,
        normalization=normalization,
        state_feature="state.proprioception",
        camera_feature="camera.primary",
        temporal_ensemble_decay=None,
        condition_mode="frozen_feature",
        feature_cache=reader,
    )
    null = ActPolicyV3(
        TransformerChunkCVAEPolicy(_transformer_config(null_config, null_spec)),
        spec=null_spec,
        normalization=normalization,
        state_feature="state.proprioception",
        camera_feature="camera.primary",
        temporal_ensemble_decay=None,
        condition_mode="learned_null",
    )
    samples = EngineV3._samples(episodes, history=1, chunk_size=4)
    return frozen_config, null_config, frozen, null, samples, normalization


def test_conditioned_arms_have_equal_parameters_and_exact_64_token(
    tmp_path: Path,
) -> None:
    _, _, frozen, null, samples, _ = _policies(tmp_path)
    assert sum(item.numel() for item in frozen.policy.parameters()) == sum(
        item.numel() for item in null.policy.parameters()
    )
    frozen_token = frozen.condition_token(samples[0])
    null_token = null.condition_token(samples[0])
    assert frozen_token.shape == null_token.shape == (64,)
    assert frozen_token.dtype == null_token.dtype == np.float32
    assert np.all(np.isfinite(frozen_token))
    assert not np.array_equal(frozen_token, null_token)


def test_null_token_is_learned_through_the_same_projection(tmp_path: Path) -> None:
    _, _, _, null, samples, _ = _policies(tmp_path)
    before = null.condition_token(samples[0])
    result = null.train_step(
        PolicyBatchV3(tuple(samples[:2]), device="cpu"),
        learning_rate=3e-4,
        step=0,
    )
    after = null.condition_token(samples[0])
    assert result.finite is True
    assert not np.array_equal(before, after)
    projection = null.policy.instruction_projection
    assert projection is not None
    assert torch.equal(torch.from_numpy(after), projection.weight.detach().cpu()[:, 0])


def test_frozen_feature_arm_trains_and_cache_tamper_fails(tmp_path: Path) -> None:
    _, _, frozen, _, samples, _ = _policies(tmp_path)
    result = frozen.train_step(
        PolicyBatchV3(tuple(samples[:2]), device="cpu"),
        learning_rate=3e-4,
        step=0,
    )
    assert result.finite is True
    assert result.gradient_norm is not None and np.isfinite(result.gradient_norm)
    feature = tmp_path / "cache" / "features" / "00000000.npy"
    feature.write_bytes(feature.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="feature content hash mismatch"):
        FrozenFeatureCacheReaderV1(tmp_path / "cache")


def test_conditioned_act_applies_mask_and_shuffle_to_every_sample(
    tmp_path: Path,
) -> None:
    _, _, frozen, _, samples, _ = _policies(tmp_path)
    selected = tuple(samples[:6])
    control = frozen._condition_features(selected)
    assert control is not None
    frozen.feature_intervention = "feature_mask"
    masked = frozen._condition_features(selected)
    assert masked is not None and np.count_nonzero(masked) == 0
    frozen.feature_intervention = "feature_shuffle"
    first = frozen._condition_features(selected)
    first_donors = dict(frozen.last_feature_donors)
    second = frozen._condition_features(selected)
    assert first is not None and second is not None
    assert np.array_equal(first, second)
    assert not np.array_equal(first, control)
    assert len(first_donors) == len(selected)
    assert all(donor is not None for donor in first_donors.values())


def test_conditioned_checkpoint_binds_mode_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_config, _, frozen, _, samples, normalization = _policies(tmp_path)
    frozen.train_step(
        PolicyBatchV3(tuple(samples[:2]), device="cpu"),
        learning_rate=3e-4,
        step=0,
    )
    checkpoint = frozen.save_checkpoint(tmp_path / "act.pt", metadata={})
    monkeypatch.setattr(act_module, "_REPOSITORY_ROOT", tmp_path)
    restored = _restore(
        checkpoint,
        frozen_config,
        act_policy_spec(frozen_config),
        normalization,
    )
    assert restored.condition_mode == "frozen_feature"
    assert np.array_equal(
        frozen.predict_chunk(samples[0]).values,
        restored.predict_chunk(samples[0]).values,
    )
    wrong = copy.deepcopy(frozen_config.to_dict())
    wrong["policy"]["parameters"]["condition_mode"] = "learned_null"
    wrong_config = ExperimentConfig.from_mapping(wrong)
    with pytest.raises(ValueError, match="condition mode"):
        _restore(
            checkpoint,
            wrong_config,
            act_policy_spec(wrong_config),
            normalization,
        )


def test_condition_config_is_fail_closed_and_old_hash_is_stable() -> None:
    base = ExperimentConfig.load("configs/v3/act_fake_libero_cpu.yaml")
    assert "condition_mode" not in base.to_dict()["policy"]["parameters"]
    backend = _backend()
    conditioned = _config("frozen_feature", backend)
    assert conditioned.policy["parameters"]["condition_input_dim"] == 16
    payload = conditioned.to_dict()
    payload["policy"]["parameters"]["d_model"] = 32
    with pytest.raises(ValueError, match="d_model=64"):
        ExperimentConfig.from_mapping(payload)
    payload = conditioned.to_dict()
    payload["policy"]["parameters"]["instruction_dim"] = 8
    with pytest.raises(ValueError, match="must equal"):
        ExperimentConfig.from_mapping(payload)
    payload = base.to_dict()
    payload["policy"]["parameters"]["condition_mode"] = "frozen_feature"
    payload["policy"]["parameters"]["condition_input_dim"] = 8
    with pytest.raises(ValueError, match="contract_revision=4"):
        ExperimentConfig.from_mapping(payload)
