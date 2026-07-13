from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
pytest.importorskip("diffusers")

from lunavla.v3 import ExperimentConfig, PolicyBatchV3, verify_run_directory  # noqa: E402
from lunavla.v3.diffusion_policy import (  # noqa: E402
    DiffusionPolicyV3,
    _restore,
    diffusion_policy_spec,
)
from lunavla.v3.engine import EngineV3, dataset_for_config, run_alpha  # noqa: E402
from lunavla.v3.normalization import fit_normalization_stats  # noqa: E402


def _config(tmp_path: Path) -> ExperimentConfig:
    payload = ExperimentConfig.load("configs/v3/diffusion_fake_libero_cpu.yaml").to_dict()
    payload["artifacts"]["output_dir"] = str(tmp_path / "diffusion-run")
    return ExperimentConfig.from_mapping(payload)


def test_diffusion_config_is_strict_and_schema_driven(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert ExperimentConfig.from_mapping(config.to_dict()).sha256() == config.sha256()
    assert "state_dim" not in config.policy["parameters"]
    assert "action_dim" not in config.policy["parameters"]
    assert config.policy["parameters"]["unused_modalities"] == ("instruction",)
    payload = config.to_dict()
    payload["policy"]["parameters"]["unused_modalities"] = []
    with pytest.raises(ValueError, match="explicitly declare"):
        ExperimentConfig.from_mapping(payload)
    payload = config.to_dict()
    payload["policy"]["parameters"]["camera_features"] = []
    with pytest.raises(ValueError, match="requires non-empty camera"):
        ExperimentConfig.from_mapping(payload)


def test_diffusion_uses_unified_engine_and_finite_gradients(tmp_path: Path) -> None:
    config = _config(tmp_path)
    engine = EngineV3(config)
    policy, losses = engine.train(dataset_for_config(config).source("train"))
    assert isinstance(policy, DiffusionPolicyV3)
    assert np.all(np.isfinite(losses))
    assert all(
        result.gradient_norm is not None and np.isfinite(result.gradient_norm)
        for result in engine.train_results
    )
    assert policy.spec.camera_order == ("camera.primary",)
    assert policy.spec.required_modalities == ("image", "state")
    assert policy.policy.config.noise_scheduler_type == "DDIM"
    assert policy.policy.config.num_inference_steps == 8


def test_diffusion_fixed_32_sample_fixture_overfits_in_100_steps(tmp_path: Path) -> None:
    payload = _config(tmp_path).to_dict()
    payload["training"]["steps"] = 100
    payload["training"]["batch_size"] = 32
    config = ExperimentConfig.from_mapping(payload)
    _, losses = EngineV3(config).train(dataset_for_config(config).source("train"))
    assert losses[-1] <= losses[0] * 0.5


def test_diffusion_checkpoint_resume_and_noise_are_exact_on_cpu(tmp_path: Path) -> None:
    payload = _config(tmp_path).to_dict()
    payload["training"]["steps"] = 1
    config = ExperimentConfig.from_mapping(payload)
    result = run_alpha(config)
    manifest = verify_run_directory(result.output_dir)
    assert manifest["policy_id"] == "diffusion_v3"
    checkpoint = result.output_dir / "checkpoint"
    first = EngineV3(config).restore_policy(checkpoint)
    second = EngineV3(config).restore_policy(checkpoint)
    assert isinstance(first, DiffusionPolicyV3) and isinstance(second, DiffusionPolicyV3)

    bundle = dataset_for_config(config)
    samples = EngineV3._samples(
        bundle.source("train").load(),
        history=first.spec.history,
        chunk_size=first.spec.horizon,
    )
    batch = PolicyBatchV3(tuple(samples[: config.training["batch_size"]]), device="cpu")
    first_result = first.train_step(
        batch, learning_rate=config.training["learning_rate"], step=1
    )
    second_result = second.train_step(
        batch, learning_rate=config.training["learning_rate"], step=1
    )
    assert first_result.loss == second_result.loss
    for name, value in first.policy.state_dict().items():
        assert torch.equal(value, second.policy.state_dict()[name])

    first.reset(123)
    same_a = first.predict_chunk(samples[0]).values
    first.reset(123)
    same_b = first.predict_chunk(samples[0]).values
    first.reset(124)
    different = first.predict_chunk(samples[0]).values
    assert np.array_equal(same_a, same_b)
    assert not np.array_equal(same_a, different)

    envelope = json.loads((checkpoint / "checkpoint.v3.json").read_text())
    paths = {record["path"] for record in envelope["files"]}
    assert "policy/diffusion_v3/model/model.safetensors" in paths
    assert "policy/diffusion_v3/training_state.pt" in paths
    assert "policy/diffusion_v3/processors/preprocessor.json" in paths


def test_diffusion_restore_rejects_scheduler_contract_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_alpha(config)
    payload = config.to_dict()
    payload["policy"]["parameters"]["num_inference_steps"] = 4
    changed = ExperimentConfig.from_mapping(payload)
    spec = diffusion_policy_spec(changed)
    normalization = fit_normalization_stats(
        dataset_for_config(changed).source("train").load(), changed.feature_schema
    )
    with pytest.raises(ValueError, match="contract does not match"):
        _restore(
            result.output_dir / "checkpoint",
            changed,
            spec,
            normalization,
        )


def test_diffusion_processor_tampering_is_detected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_alpha(config)
    processor = result.output_dir / "checkpoint/policy/diffusion_v3/processors/preprocessor.json"
    processor.write_text("{}\n", encoding="utf-8")
    normalization = fit_normalization_stats(
        dataset_for_config(config).source("train").load(), config.feature_schema
    )
    with pytest.raises(ValueError, match="processor tree hash"):
        _restore(
            result.output_dir / "checkpoint/policy/diffusion_v3",
            config,
            diffusion_policy_spec(config),
            normalization,
        )
    with pytest.raises(ValueError, match="preprocessor.json"):
        verify_run_directory(result.output_dir)
