from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunavla.v3 import ExperimentConfig, PolicyBatchV3, verify_run_directory  # noqa: E402
from lunavla.transformer_policy import TransformerChunkCVAEPolicy  # noqa: E402
from lunavla.v3.act_policy import (  # noqa: E402
    ActPolicyV3,
    _restore,
    _transformer_config,
    act_policy_spec,
)
from lunavla.v3.engine import EngineV3, dataset_for_config, run_alpha  # noqa: E402
from lunavla.v3.normalization import fit_normalization_stats  # noqa: E402


def _config(tmp_path: Path) -> ExperimentConfig:
    payload = ExperimentConfig.load("configs/v3/act_fake_libero_cpu.yaml").to_dict()
    payload["artifacts"]["output_dir"] = str(tmp_path / "act-run")
    return ExperimentConfig.from_mapping(payload)


def test_act_v3_config_round_trip_and_schema_dimensions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert ExperimentConfig.from_mapping(config.to_dict()).sha256() == config.sha256()
    assert "state_dim" not in config.policy["parameters"]
    assert "action_dim" not in config.policy["parameters"]
    assert config.feature_schema.by_role("state")[0].shape == (4,)
    assert config.feature_schema.by_role("action")[0].shape == (2,)
    payload = config.to_dict()
    payload["policy"]["parameters"]["camera_feature"] = None
    with pytest.raises(ValueError, match="silently discard"):
        ExperimentConfig.from_mapping(payload)


def test_act_v3_unified_engine_checkpoint_and_finite_gradient(tmp_path: Path) -> None:
    config = _config(tmp_path)
    bundle = dataset_for_config(config)
    engine = EngineV3(config)
    policy, losses = engine.train(bundle.source("train"))
    assert isinstance(policy, ActPolicyV3)
    assert len(losses) == config.training["steps"]
    assert np.all(np.isfinite(losses))
    assert all(
        result.gradient_norm is not None and np.isfinite(result.gradient_norm)
        for result in engine.train_results
    )
    assert policy.spec.camera_order == ("camera.primary",)
    assert policy.spec.required_modalities == ("image", "state", "instruction")


def test_act_v3_fixed_32_sample_fixture_overfits_in_100_steps(tmp_path: Path) -> None:
    payload = _config(tmp_path).to_dict()
    payload["training"]["steps"] = 100
    payload["training"]["batch_size"] = 32
    payload["policy"]["parameters"]["sample_latent_during_training"] = False
    config = ExperimentConfig.from_mapping(payload)
    engine = EngineV3(config)
    _, losses = engine.train(dataset_for_config(config).source("train"))
    assert losses[-1] <= losses[0] * 0.5


def test_act_v3_run_restores_exact_next_step_on_cpu(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_alpha(config)
    manifest = verify_run_directory(result.output_dir)
    assert manifest["policy_id"] == "act_v3"
    checkpoint = result.output_dir / "checkpoint"
    first_engine = EngineV3(config)
    second_engine = EngineV3(config)
    first = first_engine.restore_policy(checkpoint)
    second = second_engine.restore_policy(checkpoint)
    assert isinstance(first, ActPolicyV3) and isinstance(second, ActPolicyV3)

    bundle = dataset_for_config(config)
    samples = EngineV3._samples(
        bundle.source("train").load(),
        history=first.spec.history,
        chunk_size=first.spec.chunk_size,
    )
    batch = PolicyBatchV3(tuple(samples[: config.training["batch_size"]]), device="cpu")
    first_result = first.train_step(
        batch,
        learning_rate=config.training["learning_rate"],
        step=config.training["steps"],
    )
    second_result = second.train_step(
        batch,
        learning_rate=config.training["learning_rate"],
        step=config.training["steps"],
    )
    assert first_result.loss == second_result.loss
    for name, value in first.policy.state_dict().items():
        assert torch.equal(value, second.policy.state_dict()[name])

    envelope = json.loads((checkpoint / "checkpoint.v3.json").read_text())
    paths = {record["path"] for record in envelope["files"]}
    assert "policy/act_v3.pt" in paths
    assert "training_state.json" in paths


def test_v2_transformer_checkpoint_cannot_be_relabeled_act_v3(tmp_path: Path) -> None:
    config = _config(tmp_path)
    spec = act_policy_spec(config)
    episodes = dataset_for_config(config).source("train").load()
    normalization = fit_normalization_stats(episodes, config.feature_schema)
    legacy = TransformerChunkCVAEPolicy(_transformer_config(config, spec))
    checkpoint = legacy.save_checkpoint(tmp_path / "v2-transformer.pt")
    with pytest.raises(ValueError, match="cannot be relabeled"):
        _restore(checkpoint, config, spec, normalization)
