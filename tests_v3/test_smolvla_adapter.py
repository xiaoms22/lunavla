from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
pytest.importorskip("transformers")

from lerobot.policies.smolvla import (  # noqa: E402
    SmolVLAPolicy,
    make_smolvla_pre_post_processors,
)
from lunavla.v3 import ExperimentConfig, PolicyRegistryV3  # noqa: E402
from lunavla.v3.engine import EngineV3, dataset_for_config  # noqa: E402
from lunavla.v3.fake_tasks import FakePointEnvV3  # noqa: E402
from lunavla.v3.smolvla_adapter import (  # noqa: E402
    MODEL_REVISION,
    MODEL_SHA256,
    SmolVLAAdapterV3,
    blocked_smolvla_restorer,
    register_smolvla_policy,
    smolvla_conformance_factory,
    smolvla_policy_spec,
)


class PublicProcessorFixture:
    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0

    def __call__(self, value: Any) -> Any:
        self.calls += 1
        if isinstance(value, torch.Tensor):
            return value
        tensors = {
            name: item.unsqueeze(0)
            for name, item in value.items()
            if isinstance(item, torch.Tensor)
        }
        if "task" in value:
            tensors["observation.language.tokens"] = torch.ones((1, 3), dtype=torch.long)
            tensors["observation.language.attention_mask"] = torch.ones(
                (1, 3), dtype=torch.bool
            )
        return tensors

    def reset(self) -> None:
        self.resets += 1

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        config_filename: str | None = None,
        push_to_hub: bool = False,
    ) -> None:
        assert push_to_hub is False
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / str(config_filename)).write_text('{"fixture":true}\n', encoding="utf-8")


class PublicPolicyFixture:
    def __init__(self, *, chunk_size: int, action_dim: int) -> None:
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.forward_calls = 0
        self.predict_calls = 0
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Mapping[str, Any]]:
        self.forward_calls += 1
        required = {
            "observation.state",
            "observation.images.camera.primary",
            "observation.language.tokens",
            "observation.language.attention_mask",
            "action",
            "action_is_pad",
        }
        assert required <= set(batch)
        loss = batch["action"].square().mean() + 0.125
        return loss, {"loss": float(loss)}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.predict_calls += 1
        batch_size = batch["observation.state"].shape[0]
        return torch.zeros((batch_size, self.chunk_size, self.action_dim))

    def save_pretrained(
        self, save_directory: str | Path, *, push_to_hub: bool = False
    ) -> None:
        assert push_to_hub is False
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "model.safetensors").write_bytes(b"public-api-fixture")


def _config(tmp_path: Path) -> ExperimentConfig:
    payload = ExperimentConfig.load("configs/v3/smolvla_conformance_cpu.yaml").to_dict()
    payload["artifacts"]["output_dir"] = str(tmp_path / "smolvla-run")
    return ExperimentConfig.from_mapping(payload)


def _registry() -> PolicyRegistryV3:
    registry = PolicyRegistryV3()

    def policy_factory(config: ExperimentConfig, spec: object, normalization: object) -> Any:
        del spec, normalization
        action_dim = config.feature_schema.by_role("action")[0].shape[0]
        return PublicPolicyFixture(
            chunk_size=int(config.policy["parameters"]["chunk_size"]),
            action_dim=action_dim,
        )

    def processor_factory(config: object, spec: object, normalization: object) -> Any:
        del config, spec, normalization
        return PublicProcessorFixture(), PublicProcessorFixture()

    register_smolvla_policy(
        registry,
        factory=smolvla_conformance_factory(policy_factory, processor_factory),
        restorer=blocked_smolvla_restorer,
    )
    return registry


def test_smolvla_config_and_model_source_are_fail_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    spec = smolvla_policy_spec(config)
    assert spec.model_source.revision == MODEL_REVISION
    assert spec.model_source.file_hashes["model.safetensors"] == MODEL_SHA256
    assert spec.model_source.license_status == "unverified"
    assert spec.model_source.pretrained_enabled is False
    assert spec.required_modalities == ("image", "state", "instruction")

    payload = config.to_dict()
    payload["policy"]["parameters"]["pretrained_enabled"] = True
    with pytest.raises(ValueError, match="weights must remain disabled"):
        ExperimentConfig.from_mapping(payload)
    payload = config.to_dict()
    payload["policy"]["parameters"]["license_status"] = "verified"
    with pytest.raises(ValueError, match="license_status must be pinned"):
        ExperimentConfig.from_mapping(payload)
    payload = config.to_dict()
    payload["policy"]["parameters"]["file_hashes"]["model.safetensors"] = "0" * 64
    with pytest.raises(ValueError, match="hash is not pinned"):
        ExperimentConfig.from_mapping(payload)


def test_default_registry_blocks_weights_and_optimizer(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="pretrained and optimizer gates are closed"):
        EngineV3(config).train(dataset_for_config(config).source("train"))


def test_public_api_fixture_runs_through_registry_and_engine(tmp_path: Path) -> None:
    config = _config(tmp_path)
    engine = EngineV3(config, _registry())
    policy, losses = engine.train(dataset_for_config(config).source("train"))
    assert isinstance(policy, SmolVLAAdapterV3)
    assert len(losses) == 1 and losses[0] > 0
    assert engine.train_results[0].gradient_norm is None
    assert "conformance_forward" in engine.train_results[0].timing_ms
    metrics = engine.evaluate(
        policy, FakePointEnvV3("fake_libero", config.evaluation["max_steps"])
    )
    assert metrics["episodes"] == 2
    assert policy.policy.forward_calls == 1
    assert policy.policy.predict_calls > 0


def test_public_adapter_saves_only_gated_conformance_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    engine = EngineV3(config, _registry())
    policy, _ = engine.train(dataset_for_config(config).source("train"))
    checkpoint = tmp_path / "checkpoint"
    marker = policy.save_checkpoint(checkpoint, metadata={"source": "fixture"})
    payload = json.loads(marker.read_text())
    assert payload["license_status"] == "unverified"
    assert payload["pretrained_enabled"] is False
    assert payload["optimizer_step_verified"] is False
    assert (checkpoint / "model/model.safetensors").is_file()
    assert (checkpoint / "processors/preprocessor.json").is_file()
    assert (checkpoint / "processors/postprocessor.json").is_file()


def test_installed_lerobot_exposes_only_used_public_methods() -> None:
    assert "batch" in inspect.signature(SmolVLAPolicy.forward).parameters
    assert "batch" in inspect.signature(SmolVLAPolicy.predict_action_chunk).parameters
    assert "config" in inspect.signature(make_smolvla_pre_post_processors).parameters
    assert callable(SmolVLAPolicy.save_pretrained)
