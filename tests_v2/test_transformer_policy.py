from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lunavla.contracts import Observation, PolicyBatch
from model.policy_base import ActionChunk


torch = pytest.importorskip("torch")
transformer_policy = pytest.importorskip("lunavla.transformer_policy")
pytestmark = pytest.mark.torch


TransformerChunkCVAEPolicy = transformer_policy.TransformerChunkCVAEPolicy
TransformerPolicyConfig = transformer_policy.TransformerPolicyConfig
TemporalEnsembler = transformer_policy.TemporalEnsembler
load_transformer_policy = transformer_policy.load_transformer_policy


def tiny_config(**overrides: object) -> object:
    values: dict[str, object] = {
        "state_dim": 3,
        "action_dim": 2,
        "chunk_size": 3,
        "d_model": 16,
        "nhead": 4,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 32,
        "latent_dim": 4,
        "dropout": 0.0,
        "kl_weight": 1e-3,
        "sample_latent_during_training": False,
        "seed": 7,
        "device": "cpu",
    }
    values.update(overrides)
    return TransformerPolicyConfig.from_mapping(values)


def tensor_batch() -> tuple[object, object, object]:
    states = torch.tensor(
        [[0.1, 0.2, 0.3], [0.7, 0.5, 0.2]],
        dtype=torch.float32,
    )
    actions = torch.tensor(
        [
            [[0.1, -0.1], [0.2, -0.2], [9.0, 9.0]],
            [[-0.3, 0.3], [-0.2, 0.2], [-0.1, 0.1]],
        ],
        dtype=torch.float32,
    )
    valid_mask = torch.tensor([[True, True, False], [True, True, True]])
    return states, actions, valid_mask


def test_act_alias_requires_all_four_capabilities() -> None:
    required = {
        "action_query_transformer",
        "conditional_cvae_kl",
        "valid_mask_loss",
        "temporal_ensembling",
    }
    assert transformer_policy.POLICY_ID == "transformer_chunk_cvae"
    assert required <= transformer_policy.CAPABILITIES
    assert transformer_policy.act_alias_supported is True
    assert TransformerChunkCVAEPolicy.act_alias_supported is True


def test_module_registration_gates_act_alias_and_loader_round_trip(tmp_path: Path) -> None:
    from lunavla.registry import PolicyRegistry

    policies = PolicyRegistry()
    transformer_policy.register_transformer_policy(policies)
    assert policies.resolve("transformer_chunk") == "transformer_chunk_cvae"
    assert policies.resolve("act") == "transformer_chunk_cvae"

    config = tiny_config()
    policy = policies.create("act", vars(config))
    observation = Observation(np.asarray([0.2, 0.3, 0.4], dtype=np.float32))
    expected = policy.predict_chunk(observation)
    checkpoint = policy.save_checkpoint(tmp_path / "registered.pt")
    restored = policies.load_checkpoint(
        checkpoint,
        policy_id="transformer_chunk_cvae",
        config={"device": "cpu"},
    )
    actual = restored.predict_chunk(observation)
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-7)


def test_action_query_cvae_shapes_reparameterization_and_kl() -> None:
    policy = TransformerChunkCVAEPolicy(tiny_config())
    states, actions, valid_mask = tensor_batch()

    mu, logvar = policy.encode_posterior(states, actions, valid_mask)
    deterministic = policy.reparameterize(mu, logvar, sample=False)
    sampled = policy.reparameterize(mu, logvar, sample=True)
    losses = policy.compute_loss(states, actions, valid_mask, sample_latent=False)

    assert isinstance(policy.posterior_encoder, torch.nn.TransformerEncoder)
    assert isinstance(policy.action_decoder, torch.nn.TransformerDecoder)
    assert mu.shape == logvar.shape == (2, 4)
    assert torch.equal(deterministic, mu)
    assert sampled.shape == mu.shape
    assert losses["predicted_actions"].shape == (2, 3, 2)
    assert losses["kl_loss"].item() >= 0.0
    assert torch.isfinite(losses["loss"])


def test_padding_values_do_not_affect_masked_loss_or_posterior() -> None:
    policy = TransformerChunkCVAEPolicy(tiny_config())
    states, actions, valid_mask = tensor_batch()
    changed_padding = actions.clone()
    changed_padding[0, 2] = torch.tensor([-1e30, 1e30])

    first = policy.compute_loss(states, actions, valid_mask, sample_latent=False)
    second = policy.compute_loss(states, changed_padding, valid_mask, sample_latent=False)

    assert torch.allclose(first["mu"], second["mu"], atol=1e-6)
    assert torch.allclose(first["logvar"], second["logvar"], atol=1e-6)
    assert torch.allclose(
        first["reconstruction_loss"],
        second["reconstruction_loss"],
        atol=1e-6,
    )
    assert torch.isfinite(second["reconstruction_loss"])


def test_cpu_shape_dtype_and_declared_device_checks() -> None:
    policy = TransformerChunkCVAEPolicy(tiny_config())
    states, actions, valid_mask = tensor_batch()

    with pytest.raises(TypeError, match="torch.float32"):
        policy.compute_loss(states.to(torch.float64), actions, valid_mask)
    with pytest.raises(ValueError, match="states must have shape"):
        policy.compute_loss(states[:, :2], actions, valid_mask)
    with pytest.raises(TypeError, match="torch.bool"):
        policy.compute_loss(states, actions, valid_mask.to(torch.float32))

    observations = tuple(Observation(row.numpy()) for row in states)
    mismatched_device = PolicyBatch(
        observations=observations,
        targets=actions.numpy(),
        valid_mask=valid_mask.numpy(),
        device="cuda",
    )
    with pytest.raises(ValueError, match="batch declares device cuda:0"):
        policy.train_batch(mismatched_device, learning_rate=1e-3)


def test_tiny_batch_overfits_through_public_train_batch() -> None:
    policy = TransformerChunkCVAEPolicy(tiny_config(kl_weight=0.0))
    states, actions, valid_mask = tensor_batch()
    batch = PolicyBatch(
        observations=tuple(Observation(row.numpy()) for row in states),
        targets=actions.numpy(),
        valid_mask=valid_mask.numpy(),
        device="cpu",
    )
    with torch.no_grad():
        initial = policy.compute_loss(
            states,
            actions,
            valid_mask,
            sample_latent=False,
        )["reconstruction_loss"].item()

    for _ in range(160):
        policy.train_batch(batch, learning_rate=5e-3)

    with torch.no_grad():
        final = policy.compute_loss(
            states,
            actions,
            valid_mask,
            sample_latent=False,
        )["reconstruction_loss"].item()
    assert final < initial * 0.1


def test_training_seed_reproduces_losses_weights_and_checkpoint_bytes(tmp_path: Path) -> None:
    states, actions, valid_mask = tensor_batch()
    batch = PolicyBatch(
        observations=tuple(Observation(row.numpy()) for row in states),
        targets=actions.numpy(),
        valid_mask=valid_mask.numpy(),
        device="cpu",
    )
    first = TransformerChunkCVAEPolicy(
        tiny_config(sample_latent_during_training=True, dropout=0.2)
    )
    second = TransformerChunkCVAEPolicy(
        tiny_config(sample_latent_during_training=True, dropout=0.2)
    )
    first_losses = [first.train_batch(batch, learning_rate=1e-3) for _ in range(4)]
    second_losses = [second.train_batch(batch, learning_rate=1e-3) for _ in range(4)]
    assert first_losses == second_losses
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    checkpoint = tmp_path / "checkpoint.pt"
    first.save_checkpoint(checkpoint)
    first_bytes = checkpoint.read_bytes()
    second.save_checkpoint(checkpoint)
    assert first_bytes == checkpoint.read_bytes()


def test_instruction_condition_changes_prediction_for_same_state() -> None:
    policy = TransformerChunkCVAEPolicy(tiny_config(instruction_dim=32))
    state = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    move_left = Observation(state, instruction="move the block to the left target")
    move_right = Observation(state, instruction="move the block to the right target")

    left_features = policy.instruction_features(move_left.instruction, 32)
    right_features = policy.instruction_features(move_right.instruction, 32)
    left_chunk = policy.predict_chunk(move_left)
    right_chunk = policy.predict_chunk(move_right)

    assert not np.array_equal(left_features, right_features)
    assert not np.allclose(left_chunk.values, right_chunk.values)


def test_explicit_instruction_mask_has_zero_features() -> None:
    features = TransformerChunkCVAEPolicy.instruction_features("[MASK]", 16)
    np.testing.assert_array_equal(features, np.zeros(16, dtype=np.float32))


def test_image_condition_changes_prediction_for_same_state() -> None:
    policy = TransformerChunkCVAEPolicy(tiny_config(image_shape=(8, 8, 3)))
    state = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    dark = Observation(state, image=np.zeros((8, 8, 3), dtype=np.uint8))
    bright = Observation(state, image=np.full((8, 8, 3), 255, dtype=np.uint8))

    dark_chunk = policy.predict_chunk(dark)
    bright_chunk = policy.predict_chunk(bright)

    assert not np.allclose(dark_chunk.values, bright_chunk.values)


def test_enabled_and_disabled_modalities_fail_explicitly() -> None:
    state = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    instruction_policy = TransformerChunkCVAEPolicy(tiny_config(instruction_dim=16))
    with pytest.raises(ValueError, match="missing the enabled instruction"):
        instruction_policy.predict_chunk(Observation(state))

    image_policy = TransformerChunkCVAEPolicy(tiny_config(image_shape=(8, 8)))
    with pytest.raises(ValueError, match="missing the enabled image"):
        image_policy.predict_chunk(Observation(state))
    with pytest.raises(ValueError, match="image must have shape"):
        image_policy.predict_chunk(
            Observation(state, image=np.zeros((7, 8), dtype=np.uint8))
        )

    state_only_policy = TransformerChunkCVAEPolicy(tiny_config())
    with pytest.raises(ValueError, match="supplies instruction"):
        state_only_policy.predict_chunk(Observation(state, instruction="move left"))
    with pytest.raises(ValueError, match="supplies image"):
        state_only_policy.predict_chunk(
            Observation(state, image=np.zeros((8, 8), dtype=np.uint8))
        )


def test_multimodal_batch_training_and_checkpoint_round_trip(tmp_path: Path) -> None:
    policy = TransformerChunkCVAEPolicy(
        tiny_config(instruction_dim=16, image_shape=(8, 8, 3), kl_weight=0.0)
    )
    observations = (
        Observation(
            np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            instruction="reach the red goal",
            image=np.zeros((8, 8, 3), dtype=np.uint8),
        ),
        Observation(
            np.asarray([0.7, 0.5, 0.2], dtype=np.float32),
            instruction="reach the blue goal",
            image=np.full((8, 8, 3), 255, dtype=np.uint8),
        ),
    )
    targets = np.asarray(
        [
            [[0.1, -0.1], [0.2, -0.2], [0.0, 0.0]],
            [[-0.3, 0.3], [-0.2, 0.2], [-0.1, 0.1]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.asarray([[True, True, False], [True, True, True]], dtype=bool)
    batch = PolicyBatch(observations, targets, valid_mask, device="cpu")

    loss = policy.train_batch(batch, learning_rate=1e-3)
    expected = policy.predict_chunk(observations[0])
    checkpoint = policy.save_checkpoint(tmp_path / "multimodal.pt")
    restored = TransformerChunkCVAEPolicy.load_checkpoint(checkpoint)
    actual = restored.predict_chunk(observations[0])

    assert math.isfinite(loss)
    assert restored.config.instruction_dim == 16
    assert restored.config.image_shape == (8, 8, 3)
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-7)


def test_predict_chunk_and_schema_checkpoint_round_trip(tmp_path: Path) -> None:
    policy = TransformerChunkCVAEPolicy(tiny_config())
    observation = Observation(np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    expected = policy.predict_chunk(observation)
    checkpoint = policy.save_checkpoint(
        tmp_path / "policy.pt",
        metadata={"test": "round-trip"},
    )

    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = TransformerChunkCVAEPolicy.load_checkpoint(checkpoint)
    actual = restored.predict_chunk(observation)

    assert raw["schema_version"] == 2
    assert raw["format"] == "lunavla.transformer_chunk_cvae"
    assert raw["policy_id"] == "transformer_chunk_cvae"
    assert restored.checkpoint_metadata == {"test": "round-trip"}
    assert isinstance(actual, ActionChunk)
    np.testing.assert_array_equal(actual.valid_mask, expected.valid_mask)
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-7)


def test_checkpoint_resumes_optimizer_rng_and_train_step_exactly(tmp_path: Path) -> None:
    states, actions, valid_mask = tensor_batch()
    batch = PolicyBatch(
        observations=tuple(Observation(row.numpy()) for row in states),
        targets=actions.numpy(),
        valid_mask=valid_mask.numpy(),
        device="cpu",
    )
    policy = TransformerChunkCVAEPolicy(
        tiny_config(dropout=0.2, sample_latent_during_training=True)
    )
    for _ in range(3):
        policy.train_batch(batch, learning_rate=2e-3)
    restored = TransformerChunkCVAEPolicy.load_checkpoint(
        policy.save_checkpoint(tmp_path / "resume.pt")
    )
    expected_loss = policy.train_batch(batch, learning_rate=2e-3)
    actual_loss = restored.train_batch(batch, learning_rate=2e-3)
    assert actual_loss == expected_loss
    for name, value in policy.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_checkpoint_metadata_is_safe_and_image_shape_list_is_equivalent(tmp_path: Path) -> None:
    policy = TransformerChunkCVAEPolicy(tiny_config(image_shape=(8, 8, 3)))
    checkpoint = policy.save_checkpoint(tmp_path / "image.pt", metadata={"safe": [1, 2]})
    restored = load_transformer_policy(
        checkpoint,
        {"device": "cpu", "image_shape": [8, 8, 3]},
    )
    assert restored.config.image_shape == (8, 8, 3)
    with pytest.raises(TypeError, match="unsupported value type"):
        policy.save_checkpoint(tmp_path / "unsafe.pt", metadata={"path": tmp_path})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_device_is_canonical_and_public_batch_trains() -> None:
    policy = TransformerChunkCVAEPolicy(tiny_config(device="mps"))
    states, actions, valid_mask = tensor_batch()
    batch = PolicyBatch(
        observations=tuple(Observation(row.numpy()) for row in states),
        targets=actions.numpy(),
        valid_mask=valid_mask.numpy(),
        device="mps",
    )
    assert policy.device == "mps:0"
    assert math.isfinite(policy.train_batch(batch, learning_rate=1e-3))


def test_checkpoint_rejects_unknown_schema(tmp_path: Path) -> None:
    policy = TransformerChunkCVAEPolicy(tiny_config())
    checkpoint = policy.save_checkpoint(tmp_path / "policy.pt")
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    raw["schema_version"] = 999
    torch.save(raw, checkpoint)

    with pytest.raises(ValueError, match="schema_version"):
        TransformerChunkCVAEPolicy.load_checkpoint(checkpoint)


def test_temporal_ensembler_aligns_and_exponentially_weights_chunks() -> None:
    ensembler = TemporalEnsembler(decay=math.log(2.0), action_dim=1, chunk_size=3)
    first = ActionChunk(
        values=np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        valid_mask=np.ones(3, dtype=bool),
    )
    second = ActionChunk(
        values=np.asarray([[10.0], [11.0], [12.0]], dtype=np.float32),
        valid_mask=np.ones(3, dtype=bool),
    )

    np.testing.assert_allclose(ensembler.update(first), [0.0])
    # At t=1, the old prediction is 1 with weight 0.5 and the new prediction is 10 with weight 1.
    np.testing.assert_allclose(ensembler.update(second), [7.0], atol=1e-6)
    assert ensembler.step == 2
    assert ensembler.history_size == 2

    ensembler.reset()
    assert ensembler.step == 0
    assert ensembler.history_size == 0
