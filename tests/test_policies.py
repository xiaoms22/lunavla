from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from model.minivla_policy import NumpyLinearChunkPolicy
from model.policy_base import ActionChunk
from model.policy_bc import NumpyBCMLPPolicy
from model.policy_io import canonical_policy_type, load_policy


def test_action_chunk_normalizes_values_and_validates_shape() -> None:
    chunk = ActionChunk(values=np.array([[1, 2], [3, 4]]), valid_mask=np.array([1, 0]))
    assert chunk.values.shape == (2, 2)
    assert chunk.values.dtype == np.float32
    assert chunk.valid_mask.dtype == np.bool_

    with pytest.raises(ValueError, match=r"\[chunk, action\]"):
        ActionChunk(np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match="valid_mask"):
        ActionChunk(np.zeros((2, 2)), np.ones(3))
    with pytest.raises(ValueError, match="at least one"):
        ActionChunk(np.zeros((2, 2)), np.zeros(2))
    with pytest.raises(ValueError, match="NaN"):
        ActionChunk(np.array([[np.nan, 0.0]]), np.ones(1))


@pytest.mark.parametrize(
    "policy",
    [
        NumpyLinearChunkPolicy(input_dim=3, action_dim=2, chunk_size=2, seed=4),
        NumpyBCMLPPolicy(input_dim=3, action_dim=2, chunk_size=2, hidden_dim=5, seed=4),
    ],
)
def test_policy_predict_chunk_and_compatibility_action(policy: object) -> None:
    sample = np.array([0.2, -0.3, 0.5], dtype=np.float32)
    chunk = policy.predict_chunk(sample)  # type: ignore[attr-defined]
    assert isinstance(chunk, ActionChunk)
    assert chunk.values.shape == (2, 2)
    assert chunk.valid_mask.tolist() == [True, True]
    np.testing.assert_array_equal(
        policy.predict_action(sample),  # type: ignore[attr-defined]
        chunk.values[0],
    )


def test_masked_train_step_ignores_padded_targets() -> None:
    inputs = np.array([[1.0, -2.0], [0.5, 0.3]], dtype=np.float32)
    targets_a = np.zeros((2, 2, 2), dtype=np.float32)
    targets_b = targets_a.copy()
    targets_b[:, 1, :] = 10_000.0
    valid_mask = np.array([[True, False], [True, False]])
    policy_a = NumpyLinearChunkPolicy(2, action_dim=2, chunk_size=2, seed=22)
    policy_b = NumpyLinearChunkPolicy(2, action_dim=2, chunk_size=2, seed=22)

    loss_a = policy_a.train_step(inputs, targets_a, 0.01, valid_mask)
    loss_b = policy_b.train_step(inputs, targets_b, 0.01, valid_mask)

    assert loss_a == pytest.approx(loss_b)
    np.testing.assert_array_equal(policy_a.weights, policy_b.weights)
    np.testing.assert_array_equal(policy_a.bias, policy_b.bias)


@pytest.mark.parametrize(
    "policy",
    [
        NumpyLinearChunkPolicy(input_dim=3, action_dim=2, chunk_size=2, seed=9),
        NumpyBCMLPPolicy(input_dim=3, action_dim=2, chunk_size=2, hidden_dim=4, seed=9),
    ],
)
def test_checkpoint_round_trip(policy: object, tmp_path: Path) -> None:
    sample = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    before = policy.predict_chunk(sample).values  # type: ignore[attr-defined]
    checkpoint = policy.save_pretrained(tmp_path, {"purpose": "round-trip"})  # type: ignore[attr-defined]

    assert checkpoint.name == "checkpoint.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    loaded, metadata = load_policy(checkpoint)

    assert loaded.policy_name == policy.policy_name  # type: ignore[attr-defined]
    assert metadata == {"purpose": "round-trip"}
    np.testing.assert_array_equal(loaded.predict_chunk(sample).values, before)


def test_same_seed_produces_byte_identical_checkpoint(tmp_path: Path) -> None:
    hashes: list[str] = []
    for run_id in range(2):
        policy = NumpyLinearChunkPolicy(3, action_dim=2, chunk_size=2, seed=123)
        inputs = np.ones((2, 3), dtype=np.float32)
        targets = np.zeros((2, 2, 2), dtype=np.float32)
        policy.train_step(inputs, targets, learning_rate=0.05)
        checkpoint = policy.save_pretrained(tmp_path / f"run-{run_id}", {"seed": 123})
        hashes.append(hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    assert hashes[0] == hashes[1]


def test_legacy_json_checkpoint_with_pt_suffix_is_read_only_compatible(tmp_path: Path) -> None:
    legacy = tmp_path / "checkpoint.pt"
    legacy.write_text(
        json.dumps(
            {
                "policy_name": "tiny_linear",
                "input_dim": 2,
                "action_dim": 1,
                "chunk_size": 1,
                "weights": [[1.0], [2.0]],
                "bias": [0.5],
                "metadata": {"legacy": True},
            }
        ),
        encoding="utf-8",
    )

    policy, metadata = load_policy(legacy)
    assert metadata == {"legacy": True}
    np.testing.assert_allclose(policy.predict_action(np.array([1.0, 1.0])), [3.5])


def test_unknown_policy_is_never_silently_loaded(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "lunavla.numpy_policy",
                "policy": {"type": "not_a_policy"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="policy"):
        load_policy(checkpoint)


@pytest.mark.parametrize(
    ("inputs", "targets", "learning_rate", "message"),
    [
        (np.zeros((1, 3)), np.zeros((1, 2, 2)), 0.0, "learning_rate"),
        (np.zeros((1, 4)), np.zeros((1, 2, 2)), 0.1, "inputs"),
        (np.zeros((1, 3)), np.zeros((1, 3, 2)), 0.1, "targets"),
        (np.array([[np.nan, 0.0, 0.0]]), np.zeros((1, 2, 2)), 0.1, "NaN"),
    ],
)
def test_linear_policy_rejects_invalid_training_inputs(
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    message: str,
) -> None:
    policy = NumpyLinearChunkPolicy(3, action_dim=2, chunk_size=2)
    with pytest.raises(ValueError, match=message):
        policy.train_step(inputs, targets, learning_rate)


def test_bc_policy_training_reduces_tiny_batch_loss() -> None:
    policy = NumpyBCMLPPolicy(2, action_dim=1, chunk_size=1, hidden_dim=8, seed=5)
    inputs = np.array([[-1.0, 0.5], [0.0, 0.0], [1.0, -0.5]], dtype=np.float32)
    targets = np.array([[[-0.7]], [[0.0]], [[0.7]]], dtype=np.float32)
    initial = policy.forward({"inputs": inputs, "targets": targets})["loss"]
    for _ in range(80):
        policy.train_step(inputs, targets, learning_rate=0.1)
    final = policy.forward({"inputs": inputs, "targets": targets})["loss"]
    assert final < initial * 0.1


@pytest.mark.parametrize("policy_class", [NumpyLinearChunkPolicy, NumpyBCMLPPolicy])
def test_checkpoint_schema_and_suffix_errors(policy_class: type, tmp_path: Path) -> None:
    policy = policy_class(input_dim=2, action_dim=1, chunk_size=1)
    with pytest.raises(ValueError, match="checkpoint.json"):
        policy.save_pretrained(tmp_path / "checkpoint.pt")

    valid = policy.save_pretrained(tmp_path / "valid")
    payload = json.loads(valid.read_text(encoding="utf-8"))
    for field, value, message in (
        ("schema_version", 99, "schema_version"),
        ("format", "wrong", "format"),
    ):
        broken = dict(payload)
        broken[field] = value
        path = tmp_path / f"broken-{field}.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            policy_class.load(path)


def test_policy_loader_reports_malformed_checkpoints(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        load_policy(malformed)

    missing_type = tmp_path / "missing-type.json"
    missing_type.write_text(json.dumps({"schema_version": 1, "policy": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="policy.type"):
        load_policy(missing_type)

    legacy_missing_type = tmp_path / "legacy.pt"
    legacy_missing_type.write_text(json.dumps({"input_dim": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="policy_name"):
        load_policy(legacy_missing_type)

    empty_directory = tmp_path / "empty-directory"
    empty_directory.mkdir()
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        load_policy(empty_directory)


def test_policy_aliases_warn_and_preserve_canonical_names() -> None:
    assert canonical_policy_type("numpy_linear_chunk") == "numpy_linear_chunk"
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert canonical_policy_type("act") == "numpy_linear_chunk"


def test_policy_constructor_rejects_bad_dimensions_and_parameters() -> None:
    with pytest.raises(ValueError, match="positive"):
        NumpyLinearChunkPolicy(input_dim=0)
    with pytest.raises(ValueError, match="weights"):
        NumpyLinearChunkPolicy(input_dim=2, weights=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="bias"):
        NumpyLinearChunkPolicy(input_dim=2, bias=np.zeros(1))
    with pytest.raises(ValueError, match="NaN"):
        NumpyLinearChunkPolicy(input_dim=1, weights=np.array([[np.nan, 0.0]]))
    with pytest.raises(ValueError, match="dimensions"):
        NumpyBCMLPPolicy(input_dim=2, hidden_dim=0)
    with pytest.raises(ValueError, match="w1"):
        NumpyBCMLPPolicy(
            input_dim=2,
            weights={
                "w1": np.zeros((1, 32)),
                "b1": np.zeros(32),
                "w2": np.zeros((32, 2)),
                "b2": np.zeros(2),
            },
        )


@pytest.mark.parametrize(
    "policy",
    [
        NumpyLinearChunkPolicy(input_dim=2, action_dim=1, chunk_size=1),
        NumpyBCMLPPolicy(input_dim=2, action_dim=1, chunk_size=1, hidden_dim=3),
    ],
)
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("schema_version", True), "schema_version"),
        (lambda payload: payload.__setitem__("unexpected", 1), "unknown checkpoint root"),
        (lambda payload: payload.pop("metadata"), "missing checkpoint root"),
        (
            lambda payload: payload["policy"].__setitem__("unexpected", 1),
            "unknown checkpoint policy",
        ),
        (
            lambda payload: payload["policy"].__setitem__("input_dim", True),
            "positive integer",
        ),
        (
            lambda payload: payload["policy"]["parameters"].__setitem__(
                "unexpected", []
            ),
            "unknown checkpoint policy.parameters",
        ),
    ],
)
def test_numpy_schema1_rejects_root_policy_and_parameter_drift(
    policy: object,
    mutation: object,
    message: str,
    tmp_path: Path,
) -> None:
    checkpoint = policy.save_pretrained(tmp_path / "strict")  # type: ignore[attr-defined]
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    mutation(payload)  # type: ignore[operator]
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=message):
        load_policy(checkpoint)


@pytest.mark.parametrize(
    "invalid",
    [float("nan"), float("inf"), True, "not-numeric"],
)
def test_numpy_schema1_rejects_nonfinite_or_non_numeric_parameters(
    invalid: object, tmp_path: Path
) -> None:
    policy = NumpyLinearChunkPolicy(input_dim=2, action_dim=1, chunk_size=1)
    checkpoint = policy.save_pretrained(tmp_path / "strict-parameters")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["policy"]["parameters"]["weights"][0][0] = invalid
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match="numeric|finite"):
        load_policy(checkpoint)


def test_numpy_checkpoint_metadata_is_strict_on_read_and_write(tmp_path: Path) -> None:
    policy = NumpyLinearChunkPolicy(input_dim=2, action_dim=1, chunk_size=1)
    with pytest.raises(ValueError, match="finite"):
        policy.save_pretrained(tmp_path / "nan", {"value": float("nan")})
    with pytest.raises(TypeError, match="unsupported value type"):
        policy.save_pretrained(tmp_path / "path", {"value": tmp_path})

    checkpoint = policy.save_pretrained(tmp_path / "read")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["metadata"] = {"value": float("inf")}
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_policy(checkpoint)


def test_legacy_numpy_checkpoint_is_strict_read_only_input_and_resaves_schema1(
    tmp_path: Path,
) -> None:
    legacy_payload = {
        "policy_name": "tiny_linear",
        "input_dim": 2,
        "action_dim": 1,
        "chunk_size": 1,
        "weights": [[1.0], [2.0]],
        "bias": [0.5],
        "metadata": {"legacy": True},
    }
    legacy = tmp_path / "legacy.pt"
    legacy.write_text(json.dumps(legacy_payload), encoding="utf-8")
    policy, metadata = load_policy(legacy)
    upgraded = policy.save_pretrained(tmp_path / "upgraded", metadata)
    upgraded_payload = json.loads(upgraded.read_text(encoding="utf-8"))
    assert upgraded_payload["schema_version"] == 1
    assert upgraded_payload["format"] == "lunavla.numpy_policy"

    legacy_payload["unexpected"] = True
    legacy.write_text(json.dumps(legacy_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown legacy checkpoint"):
        load_policy(legacy)

    root_list = tmp_path / "root-list.json"
    root_list.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="root must be an object"):
        load_policy(root_list)
