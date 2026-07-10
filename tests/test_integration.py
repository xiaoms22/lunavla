from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from dataset import build_training_arrays, generate_mock_pusht_records, split_records_by_episode
from eval_vla import rollout_episode
from model import NumpyLinearChunkPolicy, load_policy
from trainer.train_core import train_from_config


def test_minimal_cpu_training_checkpoint_and_rollout(tmp_path: Path) -> None:
    records = generate_mock_pusht_records(
        num_episodes=5,
        steps_per_episode=4,
        seed=21,
        action_noise_std=0.0,
    )
    splits = split_records_by_episode(records, seed=22)
    arrays = build_training_arrays(splits["train"], chunk_size=2, instruction_dim=8)
    policy = NumpyLinearChunkPolicy(
        input_dim=arrays.inputs.shape[1],
        action_dim=arrays.targets.shape[2],
        chunk_size=arrays.targets.shape[1],
        seed=23,
    )
    initial_loss = policy.forward(
        {
            "inputs": arrays.inputs,
            "targets": arrays.targets,
            "valid_mask": arrays.valid_mask,
        }
    )["loss"]
    for _ in range(20):
        policy.train_step(
            arrays.inputs,
            arrays.targets,
            learning_rate=0.02,
            valid_mask=arrays.valid_mask,
        )
    final_loss = policy.forward(
        {
            "inputs": arrays.inputs,
            "targets": arrays.targets,
            "valid_mask": arrays.valid_mask,
        }
    )["loss"]
    checkpoint = policy.save_pretrained(
        tmp_path,
        {
            "dataset": {"language_instruction": "push the T block to the goal"},
            "eval": {"execution_mode": "open_loop_chunk"},
        },
    )
    loaded, metadata = load_policy(checkpoint)
    rollout = rollout_episode(
        loaded,
        seed=24,
        rollout_steps=3,
        success_distance=0.1,
        instruction=metadata["dataset"]["language_instruction"],
        execution_mode=metadata["eval"]["execution_mode"],
    )

    assert final_loss < initial_loss
    assert checkpoint.name == "checkpoint.json"
    assert rollout["steps"] >= 1
    assert rollout["execution_mode"] == "open_loop_chunk"


def engine_config(output_dir: Path, policy_type: str) -> dict[str, object]:
    policy: dict[str, object] = {
        "type": policy_type,
        "chunk_size": 2,
        "observation_dim": 4,
        "instruction_dim": 8,
        "action_dim": 2,
    }
    if policy_type == "numpy_bc_mlp":
        policy["hidden_dim"] = 4
    return {
        "schema_version": 1,
        "project_name": f"engine-{policy_type}",
        "framework": "lunavla",
        "task": "pusht_style_point_reach",
        "policy": policy,
        "dataset": {
            "source": "mock_pusht",
            "num_episodes": 5,
            "steps_per_episode": 3,
            "seed": 41,
            "split_seed": 42,
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "split": "train",
            "action_noise_std": 0.0,
        },
        "training": {
            "device": "cpu",
            "batch_size": 4,
            "num_steps": 3,
            "learning_rate": 0.02,
            "seed": 43,
            "log_interval": 1,
        },
        "eval": {
            "episodes": 1,
            "rollout_steps": 2,
            "seed": 44,
            "execution_mode": "open_loop_chunk",
        },
        "artifacts": {"output_dir": str(output_dir)},
    }


@pytest.mark.parametrize("policy_type", ["numpy_linear_chunk", "numpy_bc_mlp"])
def test_configured_training_engine_writes_complete_artifacts(
    tmp_path: Path, policy_type: str
) -> None:
    config_path = tmp_path / f"{policy_type}.yaml"
    output_dir = tmp_path / f"run-{policy_type}"
    config_path.write_text(
        yaml.safe_dump(engine_config(output_dir, policy_type), sort_keys=False),
        encoding="utf-8",
    )
    summary = train_from_config(
        config_path,
        expected_policy_type=policy_type,
    )

    assert summary["policy_name"] == policy_type
    assert summary["records"] > 0
    assert (output_dir / "checkpoint.json").is_file()
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "config.resolved.json").is_file()
    assert (output_dir / "train_records.jsonl").is_file()
    assert (output_dir / "validation_records.jsonl").is_file()
    assert (output_dir / "test_records.jsonl").is_file()

    with pytest.raises(FileExistsError, match="--overwrite"):
        train_from_config(config_path, expected_policy_type=policy_type)


def test_training_engine_is_deterministic_when_explicitly_rebuilt(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(engine_config(output_dir, "numpy_linear_chunk"), sort_keys=False),
        encoding="utf-8",
    )
    train_from_config(config_path)
    first_checkpoint_hash = hashlib.sha256(
        (output_dir / "checkpoint.json").read_bytes()
    ).hexdigest()
    first_data_hash = hashlib.sha256((output_dir / "train_records.jsonl").read_bytes()).hexdigest()

    train_from_config(config_path, overwrite=True)
    assert hashlib.sha256((output_dir / "checkpoint.json").read_bytes()).hexdigest() == (
        first_checkpoint_hash
    )
    assert hashlib.sha256((output_dir / "train_records.jsonl").read_bytes()).hexdigest() == (
        first_data_hash
    )


def test_training_entry_point_rejects_wrong_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            engine_config(tmp_path / "run", "numpy_bc_mlp"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="entry point requires"):
        train_from_config(config_path, expected_policy_type="numpy_linear_chunk")
