from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lunavla.v3 import (
    ArtifactHashRecordV1,
    CheckpointEnvelopeV4,
    CheckpointEnvelopeV4R2,
    ExperimentConfig,
    RunManifestV4,
    RunManifestV4R2,
    audit_episodes,
    split_episode_ids,
    verify_run_directory,
)
from lunavla.v3.artifacts import sha256_file
from lunavla.v3.data import episode_sha256
from lunavla.v3.engine import EngineV3, dataset_for_config, run_alpha
from lunavla.v3.fake_tasks import FakePointEnvV3, fake_feature_schema, make_fake_episodes


def _mapping(tmp_path: Path, *, task_id: str = "fake_pusht") -> dict[str, Any]:
    config_path = Path("configs/v3") / f"{task_id}_alpha.yaml"
    payload = ExperimentConfig.load(config_path).to_dict()
    payload["artifacts"]["output_dir"] = str(tmp_path / task_id)
    return payload


def test_split_and_data_audit_are_deterministic_and_disjoint() -> None:
    episodes = make_fake_episodes(task_id="fake_pusht", seed=42, episode_count=6, steps=3)
    first = split_episode_ids(episodes, seed=42)
    second = split_episode_ids(episodes, seed=42)
    assert first == second
    assert not (set(first["train"]) & set(first["validation"]))
    assert not (set(first["train"]) & set(first["test"]))
    audit = audit_episodes(episodes, feature_schema=fake_feature_schema("fake_pusht"), split=first)
    assert audit.episode_count == 6
    assert audit.transition_count == 18
    assert audit.sha256() == audit_episodes(
        episodes, feature_schema=fake_feature_schema("fake_pusht"), split=second
    ).sha256()


def test_episode_hash_covers_final_next_observation() -> None:
    episode = make_fake_episodes(task_id="fake_pusht", seed=42, episode_count=3, steps=2)[0]
    final = episode.transitions[-1]
    changed_state = {name: value.copy() for name, value in final.next_observation.state.items()}
    next(iter(changed_state.values()))[0] += 0.25
    changed_observation = replace(final.next_observation, state=changed_state)
    changed_transition = replace(final, next_observation=changed_observation)
    changed_episode = replace(episode, transitions=(*episode.transitions[:-1], changed_transition))
    assert episode_sha256(episode) != episode_sha256(changed_episode)


def test_episode_rejects_discontinuous_transition_chain() -> None:
    episode = make_fake_episodes(task_id="fake_pusht", seed=42, episode_count=3, steps=2)[0]
    second = episode.transitions[1]
    state = {name: value.copy() for name, value in second.observation.state.items()}
    next(iter(state.values()))[0] += 0.25
    changed = replace(second.observation, state=state)
    with pytest.raises(ValueError, match="previous next_observation"):
        replace(episode, transitions=(episode.transitions[0], replace(second, observation=changed)))


def test_data_audit_preserves_mixed_episode_id_types() -> None:
    episodes = make_fake_episodes(task_id="fake_pusht", seed=42, episode_count=3, steps=2)

    def with_id(episode: object, identifier: str | int) -> object:
        value = episode
        transitions = tuple(
            replace(
                item,
                observation=replace(item.observation, episode_id=identifier),
                next_observation=replace(item.next_observation, episode_id=identifier),
            )
            for item in value.transitions
        )
        return replace(value, episode_id=identifier, transitions=transitions)

    mixed = (with_id(episodes[0], 1), with_id(episodes[1], "1"), episodes[2])
    split = {"train": [1], "validation": ["1"], "test": [episodes[2].episode_id]}
    audit = audit_episodes(mixed, feature_schema=fake_feature_schema("fake_pusht"), split=split)
    assert len(audit.episode_hashes) == 3
    assert [item.to_dict()["episode_id_type"] for item in audit.episode_hashes[:2]] == [
        "integer",
        "string",
    ]


def test_engine_trains_only_the_declared_train_split(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(_mapping(tmp_path))
    bundle = dataset_for_config(config)
    assert len(bundle.episodes) == 6
    assert len(bundle.select("train")) == 4
    observations, _, _ = EngineV3._supervision(
        bundle.source("train").load(), int(config.policy["parameters"]["chunk_size"])
    )
    assert len(observations) == 4 * 5


def test_data_audit_rejects_split_overlap_and_schema_mismatch() -> None:
    episodes = make_fake_episodes(task_id="fake_pusht", seed=42, episode_count=3, steps=2)
    bad_split = {"train": [episodes[0].episode_id], "validation": [episodes[0].episode_id], "test": [episodes[2].episode_id]}
    with pytest.raises(ValueError, match="overlap"):
        audit_episodes(episodes, feature_schema=fake_feature_schema("fake_pusht"), split=bad_split)
    split = split_episode_ids(episodes, seed=42)
    with pytest.raises(ValueError, match="image order/names"):
        audit_episodes(episodes, feature_schema=fake_feature_schema("fake_libero"), split=split)


@pytest.mark.parametrize("task_id", ["fake_pusht", "fake_libero"])
def test_alpha_run_is_deterministic_and_verifiable(tmp_path: Path, task_id: str) -> None:
    config = ExperimentConfig.from_mapping(_mapping(tmp_path, task_id=task_id))
    first = run_alpha(config)
    first_hashes = {
        name: (first.output_dir / name).read_bytes()
        for name in (
            "checkpoint/policy/policy.json",
            "checkpoint/checkpoint.v3.json",
            "checkpoint/training_state.json",
            "policy_spec.json",
            "normalization.json",
            "dependency_lock.json",
            "model_source.json",
            "metrics.json",
            "data_audit.json",
        )
    }
    assert verify_run_directory(first.output_dir)["claim_allowed"] is False
    manifest = RunManifestV4R2.from_mapping(
        json.loads((first.output_dir / "manifest.json").read_text())
    )
    assert manifest.policy_id == config.policy["type"]
    second = run_alpha(config, overwrite=True)
    for name, value in first_hashes.items():
        assert (second.output_dir / name).read_bytes() == value
    assert first.metrics == second.metrics


def test_verify_run_detects_tampering(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(_mapping(tmp_path))
    result = run_alpha(config)
    metrics = result.output_dir / "metrics.json"
    payload = json.loads(metrics.read_text())
    payload["success_rate"] = 0.999
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics.json"):
        verify_run_directory(result.output_dir)


def test_verify_run_detects_nested_checkpoint_tampering(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(_mapping(tmp_path))
    result = run_alpha(config)
    processor = result.output_dir / "checkpoint/processors/processor.json"
    processor.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint/processors/processor.json"):
        verify_run_directory(result.output_dir)


def test_checkpoint_directory_restores_in_a_fresh_engine(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(_mapping(tmp_path))
    result = run_alpha(config)
    fresh = EngineV3(config)
    restored = fresh.restore_policy(result.output_dir / "checkpoint")
    metrics = fresh.evaluate(
        restored, FakePointEnvV3("fake_pusht", config.evaluation["max_steps"])
    )
    for name, value in metrics.items():
        assert value == result.metrics[name]


def test_output_directory_requires_explicit_overwrite(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(_mapping(tmp_path))
    run_alpha(config)
    with pytest.raises(FileExistsError, match="already exists"):
        run_alpha(config)


def test_overwrite_replaces_the_complete_run_generation(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(_mapping(tmp_path))
    result = run_alpha(config)
    stale = result.output_dir / "stale-checkpoint.json"
    stale.write_text("stale", encoding="utf-8")
    result = run_alpha(config, overwrite=True)
    assert not stale.exists()
    assert verify_run_directory(result.output_dir)["contract_revision"] == 2


def test_versioned_artifact_contracts_are_strict() -> None:
    envelope = CheckpointEnvelopeV4(
        "numpy_linear_chunk", "policy.json", "a" * 64, "b" * 64, "c" * 64
    ).to_dict()
    envelope["checkpoint_file"] = "../escape.json"
    with pytest.raises(ValueError, match="relative basename"):
        CheckpointEnvelopeV4.from_mapping(envelope)

    manifest = RunManifestV4(
        "a" * 40, False, "b" * 64, "c" * 64, "d" * 64, "e" * 64,
        "f" * 64, "numpy_linear_chunk", "fake_pusht", 1, (2,), True,
    ).to_dict()
    manifest["unknown"] = True
    with pytest.raises(ValueError, match="unknown field"):
        RunManifestV4.from_mapping(manifest)

    revision2 = CheckpointEnvelopeV4R2(
        "numpy_linear_chunk",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        (ArtifactHashRecordV1("policy/policy.json", "f" * 64),),
    ).to_dict()
    revision2["files"][0]["path"] = "../escape.json"
    with pytest.raises(ValueError, match="contained relative path"):
        CheckpointEnvelopeV4R2.from_mapping(revision2)


def test_revision1_run_artifacts_remain_read_only_compatible(tmp_path: Path) -> None:
    root = tmp_path / "revision1"
    root.mkdir()
    for name in ("resolved_config.json", "data_audit.json", "metrics.json"):
        (root / name).write_text("{}\n", encoding="utf-8")
    checkpoint = root / "policy.json"
    checkpoint.write_text('{"policy_id":"numpy_linear_chunk"}\n', encoding="utf-8")
    envelope = CheckpointEnvelopeV4(
        "numpy_linear_chunk",
        checkpoint.name,
        sha256_file(checkpoint),
        sha256_file(root / "resolved_config.json"),
        "a" * 64,
    )
    envelope_path = envelope.save(root / "checkpoint.v3.json")
    manifest = RunManifestV4(
        "a" * 40,
        False,
        sha256_file(root / "resolved_config.json"),
        "a" * 64,
        sha256_file(root / "data_audit.json"),
        sha256_file(envelope_path),
        sha256_file(root / "metrics.json"),
        "numpy_linear_chunk",
        "fake_pusht",
        1,
        (2,),
        True,
    )
    manifest.save(root / "manifest.json")
    assert verify_run_directory(root)["contract_revision"] == 1


def test_environment_closes_when_prediction_fails(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(_mapping(tmp_path))
    engine = EngineV3(config)
    env = FakePointEnvV3("fake_pusht", 4)

    class BrokenPolicy:
        spec = engine._policy_spec()

        def reset(self, seed: int) -> None:
            del seed

        def predict_chunk(self, observation: object) -> np.ndarray:
            del observation
            raise RuntimeError("prediction failed")

    with pytest.raises(RuntimeError, match="prediction failed"):
        engine.evaluate(BrokenPolicy(), env)  # type: ignore[arg-type]
    assert env.closed


def test_unknown_unused_modality_cannot_hide_image_drop(tmp_path: Path) -> None:
    payload = _mapping(tmp_path, task_id="fake_libero")
    payload = copy.deepcopy(payload)
    payload["policy"]["parameters"]["unused_modalities"] = []
    config = ExperimentConfig.from_mapping(payload)
    with pytest.raises(ValueError, match="silently discard image"):
        run_alpha(config)
