from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lunavla.v3 import ExperimentConfig, audit_episodes, split_episode_ids, verify_run_directory
from lunavla.v3.data import episode_sha256
from lunavla.v3.engine import EngineV3, run_alpha
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
        for name in ("policy.json", "checkpoint.v3.json", "metrics.json", "data_audit.json")
    }
    assert verify_run_directory(first.output_dir)["claim_allowed"] is False
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


def test_output_directory_requires_explicit_overwrite(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(_mapping(tmp_path))
    run_alpha(config)
    with pytest.raises(FileExistsError, match="already exists"):
        run_alpha(config)


def test_environment_closes_when_prediction_fails(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(_mapping(tmp_path))
    engine = EngineV3(config)
    env = FakePointEnvV3("fake_pusht", 4)

    class BrokenPolicy:
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
