from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class VLADatum:
    observation: list[float]
    action: list[float]
    episode_id: int
    timestep: int
    success: bool
    language_instruction: str | None
    metadata: dict[str, Any]


def generate_mock_pusht_records(
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    language_instruction: str | None = "push the T block to the goal",
) -> list[VLADatum]:
    rng = np.random.default_rng(seed)
    goal = np.array([0.80, 0.20], dtype=np.float32)
    records: list[VLADatum] = []

    for episode_id in range(num_episodes):
        position = rng.uniform(low=0.05, high=0.95, size=2).astype(np.float32)
        for timestep in range(steps_per_episode):
            observation = position.copy()
            delta = goal - position
            expert_action = np.clip(delta * 0.35, -0.12, 0.12)
            expert_action += rng.normal(0.0, 0.004, size=2).astype(np.float32)
            position = np.clip(position + expert_action, 0.0, 1.0)
            distance = float(np.linalg.norm(goal - position))
            records.append(
                VLADatum(
                    observation=[
                        float(observation[0]),
                        float(observation[1]),
                        float(goal[0]),
                        float(goal[1]),
                    ],
                    action=[float(expert_action[0]), float(expert_action[1])],
                    episode_id=episode_id,
                    timestep=timestep,
                    success=distance < 0.08,
                    language_instruction=language_instruction,
                    metadata={"task": "pusht_mock", "distance_to_goal": distance},
                )
            )

    return records


def save_jsonl(records: Iterable[VLADatum], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[VLADatum]:
    records: list[VLADatum] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(VLADatum(**item))
    return records


def load_dataset_from_config(dataset_config: dict[str, Any]) -> list[VLADatum]:
    source = dataset_config.get("source", "mock_pusht")
    if source == "mock_pusht":
        return generate_mock_pusht_records(
            num_episodes=int(dataset_config.get("num_episodes", 16)),
            steps_per_episode=int(dataset_config.get("steps_per_episode", 32)),
            seed=int(dataset_config.get("seed", 42)),
            language_instruction=dataset_config.get(
                "language_instruction", "push the T block to the goal"
            ),
        )

    if source == "jsonl":
        data_path = dataset_config.get("path")
        if not data_path:
            raise ValueError("dataset.source=jsonl requires dataset.path")
        return load_jsonl(data_path)

    raise ValueError(f"Unsupported dataset source: {source}")


def _instruction_features(text: str | None, dim: int = 8) -> np.ndarray:
    features = np.zeros(dim, dtype=np.float32)
    if not text:
        return features
    for token in text.lower().split():
        idx = sum(ord(ch) for ch in token) % dim
        features[idx] += 1.0
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features


def build_training_batch(
    records: list[VLADatum],
    chunk_size: int,
    instruction_dim: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    by_episode: dict[int, list[VLADatum]] = {}
    for record in records:
        by_episode.setdefault(record.episode_id, []).append(record)

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for episode_records in by_episode.values():
        episode_records.sort(key=lambda item: item.timestep)
        actions = [np.asarray(item.action, dtype=np.float32) for item in episode_records]
        for idx, record in enumerate(episode_records):
            observation = np.asarray(record.observation, dtype=np.float32)
            instruction = _instruction_features(record.language_instruction, instruction_dim)
            chunk: list[np.ndarray] = []
            for offset in range(chunk_size):
                action_idx = min(idx + offset, len(actions) - 1)
                chunk.append(actions[action_idx])
            inputs.append(np.concatenate([observation, instruction]))
            targets.append(np.concatenate(chunk))

    return np.vstack(inputs).astype(np.float32), np.vstack(targets).astype(np.float32)
