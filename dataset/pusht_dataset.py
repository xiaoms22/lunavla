from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt

from dataset.task_context import build_pusht_task_context
from dataset.vla_dataset import (
    VLARecord,
    build_training_arrays,
    instruction_features,
    split_records_by_episode,
    validate_records,
)


VLADatum = VLARecord
Float32Array = npt.NDArray[np.float32]


def generate_mock_pusht_records(
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    language_instruction: str | None = "push the T block to the goal",
    goal: list[float] | tuple[float, float] = (0.80, 0.20),
    start_low: float = 0.05,
    start_high: float = 0.95,
    action_gain: float = 0.35,
    action_clip: float = 0.12,
    action_noise_std: float = 0.004,
    success_distance: float = 0.08,
) -> list[VLARecord]:
    if num_episodes <= 0 or steps_per_episode <= 0:
        raise ValueError("num_episodes and steps_per_episode must be positive")
    goal_array = np.asarray(goal, dtype=np.float32)
    if goal_array.shape != (2,) or not np.all(np.isfinite(goal_array)):
        raise ValueError("goal must contain two finite coordinates")
    if not 0 <= start_low < start_high <= 1:
        raise ValueError("start range must satisfy 0 <= start_low < start_high <= 1")
    if action_clip <= 0 or action_noise_std < 0 or success_distance <= 0:
        raise ValueError("action_clip/success_distance must be positive and noise non-negative")

    rng = np.random.default_rng(seed)
    records: list[VLARecord] = []
    for episode_id in range(num_episodes):
        position = rng.uniform(low=start_low, high=start_high, size=2).astype(np.float32)
        for timestep in range(steps_per_episode):
            observation_position = position.copy()
            current_distance = float(np.linalg.norm(goal_array - observation_position))
            task_context = build_pusht_task_context(
                position=observation_position,
                goal=goal_array,
                instruction=language_instruction,
                success_distance=success_distance,
            )
            delta = goal_array - observation_position
            expert_action = np.clip(delta * action_gain, -action_clip, action_clip)
            expert_action += rng.normal(0.0, action_noise_std, size=2).astype(np.float32)
            next_position = np.clip(observation_position + expert_action, 0.0, 1.0)
            next_distance = float(np.linalg.norm(goal_array - next_position))
            success = next_distance < success_distance
            terminated = success or timestep == steps_per_episode - 1
            observation = [
                float(observation_position[0]),
                float(observation_position[1]),
                float(goal_array[0]),
                float(goal_array[1]),
            ]
            next_observation = [
                float(next_position[0]),
                float(next_position[1]),
                float(goal_array[0]),
                float(goal_array[1]),
            ]
            records.append(
                VLARecord(
                    observation=observation,
                    next_observation=next_observation,
                    action=[float(expert_action[0]), float(expert_action[1])],
                    episode_id=episode_id,
                    timestep=timestep,
                    success=success,
                    terminated=terminated,
                    language_instruction=language_instruction,
                    metadata={
                        "task": "pusht_style_point_reach",
                        "distance_to_goal": current_distance,
                        "next_distance_to_goal": next_distance,
                        "generation": {
                            "start_low": start_low,
                            "start_high": start_high,
                            "action_gain": action_gain,
                            "action_clip": action_clip,
                            "action_noise_std": action_noise_std,
                            "success_distance": success_distance,
                        },
                        "task_context": task_context.to_dict(),
                    },
                    task_id=task_context.task_id,
                    subtask_id=task_context.subtask_id,
                    phase=task_context.phase,
                )
            )
            position = next_position
            if terminated:
                break
    validate_records(records, observation_dim=4, action_dim=2)
    return records


def save_jsonl(records: Iterable[VLARecord], path: str | Path) -> None:
    record_list = list(records)
    validate_records(record_list)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        for record in record_list:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def _fill_legacy_next_observations(records: list[VLARecord]) -> None:
    by_episode: dict[int, list[VLARecord]] = {}
    for record in records:
        by_episode.setdefault(record.episode_id, []).append(record)
    for episode_records in by_episode.values():
        episode_records.sort(key=lambda item: item.timestep)
        for index, record in enumerate(episode_records):
            if not record.next_observation:
                if index + 1 < len(episode_records):
                    record.next_observation = list(episode_records[index + 1].observation)
                else:
                    record.next_observation = list(record.observation)
            if index == len(episode_records) - 1:
                record.terminated = True


def load_jsonl(path: str | Path) -> list[VLARecord]:
    records: list[VLARecord] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"line {line_number} of {path} is not a JSON object")
            try:
                records.append(VLARecord.from_mapping(item))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid record on line {line_number} of {path}: {exc}") from exc
    _fill_legacy_next_observations(records)
    validate_records(records)
    return records


def load_dataset_from_config(dataset_config: dict[str, Any]) -> list[VLARecord]:
    source = str(dataset_config.get("source", "mock_pusht"))
    if source in {"mock_pusht", "generated"}:
        return generate_mock_pusht_records(
            num_episodes=int(dataset_config.get("num_episodes", 16)),
            steps_per_episode=int(dataset_config.get("steps_per_episode", 32)),
            seed=int(dataset_config.get("seed", 42)),
            language_instruction=dataset_config.get(
                "language_instruction", "push the T block to the goal"
            ),
            goal=dataset_config.get("goal", [0.80, 0.20]),
            start_low=float(dataset_config.get("start_low", 0.05)),
            start_high=float(dataset_config.get("start_high", 0.95)),
            action_gain=float(dataset_config.get("action_gain", 0.35)),
            action_clip=float(dataset_config.get("action_clip", 0.12)),
            action_noise_std=float(dataset_config.get("action_noise_std", 0.004)),
            success_distance=float(dataset_config.get("success_distance", 0.08)),
        )
    if source == "jsonl":
        data_path = dataset_config.get("path")
        if not data_path:
            raise ValueError("dataset.source=jsonl requires dataset.path")
        return load_jsonl(data_path)
    raise ValueError(f"unsupported dataset source: {source}")


def load_dataset_splits_from_config(
    dataset_config: dict[str, Any],
) -> dict[str, list[VLARecord]]:
    records = load_dataset_from_config(dataset_config)
    return split_records_by_episode(
        records,
        train_fraction=float(dataset_config.get("train_fraction", 0.8)),
        validation_fraction=float(dataset_config.get("validation_fraction", 0.1)),
        test_fraction=float(dataset_config.get("test_fraction", 0.1)),
        seed=int(dataset_config.get("split_seed", dataset_config.get("seed", 42))),
    )


def build_training_batch(
    records: list[VLARecord],
    chunk_size: int,
    instruction_dim: int = 8,
) -> tuple[Float32Array, Float32Array]:
    """v1.0 adapter; new code should use :func:`build_training_arrays`."""

    arrays = build_training_arrays(records, chunk_size, instruction_dim)
    return arrays.inputs, arrays.targets.reshape(len(arrays.targets), -1)


_instruction_features = instruction_features
