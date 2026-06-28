from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Protocol

import numpy as np


@dataclass
class VLARecord:
    observation: list[float]
    action: list[float]
    episode_id: int
    timestep: int
    success: bool
    language_instruction: str | None
    metadata: dict[str, Any]

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "VLARecord":
        return cls(
            observation=list(item["observation"]),
            action=list(item["action"]),
            episode_id=int(item["episode_id"]),
            timestep=int(item["timestep"]),
            success=bool(item["success"]),
            language_instruction=item.get("language_instruction"),
            metadata=dict(item.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLASource(Protocol):
    def load_records(self) -> list[VLARecord]:
        ...


def instruction_features(text: str | None, dim: int = 8) -> np.ndarray:
    features = np.zeros(dim, dtype=np.float32)
    if not text:
        return features
    for token in text.lower().split():
        idx = sum(ord(ch) for ch in token) % dim
        features[idx] += 1.0
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features


def records_to_arrays(
    records: Iterable[VLARecord],
    chunk_size: int,
    instruction_dim: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    by_episode: dict[int, list[VLARecord]] = {}
    for record in records:
        by_episode.setdefault(record.episode_id, []).append(record)

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for episode_records in by_episode.values():
        episode_records.sort(key=lambda item: item.timestep)
        actions = [np.asarray(item.action, dtype=np.float32) for item in episode_records]
        for idx, record in enumerate(episode_records):
            observation = np.asarray(record.observation, dtype=np.float32)
            instruction = instruction_features(record.language_instruction, instruction_dim)
            chunk = []
            for offset in range(chunk_size):
                action_idx = min(idx + offset, len(actions) - 1)
                chunk.append(actions[action_idx])
            inputs.append(np.concatenate([observation, instruction]))
            targets.append(np.concatenate(chunk))

    return np.vstack(inputs).astype(np.float32), np.vstack(targets).astype(np.float32)


def validate_records(records: Iterable[VLARecord]) -> None:
    for idx, record in enumerate(records):
        if not record.observation:
            raise ValueError(f"record {idx} has empty observation")
        if not record.action:
            raise ValueError(f"record {idx} has empty action")
        if record.timestep < 0:
            raise ValueError(f"record {idx} has negative timestep")
