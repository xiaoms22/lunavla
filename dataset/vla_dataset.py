from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol

import numpy as np
import numpy.typing as npt


Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


@dataclass
class VLARecord:
    """One transition in the versioned LunaVLA JSONL record contract."""

    observation: list[float]
    action: list[float]
    episode_id: int
    timestep: int
    success: bool
    language_instruction: str | None
    metadata: dict[str, Any]
    next_observation: list[float] = field(default_factory=list)
    terminated: bool = False
    task_id: str = "pusht_style_point_reach"
    subtask_id: str = "unknown"
    phase: str = "unknown"

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "VLARecord":
        metadata = dict(item.get("metadata", {}))
        return cls(
            observation=list(item["observation"]),
            action=list(item["action"]),
            episode_id=int(item["episode_id"]),
            timestep=int(item["timestep"]),
            success=bool(item.get("success", False)),
            language_instruction=item.get("language_instruction"),
            metadata=metadata,
            next_observation=list(item.get("next_observation", [])),
            terminated=bool(item.get("terminated", item.get("success", False))),
            task_id=str(item.get("task_id", metadata.get("task", "pusht_style_point_reach"))),
            subtask_id=str(item.get("subtask_id", metadata.get("subtask_id", "unknown"))),
            phase=str(item.get("phase", metadata.get("phase", "unknown"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingArrays:
    """Dense policy arrays with explicit chunk padding."""

    inputs: Float32Array
    targets: Float32Array
    valid_mask: BoolArray

    def __post_init__(self) -> None:
        if self.inputs.ndim != 2:
            raise ValueError(f"inputs must be [batch, features]; got {self.inputs.shape}")
        if self.targets.ndim != 3:
            raise ValueError(f"targets must be [batch, chunk, action]; got {self.targets.shape}")
        if self.valid_mask.shape != self.targets.shape[:2]:
            raise ValueError(
                f"valid_mask must have shape {self.targets.shape[:2]}; got {self.valid_mask.shape}"
            )
        if len(self.inputs) != len(self.targets):
            raise ValueError("inputs and targets must contain the same number of samples")


class VLASource(Protocol):
    def load_records(self) -> list[VLARecord]:
        ...


def instruction_features(text: str | None, dim: int = 8) -> Float32Array:
    if dim < 0:
        raise ValueError("instruction feature dimension cannot be negative")
    features = np.zeros(dim, dtype=np.float32)
    if not text or dim == 0:
        return features
    for token in text.lower().split():
        idx = sum(ord(ch) for ch in token) % dim
        features[idx] += 1.0
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features


def build_training_arrays(
    records: Iterable[VLARecord],
    chunk_size: int,
    instruction_dim: int = 8,
) -> TrainingArrays:
    """Build padded ``[B, K, A]`` targets; padding is always masked out."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    record_list = list(records)
    validate_records(record_list)
    by_episode: dict[int, list[VLARecord]] = {}
    for record in record_list:
        by_episode.setdefault(record.episode_id, []).append(record)

    inputs: list[Float32Array] = []
    targets: list[Float32Array] = []
    valid_masks: list[BoolArray] = []
    action_dim = len(record_list[0].action)
    for episode_id in sorted(by_episode):
        episode_records = sorted(by_episode[episode_id], key=lambda item: item.timestep)
        actions = [np.asarray(item.action, dtype=np.float32) for item in episode_records]
        for idx, record in enumerate(episode_records):
            observation = np.asarray(record.observation, dtype=np.float32)
            instruction = instruction_features(record.language_instruction, instruction_dim)
            chunk = np.zeros((chunk_size, action_dim), dtype=np.float32)
            valid_mask = np.zeros(chunk_size, dtype=bool)
            remaining = min(chunk_size, len(actions) - idx)
            if remaining:
                chunk[:remaining] = np.stack(actions[idx : idx + remaining])
                valid_mask[:remaining] = True
            inputs.append(np.concatenate([observation, instruction]))
            targets.append(chunk)
            valid_masks.append(valid_mask)

    return TrainingArrays(
        inputs=np.stack(inputs).astype(np.float32),
        targets=np.stack(targets).astype(np.float32),
        valid_mask=np.stack(valid_masks),
    )


def records_to_arrays(
    records: Iterable[VLARecord],
    chunk_size: int,
    instruction_dim: int = 8,
) -> tuple[Float32Array, Float32Array]:
    """v1.0 compatibility adapter returning flattened targets without the mask."""

    arrays = build_training_arrays(records, chunk_size, instruction_dim)
    return arrays.inputs, arrays.targets.reshape(len(arrays.targets), -1)


def validate_records(
    records: Iterable[VLARecord],
    *,
    observation_dim: int | None = None,
    action_dim: int | None = None,
) -> None:
    """Validate dimensions, finite values, and per-episode timestep continuity."""

    record_list = list(records)
    if not record_list:
        raise ValueError("dataset contains no records")
    expected_observation_dim = observation_dim or len(record_list[0].observation)
    expected_action_dim = action_dim or len(record_list[0].action)
    if expected_observation_dim <= 0 or expected_action_dim <= 0:
        raise ValueError("observation and action dimensions must be positive")

    seen: set[tuple[int, int]] = set()
    episode_timesteps: dict[int, list[int]] = {}
    for idx, record in enumerate(record_list):
        if record.episode_id < 0:
            raise ValueError(f"record {idx} has negative episode_id")
        if record.timestep < 0:
            raise ValueError(f"record {idx} has negative timestep")
        key = (record.episode_id, record.timestep)
        if key in seen:
            raise ValueError(f"duplicate record for episode {key[0]} timestep {key[1]}")
        seen.add(key)
        episode_timesteps.setdefault(record.episode_id, []).append(record.timestep)

        observation = np.asarray(record.observation)
        action = np.asarray(record.action)
        next_observation = np.asarray(record.next_observation)
        if observation.dtype.kind not in "fiu" or action.dtype.kind not in "fiu":
            raise TypeError(f"record {idx} observation and action must be numeric")
        if observation.shape != (expected_observation_dim,):
            raise ValueError(
                f"record {idx} observation must have shape {(expected_observation_dim,)}; "
                f"got {observation.shape}"
            )
        if action.shape != (expected_action_dim,):
            raise ValueError(
                f"record {idx} action must have shape {(expected_action_dim,)}; got {action.shape}"
            )
        if next_observation.size and next_observation.shape != (expected_observation_dim,):
            raise ValueError(
                f"record {idx} next_observation must have shape {(expected_observation_dim,)}; "
                f"got {next_observation.shape}"
            )
        for name, value in (
            ("observation", observation),
            ("action", action),
            ("next_observation", next_observation),
        ):
            if value.size and not np.all(np.isfinite(value.astype(np.float64))):
                raise ValueError(f"record {idx} {name} contains NaN or infinite values")

    for episode_id, timesteps in episode_timesteps.items():
        ordered = sorted(timesteps)
        expected = list(range(len(ordered)))
        if ordered != expected:
            raise ValueError(
                f"episode {episode_id} timesteps must be contiguous from 0; got {ordered}"
            )


def split_records_by_episode(
    records: Iterable[VLARecord],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[VLARecord]]:
    """Deterministically split whole episodes with no ID leakage."""

    record_list = list(records)
    validate_records(record_list)
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=np.float64
    )
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0):
        raise ValueError("split fractions must be finite and non-negative")
    if not np.isclose(float(np.sum(fractions)), 1.0, atol=1e-8):
        raise ValueError("train, validation, and test fractions must sum to 1")

    episode_ids = np.asarray(sorted({record.episode_id for record in record_list}), dtype=int)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episode_ids)
    raw_counts = fractions * len(shuffled)
    counts = np.floor(raw_counts).astype(int)
    for index in np.argsort(-(raw_counts - counts))[: len(shuffled) - int(np.sum(counts))]:
        counts[index] += 1
    if len(shuffled) >= 3:
        for index in np.flatnonzero((fractions > 0) & (counts == 0)):
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[index] += 1

    train_end = counts[0]
    validation_end = train_end + counts[1]
    split_ids = {
        "train": set(shuffled[:train_end].tolist()),
        "validation": set(shuffled[train_end:validation_end].tolist()),
        "test": set(shuffled[validation_end:].tolist()),
    }
    return {
        split: [record for record in record_list if record.episode_id in ids]
        for split, ids in split_ids.items()
    }
