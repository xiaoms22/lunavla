from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import EpisodeRecordV3, FeatureSchema


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _update_observation_hash(digest: Any, observation: Any) -> None:
    for collection in (observation.images, observation.state):
        for name, value in collection.items():
            digest.update(name.encode())
            digest.update(value.dtype.str.encode())
            digest.update(_canonical_json(list(value.shape)))
            digest.update(value.tobytes(order="C"))
    digest.update((observation.instruction or "").encode())
    digest.update(
        _canonical_json(
            {
                "episode_id": observation.episode_id,
                "timestamp_s": observation.timestamp_s,
                "step_index": observation.step_index,
                "metadata": _json_value(observation.metadata),
            }
        )
    )


def episode_sha256(episode: EpisodeRecordV3) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(episode.to_dict()))
    for transition in episode.transitions:
        _update_observation_hash(digest, transition.observation)
        _update_observation_hash(digest, transition.next_observation)
        digest.update(transition.action.tobytes(order="C"))
        digest.update(
            _canonical_json(
                {
                    "reward": transition.reward,
                    "terminated": transition.terminated,
                    "truncated": transition.truncated,
                    "info": _json_value(transition.info),
                }
            )
        )
    return digest.hexdigest()


@dataclass(frozen=True)
class InMemoryDatasetSourceV3:
    episodes: tuple[EpisodeRecordV3, ...]

    def __post_init__(self) -> None:
        episodes = tuple(self.episodes)
        if not episodes or any(not isinstance(item, EpisodeRecordV3) for item in episodes):
            raise ValueError("episodes must contain at least one EpisodeRecordV3")
        object.__setattr__(self, "episodes", episodes)

    def load(self) -> Sequence[EpisodeRecordV3]:
        return self.episodes


def split_episode_ids(
    episodes: Sequence[EpisodeRecordV3],
    *,
    seed: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> Mapping[str, tuple[str | int, ...]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    fractions = np.asarray([train_fraction, validation_fraction, test_fraction], dtype=np.float64)
    if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("split fractions must be positive and sum to one")
    identifiers = [item.episode_id for item in episodes]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("episode IDs must be non-empty and unique")
    order = np.random.default_rng(seed).permutation(len(identifiers))
    shuffled = [identifiers[int(index)] for index in order]
    raw_counts = fractions * len(shuffled)
    counts = np.floor(raw_counts).astype(int)
    for index in np.argsort(-(raw_counts - counts))[: len(shuffled) - int(counts.sum())]:
        counts[int(index)] += 1
    if len(shuffled) >= 3:
        for empty in np.flatnonzero(counts == 0):
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                raise ValueError("each split must receive an episode")
            counts[donor] -= 1
            counts[int(empty)] += 1
    train_end = int(counts[0])
    validation_end = train_end + int(counts[1])
    return MappingProxyType(
        {
            "train": tuple(shuffled[:train_end]),
            "validation": tuple(shuffled[train_end:validation_end]),
            "test": tuple(shuffled[validation_end:]),
        }
    )


@dataclass(frozen=True)
class EpisodeHashRecord:
    episode_id: str | int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "episode_id_type": "integer" if isinstance(self.episode_id, int) else "string",
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class DataAuditManifest:
    schema_version: int
    feature_schema_sha256: str
    episode_hashes: tuple[EpisodeHashRecord, ...]
    split: Mapping[str, tuple[str | int, ...]]
    episode_count: int
    transition_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_revision": 2,
            "feature_schema_sha256": self.feature_schema_sha256,
            "episode_hashes": [item.to_dict() for item in self.episode_hashes],
            "split": {name: list(values) for name, values in self.split.items()},
            "episode_count": self.episode_count,
            "transition_count": self.transition_count,
        }

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


def audit_episodes(
    episodes: Sequence[EpisodeRecordV3],
    *,
    feature_schema: FeatureSchema,
    split: Mapping[str, Sequence[str | int]],
) -> DataAuditManifest:
    records = tuple(episodes)
    if not records:
        raise ValueError("dataset cannot be empty")
    identifiers = [item.episode_id for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("dataset contains duplicate episode IDs")
    all_split_ids: list[str | int] = []
    for name in ("train", "validation", "test"):
        if name not in split:
            raise ValueError("split must contain train, validation, and test")
        values = list(split[name])
        if len(values) != len(set(values)):
            raise ValueError(f"split {name} contains duplicate episode IDs")
        all_split_ids.extend(values)
    if len(all_split_ids) != len(set(all_split_ids)):
        raise ValueError("episode IDs overlap across splits")
    if set(all_split_ids) != set(identifiers):
        raise ValueError("split episode IDs must exactly match the dataset")
    hashes: list[EpisodeHashRecord] = []
    transitions = 0
    for episode in records:
        for transition in episode.transitions:
            feature_schema.validate_observation(transition.observation)
            feature_schema.validate_observation(transition.next_observation)
            feature_schema.validate_action(transition.action)
            if transition.next_observation.timestamp_s < transition.observation.timestamp_s:
                raise ValueError("observation timestamps must be monotonic")
        hashes.append(EpisodeHashRecord(episode.episode_id, episode_sha256(episode)))
        transitions += len(episode.transitions)
    return DataAuditManifest(
        schema_version=1,
        feature_schema_sha256=feature_schema.sha256(),
        episode_hashes=tuple(hashes),
        split=MappingProxyType({name: tuple(split[name]) for name in ("train", "validation", "test")}),
        episode_count=len(records),
        transition_count=transitions,
    )


@dataclass(frozen=True)
class DatasetBundle:
    episodes: tuple[EpisodeRecordV3, ...]
    split: Mapping[str, tuple[str | int, ...]]
    audit: DataAuditManifest

    def __post_init__(self) -> None:
        episodes = tuple(self.episodes)
        if not episodes:
            raise ValueError("dataset bundle cannot be empty")
        by_id = {episode.episode_id: episode for episode in episodes}
        if len(by_id) != len(episodes):
            raise ValueError("dataset bundle episode IDs must be unique")
        normalized: dict[str, tuple[str | int, ...]] = {}
        for name in ("train", "validation", "test"):
            identifiers = tuple(self.split.get(name, ()))
            if not identifiers:
                raise ValueError(f"dataset split {name} cannot be empty")
            if any(identifier not in by_id for identifier in identifiers):
                raise ValueError(f"dataset split {name} contains an unknown episode ID")
            normalized[name] = identifiers
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "split", MappingProxyType(normalized))

    def select(self, name: str) -> tuple[EpisodeRecordV3, ...]:
        if name not in self.split:
            raise ValueError("split must be train, validation, or test")
        by_id = {episode.episode_id: episode for episode in self.episodes}
        return tuple(by_id[identifier] for identifier in self.split[name])

    def source(self, name: str) -> InMemoryDatasetSourceV3:
        return InMemoryDatasetSourceV3(self.select(name))
