from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
import numpy as np

from .stable_contracts import wilson_interval


V31_EVIDENCE_ARMS = (
    "baseline",
    "smol_control",
    "feature_mask",
    "feature_shuffle",
)
V31_EVIDENCE_TASKS = (
    "direct_pick_place",
    "waypoint_sequence",
    "failure_recovery",
)
V31_EVIDENCE_STRATA = ("composition", "paraphrase")
V31_TRAIN_SEEDS = (11, 22, 33, 44, 55)


@dataclass(frozen=True)
class V31EvidenceDesignV1:
    design_id: str
    train_seeds: tuple[int, ...]
    data_seed: int
    analysis_seed: int
    bootstrap_samples: int
    arms: tuple[str, ...]
    task_ids: tuple[str, ...]
    held_out_strata: tuple[str, ...]
    episodes_per_cell: int
    expected_rows: int
    claim_allowed: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("V31EvidenceDesignV1 schema_version must be integer 1")
        if self.design_id != "v31_frozen_vlm_feature_contribution_v1":
            raise ValueError("unexpected v3.1 evidence design_id")
        if self.train_seeds != V31_TRAIN_SEEDS:
            raise ValueError("v3.1 evidence requires train seeds 11,22,33,44,55")
        if self.data_seed != 42 or self.analysis_seed != 202701:
            raise ValueError("v3.1 evidence seeds are fixed")
        if self.bootstrap_samples != 10_000:
            raise ValueError("v3.1 evidence requires 10,000 bootstrap samples")
        if self.arms != V31_EVIDENCE_ARMS:
            raise ValueError("v3.1 evidence arms are fixed and ordered")
        if self.task_ids != V31_EVIDENCE_TASKS:
            raise ValueError("v3.1 evidence tasks are fixed and ordered")
        if self.held_out_strata != V31_EVIDENCE_STRATA:
            raise ValueError("v3.1 evidence strata are fixed and ordered")
        if self.episodes_per_cell != 20:
            raise ValueError("v3.1 evidence requires 20 paired episodes per cell")
        computed = (
            len(self.train_seeds)
            * len(self.arms)
            * len(self.task_ids)
            * len(self.held_out_strata)
            * self.episodes_per_cell
        )
        if computed != 2_400 or self.expected_rows != computed:
            raise ValueError("v3.1 evidence matrix must contain exactly 2,400 rows")
        if self.claim_allowed is not False:
            raise ValueError("a design alone cannot open a scientific claim")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> V31EvidenceDesignV1:
        fields = {
            "schema_version",
            "design_id",
            "train_seeds",
            "data_seed",
            "analysis_seed",
            "bootstrap_samples",
            "arms",
            "task_ids",
            "held_out_strata",
            "episodes_per_cell",
            "expected_rows",
            "claim_allowed",
        }
        if not isinstance(value, Mapping):
            raise TypeError("v3.1 evidence design must be a mapping")
        unknown = sorted(set(value) - fields)
        missing = sorted(fields - set(value))
        if unknown or missing:
            raise ValueError(f"invalid v3.1 evidence fields; unknown={unknown}, missing={missing}")
        for name in (
            "train_seeds",
            "arms",
            "task_ids",
            "held_out_strata",
        ):
            raw = value[name]
            if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, (list, tuple)):
                raise TypeError(f"{name} must be a sequence")
        return cls(
            schema_version=value["schema_version"],
            design_id=value["design_id"],
            train_seeds=tuple(value["train_seeds"]),
            data_seed=value["data_seed"],
            analysis_seed=value["analysis_seed"],
            bootstrap_samples=value["bootstrap_samples"],
            arms=tuple(value["arms"]),
            task_ids=tuple(value["task_ids"]),
            held_out_strata=tuple(value["held_out_strata"]),
            episodes_per_cell=value["episodes_per_cell"],
            expected_rows=value["expected_rows"],
            claim_allowed=value["claim_allowed"],
        )

    @classmethod
    def load(cls, path: str | Path) -> V31EvidenceDesignV1:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "design_id": self.design_id,
            "train_seeds": list(self.train_seeds),
            "data_seed": self.data_seed,
            "analysis_seed": self.analysis_seed,
            "bootstrap_samples": self.bootstrap_samples,
            "arms": list(self.arms),
            "task_ids": list(self.task_ids),
            "held_out_strata": list(self.held_out_strata),
            "episodes_per_cell": self.episodes_per_cell,
            "expected_rows": self.expected_rows,
            "claim_allowed": self.claim_allowed,
        }

    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class V31EvidenceRowV1:
    train_seed: int
    arm: str
    task_id: str
    held_out_stratum: str
    episode_index: int
    pair_id: str
    git_sha: str
    dependency_lock_sha256: str
    feature_source_sha256: str
    checkpoint_sha256: str
    success: bool
    final_distance: float
    first_action_mse: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("V31EvidenceRowV1 schema_version must be integer 1")
        if self.train_seed not in V31_TRAIN_SEEDS:
            raise ValueError("row train_seed is not preregistered")
        if self.arm not in V31_EVIDENCE_ARMS:
            raise ValueError("row arm is not preregistered")
        if self.task_id not in V31_EVIDENCE_TASKS:
            raise ValueError("row task_id is not preregistered")
        if self.held_out_stratum not in V31_EVIDENCE_STRATA:
            raise ValueError("row held_out_stratum is not preregistered")
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, int)
            or not 0 <= self.episode_index < 20
        ):
            raise ValueError("row episode_index must be in [0, 20)")
        expected_pair = f"{self.task_id}:{self.held_out_stratum}:{self.episode_index}"
        if self.pair_id != expected_pair:
            raise ValueError("row pair_id does not match task/stratum/episode")
        if (
            not isinstance(self.git_sha, str)
            or len(self.git_sha) != 40
            or any(character not in "0123456789abcdef" for character in self.git_sha)
        ):
            raise ValueError("row git_sha must be a full lowercase Git SHA")
        for name in (
            "dependency_lock_sha256",
            "feature_source_sha256",
            "checkpoint_sha256",
        ):
            _sha256(getattr(self, name), name)
        if not isinstance(self.success, bool):
            raise TypeError("row success must be boolean")
        for name in ("final_distance", "first_action_mse"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"row {name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"row {name} must be finite and non-negative")

    @property
    def matrix_key(self) -> tuple[int, str, str, str, int]:
        return (
            self.train_seed,
            self.arm,
            self.task_id,
            self.held_out_stratum,
            self.episode_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "train_seed": self.train_seed,
            "arm": self.arm,
            "task_id": self.task_id,
            "held_out_stratum": self.held_out_stratum,
            "episode_index": self.episode_index,
            "pair_id": self.pair_id,
            "git_sha": self.git_sha,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "feature_source_sha256": self.feature_source_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "success": self.success,
            "final_distance": self.final_distance,
            "first_action_mse": self.first_action_mse,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> V31EvidenceRowV1:
        fields = set(cls.__dataclass_fields__)
        if set(value) != fields:
            raise ValueError("invalid V31EvidenceRowV1 fields")
        return cls(**value)


def v31_expected_matrix_keys(
    design: V31EvidenceDesignV1,
) -> tuple[tuple[int, str, str, str, int], ...]:
    return tuple(
        (seed, arm, task, stratum, episode)
        for seed in design.train_seeds
        for arm in design.arms
        for task in design.task_ids
        for stratum in design.held_out_strata
        for episode in range(design.episodes_per_cell)
    )


def v31_row_inventory_sha256(rows: tuple[V31EvidenceRowV1, ...]) -> str:
    ordered = sorted(rows, key=lambda row: row.matrix_key)
    return _hash({"rows": [row.to_dict() for row in ordered]})


@dataclass(frozen=True)
class V31RepeatSentinelV1:
    train_seed: int
    source_rows_sha256: str
    repeat_rows_sha256: str
    source_checkpoints_sha256: str
    repeat_checkpoints_sha256: str
    source_metrics_sha256: str
    repeat_metrics_sha256: str
    verified: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("V31RepeatSentinelV1 schema_version must be integer 1")
        if self.train_seed != 11:
            raise ValueError("v3.1 repeat sentinel must use train seed 11")
        names = (
            "source_rows_sha256",
            "repeat_rows_sha256",
            "source_checkpoints_sha256",
            "repeat_checkpoints_sha256",
            "source_metrics_sha256",
            "repeat_metrics_sha256",
        )
        for name in names:
            _sha256(getattr(self, name), name)
        equal = all(
            getattr(self, source) == getattr(self, repeat)
            for source, repeat in (
                ("source_rows_sha256", "repeat_rows_sha256"),
                ("source_checkpoints_sha256", "repeat_checkpoints_sha256"),
                ("source_metrics_sha256", "repeat_metrics_sha256"),
            )
        )
        if not isinstance(self.verified, bool) or self.verified != equal:
            raise ValueError("sentinel verified must reflect exact hash equality")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def sha256(self) -> str:
        return _hash(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> V31RepeatSentinelV1:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("invalid V31RepeatSentinelV1 fields")
        return cls(**value)


@dataclass(frozen=True)
class V31EvidenceManifestV1:
    design_sha256: str
    row_inventory_sha256: str
    sentinel_sha256: str
    statistics_sha256: str
    claim_gate_sha256: str
    git_sha: str
    feature_source: str
    expected_rows: int
    observed_rows: int
    matrix_complete: bool
    homogeneous: bool
    sentinel_verified: bool
    claim_allowed: bool
    release_eligible: bool
    gate_reasons: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("V31EvidenceManifestV1 schema_version must be integer 1")
        for name in (
            "design_sha256",
            "row_inventory_sha256",
            "sentinel_sha256",
            "statistics_sha256",
            "claim_gate_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            not isinstance(self.git_sha, str)
            or len(self.git_sha) != 40
            or any(character not in "0123456789abcdef" for character in self.git_sha)
        ):
            raise ValueError("manifest git_sha must be a full lowercase Git SHA")
        if self.feature_source not in {"real_frozen_vlm", "deterministic_fixture"}:
            raise ValueError("invalid feature_source")
        if self.expected_rows != 2400 or self.observed_rows < 0:
            raise ValueError("manifest row counts are invalid")
        for name in (
            "matrix_complete",
            "homogeneous",
            "sentinel_verified",
            "claim_allowed",
            "release_eligible",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        allowed = {
            "incomplete_matrix",
            "mixed_source",
            "sentinel_failure",
            "fixture_source",
            "claim_threshold_not_met",
        }
        if not set(self.gate_reasons).issubset(allowed):
            raise ValueError("manifest contains an unknown gate reason")
        integrity = self.matrix_complete and self.homogeneous and self.sentinel_verified
        releasable = integrity and self.feature_source == "real_frozen_vlm"
        if self.release_eligible != releasable:
            raise ValueError("release_eligible requires real evidence integrity")
        if self.claim_allowed and (
            not integrity
            or self.feature_source != "real_frozen_vlm"
            or "claim_threshold_not_met" in self.gate_reasons
        ):
            raise ValueError("claim gate cannot open without real complete evidence")

    def to_dict(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        value["gate_reasons"] = list(self.gate_reasons)
        return value

    def sha256(self) -> str:
        return _hash(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> V31EvidenceManifestV1:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("invalid V31EvidenceManifestV1 fields")
        payload = dict(value)
        payload["gate_reasons"] = tuple(payload["gate_reasons"])
        return cls(**payload)


def v31_statistics(
    rows: tuple[V31EvidenceRowV1, ...], design: V31EvidenceDesignV1
) -> dict[str, Any]:
    indexed = {row.matrix_key: row for row in rows}

    def paired(metric: str, left: str, right: str, *, sign: float = 1.0) -> list[float]:
        result: list[float] = []
        for seed in design.train_seeds:
            for task in design.task_ids:
                for stratum in design.held_out_strata:
                    for episode in range(design.episodes_per_cell):
                        left_row = indexed[(seed, left, task, stratum, episode)]
                        right_row = indexed[(seed, right, task, stratum, episode)]
                        left_value = float(getattr(left_row, metric))
                        right_value = float(getattr(right_row, metric))
                        result.append(sign * (left_value - right_value))
        return result

    comparisons: dict[str, dict[str, Any]] = {}
    definitions = {
        "control_vs_baseline_success": ("success", "smol_control", "baseline", 1.0),
        "baseline_vs_control_distance": (
            "final_distance",
            "baseline",
            "smol_control",
            1.0,
        ),
        "control_vs_shuffle_success": (
            "success",
            "smol_control",
            "feature_shuffle",
            1.0,
        ),
    }
    for name, (metric, left, right, sign) in definitions.items():
        clusters: dict[int, list[float]] = {seed: [] for seed in design.train_seeds}
        for seed in design.train_seeds:
            for task in design.task_ids:
                for stratum in design.held_out_strata:
                    for episode in range(design.episodes_per_cell):
                        a = indexed[(seed, left, task, stratum, episode)]
                        b = indexed[(seed, right, task, stratum, episode)]
                        av = float(getattr(a, metric))
                        bv = float(getattr(b, metric))
                        clusters[seed].append(sign * (av - bv))
        interval = _clustered_bootstrap(clusters, design)
        flat = paired(metric, left, right, sign=sign)
        comparisons[name] = {"estimate": sum(flat) / len(flat), "ci95": list(interval)}
    strata: list[dict[str, Any]] = []
    for task in design.task_ids:
        for stratum in design.held_out_strata:
            clusters = {seed: [] for seed in design.train_seeds}
            for seed in design.train_seeds:
                for episode in range(design.episodes_per_cell):
                    control = indexed[(seed, "smol_control", task, stratum, episode)]
                    baseline = indexed[(seed, "baseline", task, stratum, episode)]
                    clusters[seed].append(float(control.success) - float(baseline.success))
            interval = _clustered_bootstrap(clusters, design)
            strata.append({"task_id": task, "stratum": stratum, "ci95": list(interval)})
    wilson: list[dict[str, Any]] = []
    for arm in design.arms:
        arm_rows = [row for row in rows if row.arm == arm]
        successes = sum(row.success for row in arm_rows)
        wilson.append(
            {
                "arm": arm,
                "successes": successes,
                "trials": len(arm_rows),
                "interval": list(wilson_interval(successes, len(arm_rows))),
            }
        )
    return {
        "schema_version": 1,
        "cluster_unit": "train_seed",
        "pair_unit": "task_stratum_episode",
        "bootstrap_samples": design.bootstrap_samples,
        "comparisons": comparisons,
        "strata_noninferiority": strata,
        "wilson": wilson,
    }


def _clustered_bootstrap(
    clusters: Mapping[int, list[float]], design: V31EvidenceDesignV1
) -> tuple[float, float]:
    keys = tuple(sorted(clusters))
    arrays = [np.asarray(clusters[key], dtype=np.float64) for key in keys]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1 or not arrays or not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("v3.1 bootstrap requires equal non-empty finite seed clusters")
    matrix = np.stack(arrays)
    rng = np.random.default_rng(design.analysis_seed)
    estimates = np.empty(design.bootstrap_samples, dtype=np.float64)
    chunk_size = 1_000
    cluster_count, pair_count = matrix.shape
    for start in range(0, design.bootstrap_samples, chunk_size):
        count = min(chunk_size, design.bootstrap_samples - start)
        sampled_clusters = rng.integers(0, cluster_count, size=(count, cluster_count))
        sampled_pairs = rng.integers(0, pair_count, size=(count, cluster_count, pair_count))
        selected_matrices = matrix[sampled_clusters]
        selected = np.take_along_axis(selected_matrices, sampled_pairs, axis=2)
        estimates[start : start + count] = selected.mean(axis=(1, 2))
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def v31_claim_passes(statistics: Mapping[str, Any]) -> bool:
    comparisons = statistics["comparisons"]
    required = (
        comparisons["control_vs_baseline_success"]["ci95"][0] > 0,
        comparisons["baseline_vs_control_distance"]["ci95"][0] > 0,
        comparisons["control_vs_shuffle_success"]["ci95"][0] > 0,
        all(item["ci95"][0] >= -0.05 for item in statistics["strata_noninferiority"]),
    )
    return all(required)
