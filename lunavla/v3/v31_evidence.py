from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


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
