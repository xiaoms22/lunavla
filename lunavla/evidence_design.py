"""Strict, versioned contracts for predeclared v2 evidence studies."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TypeVar, cast

import yaml


EVIDENCE_DESIGN_SCHEMA_VERSION = 1

EvidenceSuite = Literal["language", "visual"]
ArmRole = Literal["control", "intervention", "baseline"]
MetricKind = Literal["binary", "continuous"]
MetricDirection = Literal["positive", "negative"]

_ROOT_FIELDS = {
    "schema_version",
    "design_id",
    "suite",
    "base_config",
    "seeds",
    "arms",
    "metrics",
    "budget",
    "output",
}
_SEED_FIELDS = {"train", "data", "split", "evaluation", "bootstrap"}
_ARM_FIELDS = {"id", "role", "mode"}
_METRIC_FIELDS = {"name", "kind", "direction"}
_BUDGET_FIELDS = {
    "dataset_episodes",
    "batch_size",
    "training_steps",
    "learning_rate",
    "evaluation_episodes",
    "bootstrap_samples",
}
_OUTPUT_FIELDS = {"run_root", "snapshot_root"}
_SUITE_MODES = {
    "language": {"none", "mask", "shuffle", "counterfactual"},
    "visual": {"none", "occlusion", "shuffle", "state_only"},
}
_SLUG = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
_Resolved = TypeVar("_Resolved")


def _resolved_instance(cls: type[_Resolved], **values: Any) -> _Resolved:
    """Construct an already-validated frozen value without exposing public init."""

    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance


def _reject_direct_construction(name: str) -> None:
    raise TypeError(f"{name} cannot be constructed directly; use {name}.from_mapping()")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    return dict(value)


def _strict_fields(
    name: str,
    value: Mapping[str, Any],
    expected: set[str],
) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ValueError(f"unknown field(s) in {name}: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing field(s) in {name}: {', '.join(missing)}")


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if not positive and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _slug(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not _SLUG.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase slug")
    return result


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _repository_path(value: Any, name: str, *, prefix: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    raw = value.strip()
    path = Path(raw)
    normalized = path.as_posix()
    if (
        not raw
        or path.is_absolute()
        or "\\" in raw
        or ".." in path.parts
        or path.parts[: len(prefix)] != prefix
        or len(path.parts) <= len(prefix)
        or normalized != raw
    ):
        prefix_text = "/".join(prefix) + "/"
        raise ValueError(
            f"{name} must be a normalized repository-relative path under {prefix_text}"
        )
    return normalized


@dataclass(frozen=True, init=False)
class EvidenceSeeds:
    train: tuple[int, ...]
    data: int
    split: int
    evaluation: tuple[int, ...]
    bootstrap: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        _reject_direct_construction(type(self).__name__)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "EvidenceSeeds":
        payload = _mapping(source, "seeds")
        _strict_fields("seeds", payload, _SEED_FIELDS)
        train = tuple(
            _integer(value, f"seeds.train[{index}]")
            for index, value in enumerate(_sequence(payload["train"], "seeds.train"))
        )
        evaluation = tuple(
            _integer(value, f"seeds.evaluation[{index}]")
            for index, value in enumerate(_sequence(payload["evaluation"], "seeds.evaluation"))
        )
        if len(train) < 2:
            raise ValueError("seeds.train must contain at least two training seeds")
        if not evaluation:
            raise ValueError("seeds.evaluation cannot be empty")
        if len(set(train)) != len(train):
            raise ValueError("seeds.train values must be unique")
        if len(set(evaluation)) != len(evaluation):
            raise ValueError("seeds.evaluation values must be unique")
        return _resolved_instance(
            cls,
            train=train,
            data=_integer(payload["data"], "seeds.data"),
            split=_integer(payload["split"], "seeds.split"),
            evaluation=evaluation,
            bootstrap=_integer(payload["bootstrap"], "seeds.bootstrap"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": list(self.train),
            "data": self.data,
            "split": self.split,
            "evaluation": list(self.evaluation),
            "bootstrap": self.bootstrap,
        }


@dataclass(frozen=True, init=False)
class EvidenceArm:
    id: str
    role: ArmRole
    mode: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        _reject_direct_construction(type(self).__name__)

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, Any],
        *,
        suite: EvidenceSuite,
        index: int,
    ) -> "EvidenceArm":
        name = f"arms[{index}]"
        payload = _mapping(source, name)
        _strict_fields(name, payload, _ARM_FIELDS)
        arm_id = _slug(payload["id"], f"{name}.id")
        raw_role = str(payload["role"])
        if raw_role not in {"control", "intervention", "baseline"}:
            raise ValueError(f"{name}.role must be control, intervention, or baseline")
        role = cast(ArmRole, raw_role)
        mode = _slug(payload["mode"], f"{name}.mode")
        if mode not in _SUITE_MODES[suite]:
            raise ValueError(f"{name}.mode={mode!r} is not valid for the {suite} suite")
        expected_role: ArmRole
        if mode == "none":
            expected_role = "control"
        elif mode == "state_only":
            expected_role = "baseline"
        else:
            expected_role = "intervention"
        if role != expected_role:
            raise ValueError(f"{name}.mode={mode!r} requires role={expected_role!r}")
        return _resolved_instance(cls, id=arm_id, role=role, mode=mode)

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "role": self.role, "mode": self.mode}


@dataclass(frozen=True, init=False)
class EvidenceMetric:
    name: str
    kind: MetricKind
    direction: MetricDirection

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        _reject_direct_construction(type(self).__name__)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any], *, index: int) -> "EvidenceMetric":
        name = f"metrics[{index}]"
        payload = _mapping(source, name)
        _strict_fields(name, payload, _METRIC_FIELDS)
        metric_name = _slug(payload["name"], f"{name}.name")
        raw_kind = str(payload["kind"])
        if raw_kind not in {"binary", "continuous"}:
            raise ValueError(f"{name}.kind must be binary or continuous")
        raw_direction = str(payload["direction"])
        if raw_direction not in {"positive", "negative"}:
            raise ValueError(f"{name}.direction must be positive or negative")
        return _resolved_instance(
            cls,
            name=metric_name,
            kind=cast(MetricKind, raw_kind),
            direction=cast(MetricDirection, raw_direction),
        )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind, "direction": self.direction}


@dataclass(frozen=True, init=False)
class EvidenceBudget:
    dataset_episodes: int
    batch_size: int
    training_steps: int
    learning_rate: float
    evaluation_episodes: int
    bootstrap_samples: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        _reject_direct_construction(type(self).__name__)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "EvidenceBudget":
        payload = _mapping(source, "budget")
        _strict_fields("budget", payload, _BUDGET_FIELDS)
        return _resolved_instance(
            cls,
            dataset_episodes=_integer(
                payload["dataset_episodes"], "budget.dataset_episodes", positive=True
            ),
            batch_size=_integer(payload["batch_size"], "budget.batch_size", positive=True),
            training_steps=_integer(
                payload["training_steps"], "budget.training_steps", positive=True
            ),
            learning_rate=_positive_float(payload["learning_rate"], "budget.learning_rate"),
            evaluation_episodes=_integer(
                payload["evaluation_episodes"],
                "budget.evaluation_episodes",
                positive=True,
            ),
            bootstrap_samples=_integer(
                payload["bootstrap_samples"], "budget.bootstrap_samples", positive=True
            ),
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "dataset_episodes": self.dataset_episodes,
            "batch_size": self.batch_size,
            "training_steps": self.training_steps,
            "learning_rate": self.learning_rate,
            "evaluation_episodes": self.evaluation_episodes,
            "bootstrap_samples": self.bootstrap_samples,
        }


@dataclass(frozen=True, init=False)
class EvidenceOutput:
    run_root: str
    snapshot_root: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        _reject_direct_construction(type(self).__name__)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "EvidenceOutput":
        payload = _mapping(source, "output")
        _strict_fields("output", payload, _OUTPUT_FIELDS)
        return _resolved_instance(
            cls,
            run_root=_repository_path(payload["run_root"], "output.run_root", prefix=("outputs",)),
            snapshot_root=_repository_path(
                payload["snapshot_root"],
                "output.snapshot_root",
                prefix=("results", "v2"),
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {"run_root": self.run_root, "snapshot_root": self.snapshot_root}


@dataclass(frozen=True, init=False)
class EvidenceDesign:
    """A complete, immutable declaration of one controlled evidence matrix."""

    schema_version: int
    design_id: str
    suite: EvidenceSuite
    base_config: str
    seeds: EvidenceSeeds
    arms: tuple[EvidenceArm, ...]
    metrics: tuple[EvidenceMetric, ...]
    budget: EvidenceBudget
    output: EvidenceOutput

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        _reject_direct_construction(type(self).__name__)

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "EvidenceDesign":
        payload = _mapping(source, "root")
        _strict_fields("root", payload, _ROOT_FIELDS)
        schema_version = _integer(payload["schema_version"], "schema_version")
        if schema_version != EVIDENCE_DESIGN_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence design schema_version: {schema_version}")
        design_id = _slug(payload["design_id"], "design_id")
        raw_suite = str(payload["suite"])
        if raw_suite not in _SUITE_MODES:
            raise ValueError("suite must be language or visual")
        suite = cast(EvidenceSuite, raw_suite)
        base_config = _repository_path(
            payload["base_config"], "base_config", prefix=("configs", "v2")
        )
        if Path(base_config).suffix not in {".yaml", ".yml"}:
            raise ValueError("base_config must name a YAML file")
        seeds = EvidenceSeeds.from_mapping(_mapping(payload["seeds"], "seeds"))
        arms_source = _sequence(payload["arms"], "arms")
        arms = tuple(
            EvidenceArm.from_mapping(_mapping(value, f"arms[{index}]"), suite=suite, index=index)
            for index, value in enumerate(arms_source)
        )
        if len(arms) < 2:
            raise ValueError("arms must contain a control and at least one comparison")
        if len({arm.id for arm in arms}) != len(arms):
            raise ValueError("arms.id values must be unique")
        if sum(arm.role == "control" for arm in arms) != 1:
            raise ValueError("arms must contain exactly one control")
        metrics_source = _sequence(payload["metrics"], "metrics")
        metrics = tuple(
            EvidenceMetric.from_mapping(_mapping(value, f"metrics[{index}]"), index=index)
            for index, value in enumerate(metrics_source)
        )
        if not metrics:
            raise ValueError("metrics cannot be empty")
        if len({metric.name for metric in metrics}) != len(metrics):
            raise ValueError("metrics.name values must be unique")
        budget = EvidenceBudget.from_mapping(_mapping(payload["budget"], "budget"))
        if budget.evaluation_episodes != len(seeds.evaluation):
            raise ValueError("budget.evaluation_episodes must equal the number of evaluation seeds")
        output = EvidenceOutput.from_mapping(_mapping(payload["output"], "output"))
        return _resolved_instance(
            cls,
            schema_version=schema_version,
            design_id=design_id,
            suite=suite,
            base_config=base_config,
            seeds=seeds,
            arms=arms,
            metrics=metrics,
            budget=budget,
            output=output,
        )

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceDesign":
        source = Path(path)
        try:
            payload = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
        except yaml.YAMLError as exc:
            problem = getattr(exc, "problem", None) or "malformed YAML"
            raise ValueError(f"invalid YAML in {source}: {problem}") from exc
        if not isinstance(payload, Mapping):
            raise TypeError("evidence design file must contain a mapping")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "design_id": self.design_id,
            "suite": self.suite,
            "base_config": self.base_config,
            "seeds": self.seeds.to_dict(),
            "arms": [arm.to_dict() for arm in self.arms],
            "metrics": [metric.to_dict() for metric in self.metrics],
            "budget": self.budget.to_dict(),
            "output": self.output.to_dict(),
        }

    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def evidence_design_schema_descriptor() -> dict[str, Any]:
    """Describe the versioned public EvidenceDesign surface."""

    return {
        "contract": "EvidenceDesign",
        "schema_version": EVIDENCE_DESIGN_SCHEMA_VERSION,
        "root_fields": sorted(_ROOT_FIELDS),
        "required_root_fields": sorted(_ROOT_FIELDS),
        "section_fields": {
            "seeds": sorted(_SEED_FIELDS),
            "arm": sorted(_ARM_FIELDS),
            "metric": sorted(_METRIC_FIELDS),
            "budget": sorted(_BUDGET_FIELDS),
            "output": sorted(_OUTPUT_FIELDS),
        },
        "required_section_fields": {
            "seeds": sorted(_SEED_FIELDS),
            "arm": sorted(_ARM_FIELDS),
            "metric": sorted(_METRIC_FIELDS),
            "budget": sorted(_BUDGET_FIELDS),
            "output": sorted(_OUTPUT_FIELDS),
        },
        "registries": {
            "suites": sorted(_SUITE_MODES),
            "suite_modes": {suite: sorted(modes) for suite, modes in sorted(_SUITE_MODES.items())},
            "arm_roles": ["baseline", "control", "intervention"],
            "metric_kinds": ["binary", "continuous"],
            "metric_directions": ["negative", "positive"],
        },
    }
