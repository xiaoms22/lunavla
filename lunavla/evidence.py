"""Small deterministic statistics used to gate v2 modality claims."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np


Direction = Literal["positive", "negative"]
StatisticMethod = Literal["wilson", "hierarchical_paired_bootstrap"]

EVIDENCE_MANIFEST_SCHEMA_VERSION = 1
_SLUG = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binomial proportion."""

    if isinstance(successes, bool) or not isinstance(successes, Integral):
        raise TypeError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, Integral):
        raise TypeError("trials must be an integer")
    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must satisfy 0 <= successes <= trials")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be positive and finite")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True)
class PairedInterval:
    """A treatment-minus-control paired bootstrap result."""

    metric: str
    paired_n: int
    mean_difference: float
    lower: float
    upper: float
    bootstrap_samples: int
    seed: int

    def supports(self, direction: Direction) -> bool:
        """Whether the interval excludes zero in the declared direction."""

        if direction == "positive":
            return self.lower > 0
        if direction == "negative":
            return self.upper < 0
        raise ValueError(f"unsupported direction: {direction!r}")

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0 or self.upper < 0


def paired_bootstrap_interval(
    control: Sequence[float],
    treatment: Sequence[float],
    *,
    metric: str,
    samples: int = 10_000,
    seed: int = 202_601,
) -> PairedInterval:
    """Bootstrap paired treatment-minus-control differences.

    Inputs must already be aligned by a stable pair identifier. This function
    deliberately does not guess how to join unrelated rows.
    """

    control_array = np.asarray(control, dtype=np.float64)
    treatment_array = np.asarray(treatment, dtype=np.float64)
    if control_array.ndim != 1 or treatment_array.ndim != 1:
        raise ValueError("control and treatment must be rank-1")
    if control_array.shape != treatment_array.shape or control_array.size == 0:
        raise ValueError("control and treatment must be non-empty and equally sized")
    if not np.all(np.isfinite(control_array)) or not np.all(np.isfinite(treatment_array)):
        raise ValueError("control and treatment must contain only finite values")
    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not str(metric).strip():
        raise ValueError("metric cannot be empty")

    differences = treatment_array - control_array
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    means = np.mean(differences[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return PairedInterval(
        metric=str(metric),
        paired_n=int(differences.size),
        mean_difference=float(np.mean(differences)),
        lower=float(lower),
        upper=float(upper),
        bootstrap_samples=samples,
        seed=int(seed),
    )


@dataclass(frozen=True)
class HierarchicalPairedInterval:
    """A seed-clustered, episode-paired treatment-minus-control interval."""

    metric: str
    train_seeds: tuple[int, ...]
    train_seed_n: int
    episodes_per_seed: int
    paired_n: int
    mean_difference: float
    lower: float
    upper: float
    bootstrap_samples: int
    seed: int

    def supports(self, direction: Direction) -> bool:
        if direction == "positive":
            return self.lower > 0
        if direction == "negative":
            return self.upper < 0
        raise ValueError(f"unsupported direction: {direction!r}")

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0 or self.upper < 0


def _strict_integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _episode_key(value: Any, name: str) -> tuple[str, str]:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or string")
    if isinstance(value, Integral):
        return "integer", str(int(value))
    if isinstance(value, str) and value and value == value.strip():
        return "string", value
    if isinstance(value, str):
        raise ValueError(f"{name} must be non-empty and normalized")
    raise TypeError(f"{name} must be an integer or string")


def hierarchical_paired_bootstrap_interval(
    control: Sequence[float],
    treatment: Sequence[float],
    *,
    train_seeds: Sequence[int],
    episode_ids: Sequence[int | str],
    metric: str,
    samples: int = 10_000,
    seed: int = 202_611,
) -> HierarchicalPairedInterval:
    """Bootstrap paired differences by sampling train seeds, then episodes.

    Every training seed must contain exactly the same unique episode IDs. This
    fail-closed requirement prevents an incomplete or duplicated evaluation
    matrix from silently changing the estimand.
    """

    control_array = np.asarray(control, dtype=np.float64)
    treatment_array = np.asarray(treatment, dtype=np.float64)
    if control_array.ndim != 1 or treatment_array.ndim != 1:
        raise ValueError("control and treatment must be rank-1")
    if control_array.shape != treatment_array.shape or control_array.size == 0:
        raise ValueError("control and treatment must be non-empty and equally sized")
    if not np.all(np.isfinite(control_array)) or not np.all(np.isfinite(treatment_array)):
        raise ValueError("control and treatment must contain only finite values")
    if len(train_seeds) != control_array.size or len(episode_ids) != control_array.size:
        raise ValueError("train_seeds and episode_ids must align with every paired value")
    bootstrap_samples = _strict_integer(samples, "samples", positive=True)
    bootstrap_seed = _strict_integer(seed, "seed")
    metric_name = str(metric).strip()
    if not metric_name:
        raise ValueError("metric cannot be empty")

    differences = treatment_array - control_array
    rows: dict[tuple[int, tuple[str, str]], float] = {}
    episodes_by_seed: dict[int, set[tuple[str, str]]] = {}
    for index, (raw_train_seed, raw_episode_id, difference) in enumerate(
        zip(train_seeds, episode_ids, differences, strict=True)
    ):
        train_seed = _strict_integer(raw_train_seed, f"train_seeds[{index}]")
        episode_key = _episode_key(raw_episode_id, f"episode_ids[{index}]")
        pair_key = (train_seed, episode_key)
        if pair_key in rows:
            raise ValueError(
                "each (train_seed, episode_id) pair must be unique; "
                f"duplicate at index {index}"
            )
        rows[pair_key] = float(difference)
        episodes_by_seed.setdefault(train_seed, set()).add(episode_key)

    ordered_seeds = tuple(sorted(episodes_by_seed))
    if len(ordered_seeds) < 2:
        raise ValueError("hierarchical bootstrap requires at least two training seeds")
    expected_episodes = episodes_by_seed[ordered_seeds[0]]
    if not expected_episodes:
        raise ValueError("each training seed must contain at least one episode")
    for train_seed in ordered_seeds[1:]:
        if episodes_by_seed[train_seed] != expected_episodes:
            raise ValueError("every training seed must contain the same episode_id set")
    ordered_episodes = tuple(sorted(expected_episodes))
    difference_matrix = np.asarray(
        [
            [rows[(train_seed, episode_id)] for episode_id in ordered_episodes]
            for train_seed in ordered_seeds
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(bootstrap_seed)
    cluster_indices = rng.integers(
        0,
        len(ordered_seeds),
        size=(bootstrap_samples, len(ordered_seeds)),
    )
    episode_indices = rng.integers(
        0,
        len(ordered_episodes),
        size=(bootstrap_samples, len(ordered_seeds), len(ordered_episodes)),
    )
    selected_clusters = difference_matrix[cluster_indices]
    selected_differences = np.take_along_axis(
        selected_clusters,
        episode_indices,
        axis=2,
    )
    bootstrap_means = np.mean(selected_differences, axis=(1, 2))
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    paired_n = len(ordered_seeds) * len(ordered_episodes)
    return HierarchicalPairedInterval(
        metric=metric_name,
        train_seeds=ordered_seeds,
        train_seed_n=len(ordered_seeds),
        episodes_per_seed=len(ordered_episodes),
        paired_n=paired_n,
        mean_difference=float(np.mean(difference_matrix)),
        lower=float(lower),
        upper=float(upper),
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _slug(value: Any, name: str) -> str:
    result = _nonempty(value, name)
    if not _SLUG.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase slug")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _strict_mapping(
    source: Any,
    name: str,
    fields: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in source):
        raise TypeError(f"{name} field names must be strings")
    result = dict(source)
    if fields is not None:
        unknown = sorted(set(result) - fields)
        missing = sorted(fields - set(result))
        if unknown:
            raise ValueError(f"unknown field(s) in {name}: {', '.join(unknown)}")
        if missing:
            raise ValueError(f"missing field(s) in {name}: {', '.join(missing)}")
    return result


@dataclass(frozen=True)
class EvidenceSource:
    run_id: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _nonempty(self.run_id, "source.run_id"))
        object.__setattr__(
            self,
            "manifest_sha256",
            _sha256(self.manifest_sha256, "source.manifest_sha256"),
        )

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "EvidenceSource":
        payload = _strict_mapping(
            source, "source", {"run_id", "manifest_sha256"}
        )
        return cls(
            run_id=payload["run_id"],
            manifest_sha256=payload["manifest_sha256"],
        )

    def to_dict(self) -> dict[str, str]:
        return {"run_id": self.run_id, "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True)
class EvidenceStatistic:
    statistic_id: str
    metric: str
    scope: str
    method: StatisticMethod
    estimate: float
    lower: float
    upper: float
    sample_n: int
    train_seed_n: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "statistic_id", _slug(self.statistic_id, "statistic_id")
        )
        object.__setattr__(self, "metric", _slug(self.metric, "metric"))
        object.__setattr__(self, "scope", _nonempty(self.scope, "scope"))
        if self.method not in {"wilson", "hierarchical_paired_bootstrap"}:
            raise ValueError("method must be wilson or hierarchical_paired_bootstrap")
        for name in ("estimate", "lower", "upper"):
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError(f"{name} must be a number")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.lower > self.upper:
            raise ValueError("lower cannot exceed upper")
        object.__setattr__(
            self, "sample_n", _strict_integer(self.sample_n, "sample_n", positive=True)
        )
        if self.method == "wilson":
            if self.train_seed_n is not None:
                raise ValueError("Wilson statistics cannot set train_seed_n")
            if self.lower < 0 or self.upper > 1 or not 0 <= self.estimate <= 1:
                raise ValueError("Wilson statistics must remain within [0, 1]")
        else:
            if self.train_seed_n is None:
                raise ValueError("hierarchical statistics require train_seed_n")
            object.__setattr__(
                self,
                "train_seed_n",
                _strict_integer(self.train_seed_n, "train_seed_n", positive=True),
            )
            if self.train_seed_n < 2:
                raise ValueError("hierarchical statistics require at least two train seeds")

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "EvidenceStatistic":
        payload = _strict_mapping(
            source,
            "statistic",
            {
                "statistic_id",
                "metric",
                "scope",
                "method",
                "estimate",
                "lower",
                "upper",
                "sample_n",
                "train_seed_n",
            },
        )
        return cls(
            statistic_id=payload["statistic_id"],
            metric=payload["metric"],
            scope=payload["scope"],
            method=cast(StatisticMethod, payload["method"]),
            estimate=payload["estimate"],
            lower=payload["lower"],
            upper=payload["upper"],
            sample_n=payload["sample_n"],
            train_seed_n=payload["train_seed_n"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic_id": self.statistic_id,
            "metric": self.metric,
            "scope": self.scope,
            "method": self.method,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "sample_n": self.sample_n,
            "train_seed_n": self.train_seed_n,
        }


@dataclass(frozen=True)
class ClaimDecision:
    """A claim gate whose allowed value is derived solely from named checks."""

    claim_id: str
    checks: tuple[tuple[str, bool], ...]
    allowed_statement: str
    denied_statement: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _slug(self.claim_id, "claim_id"))
        if not self.checks:
            raise ValueError("claim checks cannot be empty")
        normalized: list[tuple[str, bool]] = []
        for index, (raw_name, raw_passed) in enumerate(self.checks):
            name = _slug(raw_name, f"checks[{index}].name")
            passed = _strict_bool(raw_passed, f"checks[{index}].passed")
            normalized.append((name, passed))
        normalized.sort(key=lambda item: item[0])
        if len({name for name, _ in normalized}) != len(normalized):
            raise ValueError("claim check names must be unique")
        object.__setattr__(self, "checks", tuple(normalized))
        object.__setattr__(
            self,
            "allowed_statement",
            _nonempty(self.allowed_statement, "allowed_statement"),
        )
        object.__setattr__(
            self,
            "denied_statement",
            _nonempty(self.denied_statement, "denied_statement"),
        )

    @classmethod
    def from_checks(
        cls,
        *,
        claim_id: str,
        checks: Mapping[str, bool],
        allowed_statement: str,
        denied_statement: str,
    ) -> "ClaimDecision":
        return cls(
            claim_id=claim_id,
            checks=tuple(checks.items()),
            allowed_statement=allowed_statement,
            denied_statement=denied_statement,
        )

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "ClaimDecision":
        payload = _strict_mapping(
            source,
            "claim",
            {
                "claim_id",
                "checks",
                "allowed",
                "failed_checks",
                "allowed_statement",
                "denied_statement",
                "statement",
            },
        )
        checks = _strict_mapping(payload["checks"], "claim.checks")
        decision = cls.from_checks(
            claim_id=payload["claim_id"],
            checks={name: _strict_bool(value, f"claim.checks.{name}") for name, value in checks.items()},
            allowed_statement=payload["allowed_statement"],
            denied_statement=payload["denied_statement"],
        )
        if _strict_bool(payload["allowed"], "claim.allowed") != decision.allowed:
            raise ValueError("claim.allowed does not match its checks")
        failed = payload["failed_checks"]
        if isinstance(failed, (str, bytes)) or not isinstance(failed, Sequence):
            raise TypeError("claim.failed_checks must be a sequence")
        if tuple(failed) != decision.failed_checks:
            raise ValueError("claim.failed_checks does not match its checks")
        if str(payload["statement"]) != decision.statement:
            raise ValueError("claim.statement does not match its checks")
        return decision

    @property
    def allowed(self) -> bool:
        return all(passed for _, passed in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.checks if not passed)

    @property
    def statement(self) -> str:
        return self.allowed_statement if self.allowed else self.denied_statement

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "checks": dict(self.checks),
            "allowed": self.allowed,
            "failed_checks": list(self.failed_checks),
            "allowed_statement": self.allowed_statement,
            "denied_statement": self.denied_statement,
            "statement": self.statement,
        }


@dataclass(frozen=True)
class EvidenceManifest:
    """Aggregate evidence with integrity-aware, fail-closed claim decisions."""

    schema_version: int
    design_id: str
    design_sha256: str
    reduced_design: bool
    matrix_complete: bool
    integrity_checks: tuple[tuple[str, bool], ...]
    sources: tuple[EvidenceSource, ...]
    statistics: tuple[EvidenceStatistic, ...]
    claims: tuple[ClaimDecision, ...]

    def __post_init__(self) -> None:
        version = _strict_integer(self.schema_version, "schema_version")
        if version != EVIDENCE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence manifest schema_version: {version}")
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "design_id", _slug(self.design_id, "design_id"))
        object.__setattr__(
            self, "design_sha256", _sha256(self.design_sha256, "design_sha256")
        )
        reduced_design = _strict_bool(self.reduced_design, "reduced_design")
        matrix_complete = _strict_bool(self.matrix_complete, "matrix_complete")
        object.__setattr__(self, "reduced_design", reduced_design)
        object.__setattr__(self, "matrix_complete", matrix_complete)
        if not self.integrity_checks:
            raise ValueError("integrity_checks cannot be empty")
        normalized_checks = tuple(
            sorted(
                (
                    _slug(name, f"integrity_checks[{index}].name"),
                    _strict_bool(passed, f"integrity_checks[{index}].passed"),
                )
                for index, (name, passed) in enumerate(self.integrity_checks)
            )
        )
        if len({name for name, _ in normalized_checks}) != len(normalized_checks):
            raise ValueError("integrity check names must be unique")
        object.__setattr__(self, "integrity_checks", normalized_checks)
        if matrix_complete and not self.sources:
            raise ValueError("a complete evidence matrix must include source manifests")
        if len({source.run_id for source in self.sources}) != len(self.sources):
            raise ValueError("source run_id values must be unique")
        object.__setattr__(self, "sources", tuple(sorted(self.sources, key=lambda item: item.run_id)))
        if len({stat.statistic_id for stat in self.statistics}) != len(self.statistics):
            raise ValueError("statistic_id values must be unique")
        object.__setattr__(
            self,
            "statistics",
            tuple(sorted(self.statistics, key=lambda item: item.statistic_id)),
        )
        if not self.claims:
            raise ValueError("claims cannot be empty")
        if len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError("claim_id values must be unique")
        object.__setattr__(self, "claims", tuple(sorted(self.claims, key=lambda item: item.claim_id)))
        global_gate = (
            not reduced_design
            and matrix_complete
            and all(passed for _, passed in normalized_checks)
        )
        if not global_gate and any(claim.allowed for claim in self.claims):
            raise ValueError(
                "claims must fail closed for reduced, incomplete, or integrity-failed evidence"
            )
        if any(claim.allowed for claim in self.claims) and not self.statistics:
            raise ValueError("an allowed claim requires aggregate statistics")

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "EvidenceManifest":
        payload = _strict_mapping(
            source,
            "evidence manifest",
            {
                "schema_version",
                "design_id",
                "design_sha256",
                "reduced_design",
                "matrix_complete",
                "integrity_checks",
                "sources",
                "statistics",
                "claims",
            },
        )
        checks = _strict_mapping(payload["integrity_checks"], "integrity_checks")
        sequences: dict[str, Sequence[Any]] = {}
        for name in ("sources", "statistics", "claims"):
            value = payload[name]
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"{name} must be a sequence")
            sequences[name] = value
        return cls(
            schema_version=payload["schema_version"],
            design_id=payload["design_id"],
            design_sha256=payload["design_sha256"],
            reduced_design=payload["reduced_design"],
            matrix_complete=payload["matrix_complete"],
            integrity_checks=tuple(
                (name, _strict_bool(value, f"integrity_checks.{name}"))
                for name, value in checks.items()
            ),
            sources=tuple(
                EvidenceSource.from_mapping(_strict_mapping(item, "source"))
                for item in sequences["sources"]
            ),
            statistics=tuple(
                EvidenceStatistic.from_mapping(_strict_mapping(item, "statistic"))
                for item in sequences["statistics"]
            ),
            claims=tuple(
                ClaimDecision.from_mapping(_strict_mapping(item, "claim"))
                for item in sequences["claims"]
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("evidence manifest root must be a mapping")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "design_id": self.design_id,
            "design_sha256": self.design_sha256,
            "reduced_design": self.reduced_design,
            "matrix_complete": self.matrix_complete,
            "integrity_checks": dict(self.integrity_checks),
            "sources": [source.to_dict() for source in self.sources],
            "statistics": [statistic.to_dict() for statistic in self.statistics],
            "claims": [claim.to_dict() for claim in self.claims],
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

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return target
