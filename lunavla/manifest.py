"""Versioned, hash-verifiable run manifest for the v2 engine."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn, Sequence

import numpy as np

from lunavla.config import ExperimentConfig
from lunavla.contracts import Transition
from lunavla.artifact_contracts import (
    RUN_MANIFEST_READ_ONLY_SCHEMAS,
    RUN_MANIFEST_SCHEMA2_FIELDS,
    RUN_MANIFEST_SCHEMA3_FIELDS,
    RUN_MANIFEST_SCHEMA_VERSION,
)


MANIFEST_SCHEMA_VERSION = RUN_MANIFEST_SCHEMA_VERSION
_READ_ONLY_MANIFEST_SCHEMA_VERSIONS = RUN_MANIFEST_READ_ONLY_SCHEMAS
_SHA256_HEX = frozenset("0123456789abcdef")


class _FrozenDict(dict[str, Any]):
    """A JSON-compatible dictionary that rejects every mutation path."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        raise TypeError("manifest values are read-only")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        return {
            copy.deepcopy(key, memo): copy.deepcopy(value, memo)
            for key, value in self.items()
        }


class _FrozenList(list[Any]):
    """A JSON-compatible list that rejects every mutation path."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        raise TypeError("manifest values are read-only")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        return [copy.deepcopy(value, memo) for value in self]


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenDict({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return _FrozenList(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_thaw(item) for item in value]
    return value


def _validate_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_git_sha(value: Any) -> str:
    return _require_non_empty_string(value, "git_sha")


def _validate_embedded_sha256(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} mapping keys must be strings")
            item_name = f"{name}.{key}"
            if key.endswith("sha256") and item is not None:
                _validate_sha256(item, item_name)
            _validate_embedded_sha256(item, item_name)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_embedded_sha256(item, f"{name}[{index}]")


def _validate_seed_sequence(value: Any, name: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a non-empty sequence of non-negative integers")
    for index, seed in enumerate(value):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"{name}[{index}] must be a non-negative integer")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicate seeds")


def _validate_dataset_split(value: Any, *, schema_version: int) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("dataset_split must be a mapping")
    if schema_version == MANIFEST_SCHEMA_VERSION:
        expected = {"train", "validation", "test"}
        if set(value) != expected:
            raise ValueError("dataset_split must contain exactly train, validation, and test")
    all_ids: set[int | str] = set()
    for name, identifiers in value.items():
        _require_non_empty_string(name, "dataset_split key")
        if isinstance(identifiers, (str, bytes)) or not isinstance(identifiers, Sequence):
            raise TypeError(f"dataset_split.{name} must be a sequence")
        local_ids: set[int | str] = set()
        for index, identifier in enumerate(identifiers):
            if isinstance(identifier, bool) or not isinstance(identifier, (int, str)):
                raise TypeError(
                    f"dataset_split.{name}[{index}] must be an integer or string"
                )
            if isinstance(identifier, str) and not identifier:
                raise ValueError(f"dataset_split.{name}[{index}] must not be empty")
            if identifier in local_ids:
                raise ValueError(f"dataset_split.{name} contains duplicate ID {identifier!r}")
            if identifier in all_ids:
                raise ValueError(f"dataset split ID {identifier!r} appears in multiple splits")
            local_ids.add(identifier)
            all_ids.add(identifier)


def _validate_dependencies(value: Any) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("dependencies must be a non-empty string mapping")
    for name, version in value.items():
        _require_non_empty_string(name, "dependency name")
        _require_non_empty_string(version, f"dependencies.{name}")


def _validate_command(value: Any) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError("command must be a non-empty sequence of strings")
    for index, argument in enumerate(value):
        _require_non_empty_string(argument, f"command[{index}]")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("manifest mapping keys must be strings")
        return {key: _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("manifest values must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"manifest contains unsupported value type {type(value).__name__}")


def _manifest_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    converted = _jsonable(value)
    assert isinstance(converted, dict)
    return converted


def _manifest_mapping_list(
    value: Sequence[Mapping[str, Any]],
    name: str,
) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of mappings")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{name}[{index}] must be a mapping")
        result.append(_manifest_mapping(item, f"{name}[{index}]"))
    return result


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_transitions(transitions: Sequence[Transition]) -> str:
    """Hash in-memory transitions without writing a transient dataset file."""

    digest = hashlib.sha256()
    if not transitions:
        raise ValueError("cannot hash an empty transition sequence")
    for transition in transitions:
        if not isinstance(transition, Transition):
            raise TypeError("all items must be Transition instances")
        image = transition.observation.image
        next_image = transition.next_observation.image
        payload = {
            "observation": {
                "state": transition.observation.state.tolist(),
                "instruction": transition.observation.instruction,
                "image_shape": None if image is None else list(image.shape),
                "image_dtype": None if image is None else str(image.dtype),
                "image_sha256": (
                    None if image is None else hashlib.sha256(image.tobytes()).hexdigest()
                ),
            },
            "action": transition.action.tolist(),
            "reward": transition.reward,
            "next_observation": {
                "state": transition.next_observation.state.tolist(),
                "instruction": transition.next_observation.instruction,
                "image_shape": None if next_image is None else list(next_image.shape),
                "image_dtype": None if next_image is None else str(next_image.dtype),
                "image_sha256": (
                    None if next_image is None else hashlib.sha256(next_image.tobytes()).hexdigest()
                ),
            },
            "terminated": transition.terminated,
            "info": _jsonable(transition.info),
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_source_state(root: Path) -> tuple[bool, str | None]:
    """Return dirty state and a digest of tracked diffs plus untracked source files."""

    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if status.returncode != 0:
        unavailable = hashlib.sha256(b"git-source-state-unavailable").hexdigest()
        return True, unavailable
    if not status.stdout:
        return False, None
    digest = hashlib.sha256()
    for command in (
        ["git", "diff", "--binary", "HEAD"],
        ["git", "diff", "--binary", "--cached", "HEAD"],
    ):
        result = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
        )
        digest.update(result.stdout)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if untracked.returncode == 0:
        for raw_path in sorted(item for item in untracked.stdout.split(b"\0") if item):
            digest.update(raw_path)
            path = root / raw_path.decode("utf-8", errors="surrogateescape")
            if path.is_file():
                digest.update(path.read_bytes())
    return True, digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in ("numpy", "Pillow", "PyYAML", "torch", "torchvision", "lerobot"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "not-installed"
    return result


def _runtime_determinism(device: str) -> dict[str, Any]:
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    torch_version: str = "not-installed"
    torch_deterministic_algorithms: bool | None = None
    cudnn_deterministic: bool | None = None
    cudnn_benchmark: bool | None = None
    try:
        import torch

        torch_version = str(torch.__version__)
        torch_deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())
        cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    except (ImportError, OSError):
        pass

    torch_ready = torch_version == "not-installed" or torch_deterministic_algorithms is True
    cudnn_ready = not str(device).startswith("cuda") or (
        cudnn_deterministic is True and cudnn_benchmark is False
    )
    flags_satisfied = (
        python_hash_seed is not None and str(device) == "cpu" and torch_ready and cudnn_ready
    )
    return {
        # One process can record its flags; repeatability is verified separately.
        "status": "unverified",
        "deterministic_flags_satisfied": flags_satisfied,
        "device": str(device),
        "PYTHONHASHSEED": python_hash_seed,
        "numpy_bit_generator": np.random.PCG64.__name__,
        "torch_version": torch_version,
        "torch_deterministic_algorithms": torch_deterministic_algorithms,
        "cudnn_deterministic": cudnn_deterministic,
        "cudnn_benchmark": cudnn_benchmark,
    }


def _merge_recorded_mapping(
    recorded: Mapping[str, Any],
    supplied: Mapping[str, Any] | None,
    name: str,
) -> dict[str, Any]:
    result = _manifest_mapping(recorded, name)
    if supplied is None:
        return result
    additions = _manifest_mapping(supplied, name)
    for key, value in additions.items():
        if key in result and result[key] != value:
            raise ValueError(f"{name}.{key} conflicts with the recorded run value")
        result[key] = value
    return result


def _portable_command(root: Path, command: Iterable[str]) -> list[str]:
    resolved_root = root.resolve()
    result: list[str] = []
    for index, raw in enumerate(command):
        value = str(raw)
        candidate = Path(value)
        if index == 0 and candidate.name.startswith("python"):
            result.append("python")
        elif candidate.is_absolute():
            try:
                result.append(candidate.resolve().relative_to(resolved_root).as_posix())
            except ValueError:
                result.append(candidate.name)
        else:
            result.append(value)
    return result


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    git_sha: str
    git_dirty: bool
    source_diff_sha256: str | None
    config: dict[str, Any]
    config_sha256: str
    data_sha256: str
    dataset_split: dict[str, list[int | str]]
    data_seeds: list[int]
    train_seeds: list[int]
    eval_seeds: list[int]
    python_version: str
    dependencies: dict[str, str]
    policy_id: str
    task_id: str
    checkpoint_sha256: str
    artifact_sha256: dict[str, str]
    command: list[str]
    interventions: dict[str, str]
    pair_ids: list[str]
    paired_intervals: list[dict[str, Any]]
    metrics: dict[str, Any] = field(default_factory=dict)
    design_id: str | None = None
    design_sha256: str | None = None
    condition: dict[str, Any] = field(default_factory=dict)
    eval_fixture: dict[str, Any] = field(default_factory=dict)
    eval_fixture_sha256: str | None = None
    paired_data: dict[str, Any] = field(default_factory=dict)
    paired_data_sha256: str | None = None
    arms: list[dict[str, Any]] = field(default_factory=list)
    pairs: list[dict[str, Any]] = field(default_factory=list)
    runtime_determinism: dict[str, Any] = field(default_factory=lambda: {"status": "unverified"})

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version
            not in {MANIFEST_SCHEMA_VERSION, *_READ_ONLY_MANIFEST_SCHEMA_VERSIONS}
        ):
            raise ValueError(f"unsupported manifest schema_version: {self.schema_version}")
        for name in ("run_id", "python_version", "policy_id", "task_id"):
            _require_non_empty_string(getattr(self, name), name)
        _validate_git_sha(self.git_sha)
        if not isinstance(self.git_dirty, bool):
            raise TypeError("git_dirty must be boolean")
        if self.git_dirty:
            if self.source_diff_sha256 is None:
                raise ValueError("dirty manifests require source_diff_sha256")
            _validate_sha256(self.source_diff_sha256, "source_diff_sha256")
        elif self.source_diff_sha256 is not None:
            raise ValueError("clean manifests must set source_diff_sha256 to null")

        for name in (
            "config_sha256",
            "data_sha256",
            "checkpoint_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        _validate_dataset_split(self.dataset_split, schema_version=self.schema_version)
        _validate_seed_sequence(self.data_seeds, "data_seeds")
        _validate_seed_sequence(self.train_seeds, "train_seeds")
        _validate_seed_sequence(self.eval_seeds, "eval_seeds")
        _validate_dependencies(self.dependencies)
        _validate_command(self.command)

        object.__setattr__(self, "config", _manifest_mapping(self.config, "config"))
        object.__setattr__(
            self,
            "dataset_split",
            _manifest_mapping(self.dataset_split, "dataset_split"),
        )
        object.__setattr__(
            self,
            "dependencies",
            _manifest_mapping(self.dependencies, "dependencies"),
        )
        object.__setattr__(
            self,
            "artifact_sha256",
            _manifest_mapping(self.artifact_sha256, "artifact_sha256"),
        )
        object.__setattr__(
            self,
            "interventions",
            _manifest_mapping(self.interventions, "interventions"),
        )
        object.__setattr__(self, "metrics", _manifest_mapping(self.metrics, "metrics"))
        for name in ("data_seeds", "train_seeds", "eval_seeds", "command", "pair_ids"):
            normalized = _jsonable(getattr(self, name))
            if not isinstance(normalized, list):
                raise TypeError(f"{name} must be a sequence")
            object.__setattr__(self, name, normalized)
        object.__setattr__(
            self,
            "paired_intervals",
            _manifest_mapping_list(self.paired_intervals, "paired_intervals"),
        )

        resolved = ExperimentConfig.from_mapping(self.config)
        if resolved.sha256() != self.config_sha256:
            raise ValueError("manifest config_sha256 does not match its resolved config")
        if self.schema_version == MANIFEST_SCHEMA_VERSION:
            expected_data_seeds = [int(resolved.dataset["seed"])]
            expected_train_seeds = [int(resolved.training["seed"])]
            configured_eval_seeds = resolved.evaluation.get("seeds")
            if configured_eval_seeds is None:
                first_eval_seed = int(resolved.evaluation["seed"])
                configured_eval_seeds = list(
                    range(
                        first_eval_seed,
                        first_eval_seed + int(resolved.evaluation["episodes"]),
                    )
                )
            if self.data_seeds != expected_data_seeds:
                raise ValueError("data_seeds conflicts with config.dataset.seed")
            if self.train_seeds != expected_train_seeds:
                raise ValueError("train_seeds conflicts with config.training.seed")
            if self.eval_seeds != [int(seed) for seed in configured_eval_seeds]:
                raise ValueError("eval_seeds conflicts with config.evaluation seeds")
            if self.policy_id != str(resolved.policy["type"]):
                raise ValueError("policy_id conflicts with config.policy.type")
            if self.task_id != str(resolved.task["id"]):
                raise ValueError("task_id conflicts with config.task.id")

        for name, digest in self.artifact_sha256.items():
            _require_non_empty_string(name, "artifact_sha256 key")
            _validate_sha256(digest, f"artifact_sha256.{name}")
        for name, mode in self.interventions.items():
            _require_non_empty_string(name, "interventions key")
            _require_non_empty_string(mode, f"interventions.{name}")
        seen_pair_ids: set[str] = set()
        for index, pair_id in enumerate(self.pair_ids):
            normalized_pair_id = _require_non_empty_string(pair_id, f"pair_ids[{index}]")
            if normalized_pair_id in seen_pair_ids:
                raise ValueError("pair_ids must not contain duplicates")
            seen_pair_ids.add(normalized_pair_id)
        _validate_embedded_sha256(self.config, "config")
        _validate_embedded_sha256(self.metrics, "metrics")

        if (self.design_id is None) != (self.design_sha256 is None):
            raise ValueError("design_id and design_sha256 must be provided together")
        if self.design_id is not None:
            _require_non_empty_string(self.design_id, "design_id")
            assert self.design_sha256 is not None
            _validate_sha256(self.design_sha256, "design_sha256")

        object.__setattr__(self, "condition", _manifest_mapping(self.condition, "condition"))
        object.__setattr__(
            self,
            "eval_fixture",
            _manifest_mapping(self.eval_fixture, "eval_fixture"),
        )
        object.__setattr__(
            self,
            "paired_data",
            _manifest_mapping(self.paired_data, "paired_data"),
        )
        object.__setattr__(self, "arms", _manifest_mapping_list(self.arms, "arms"))
        object.__setattr__(self, "pairs", _manifest_mapping_list(self.pairs, "pairs"))
        object.__setattr__(
            self,
            "runtime_determinism",
            _manifest_mapping(self.runtime_determinism, "runtime_determinism"),
        )
        for name in ("condition", "eval_fixture", "paired_data", "arms", "pairs"):
            _validate_embedded_sha256(getattr(self, name), name)
        if self.schema_version == MANIFEST_SCHEMA_VERSION:
            if self.eval_fixture_sha256 is None:
                raise ValueError("eval_fixture_sha256 is required for schema 3")
            if self.paired_data_sha256 is None:
                raise ValueError("paired_data_sha256 is required for schema 3")
            _validate_sha256(self.eval_fixture_sha256, "eval_fixture_sha256")
            _validate_sha256(self.paired_data_sha256, "paired_data_sha256")
            if self.eval_fixture_sha256 != _sha256_json(self.eval_fixture):
                raise ValueError("eval_fixture_sha256 does not match eval_fixture")
            if self.paired_data_sha256 != _sha256_json(self.paired_data):
                raise ValueError("paired_data_sha256 does not match paired_data")
            required_fields = {
                "condition": {"language_ablation", "image_ablation"},
                "eval_fixture": {
                    "task_id",
                    "family",
                    "execution_mode",
                    "eval_seeds",
                    "max_steps",
                },
                "paired_data": {"pair_ids", "paired_intervals"},
                "runtime_determinism": {
                    "status",
                    "device",
                    "PYTHONHASHSEED",
                    "numpy_bit_generator",
                    "torch_version",
                    "torch_deterministic_algorithms",
                    "cudnn_deterministic",
                    "cudnn_benchmark",
                    "deterministic_flags_satisfied",
                },
            }
            for name, required in required_fields.items():
                value = getattr(self, name)
                missing = sorted(required - set(value))
                if missing:
                    raise ValueError(f"{name} is missing required field(s): {', '.join(missing)}")
            if self.runtime_determinism["status"] not in {"verified", "unverified"}:
                raise ValueError("runtime_determinism.status must be verified or unverified")
            if not isinstance(
                self.runtime_determinism["deterministic_flags_satisfied"], bool
            ):
                raise TypeError(
                    "runtime_determinism.deterministic_flags_satisfied must be boolean"
                )
            _require_non_empty_string(
                self.runtime_determinism["device"], "runtime_determinism.device"
            )
            _require_non_empty_string(
                self.runtime_determinism["numpy_bit_generator"],
                "runtime_determinism.numpy_bit_generator",
            )
            _require_non_empty_string(
                self.runtime_determinism["torch_version"],
                "runtime_determinism.torch_version",
            )
            python_hash_seed = self.runtime_determinism["PYTHONHASHSEED"]
            if python_hash_seed is not None and not isinstance(python_hash_seed, str):
                raise TypeError("runtime_determinism.PYTHONHASHSEED must be a string or null")
            for name in (
                "torch_deterministic_algorithms",
                "cudnn_deterministic",
                "cudnn_benchmark",
            ):
                if self.runtime_determinism[name] is not None and not isinstance(
                    self.runtime_determinism[name], bool
                ):
                    raise TypeError(f"runtime_determinism.{name} must be boolean or null")
            if (
                self.runtime_determinism["status"] == "verified"
                and self.runtime_determinism["deterministic_flags_satisfied"] is not True
            ):
                raise ValueError("verified runtime_determinism requires deterministic flags")

        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, (Mapping, list, tuple)):
                object.__setattr__(self, item.name, _deep_freeze(value))

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        config: ExperimentConfig,
        data_sha256: str,
        checkpoint_path: str | Path,
        dataset_split: Mapping[str, Iterable[int | str]],
        command: Iterable[str],
        metrics: Mapping[str, Any],
        ablation: Mapping[str, Any] | None = None,
        artifact_paths: Mapping[str, Path] | None = None,
        design_id: str | None = None,
        design_sha256: str | None = None,
        condition: Mapping[str, Any] | None = None,
        eval_fixture: Mapping[str, Any] | None = None,
        paired_data: Mapping[str, Any] | None = None,
        arms: Sequence[Mapping[str, Any]] | None = None,
        pairs: Sequence[Mapping[str, Any]] | None = None,
        runtime_determinism: Mapping[str, Any] | None = None,
    ) -> "RunManifest":
        _validate_sha256(data_sha256, "data_sha256")
        eval_seeds = config.evaluation.get("seeds")
        if eval_seeds is None:
            start = int(config.evaluation["seed"])
            eval_seeds = list(range(start, start + int(config.evaluation["episodes"])))
        git_dirty, source_diff_sha256 = git_source_state(root)
        ablation_payload = dict(ablation or {})
        interval = ablation_payload.get("interval")
        paired_intervals = [dict(interval)] if isinstance(interval, Mapping) else []
        mode = ablation_payload.get("ablation_mode")
        interventions = {"mode": str(mode)} if mode is not None else {}
        pair_ids = [str(value) for value in ablation_payload.get("pair_ids", [])]
        recorded_condition = {
            "language_ablation": str(config.evaluation["language_ablation"]),
            "image_ablation": str(config.evaluation["image_ablation"]),
        }
        recorded_fixture = {
            "task_id": str(config.task["id"]),
            "family": str(config.task["family"]),
            "execution_mode": str(config.evaluation["execution_mode"]),
            "eval_seeds": [int(seed) for seed in eval_seeds],
            "max_steps": int(config.task["max_steps"]),
        }
        recorded_paired_data = {
            "pair_ids": pair_ids,
            "paired_intervals": paired_intervals,
        }
        recorded_runtime = _runtime_determinism(str(config.training["device"]))
        resolved_condition = _merge_recorded_mapping(
            recorded_condition,
            condition,
            "condition",
        )
        resolved_fixture = _merge_recorded_mapping(
            recorded_fixture,
            eval_fixture,
            "eval_fixture",
        )
        resolved_paired_data = _merge_recorded_mapping(
            recorded_paired_data,
            paired_data,
            "paired_data",
        )
        return cls(
            schema_version=MANIFEST_SCHEMA_VERSION,
            run_id=config.project_name,
            git_sha=_git_sha(root),
            git_dirty=git_dirty,
            source_diff_sha256=source_diff_sha256,
            config=config.to_dict(),
            config_sha256=config.sha256(),
            data_sha256=data_sha256,
            dataset_split={
                str(name): sorted(set(values), key=lambda value: str(value))
                for name, values in dataset_split.items()
            },
            data_seeds=[int(config.dataset["seed"])],
            train_seeds=[int(config.training["seed"])],
            eval_seeds=[int(seed) for seed in eval_seeds],
            python_version=platform.python_version(),
            dependencies=_dependency_versions(),
            policy_id=str(config.policy["type"]),
            task_id=str(config.task["id"]),
            checkpoint_sha256=sha256_file(checkpoint_path),
            artifact_sha256={
                str(name): sha256_file(path)
                for name, path in sorted((artifact_paths or {}).items())
            },
            command=_portable_command(root, command),
            interventions=interventions,
            pair_ids=pair_ids,
            paired_intervals=paired_intervals,
            metrics=_jsonable(dict(metrics)),
            design_id=design_id,
            design_sha256=design_sha256,
            condition=resolved_condition,
            eval_fixture=resolved_fixture,
            eval_fixture_sha256=_sha256_json(resolved_fixture),
            paired_data=resolved_paired_data,
            paired_data_sha256=_sha256_json(resolved_paired_data),
            arms=_manifest_mapping_list(arms or (), "arms"),
            pairs=_manifest_mapping_list(
                pairs if pairs is not None else tuple({"pair_id": pair_id} for pair_id in pair_ids),
                "pairs",
            ),
            runtime_determinism=_merge_recorded_mapping(
                recorded_runtime,
                runtime_determinism,
                "runtime_determinism",
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("manifest root must be an object")
        schema_version = payload.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version
            not in {MANIFEST_SCHEMA_VERSION, *_READ_ONLY_MANIFEST_SCHEMA_VERSIONS}
        ):
            raise ValueError(f"unsupported manifest schema_version: {schema_version}")
        expected = (
            RUN_MANIFEST_SCHEMA3_FIELDS
            if schema_version == MANIFEST_SCHEMA_VERSION
            else RUN_MANIFEST_SCHEMA2_FIELDS
        )
        unknown = sorted(set(payload) - expected)
        if unknown:
            raise ValueError("unknown manifest field(s): " + ", ".join(unknown))
        missing = sorted(expected - set(payload))
        if missing:
            raise ValueError("missing manifest field(s): " + ", ".join(missing))
        manifest = cls(**payload)
        resolved = ExperimentConfig.from_mapping(manifest.config)
        if manifest.schema_version == MANIFEST_SCHEMA_VERSION:
            expected_condition = {
                "language_ablation": str(resolved.evaluation["language_ablation"]),
                "image_ablation": str(resolved.evaluation["image_ablation"]),
            }
            for key, expected_value in expected_condition.items():
                if manifest.condition[key] != expected_value:
                    raise ValueError(f"manifest condition.{key} conflicts with config")
            expected_fixture: dict[str, Any] = {
                "task_id": str(resolved.task["id"]),
                "family": str(resolved.task["family"]),
                "execution_mode": str(resolved.evaluation["execution_mode"]),
                "eval_seeds": manifest.eval_seeds,
                "max_steps": int(resolved.task["max_steps"]),
            }
            for key, expected_value in expected_fixture.items():
                if manifest.eval_fixture[key] != expected_value:
                    raise ValueError(f"manifest eval_fixture.{key} conflicts with config")
            if manifest.paired_data["pair_ids"] != manifest.pair_ids:
                raise ValueError("manifest paired_data.pair_ids conflicts with pair_ids")
            if manifest.paired_data["paired_intervals"] != manifest.paired_intervals:
                raise ValueError(
                    "manifest paired_data.paired_intervals conflicts with paired_intervals"
                )
            if manifest.runtime_determinism["device"] != str(resolved.training["device"]):
                raise ValueError("manifest runtime_determinism.device conflicts with config")
        return manifest

    def to_dict(self) -> dict[str, Any]:
        allowed = (
            RUN_MANIFEST_SCHEMA3_FIELDS
            if self.schema_version == MANIFEST_SCHEMA_VERSION
            else RUN_MANIFEST_SCHEMA2_FIELDS
        )
        return {
            item.name: _deep_thaw(getattr(self, item.name))
            for item in fields(self)
            if item.name in allowed
        }

    def write(self, path: str | Path) -> Path:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("schema 2 manifests are read-only and cannot be written")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def verify_run_dir(cls, run_dir: str | Path) -> "RunManifest":
        root = Path(run_dir)
        manifest = cls.load(root / "manifest.json")
        checkpoint_name = str(manifest.config["artifacts"]["checkpoint_name"])
        checkpoint_path = (root / checkpoint_name).resolve()
        if root.resolve() not in checkpoint_path.parents:
            raise ValueError("manifest checkpoint path escapes the run directory")
        if sha256_file(checkpoint_path) != manifest.checkpoint_sha256:
            raise ValueError("manifest checkpoint_sha256 does not match the checkpoint")
        for name, expected in manifest.artifact_sha256.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"manifest artifact path is unsafe: {name}")
            path = (root / relative).resolve()
            if root.resolve() not in path.parents:
                raise ValueError(f"manifest artifact path escapes the run directory: {name}")
            if not path.is_file():
                raise FileNotFoundError(f"manifest artifact is missing: {name}")
            if sha256_file(path) != expected:
                raise ValueError(f"manifest artifact hash mismatch: {name}")
        return manifest
