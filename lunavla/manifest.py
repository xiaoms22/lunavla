"""Versioned, hash-verifiable run manifest for the v2 engine."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from lunavla.config import ExperimentConfig
from lunavla.contracts import Transition


MANIFEST_SCHEMA_VERSION = 2


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
                    None
                    if next_image is None
                    else hashlib.sha256(next_image.tobytes()).hexdigest()
                ),
            },
            "terminated": transition.terminated,
            "info": _jsonable(transition.info),
        }
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
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
        return True, "unknown"
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
    ) -> "RunManifest":
        if len(data_sha256) != 64 or any(character not in "0123456789abcdef" for character in data_sha256):
            raise ValueError("data_sha256 must be a lowercase SHA-256 hex digest")
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
        )

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("manifest root must be an object")
        if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema_version: {payload.get('schema_version')}"
            )
        manifest = cls(**payload)
        resolved = ExperimentConfig.from_mapping(manifest.config)
        if resolved.sha256() != manifest.config_sha256:
            raise ValueError("manifest config_sha256 does not match its resolved config")
        return manifest

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
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
