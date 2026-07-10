from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from dataset import VLARecord
from trainer.config import ExperimentConfig


MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
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


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("numpy", "PyYAML"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


@dataclass
class RunManifest:
    """Traceability record for a training/evaluation run."""

    schema_version: int
    run_id: str
    git_sha: str
    config: dict[str, Any]
    config_sha256: str
    data_sha256: str
    dataset_split: dict[str, list[int]]
    train_seeds: list[int]
    eval_seeds: list[int]
    python_version: str
    dependencies: dict[str, str]
    policy_id: str
    task_id: str
    checkpoint_sha256: str
    command: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        config: ExperimentConfig,
        data_path: Path,
        checkpoint_path: Path,
        splits: Mapping[str, Iterable[VLARecord]],
        command: list[str],
        metrics: dict[str, Any],
    ) -> "RunManifest":
        eval_seeds = config.eval.get("seeds")
        if eval_seeds is None:
            start = int(config.eval["seed"])
            eval_seeds = list(range(start, start + int(config.eval["episodes"])))
        split_ids = {
            name: sorted({record.episode_id for record in records})
            for name, records in splits.items()
        }
        return cls(
            schema_version=MANIFEST_SCHEMA_VERSION,
            run_id=config.project_name,
            git_sha=_git_sha(root),
            config=config.to_dict(),
            config_sha256=config.sha256(),
            data_sha256=sha256_file(data_path),
            dataset_split=split_ids,
            train_seeds=[int(config.dataset["seed"]), int(config.training["seed"])],
            eval_seeds=[int(seed) for seed in eval_seeds],
            python_version=platform.python_version(),
            dependencies=_dependency_versions(),
            policy_id=str(config.policy["type"]),
            task_id=config.task,
            checkpoint_sha256=sha256_file(checkpoint_path),
            command=list(command),
            metrics=dict(metrics),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema_version: {payload.get('schema_version')}")
        if "run_id" not in payload:
            config = payload.get("config")
            project_name = config.get("project_name") if isinstance(config, dict) else None
            if not project_name:
                raise ValueError("schema-1 manifest is missing run_id and config.project_name")
            warnings.warn(
                "schema-1 manifest without run_id is deprecated; deriving it from config.project_name",
                DeprecationWarning,
                stacklevel=2,
            )
            payload["run_id"] = str(project_name)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def add_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics.update(metrics)
