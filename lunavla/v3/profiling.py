from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .artifacts import sha256_file
from .config import ExperimentConfig
from .engine import EngineV3, dataset_for_config
from .policy import PolicyBatchV3


PROFILE_STATEMENT = (
    "CPU measurements describe one recorded environment and do not establish policy superiority."
)
_ROOT = Path(__file__).resolve().parents[2]
_SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")
_GIT_SHA = __import__("re").compile(r"^[0-9a-f]{40}$")


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return value


def _relative_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or path in {Path("."), Path("..")} or ".." in path.parts:
        raise ValueError(f"{name} must be a contained relative path")
    return path.as_posix()


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True)
class PolicyProfileDesignV1:
    config: str
    warmup: int
    measurements: int
    batch_size: int
    device: str
    threads: int
    python_version: str
    platforms: tuple[str, ...]
    output_dir: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PolicyProfileDesignV1 schema_version must be integer 1")
        config = _relative_path(self.config, "config")
        output = _relative_path(self.output_dir, "output_dir")
        warmup = _integer(self.warmup, "warmup", positive=True)
        measurements = _integer(self.measurements, "measurements", positive=True)
        batch = _integer(self.batch_size, "batch_size", positive=True)
        threads = _integer(self.threads, "threads", positive=True)
        if warmup != 5 or measurements != 20:
            raise ValueError("the standard CPU profile requires 5 warmup and 20 measurements")
        if self.device != "cpu":
            raise ValueError("v3.0 policy profiling supports CPU only")
        if self.python_version != "3.12":
            raise ValueError("policy profile requires Python 3.12")
        platforms = tuple(self.platforms)
        if not platforms or any(item not in {"linux", "darwin"} for item in platforms):
            raise ValueError("platforms must contain linux and/or darwin")
        if len(platforms) != len(set(platforms)):
            raise ValueError("platforms cannot contain duplicates")
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "output_dir", output)
        object.__setattr__(self, "warmup", warmup)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "batch_size", batch)
        object.__setattr__(self, "threads", threads)
        object.__setattr__(self, "platforms", platforms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": self.config,
            "warmup": self.warmup,
            "measurements": self.measurements,
            "batch_size": self.batch_size,
            "device": self.device,
            "threads": self.threads,
            "python_version": self.python_version,
            "platforms": list(self.platforms),
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PolicyProfileDesignV1":
        fields = {
            "schema_version", "config", "warmup", "measurements", "batch_size",
            "device", "threads", "python_version", "platforms", "output_dir",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("PolicyProfileDesignV1 requires exact fields")
        platforms = value["platforms"]
        if isinstance(platforms, (str, bytes, Mapping)) or not isinstance(platforms, Sequence):
            raise TypeError("platforms must be a sequence")
        payload = dict(value)
        payload["platforms"] = tuple(platforms)
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class PolicyProfileManifestV1:
    git_sha: str
    git_dirty: bool
    design_sha256: str
    config_sha256: str
    dependency_lock_sha256: str
    environment_sha256: str
    metrics_sha256: str
    policy_id: str
    device: str
    warmup: int
    measurements: int
    batch_size: int
    threads: int
    release_eligible: bool
    comparative_claim_allowed: bool
    statement: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PolicyProfileManifestV1 schema_version must be integer 1")
        if not _GIT_SHA.fullmatch(self.git_sha):
            raise ValueError("git_sha must be a full lowercase Git SHA")
        if not isinstance(self.git_dirty, bool):
            raise TypeError("git_dirty must be boolean")
        for name in (
            "design_sha256", "config_sha256", "dependency_lock_sha256",
            "environment_sha256", "metrics_sha256",
        ):
            if not _SHA256.fullmatch(str(getattr(self, name))):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if not self.policy_id or self.device != "cpu":
            raise ValueError("profile policy_id must be set and device must be cpu")
        for name in ("warmup", "measurements", "batch_size", "threads"):
            _integer(getattr(self, name), name, positive=True)
        if not isinstance(self.release_eligible, bool):
            raise TypeError("release_eligible must be boolean")
        if self.release_eligible and self.git_dirty:
            raise ValueError("dirty profiles cannot be release eligible")
        if self.comparative_claim_allowed is not False:
            raise ValueError("CPU profiles cannot open comparative claims")
        if self.statement != PROFILE_STATEMENT:
            raise ValueError("profile statement must use the fixed wording")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PolicyProfileManifestV1":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("PolicyProfileManifestV1 requires exact fields")
        return cls(**dict(value))

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


def _git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=_ROOT, text=True
            ).strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40, True


def _resolve_input(design_path: Path, relative: str) -> Path:
    candidates = (
        (design_path.resolve().parent / relative).resolve(),
        (_ROOT / relative).resolve(),
    )
    for candidate in candidates:
        if candidate.is_file() and (
            candidate.is_relative_to(design_path.resolve().parent)
            or candidate.is_relative_to(_ROOT)
        ):
            return candidate
    raise FileNotFoundError(f"profile config does not exist: {relative}")


def _resolve_output(design_path: Path, relative: str) -> Path:
    resolved = design_path.resolve()
    root = _ROOT if resolved.is_relative_to(_ROOT) else resolved.parent
    output = (root / relative).resolve()
    if not output.is_relative_to(root):
        raise ValueError("profile output escapes its allowed root")
    return output


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (20,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("profile measurements must contain 20 positive finite values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _measure_profile(
    config: ExperimentConfig, design: PolicyProfileDesignV1
) -> tuple[dict[str, Any], dict[str, Any]]:
    os.environ["OMP_NUM_THREADS"] = str(design.threads)
    os.environ["MKL_NUM_THREADS"] = str(design.threads)
    try:
        import torch

        torch.set_num_threads(design.threads)
    except ModuleNotFoundError:
        pass
    bundle = dataset_for_config(config)
    engine = EngineV3(config)
    policy, _losses = engine.train(bundle.source("train"))
    if engine.policy_spec is None:
        raise RuntimeError("profile engine did not resolve PolicySpecV3")
    episodes = tuple(bundle.source("train").load())
    samples = engine._samples(
        episodes,
        history=engine.policy_spec.history,
        chunk_size=engine.policy_spec.horizon
        if engine.policy_spec.policy_id == "diffusion_v3"
        else engine.policy_spec.chunk_size,
    )
    selected = tuple(samples[index % len(samples)] for index in range(design.batch_size))
    batch = PolicyBatchV3(selected, device="cpu")
    inference_sample = selected[0]
    step = int(config.training["steps"])

    def train_once() -> float:
        nonlocal step
        started = time.perf_counter_ns()
        result = policy.train_step(
            batch, learning_rate=config.training["learning_rate"], step=step
        )
        step += 1
        if not result.finite or not math.isfinite(result.loss):
            raise FloatingPointError("profile train step was non-finite")
        return (time.perf_counter_ns() - started) / 1_000_000.0

    def infer_once() -> float:
        started = time.perf_counter_ns()
        for _ in range(design.batch_size):
            chunk = policy.predict_chunk(inference_sample)
            if not np.all(np.isfinite(chunk.values)):
                raise FloatingPointError("profile inference returned non-finite actions")
        return (time.perf_counter_ns() - started) / 1_000_000.0

    for _ in range(design.warmup):
        train_once()
        infer_once()
    training_ms = [train_once() for _ in range(design.measurements)]
    inference_ms = [infer_once() for _ in range(design.measurements)]
    metrics = {
        "training_latency_ms": training_ms,
        "inference_latency_ms": inference_ms,
        "training_summary_ms": _summary(training_ms),
        "inference_summary_ms": _summary(inference_ms),
        "training_samples_per_second": [
            design.batch_size / (value / 1000.0) for value in training_ms
        ],
        "inference_samples_per_second": [
            design.batch_size / (value / 1000.0) for value in inference_ms
        ],
        "peak_rss_bytes": _peak_rss_bytes(),
        "comparative_claim_allowed": False,
        "statement": PROFILE_STATEMENT,
    }
    environment = {
        "platform": sys.platform,
        "architecture": platform.machine().lower(),
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "numpy_version": np.__version__,
        "cpu_count": os.cpu_count(),
        "threads": design.threads,
        "device": "cpu",
    }
    return metrics, environment


def run_profile(design_path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(design_path)
    source = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(source, Mapping):
        raise TypeError("profile design YAML must contain a mapping")
    design = PolicyProfileDesignV1.from_mapping(source)
    if sys.platform not in design.platforms:
        raise RuntimeError(f"profile platform {sys.platform!r} is not permitted")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("profile runtime must use Python 3.12")
    config_path = _resolve_input(path, design.config)
    config = ExperimentConfig.load(config_path)
    if config.training["device"] != "cpu":
        raise ValueError("profile base config must use CPU")
    payload = config.to_dict()
    payload["training"]["batch_size"] = design.batch_size
    payload["training"]["device"] = "cpu"
    config = ExperimentConfig.from_mapping(payload)
    output = _resolve_output(path, design.output_dir)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("profile output already exists; use --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    backup = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
    try:
        staging.mkdir()
        design_file = _write_json(staging / "design.json", design.to_dict())
        config_file = _write_json(staging / "resolved_config.json", config.to_dict())
        metrics, environment = _measure_profile(config, design)
        metrics_file = _write_json(staging / "metrics.json", metrics)
        environment_file = _write_json(staging / "environment.json", environment)
        lock_name = (
            "requirements-v3-diffusion-cpu.lock"
            if config.policy["type"] == "diffusion_v3"
            else "requirements-v3-core-cpu.lock"
        )
        lock_source = _ROOT / lock_name
        lock_target = staging / "dependency-lock.txt"
        shutil.copyfile(lock_source, lock_target)
        git_sha, dirty = _git_state()
        manifest = PolicyProfileManifestV1(
            git_sha=git_sha,
            git_dirty=dirty,
            design_sha256=sha256_file(design_file),
            config_sha256=sha256_file(config_file),
            dependency_lock_sha256=sha256_file(lock_target),
            environment_sha256=sha256_file(environment_file),
            metrics_sha256=sha256_file(metrics_file),
            policy_id=str(config.policy["type"]),
            device="cpu",
            warmup=design.warmup,
            measurements=design.measurements,
            batch_size=design.batch_size,
            threads=design.threads,
            release_eligible=not dirty,
            comparative_claim_allowed=False,
            statement=PROFILE_STATEMENT,
        )
        _write_json(staging / "profile_manifest.json", manifest.to_dict())
        verify_profile(staging)
        if output.exists():
            output.replace(backup)
        staging.replace(output)
        if backup.exists():
            shutil.rmtree(backup)
        return output
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise


def verify_profile(output_root: str | Path) -> PolicyProfileManifestV1:
    root = Path(output_root)
    expected = {
        "design.json", "resolved_config.json", "dependency-lock.txt",
        "environment.json", "metrics.json", "profile_manifest.json",
    }
    if not root.is_dir() or {item.name for item in root.iterdir()} != expected:
        raise ValueError("profile output file set is incomplete or contains extras")
    manifest_value = json.loads((root / "profile_manifest.json").read_text(encoding="utf-8"))
    manifest = PolicyProfileManifestV1.from_mapping(manifest_value)
    checks = {
        "design_sha256": "design.json",
        "config_sha256": "resolved_config.json",
        "dependency_lock_sha256": "dependency-lock.txt",
        "environment_sha256": "environment.json",
        "metrics_sha256": "metrics.json",
    }
    for field, filename in checks.items():
        if sha256_file(root / filename) != getattr(manifest, field):
            raise ValueError(f"profile {field} mismatch")
    design = PolicyProfileDesignV1.from_mapping(
        json.loads((root / "design.json").read_text(encoding="utf-8"))
    )
    config = ExperimentConfig.from_mapping(
        json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    )
    if manifest.policy_id != config.policy["type"]:
        raise ValueError("profile policy identity mismatch")
    if (
        manifest.warmup,
        manifest.measurements,
        manifest.batch_size,
        manifest.threads,
    ) != (design.warmup, design.measurements, design.batch_size, design.threads):
        raise ValueError("profile design and manifest settings mismatch")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if set(metrics) != {
        "training_latency_ms", "inference_latency_ms", "training_summary_ms",
        "inference_summary_ms", "training_samples_per_second",
        "inference_samples_per_second", "peak_rss_bytes",
        "comparative_claim_allowed", "statement",
    }:
        raise ValueError("profile metrics require exact fields")
    training = tuple(float(item) for item in metrics["training_latency_ms"])
    inference = tuple(float(item) for item in metrics["inference_latency_ms"])
    if metrics["training_summary_ms"] != _summary(training):
        raise ValueError("profile training summary mismatch")
    if metrics["inference_summary_ms"] != _summary(inference):
        raise ValueError("profile inference summary mismatch")
    if metrics["comparative_claim_allowed"] is not False or metrics["statement"] != PROFILE_STATEMENT:
        raise ValueError("profile claim boundary mismatch")
    if isinstance(metrics["peak_rss_bytes"], bool) or int(metrics["peak_rss_bytes"]) <= 0:
        raise ValueError("profile peak RSS must be positive")
    return manifest
