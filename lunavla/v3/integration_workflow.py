from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .contracts import EpisodeRecordV3
from .data import InMemoryDatasetSourceV3
from .engine import EngineV3
from .integration_contracts import (
    CONNECTIVITY_STATEMENT,
    LIBERO_REPO_ID,
    LIBERO_SPATIAL_DATASET_TASK_IDS,
    LIBERO_SPATIAL_MIN_EPISODES,
    LIBERO_SPATIAL_TASK_LANGUAGES,
    PUSHT_REPO_ID,
    ExternalDatasetSpecV1,
    IntegrationManifestV1,
)
from .real_adapters import LeRobotDatasetSourceV3, LiberoSpatialEnvV3, PushTEnvV3
_ROOT = Path(__file__).resolve().parents[2]
_LOCK = _ROOT / "requirements-v3-beta2-integration-cpu.lock"
_SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _safe_output(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    if target == _ROOT or _ROOT in target.parents:
        raise ValueError("integration output must be outside the Git checkout")
    return target


@dataclass(frozen=True)
class SourceFileRecordV1:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.path)
        if path.is_absolute() or self.path in {"", ".", ".."} or ".." in path.parts:
            raise ValueError("source file path must be repository-relative")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0:
            raise ValueError("source file size must be positive")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError("source file sha256 must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceFileRecordV1":
        if set(value) != {"path", "size", "sha256"}:
            raise ValueError("SourceFileRecordV1 requires exact fields")
        return cls(**dict(value))


@dataclass(frozen=True)
class SourceInventoryV1:
    repo_id: str
    revision: str
    planned_download_bytes: int
    max_download_bytes: int
    files: tuple[SourceFileRecordV1, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("SourceInventoryV1 schema_version must be 1")
        if not self.repo_id or not self.revision:
            raise ValueError("source inventory repo and revision are required")
        if isinstance(self.planned_download_bytes, bool) or not isinstance(
            self.planned_download_bytes, int
        ):
            raise TypeError("planned_download_bytes must be an integer")
        if self.planned_download_bytes <= 0:
            raise ValueError("planned_download_bytes must be positive")
        if self.planned_download_bytes > self.max_download_bytes:
            raise ValueError("source inventory exceeds the configured download limit")
        files = tuple(self.files)
        if not files or len({item.path for item in files}) != len(files):
            raise ValueError("source inventory files must be non-empty and unique")
        if sum(item.size for item in files) != self.planned_download_bytes:
            raise ValueError("source inventory byte total does not match its files")
        object.__setattr__(self, "files", tuple(sorted(files, key=lambda item: item.path)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "planned_download_bytes": self.planned_download_bytes,
            "max_download_bytes": self.max_download_bytes,
            "files": [item.to_dict() for item in self.files],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceInventoryV1":
        fields = {
            "schema_version", "repo_id", "revision", "planned_download_bytes",
            "max_download_bytes", "files",
        }
        if set(value) != fields:
            raise ValueError("SourceInventoryV1 requires exact fields")
        payload = dict(value)
        raw_files = payload["files"]
        if isinstance(raw_files, (str, bytes, Mapping)) or not isinstance(raw_files, Sequence):
            raise TypeError("source inventory files must be a sequence")
        payload["files"] = tuple(SourceFileRecordV1.from_mapping(item) for item in raw_files)
        return cls(**payload)

    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _selected_hub_paths(spec: ExternalDatasetSpecV1, siblings: Sequence[Any]) -> tuple[Any, ...]:
    by_path = {str(getattr(item, "rfilename", "")): item for item in siblings}
    if spec.repo_id == PUSHT_REPO_ID:
        required = tuple(spec.file_hashes)
    else:
        required = (
            "meta/info.json",
            "meta/tasks.parquet",
            "meta/episodes/chunk-000/file-000.parquet",
            "data/chunk-000/file-309.parquet",
            "data/chunk-000/file-310.parquet",
            "data/chunk-000/file-311.parquet",
            "videos/observation.images.image/chunk-000/file-027.mp4",
            "videos/observation.images.image/chunk-000/file-028.mp4",
            "videos/observation.images.image2/chunk-000/file-027.mp4",
            "videos/observation.images.image2/chunk-000/file-028.mp4",
        )
    missing = sorted(set(required) - set(by_path))
    if missing:
        raise FileNotFoundError("pinned Hub revision is missing: " + ", ".join(missing))
    return tuple(by_path[path] for path in required)


def preflight_source(
    config: ExperimentConfig,
    *,
    api: Any | None = None,
    metadata_cache: str | Path | None = None,
) -> SourceInventoryV1:
    spec = config.external_dataset_spec
    if config.contract_revision != 3 or spec is None:
        raise ValueError("source preflight requires a revision-3 real-source config")
    if api is None:
        hub = importlib.import_module("huggingface_hub")
        api = hub.HfApi()
    info = api.dataset_info(spec.repo_id, revision=spec.revision, files_metadata=True)
    if str(getattr(info, "sha", "")) != spec.revision:
        raise ValueError("Hub resolved a revision different from the pinned source")
    selected = _selected_hub_paths(spec, tuple(getattr(info, "siblings", ())))
    cache = Path(metadata_cache) if metadata_cache is not None else None
    records: list[SourceFileRecordV1] = []
    for sibling in selected:
        path = str(getattr(sibling, "rfilename", ""))
        lfs = getattr(sibling, "lfs", None)
        size = getattr(sibling, "size", None)
        if size is None:
            size = getattr(lfs, "size", None)
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"Hub file {path!r} has no valid size")
        digest = str(getattr(lfs, "sha256", ""))
        if path in spec.file_hashes and digest != spec.file_hashes[path]:
            raise ValueError(f"Hub SHA-256 drift for {path}")
        if not _SHA256.fullmatch(digest):
            if cache is None:
                raise ValueError(f"Hub metadata has no SHA-256 for {path}; metadata_cache is required")
            hub = importlib.import_module("huggingface_hub")
            local = Path(
                hub.hf_hub_download(
                    repo_id=spec.repo_id,
                    repo_type="dataset",
                    revision=spec.revision,
                    filename=path,
                    cache_dir=cache,
                )
            )
            if local.stat().st_size != size:
                raise ValueError(f"downloaded metadata size drift for {path}")
            digest = _sha256_file(local)
        records.append(SourceFileRecordV1(path, size, digest))
    inventory = SourceInventoryV1(
        spec.repo_id,
        spec.revision,
        sum(item.size for item in records),
        spec.max_download_bytes,
        tuple(records),
    )
    return inventory


def _metadata_for_libero(cache_dir: Path, spec: ExternalDatasetSpecV1) -> tuple[list[dict[str, Any]], dict[int, str]]:
    module = importlib.import_module("lerobot.datasets.dataset_metadata")
    metadata = module.LeRobotDatasetMetadata(spec.repo_id, cache_dir, spec.revision)
    tasks = {
        int(row.task_index): str(task)
        for task, row in metadata.tasks.iterrows()
        if int(row.task_index) in spec.task_ids
    }
    if set(tasks) != set(spec.task_ids):
        raise ValueError("LeRobot metadata does not contain the pinned global LIBERO task IDs")
    expected_languages = dict(
        zip(LIBERO_SPATIAL_DATASET_TASK_IDS, LIBERO_SPATIAL_TASK_LANGUAGES, strict=True)
    )
    if tasks != expected_languages:
        raise ValueError("LeRobot global task language mapping drifted from LIBERO-Spatial")
    episode_metadata = {
        int(row["episode_index"]): dict(row)
        for row in (metadata.episodes[index] for index in range(len(metadata.episodes)))
    }
    task_to_episode = dict(
        zip(LIBERO_SPATIAL_DATASET_TASK_IDS, LIBERO_SPATIAL_MIN_EPISODES, strict=True)
    )
    rows: list[dict[str, Any]] = []
    for task, episode in task_to_episode.items():
        if episode not in episode_metadata:
            raise ValueError("LeRobot metadata is missing a pinned minimum Spatial episode")
        rows.append({"task_index": task, "episode_index": episode, "task": tasks[task]})
    return rows, tasks


def load_real_episodes(config: ExperimentConfig, cache_dir: Path) -> tuple[EpisodeRecordV3, ...]:
    spec = config.external_dataset_spec
    if spec is None:
        raise ValueError("real episode loading requires an external dataset spec")
    metadata: Sequence[Mapping[str, Any]] = ()
    languages: Mapping[int, str] | None = None
    if spec.repo_id == LIBERO_REPO_ID:
        metadata, languages = _metadata_for_libero(cache_dir, spec)
    source = LeRobotDatasetSourceV3(
        spec,
        config.feature_schema,
        root=cache_dir,
        metadata=metadata,
        expected_task_languages=languages,
    )
    return source.load()


def _policy_payload(config: ExperimentConfig, policy_id: str) -> ExperimentConfig:
    payload = config.to_dict()
    images = [item.name for item in config.feature_schema.by_role("image")]
    instruction_dim = 8
    if policy_id == "act_v3":
        camera_parameters: dict[str, Any] = (
            {"camera_feature": images[0]}
            if len(images) == 1
            else {"camera_feature": None, "camera_features": images}
        )
        payload["policy"] = {
            "type": "act_v3",
            "parameters": {
                "state_feature": "state.proprioception",
                **camera_parameters,
                "instruction_dim": instruction_dim,
                "history": 1,
                "chunk_size": 4,
                "horizon": 4,
                "execution_steps": 1,
                "d_model": 64,
                "nhead": 4,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "dim_feedforward": 128,
                "latent_dim": 16,
                "dropout": 0.0,
                "kl_weight": 0.01,
                "sample_latent_during_training": True,
                "temporal_ensemble_decay": 0.01,
            },
        }
        payload["training"]["optimizer"] = {"type": "adam", "parameters": {}}
        payload["evaluation"]["execution_mode"] = "receding_horizon"
        payload["artifacts"]["checkpoint_name"] = "act_v3.pt"
    else:
        payload["policy"] = {
            "type": "diffusion_v3",
            "parameters": {
                "state_feature": "state.proprioception",
                "camera_features": images,
                "unused_modalities": ["instruction"],
                "history": 2,
                "chunk_size": 4,
                "horizon": 8,
                "execution_steps": 4,
                "n_action_steps": 4,
                "noise_scheduler_type": "DDIM",
                "num_train_timesteps": 16,
                "num_inference_steps": 8,
                "prediction_type": "epsilon",
                "do_mask_loss_for_padding": True,
                "noise_seed": 202611,
                "down_dims": [32, 64, 128],
                "kernel_size": 3,
                "n_groups": 8,
                "diffusion_step_embed_dim": 32,
                "spatial_softmax_num_keypoints": 8,
                "vision_backbone": "resnet18",
                "use_group_norm": True,
                "use_separate_rgb_encoder_per_camera": True,
                "pretrained_backbone_weights": None,
                "beta_schedule": "squaredcos_cap_v2",
                "clip_sample": True,
                "clip_sample_range": 1.0,
            },
        }
        payload["training"]["optimizer"] = {
            "type": "adamw",
            "parameters": {"weight_decay": 1e-6, "betas": [0.95, 0.999], "eps": 1e-8},
        }
        payload["evaluation"]["execution_mode"] = "open_loop_chunk"
        payload["artifacts"]["checkpoint_name"] = "diffusion_v3"
    payload["training"]["steps"] = 1
    payload["training"]["batch_size"] = 1
    # Beta 2's authoritative integration profile is hosted Linux CPU.  Do not
    # inherit the retired external-GPU plan into a connectivity-only smoke.
    payload["training"]["device"] = "cpu"
    payload["prompt"]["camera_order"] = images
    payload["artifacts"]["output_dir"] = f"outputs/v3/beta2-{policy_id}"
    return ExperimentConfig.from_mapping(payload)


def run_policy_smokes(
    config: ExperimentConfig, episodes: Sequence[EpisodeRecordV3]
) -> tuple[dict[str, Any], ...]:
    results: list[dict[str, Any]] = []
    for policy_id in ("act_v3", "diffusion_v3"):
        resolved = _policy_payload(config, policy_id)
        engine = EngineV3(resolved)
        _policy, losses = engine.train(InMemoryDatasetSourceV3(tuple(episodes)))
        result = engine.train_results[-1]
        if not result.finite or not np.isfinite(result.loss):
            raise FloatingPointError(f"{policy_id} bounded optimizer smoke was non-finite")
        results.append(
            {
                "policy_id": policy_id,
                "config_sha256": resolved.sha256(),
                "loss": result.loss,
                "gradient_norm": result.gradient_norm,
                "finite": result.finite,
                "state_shape": list(config.feature_schema.by_role("state")[0].shape),
                "action_shape": list(config.feature_schema.by_role("action")[0].shape),
                "camera_order": [item.name for item in config.feature_schema.by_role("image")],
            }
        )
    return tuple(results)


def run_environment_smoke(config: ExperimentConfig) -> dict[str, Any]:
    spec = config.simulation_task_spec
    if spec is None:
        raise ValueError("environment smoke requires a simulation task spec")
    action = np.zeros(config.feature_schema.by_role("action")[0].shape, dtype=np.float32)
    if config.task["id"] == "lerobot_pusht":
        environment = PushTEnvV3(config.feature_schema, spec)
        environments: list[Any] = [environment]
    else:
        module = importlib.import_module("lerobot.envs.libero")
        suite = module._get_suite("libero_spatial")
        suite_languages = tuple(str(suite.get_task(task).language) for task in spec.task_ids)
        if suite_languages != LIBERO_SPATIAL_TASK_LANGUAGES:
            raise ValueError("hf-libero Spatial task language mapping drift")

        def factory(**kwargs: Any) -> Any:
            return module.LiberoEnv(
                task_suite=suite,
                task_id=kwargs["task_id"],
                task_suite_name="libero_spatial",
                episode_length=3,
                obs_type="pixels_agent_pos",
                observation_width=256,
                observation_height=256,
                episode_index=kwargs["init_state_id"],
                control_mode="relative",
            )

        environments = [
            LiberoSpatialEnvV3(
                config.feature_schema,
                spec,
                task_id=task_id,
                init_state_id=0,
                task_language=str(suite.get_task(task_id).language),
                env_factory=factory,
            )
            for task_id in spec.task_ids
        ]
    validations: list[dict[str, Any]] = []
    try:
        for environment in environments:
            observation = environment.reset(seed=1000)
            transitions = [environment.step(action) for _ in range(3)]
            validations.append(
                {
                    "episode_id": str(observation.episode_id),
                    "steps": len(transitions),
                    "camera_order": list(observation.images),
                    "state_order": list(observation.state),
                }
            )
    finally:
        for environment in environments:
            environment.close()
    return {"environments": validations, "closed": all(item.close_count == 1 for item in environments)}


@dataclass(frozen=True)
class IntegrationRuntime:
    inventory: SourceInventoryV1
    episodes: tuple[EpisodeRecordV3, ...]
    environment_validation: Mapping[str, Any]
    policy_smokes: tuple[Mapping[str, Any], ...]
    runtime_environment_path: Path
    execution_environment: str


def _git_state() -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, bool(status.strip())


def _default_runtime(config: ExperimentConfig, cache_dir: Path) -> IntegrationRuntime:
    if os.environ.get("LUNAVLA_HOSTED_CPU") != "true":
        raise RuntimeError("real integration requires LUNAVLA_HOSTED_CPU=true")
    import torch
    import torchvision

    sha, dirty = _git_state()
    if dirty:
        raise ValueError("hosted CPU integration requires a clean Git checkout")
    machine = platform.machine().lower()
    if platform.system() != "Linux" or machine not in {"x86_64", "amd64"}:
        raise RuntimeError("hosted CPU integration requires Linux x86_64")
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("hosted CPU integration forbids CUDA")
    runtime_environment_path = _write_json(
        cache_dir / "runtime-environment.json",
        {
            "schema_version": 1,
            "git_sha": sha,
            "execution_environment": "hosted_cpu",
            "platform_system": platform.system(),
            "platform_machine": machine,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_available": False,
            "cpu_count": os.cpu_count() or 1,
            "lerobot_version": metadata.version("lerobot"),
            "hf_libero_version": metadata.version("hf-libero"),
        },
    )
    inventory = preflight_source(config, metadata_cache=cache_dir / "hub-metadata")
    episodes = load_real_episodes(config, cache_dir / "dataset")
    return IntegrationRuntime(
        inventory,
        episodes,
        run_environment_smoke(config),
        run_policy_smokes(config, episodes),
        runtime_environment_path,
        "hosted_cpu",
    )


def run_integration(
    config_path: str | Path,
    *,
    cache_dir: str | Path,
    output_root: str | Path,
    runtime_factory: Callable[[ExperimentConfig, Path], IntegrationRuntime] = _default_runtime,
) -> Path:
    config = ExperimentConfig.load(config_path)
    if config.contract_revision != 3:
        raise ValueError("integration-run requires config contract_revision=3")
    if not _LOCK.is_file():
        raise FileNotFoundError(f"missing integration lock: {_LOCK.name}")
    output = _safe_output(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite integration output: {output}")
    cache = _safe_output(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        runtime = runtime_factory(config, cache)
        resolved_path = _write_json(staging / "resolved_config.json", config.to_dict())
        inventory_path = _write_json(staging / "source_inventory.json", runtime.inventory.to_dict())
        lock_path = staging / "dependency-lock.txt"
        shutil.copyfile(_LOCK, lock_path)
        runtime_environment_path = staging / "runtime-environment.json"
        shutil.copyfile(runtime.runtime_environment_path, runtime_environment_path)
        frame_counts = [len(episode.transitions) for episode in runtime.episodes]
        expected_counts = [161] if config.task["id"] == "lerobot_pusht" else None
        if expected_counts is not None and frame_counts != expected_counts:
            raise ValueError("PushT episode 0 must contain exactly 161 frames")
        if config.task["id"] == "libero_spatial_subset" and len(frame_counts) != 4:
            raise ValueError("LIBERO-Spatial integration requires four selected episodes")
        data_validation = {
            "episode_ids": [episode.episode_id for episode in runtime.episodes],
            "frame_counts": frame_counts,
            "camera_order": [item.name for item in config.feature_schema.by_role("image")],
            "state_shape": list(config.feature_schema.by_role("state")[0].shape),
            "action_shape": list(config.feature_schema.by_role("action")[0].shape),
            "frequency_hz": 10.0,
        }
        metrics = {
            "data_validation": data_validation,
            "environment_validation": dict(runtime.environment_validation),
            "policy_smokes": [dict(item) for item in runtime.policy_smokes],
        }
        metrics_path = _write_json(staging / "metrics.json", metrics)
        sha, dirty = _git_state()
        manifest = IntegrationManifestV1(
            git_sha=sha,
            git_dirty=dirty,
            config_sha256=_sha256_file(resolved_path),
            dependency_lock_sha256=_sha256_file(lock_path),
            source_spec_sha256=config.external_dataset_spec.sha256(),  # type: ignore[union-attr]
            source_inventory_sha256=_sha256_file(inventory_path),
            runtime_environment_sha256=_sha256_file(runtime_environment_path),
            metrics_sha256=_sha256_file(metrics_path),
            execution_environment=runtime.execution_environment,
            data_validation=data_validation,
            environment_validation=runtime.environment_validation,
            policy_smokes=runtime.policy_smokes,
            downloaded_bytes=runtime.inventory.planned_download_bytes,
            claim_allowed=False,
            benchmark_claim=False,
            statement=CONNECTIVITY_STATEMENT,
        )
        _write_json(staging / "integration_manifest.json", manifest.to_dict())
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def verify_integration(output_root: str | Path) -> IntegrationManifestV1:
    output = Path(output_root).expanduser().resolve()
    expected = {
        "resolved_config.json", "source_inventory.json", "dependency-lock.txt",
        "runtime-environment.json", "metrics.json", "integration_manifest.json",
    }
    if not output.is_dir() or {item.name for item in output.iterdir()} != expected:
        raise ValueError("integration output must contain the exact contracted artifact set")
    manifest = IntegrationManifestV1.from_mapping(
        json.loads((output / "integration_manifest.json").read_text(encoding="utf-8"))
    )
    config = ExperimentConfig.from_mapping(
        json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    )
    inventory = SourceInventoryV1.from_mapping(
        json.loads((output / "source_inventory.json").read_text(encoding="utf-8"))
    )
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    checks = {
        "config_sha256": _sha256_file(output / "resolved_config.json"),
        "dependency_lock_sha256": _sha256_file(output / "dependency-lock.txt"),
        "source_inventory_sha256": _sha256_file(output / "source_inventory.json"),
        "runtime_environment_sha256": _sha256_file(output / "runtime-environment.json"),
        "metrics_sha256": _sha256_file(output / "metrics.json"),
    }
    for name, digest in checks.items():
        if getattr(manifest, name) != digest:
            raise ValueError(f"integration artifact hash mismatch: {name}")
    if config.external_dataset_spec is None or manifest.source_spec_sha256 != config.external_dataset_spec.sha256():
        raise ValueError("integration source spec hash mismatch")
    if inventory.repo_id != config.external_dataset_spec.repo_id or inventory.revision != config.external_dataset_spec.revision:
        raise ValueError("source inventory identity does not match config")
    manifest_payload = manifest.to_dict()
    if metrics != {
        "data_validation": manifest_payload["data_validation"],
        "environment_validation": manifest_payload["environment_validation"],
        "policy_smokes": manifest_payload["policy_smokes"],
    }:
        raise ValueError("metrics do not independently reproduce the integration manifest")
    if manifest.execution_environment == "hosted_cpu":
        runtime_environment = json.loads(
            (output / "runtime-environment.json").read_text(encoding="utf-8")
        )
        expected_fields = {
            "schema_version", "git_sha", "execution_environment", "platform_system",
            "platform_machine", "python_version", "torch_version", "torchvision_version",
            "cuda_available", "cpu_count", "lerobot_version", "hf_libero_version",
        }
        if not isinstance(runtime_environment, Mapping) or set(runtime_environment) != expected_fields:
            raise ValueError("hosted CPU runtime environment fields are invalid")
        if (
            runtime_environment["schema_version"] != 1
            or isinstance(runtime_environment["schema_version"], bool)
            or runtime_environment["git_sha"] != manifest.git_sha
            or runtime_environment["execution_environment"] != "hosted_cpu"
            or runtime_environment["platform_system"] != "Linux"
            or runtime_environment["platform_machine"] not in {"x86_64", "amd64"}
            or runtime_environment["cuda_available"] is not False
            or runtime_environment["torch_version"] != "2.11.0+cpu"
            or runtime_environment["torchvision_version"] != "0.26.0+cpu"
            or runtime_environment["lerobot_version"] != "0.6.0"
            or runtime_environment["hf_libero_version"] != "0.1.4"
        ):
            raise ValueError("hosted CPU runtime environment does not match its contract")
    return manifest


def print_source_preflight(config_path: str | Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lunavla-v3-source-preflight-") as directory:
        inventory = preflight_source(
            ExperimentConfig.load(config_path), metadata_cache=directory
        )
    return inventory.to_dict()
