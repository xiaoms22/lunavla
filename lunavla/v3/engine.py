from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from lunavla.contracts import Observation, PolicyBatch, VLAPolicy
from lunavla.registry import default_policy_registry
from model.policy_base import ActionChunk

from .artifacts import (
    ArtifactHashRecordV1,
    CheckpointEnvelopeV4R2,
    RunManifestV4R2,
    sha256_file,
    verify_checkpoint_directory,
)
from .config import ExperimentConfig
from .contracts import EpisodeRecordV3, ObservationV3, TaskEnvV3
from .data import DatasetBundle, InMemoryDatasetSourceV3, audit_episodes, split_episode_ids
from .diagnostic_engine import DiagnosticRouterV1
from .fake_tasks import FakePointEnvV3, make_fake_episodes
from .normalization import NormalizationStatsV1, fit_normalization_stats
from .policy import (
    ModelSourceContractV1,
    PolicyBatchV3,
    PolicySampleV3,
    PolicySpecV3,
    TrainStepResultV3,
    VLAPolicyV3,
)
from .registry import PolicyRegistryV3


Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _git_identity() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40, True
    return sha, dirty


class V2PolicyBridge:
    def __init__(
        self,
        policy: VLAPolicy,
        *,
        spec: PolicySpecV3,
        normalization: NormalizationStatsV1,
        state_feature: str,
        unused_modalities: Sequence[str],
    ) -> None:
        self.policy = policy
        self.spec = spec
        self.normalization = normalization
        self.state_feature = state_feature
        self.unused_modalities = tuple(unused_modalities)
        if len(self.unused_modalities) != len(set(self.unused_modalities)):
            raise ValueError("unused_modalities cannot contain duplicates")

        if policy.policy_id != spec.policy_id:
            raise ValueError("v2 bridge policy_id does not match PolicySpecV3")

    @property
    def policy_id(self) -> str:
        return self.spec.policy_id

    def reset(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("policy reset seed must be a non-negative integer")

    def _observation(self, value: ObservationV3) -> Observation:
        if self.state_feature not in value.state:
            raise ValueError(f"missing policy state feature {self.state_feature}")
        if value.images and "image" not in self.unused_modalities:
            raise ValueError("v2 policy bridge cannot silently discard image features")
        instruction = value.instruction
        if instruction is not None and "instruction" in self.unused_modalities:
            instruction = None
        state = np.asarray(value.state[self.state_feature])
        stats = self.normalization.features.get(self.state_feature)
        if stats is not None:
            state = stats.normalize(state)
        return Observation(state, instruction=instruction)

    def train_batch(
        self,
        observations: Sequence[ObservationV3],
        targets: Float32Array,
        mask: BoolArray,
        *,
        learning_rate: float,
    ) -> float:
        batch = PolicyBatch(tuple(self._observation(item) for item in observations), targets, mask)
        return self.policy.train_batch(batch, learning_rate=learning_rate)

    def train_step(
        self, batch: PolicyBatchV3, *, learning_rate: float, step: int
    ) -> TrainStepResultV3:
        started = time.perf_counter()
        target_values: list[Float32Array] = []
        mask_values: list[BoolArray] = []
        for item in batch.samples:
            if item.action_chunk is None or item.valid_mask is None:
                raise ValueError("training samples require action supervision")
            target_values.append(item.action_chunk)
            mask_values.append(item.valid_mask)
        targets = np.stack(target_values)
        masks = np.stack(mask_values)
        loss = self.train_batch(
            [item.observation for item in batch.samples],
            targets,
            masks,
            learning_rate=learning_rate,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not np.isfinite(loss):
            raise FloatingPointError("policy returned a non-finite loss")
        return TrainStepResultV3(
            loss=loss,
            loss_components={"mse": loss},
            gradient_norm=None,
            learning_rate=learning_rate,
            step=step,
            finite=True,
            timing_ms={"train_step": elapsed_ms},
        )

    def predict_chunk(self, sample: PolicySampleV3) -> ActionChunk:
        return self.policy.predict_chunk(self._observation(sample.observation))

    def save_checkpoint(self, path: Path, *, metadata: Mapping[str, Any]) -> Path:
        return self.policy.save_checkpoint(path, metadata=metadata)


@dataclass(frozen=True)
class AlphaRunResult:
    losses: tuple[float, ...]
    metrics: Mapping[str, Any]
    output_dir: Path


class EngineV3:
    def __init__(
        self, config: ExperimentConfig, registry: PolicyRegistryV3 | None = None
    ) -> None:
        self.config = config
        self.registry = registry or self._compat_registry()
        self.policy_spec: PolicySpecV3 | None = None
        self.normalization: NormalizationStatsV1 | None = None
        self.train_results: tuple[TrainStepResultV3, ...] = ()
        self.training_state: Mapping[str, Any] = {}
        self.diagnostic_router: DiagnosticRouterV1 | None = None

    def _policy_spec(self) -> PolicySpecV3:
        if self.config.policy["type"] == "act_v3":
            try:
                from .act_policy import act_policy_spec
            except ModuleNotFoundError as exc:
                if exc.name == "torch":
                    raise RuntimeError("act_v3 requires the v3-act dependency profile") from exc
                raise

            return act_policy_spec(self.config)
        if self.config.policy["type"] == "diffusion_v3":
            try:
                from .diffusion_policy import diffusion_policy_spec
            except ModuleNotFoundError as exc:
                if exc.name in {"torch", "lerobot", "diffusers"}:
                    raise RuntimeError(
                        "diffusion_v3 requires the v3-diffusion dependency profile"
                    ) from exc
                raise

            return diffusion_policy_spec(self.config)
        if self.config.policy["type"] == "lerobot_smolvla":
            from .smolvla_adapter import smolvla_policy_spec

            return smolvla_policy_spec(self.config)
        parameters = dict(self.config.policy["parameters"])
        raw = dict(parameters.get("legacy", parameters))
        state_feature = str(raw.get("state_feature", "state.proprioception"))
        instruction_dim = int(raw.get("instruction_dim", 0))
        modalities = ("state", "instruction") if instruction_dim else ("state",)
        normalization = {
            item.name: item.normalization
            for item in self.config.feature_schema.features
            if item.name == state_feature or item.role == "action"
        }
        return PolicySpecV3(
            policy_id=self.config.policy["type"],
            backend="numpy_v2_compat",
            model_source=ModelSourceContractV1(
                repo_id="lunavla/native",
                revision="v3-alpha2-contracts",
                file_hashes={},
                license_status="not_required",
                pretrained_enabled=False,
            ),
            required_modalities=modalities,
            camera_order=(),
            state_order=(state_feature,),
            history=int(raw.get("history", 1)),
            chunk_size=int(raw["chunk_size"]),
            horizon=int(raw.get("horizon", raw["chunk_size"])),
            execution_steps=int(raw.get("execution_steps", raw["chunk_size"])),
            normalization=normalization,
            device=self.config.training["device"],
            deterministic=True,
        )

    def _create_bridge(
        self,
        config: ExperimentConfig,
        spec: PolicySpecV3,
        normalization: NormalizationStatsV1,
    ) -> VLAPolicyV3:
        parameters = dict(config.policy["parameters"])
        raw = dict(parameters.get("legacy", parameters))
        unused = tuple(raw.pop("unused_modalities", ()))
        state_feature = str(raw.pop("state_feature", "state.proprioception"))
        for name in ("history", "horizon", "execution_steps"):
            raw.pop(name, None)
        raw.setdefault("seed", config.training["seed"])
        raw.setdefault("device", config.training["device"])
        policy = default_policy_registry().create(config.policy["type"], raw)
        return V2PolicyBridge(
            policy,
            spec=spec,
            normalization=normalization,
            state_feature=state_feature,
            unused_modalities=unused,
        )

    def _restore_bridge(
        self,
        checkpoint: Path,
        config: ExperimentConfig,
        spec: PolicySpecV3,
        normalization: NormalizationStatsV1,
    ) -> VLAPolicyV3:
        parameters = dict(config.policy["parameters"])
        raw = dict(parameters.get("legacy", parameters))
        unused = tuple(raw.get("unused_modalities", ()))
        state_feature = str(raw.get("state_feature", "state.proprioception"))
        target = checkpoint
        if checkpoint.is_dir():
            target = checkpoint / "policy" / str(config.artifacts["checkpoint_name"])
        policy = default_policy_registry().load_checkpoint(
            target, policy_id=config.policy["type"]
        )
        return V2PolicyBridge(
            policy,
            spec=spec,
            normalization=normalization,
            state_feature=state_feature,
            unused_modalities=unused,
        )

    def _compat_registry(self) -> PolicyRegistryV3:
        registry = PolicyRegistryV3()
        if self.config.policy["type"] == "act_v3":
            try:
                from .act_policy import register_act_policy
            except ModuleNotFoundError as exc:
                if exc.name == "torch":
                    raise RuntimeError("act_v3 requires the v3-act dependency profile") from exc
                raise

            register_act_policy(registry)
        elif self.config.policy["type"] == "diffusion_v3":
            try:
                from .diffusion_policy import register_diffusion_policy
            except ModuleNotFoundError as exc:
                if exc.name in {"torch", "lerobot", "diffusers"}:
                    raise RuntimeError(
                        "diffusion_v3 requires the v3-diffusion dependency profile"
                    ) from exc
                raise

            register_diffusion_policy(registry)
        elif self.config.policy["type"] == "lerobot_smolvla":
            from .smolvla_adapter import register_smolvla_policy

            register_smolvla_policy(registry)
        else:
            registry.register(
                self.config.policy["type"], self._create_bridge, self._restore_bridge
            )
        return registry

    @staticmethod
    def _supervision(
        episodes: Sequence[EpisodeRecordV3], chunk_size: int
    ) -> tuple[list[ObservationV3], Float32Array, BoolArray]:
        observations: list[ObservationV3] = []
        targets: list[Float32Array] = []
        masks: list[BoolArray] = []
        for episode in episodes:
            actions = [transition.action for transition in episode.transitions]
            for index, transition in enumerate(episode.transitions):
                values = np.zeros((chunk_size, actions[0].shape[0]), dtype=np.float32)
                mask = np.zeros(chunk_size, dtype=bool)
                for offset in range(chunk_size):
                    if index + offset < len(actions):
                        values[offset] = actions[index + offset]
                        mask[offset] = True
                observations.append(transition.observation)
                targets.append(values)
                masks.append(mask)
        return observations, np.stack(targets), np.stack(masks)

    @staticmethod
    def _samples(
        episodes: Sequence[EpisodeRecordV3], *, history: int, chunk_size: int
    ) -> tuple[PolicySampleV3, ...]:
        samples: list[PolicySampleV3] = []
        for episode in episodes:
            observations = [item.observation for item in episode.transitions]
            actions = [item.action for item in episode.transitions]
            for index, observation in enumerate(observations):
                start = max(0, index - history + 1)
                available = observations[start : index + 1]
                padding = history - len(available)
                observation_history = (available[0],) * padding + tuple(available)
                history_mask = np.asarray(
                    [False] * padding + [True] * len(available), dtype=bool
                )
                values = np.zeros((chunk_size, actions[0].shape[0]), dtype=np.float32)
                mask = np.zeros(chunk_size, dtype=bool)
                for offset in range(chunk_size):
                    if index + offset < len(actions):
                        values[offset] = actions[index + offset]
                        mask[offset] = True
                samples.append(
                    PolicySampleV3(
                        observation_history,
                        history_mask,
                        values,
                        mask,
                        observation.episode_id,
                        observation.step_index,
                    )
                )
        return tuple(samples)

    def train(self, dataset: InMemoryDatasetSourceV3) -> tuple[VLAPolicyV3, tuple[float, ...]]:
        episodes = tuple(dataset.load())
        normalization = fit_normalization_stats(episodes, self.config.feature_schema)
        if self.config.diagnostics["enabled"]:
            self.diagnostic_router = DiagnosticRouterV1(self.config, normalization)
            episodes = tuple(
                self.diagnostic_router.route_episode(episode) for episode in episodes
            )
        spec = self._policy_spec()
        policy = self.registry.create(self.config, spec, normalization)
        supervision_steps = spec.horizon if spec.policy_id == "diffusion_v3" else spec.chunk_size
        samples = self._samples(
            episodes, history=spec.history, chunk_size=supervision_steps
        )
        rng = np.random.default_rng(self.config.training["seed"])
        results: list[TrainStepResultV3] = []
        policy.reset(self.config.training["seed"])
        for step in range(self.config.training["steps"]):
            indices = rng.integers(0, len(samples), size=self.config.training["batch_size"])
            batch = PolicyBatchV3(
                tuple(samples[int(index)] for index in indices),
                device=self.config.training["device"],
            )
            results.append(
                policy.train_step(
                    batch,
                    learning_rate=self.config.training["learning_rate"],
                    step=step,
                )
            )
        self.policy_spec = spec
        self.normalization = normalization
        self.train_results = tuple(results)
        training_payload = self.config.to_dict()["training"]
        self.training_state = {
            "format": "lunavla_v3_engine_sampling_json",
            "policy_id": spec.policy_id,
            "step": self.config.training["steps"],
            "optimizer": training_payload["optimizer"],
            "scheduler": training_payload["scheduler"],
            "numpy_rng_state": rng.bit_generator.state,
        }
        return policy, tuple(item.loss for item in results)

    def restore_policy(self, checkpoint: str | Path) -> VLAPolicyV3:
        checkpoint_path = Path(checkpoint)
        expected_spec = self._policy_spec()
        spec = self.policy_spec or expected_spec
        normalization = self.normalization
        if checkpoint_path.is_dir():
            envelope = verify_checkpoint_directory(checkpoint_path)
            loaded_spec = PolicySpecV3.from_mapping(
                json.loads((checkpoint_path / "policy_spec.json").read_text(encoding="utf-8"))
            )
            loaded_normalization = NormalizationStatsV1.from_mapping(
                json.loads(
                    (checkpoint_path / "normalization.json").read_text(encoding="utf-8")
                )
            )
            if envelope["policy_id"] != loaded_spec.policy_id:
                raise ValueError("checkpoint policy id does not match its policy spec")
            if loaded_spec != expected_spec:
                raise ValueError("checkpoint PolicySpecV3 does not match the resolved config")
            if loaded_normalization.feature_schema_sha256 != self.config.feature_schema.sha256():
                raise ValueError("checkpoint normalization does not match FeatureSchema")
            if self.policy_spec is not None and self.policy_spec != loaded_spec:
                raise ValueError("checkpoint policy spec conflicts with active engine state")
            if self.normalization is not None and self.normalization != loaded_normalization:
                raise ValueError("checkpoint normalization conflicts with active engine state")
            spec = loaded_spec
            normalization = loaded_normalization
            self.policy_spec = loaded_spec
            self.normalization = loaded_normalization
        if normalization is None:
            raise ValueError("restore requires fitted or checkpoint normalization statistics")
        if self.config.diagnostics["enabled"]:
            self.diagnostic_router = DiagnosticRouterV1(self.config, normalization)
        policy = self.registry.restore(
            checkpoint_path, self.config, spec, normalization
        )
        return policy

    def evaluate(self, policy: VLAPolicyV3, env: TaskEnvV3) -> dict[str, Any]:
        rewards: list[float] = []
        successes: list[bool] = []
        steps: list[int] = []
        try:
            for seed in self.config.evaluation["seeds"]:
                canonical_observation = env.reset(seed=seed)
                observation = (
                    canonical_observation
                    if self.diagnostic_router is None
                    else self.diagnostic_router.route_observation(
                        canonical_observation
                    ).observation
                )
                policy.reset(seed)
                history = [observation]
                total_reward = 0.0
                success = False
                step_count = 0
                while step_count < self.config.evaluation["max_steps"]:
                    padding = policy.spec.history - len(history)
                    window = history[-policy.spec.history :]
                    sample = PolicySampleV3(
                        (window[0],) * max(0, padding) + tuple(window),
                        np.asarray([False] * max(0, padding) + [True] * len(window)),
                        None,
                        None,
                        observation.episode_id,
                        observation.step_index,
                    )
                    chunk = policy.predict_chunk(sample)
                    valid_actions = chunk.values[chunk.valid_mask]
                    actions = (
                        valid_actions[: policy.spec.execution_steps]
                        if self.config.evaluation["execution_mode"] == "open_loop_chunk"
                        else valid_actions[:1]
                    )
                    stop = False
                    for action in actions:
                        transition = env.step(action)
                        canonical_observation = transition.next_observation
                        observation = (
                            canonical_observation
                            if self.diagnostic_router is None
                            else self.diagnostic_router.route_observation(
                                canonical_observation
                            ).observation
                        )
                        history.append(observation)
                        total_reward += transition.reward
                        step_count += 1
                        success = bool(transition.info.get("success", False))
                        if transition.terminated or transition.truncated or step_count >= self.config.evaluation["max_steps"]:
                            stop = True
                            break
                    if stop:
                        break
                rewards.append(total_reward)
                successes.append(success)
                steps.append(step_count)
        finally:
            env.close()
        return {
            "episodes": len(rewards),
            "success_rate": float(np.mean(successes)),
            "mean_total_reward": float(np.mean(rewards)),
            "mean_steps": float(np.mean(steps)),
        }


def dataset_for_config(config: ExperimentConfig) -> DatasetBundle:
    if config.task["id"] not in {"fake_pusht", "fake_libero"}:
        raise ValueError("Alpha data source supports fake_pusht and fake_libero only")
    parameters = dict(config.dataset["parameters"])
    episode_count = int(parameters.get("episode_count", 6))
    steps = int(parameters.get("steps_per_episode", 5))
    episodes = make_fake_episodes(
        task_id=config.task["id"], seed=config.dataset["seed"],
        episode_count=episode_count, steps=steps,
    )
    split = split_episode_ids(episodes, seed=config.dataset["seed"])
    audit = audit_episodes(episodes, feature_schema=config.feature_schema, split=split)
    return DatasetBundle(tuple(episodes), split, audit)


def _execute_alpha(config: ExperimentConfig, output: Path) -> AlphaRunResult:
    output.mkdir(parents=True, exist_ok=True)
    rollouts = output / "rollouts"
    rollouts.mkdir(exist_ok=True)
    bundle = dataset_for_config(config)
    engine = EngineV3(config)
    policy, losses = engine.train(bundle.source("train"))
    if engine.policy_spec is None or engine.normalization is None:
        raise RuntimeError("engine did not establish policy and normalization contracts")
    config_path = _write_json(output / "resolved_config.json", config.to_dict())
    config_file_sha256 = sha256_file(config_path)
    policy_spec_path = _write_json(output / "policy_spec.json", engine.policy_spec.to_dict())
    normalization_path = _write_json(
        output / "normalization.json", engine.normalization.to_dict()
    )
    lock_source = Path(
        "requirements-v3-diffusion-cpu.lock"
        if engine.policy_spec.policy_id == "diffusion_v3"
        else "requirements-v3-core-cpu.lock"
    )
    dependency_lock_path = _write_json(
        output / "dependency_lock.json",
        {"path": lock_source.name, "sha256": sha256_file(lock_source)},
    )
    model_source_path = _write_json(
        output / "model_source.json", engine.policy_spec.model_source.to_dict()
    )
    runtime_path = _write_json(
        output / "runtime.json",
        {
            "device": engine.policy_spec.device,
            "deterministic": engine.policy_spec.deterministic,
            "train_step_latency_ms": [
                item.timing_ms["train_step"] for item in engine.train_results
            ],
            "peak_memory_bytes": 0,
        },
    )
    checkpoint_name = config.artifacts["checkpoint_name"]
    checkpoint_root = output / "checkpoint"
    policy_root = checkpoint_root / "policy"
    processors_root = checkpoint_root / "processors"
    policy_root.mkdir(parents=True)
    processors_root.mkdir(parents=True)
    checkpoint = policy.save_checkpoint(
        policy_root / checkpoint_name,
        metadata={"v3": {"config_sha256": config_file_sha256, "feature_schema_sha256": config.feature_schema.sha256()}},
    )
    checkpoint_policy_spec = _write_json(
        checkpoint_root / "policy_spec.json", engine.policy_spec.to_dict()
    )
    checkpoint_normalization = _write_json(
        checkpoint_root / "normalization.json", engine.normalization.to_dict()
    )
    checkpoint_dependency = _write_json(
        checkpoint_root / "dependency_lock.json",
        {"path": lock_source.name, "sha256": sha256_file(lock_source)},
    )
    processor = _write_json(
        processors_root / "processor.json",
        {
            "schema_version": 1,
            "type": engine.policy_spec.backend,
            "state_order": list(engine.policy_spec.state_order),
            "camera_order": list(engine.policy_spec.camera_order),
        },
    )
    training_state = _write_json(
        checkpoint_root / "training_state.json", dict(engine.training_state)
    )
    checkpoint_files = tuple(
        sorted(
            {
                checkpoint,
                checkpoint_policy_spec,
                checkpoint_normalization,
                checkpoint_dependency,
                processor,
                training_state,
                *(
                    path
                    for path in (policy_root / checkpoint_name).rglob("*")
                    if path.is_file()
                ),
            },
            key=lambda path: path.relative_to(checkpoint_root).as_posix(),
        )
    )
    envelope = CheckpointEnvelopeV4R2(
        policy_id=engine.policy_spec.policy_id,
        policy_spec_sha256=sha256_file(policy_spec_path),
        normalization_sha256=sha256_file(normalization_path),
        config_sha256=config_file_sha256,
        feature_schema_sha256=config.feature_schema.sha256(),
        dependency_lock_sha256=sha256_file(dependency_lock_path),
        files=tuple(
            ArtifactHashRecordV1(
                path.relative_to(checkpoint_root).as_posix(), sha256_file(path)
            )
            for path in checkpoint_files
        ),
    )
    envelope_path = envelope.save(checkpoint_root / "checkpoint.v3.json")
    task_id = config.task["id"]
    restored_policy = engine.restore_policy(checkpoint_root)
    metrics = engine.evaluate(
        restored_policy, FakePointEnvV3(task_id, config.evaluation["max_steps"])
    )
    metrics = {**metrics, "final_loss": losses[-1], "claim_allowed": False}
    audit_path = bundle.audit.save(output / "data_audit.json")
    metrics_path = _write_json(output / "metrics.json", metrics)
    git_sha, dirty = _git_identity()
    manifest = RunManifestV4R2(
        git_sha=git_sha,
        git_dirty=dirty,
        config_sha256=config_file_sha256,
        feature_schema_sha256=config.feature_schema.sha256(),
        data_audit_sha256=sha256_file(audit_path),
        checkpoint_envelope_sha256=sha256_file(envelope_path),
        metrics_sha256=sha256_file(metrics_path),
        policy_spec_sha256=sha256_file(policy_spec_path),
        normalization_sha256=sha256_file(normalization_path),
        dependency_lock_sha256=sha256_file(dependency_lock_path),
        model_source_sha256=sha256_file(model_source_path),
        runtime_sha256=sha256_file(runtime_path),
        policy_id=engine.policy_spec.policy_id,
        task_id=task_id,
        device=engine.policy_spec.device,
        train_seed=config.training["seed"],
        evaluation_seeds=tuple(config.evaluation["seeds"]),
        deterministic=True,
    )
    manifest.save(output / "manifest.json")
    return AlphaRunResult(losses, metrics, output)


def run_alpha(config: ExperimentConfig, *, overwrite: bool = False) -> AlphaRunResult:
    output = Path(config.artifacts["output_dir"])
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    backup = output.with_name(f".{output.name}.previous-{uuid.uuid4().hex}")
    try:
        staged = _execute_alpha(config, staging)
        if output.exists():
            output.replace(backup)
        staging.replace(output)
        if backup.exists():
            shutil.rmtree(backup)
        return AlphaRunResult(staged.losses, staged.metrics, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and not output.exists():
            backup.replace(output)
        raise
