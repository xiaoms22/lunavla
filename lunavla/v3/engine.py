from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from lunavla.contracts import Observation, PolicyBatch, VLAPolicy
from lunavla.registry import default_policy_registry

from .artifacts import CheckpointEnvelopeV4, RunManifestV4, sha256_file
from .config import ExperimentConfig
from .contracts import EpisodeRecordV3, ObservationV3, TaskEnvV3
from .data import DatasetBundle, InMemoryDatasetSourceV3, audit_episodes, split_episode_ids
from .fake_tasks import FakePointEnvV3, make_fake_episodes


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
    def __init__(self, policy: VLAPolicy, *, state_feature: str, unused_modalities: Sequence[str]) -> None:
        self.policy = policy
        self.state_feature = state_feature
        self.unused_modalities = tuple(unused_modalities)
        if len(self.unused_modalities) != len(set(self.unused_modalities)):
            raise ValueError("unused_modalities cannot contain duplicates")

    @property
    def policy_id(self) -> str:
        return self.policy.policy_id

    def _observation(self, value: ObservationV3) -> Observation:
        if self.state_feature not in value.state:
            raise ValueError(f"missing policy state feature {self.state_feature}")
        if value.images and "image" not in self.unused_modalities:
            raise ValueError("v2 policy bridge cannot silently discard image features")
        instruction = value.instruction
        if instruction is not None and "instruction" in self.unused_modalities:
            instruction = None
        return Observation(np.asarray(value.state[self.state_feature]), instruction=instruction)

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

    def predict_chunk(self, observation: ObservationV3) -> Float32Array:
        return self.policy.predict_chunk(self._observation(observation)).values

    def save_checkpoint(self, path: Path, *, metadata: Mapping[str, Any]) -> Path:
        return self.policy.save_checkpoint(path, metadata=metadata)


@dataclass(frozen=True)
class AlphaRunResult:
    losses: tuple[float, ...]
    metrics: Mapping[str, Any]
    output_dir: Path


class EngineV3:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def _policy(self) -> V2PolicyBridge:
        parameters = dict(self.config.policy["parameters"])
        raw = dict(parameters.get("legacy", parameters))
        unused = tuple(raw.pop("unused_modalities", ()))
        state_feature = str(raw.pop("state_feature", "state.proprioception"))
        raw.setdefault("seed", self.config.training["seed"])
        raw.setdefault("device", self.config.training["device"])
        policy = default_policy_registry().create(self.config.policy["type"], raw)
        return V2PolicyBridge(policy, state_feature=state_feature, unused_modalities=unused)

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

    def train(self, dataset: InMemoryDatasetSourceV3) -> tuple[V2PolicyBridge, tuple[float, ...]]:
        policy = self._policy()
        observations, targets, masks = self._supervision(dataset.load(), policy.policy.chunk_size)
        rng = np.random.default_rng(self.config.training["seed"])
        losses: list[float] = []
        for _ in range(self.config.training["steps"]):
            indices = rng.integers(0, len(observations), size=self.config.training["batch_size"])
            losses.append(
                policy.train_batch(
                    [observations[int(index)] for index in indices], targets[indices], masks[indices],
                    learning_rate=self.config.training["learning_rate"],
                )
            )
        return policy, tuple(losses)

    def restore_policy(self, checkpoint: str | Path) -> V2PolicyBridge:
        parameters = dict(self.config.policy["parameters"])
        raw = dict(parameters.get("legacy", parameters))
        unused = tuple(raw.get("unused_modalities", ()))
        state_feature = str(raw.get("state_feature", "state.proprioception"))
        policy = default_policy_registry().load_checkpoint(
            checkpoint, policy_id=self.config.policy["type"]
        )
        return V2PolicyBridge(policy, state_feature=state_feature, unused_modalities=unused)

    def evaluate(self, policy: V2PolicyBridge, env: TaskEnvV3) -> dict[str, Any]:
        rewards: list[float] = []
        successes: list[bool] = []
        steps: list[int] = []
        try:
            for seed in self.config.evaluation["seeds"]:
                observation = env.reset(seed=seed)
                total_reward = 0.0
                success = False
                step_count = 0
                while step_count < self.config.evaluation["max_steps"]:
                    chunk = policy.predict_chunk(observation)
                    actions = chunk if self.config.evaluation["execution_mode"] == "open_loop_chunk" else chunk[:1]
                    stop = False
                    for action in actions:
                        transition = env.step(action)
                        observation = transition.next_observation
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


def run_alpha(config: ExperimentConfig, *, overwrite: bool = False) -> AlphaRunResult:
    output = Path(config.artifacts["output_dir"])
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rollouts = output / "rollouts"
    if rollouts.exists():
        for child in rollouts.iterdir():
            if child.is_file():
                child.unlink()
    rollouts.mkdir(exist_ok=True)
    bundle = dataset_for_config(config)
    engine = EngineV3(config)
    policy, losses = engine.train(bundle.source("train"))
    config_path = _write_json(output / "resolved_config.json", config.to_dict())
    config_file_sha256 = sha256_file(config_path)
    checkpoint_name = config.artifacts["checkpoint_name"]
    checkpoint = policy.save_checkpoint(
        output / checkpoint_name,
        metadata={"v3": {"config_sha256": config_file_sha256, "feature_schema_sha256": config.feature_schema.sha256()}},
    )
    envelope = CheckpointEnvelopeV4(
        policy.policy_id, checkpoint.name, sha256_file(checkpoint), config_file_sha256, config.feature_schema.sha256()
    )
    envelope_path = envelope.save(output / "checkpoint.v3.json")
    task_id = config.task["id"]
    restored_policy = engine.restore_policy(checkpoint)
    metrics = engine.evaluate(
        restored_policy, FakePointEnvV3(task_id, config.evaluation["max_steps"])
    )
    metrics = {**metrics, "final_loss": losses[-1], "claim_allowed": False}
    audit_path = bundle.audit.save(output / "data_audit.json")
    metrics_path = _write_json(output / "metrics.json", metrics)
    git_sha, dirty = _git_identity()
    manifest = RunManifestV4(
        git_sha=git_sha,
        git_dirty=dirty,
        config_sha256=config_file_sha256,
        feature_schema_sha256=config.feature_schema.sha256(),
        data_audit_sha256=sha256_file(audit_path),
        checkpoint_envelope_sha256=sha256_file(envelope_path),
        metrics_sha256=sha256_file(metrics_path),
        policy_id=policy.policy_id,
        task_id=task_id,
        train_seed=config.training["seed"],
        evaluation_seeds=tuple(config.evaluation["seeds"]),
        deterministic=True,
    )
    manifest.save(output / "manifest.json")
    return AlphaRunResult(losses, metrics, output)
