from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from model.policy_base import ActionChunk

from .contracts import (
    DatasetSource,
    Observation,
    PolicyBatch,
    TaskEnv,
    Transition,
    VLAPolicy,
    normalize_device,
)
from .registry import PolicyRegistry, default_policy_registry


_EXECUTION_MODE_ALIASES = {
    "open_loop": "open_loop",
    "open_loop_chunk": "open_loop",
    "receding": "receding",
    "receding_horizon": "receding",
}


def _json_safe_value(value: Any, name: str) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist(), name)
    if isinstance(value, np.generic):
        return _json_safe_value(value.item(), name)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains NaN or infinite values")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} keys must be strings")
            result[key] = _json_safe_value(item, f"{name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item, f"{name}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{name} contains unsupported value type {type(value).__name__}")


def _json_safe_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    result = _json_safe_value(value, name)
    assert isinstance(result, dict)
    return result


def _optional_info_string(
    info: Mapping[str, Any],
    primary: str,
    fallback: str,
) -> str | None:
    value = info.get(primary, info.get(fallback))
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"transition.info.{primary}/{fallback} must be a string")
    return value


def _optional_info_distance(info: Mapping[str, Any]) -> float | None:
    value: Any = None
    for key in ("final_distance", "distance", "distance_to_goal"):
        if key in info:
            value = info[key]
            break
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("transition.info distance must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("transition.info distance must be finite and non-negative")
    return result


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not boolean or floating point")
    result = int(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def normalize_execution_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    try:
        return _EXECUTION_MODE_ALIASES[value]
    except KeyError as exc:
        raise ValueError(
            "execution_mode must be open_loop/open_loop_chunk or receding/receding_horizon"
        ) from exc


@dataclass(frozen=True)
class EngineConfig:
    device: str = "cpu"
    seed: int = 0
    eval_seed: int | None = None
    eval_seeds: tuple[int, ...] | None = None
    batch_size: int = 32
    train_steps: int = 100
    learning_rate: float = 0.04
    execution_mode: str = "receding"
    temporal_ensemble_decay: float | None = None
    eval_episodes: int = 5
    max_steps: int = 40

    def __post_init__(self) -> None:
        seed = _integer(self.seed, "seed")
        eval_seed = seed if self.eval_seed is None else _integer(self.eval_seed, "eval_seed")
        eval_seeds = self.eval_seeds
        if eval_seeds is not None:
            eval_seeds = tuple(_integer(value, "eval_seeds item") for value in eval_seeds)
            if len(eval_seeds) != self.eval_episodes:
                raise ValueError("eval_seeds must contain exactly eval_episodes values")
        object.__setattr__(
            self, "batch_size", _integer(self.batch_size, "batch_size", positive=True)
        )
        object.__setattr__(
            self, "train_steps", _integer(self.train_steps, "train_steps", positive=True)
        )
        object.__setattr__(
            self, "eval_episodes", _integer(self.eval_episodes, "eval_episodes", positive=True)
        )
        object.__setattr__(self, "max_steps", _integer(self.max_steps, "max_steps", positive=True))
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite value")
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "device", normalize_device(self.device))
        execution_mode = normalize_execution_mode(self.execution_mode)
        temporal_decay = self.temporal_ensemble_decay
        if temporal_decay is not None:
            temporal_decay = float(temporal_decay)
            if not math.isfinite(temporal_decay) or temporal_decay < 0:
                raise ValueError("temporal_ensemble_decay must be finite and non-negative")
            if execution_mode != "receding":
                raise ValueError("temporal ensembling requires receding-horizon execution")
        object.__setattr__(self, "execution_mode", execution_mode)
        object.__setattr__(self, "temporal_ensemble_decay", temporal_decay)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "eval_seed", eval_seed)
        object.__setattr__(self, "eval_seeds", eval_seeds)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "EngineConfig":
        unknown = sorted(
            set(config)
            - {
                "device",
                "seed",
                "eval_seed",
                "eval_seeds",
                "batch_size",
                "train_steps",
                "learning_rate",
                "execution_mode",
                "temporal_ensemble_decay",
                "eval_episodes",
                "max_steps",
            }
        )
        if unknown:
            raise ValueError("unknown engine config field(s): " + ", ".join(unknown))
        return cls(**dict(config))

    @property
    def evaluation_seed(self) -> int:
        # __post_init__ resolves None to the training seed.
        assert self.eval_seed is not None
        return self.eval_seed

    @property
    def evaluation_seeds(self) -> tuple[int, ...]:
        if self.eval_seeds is not None:
            return self.eval_seeds
        return tuple(self.evaluation_seed + episode for episode in range(self.eval_episodes))


@dataclass(frozen=True)
class TrainingResult:
    policy: VLAPolicy
    losses: tuple[float, ...]
    samples: int

    @property
    def final_loss(self) -> float:
        return self.losses[-1]


@runtime_checkable
class ObservationIntervention(Protocol):
    """Transform only the policy-facing modalities during an evaluation rollout."""

    def apply(
        self,
        observation: Observation,
        *,
        episode_seed: int,
        step_index: int,
    ) -> Observation: ...


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    total_reward: float
    steps: int
    terminated: bool
    success: bool
    actions: tuple[tuple[float, ...], ...]
    final_state: tuple[float, ...]
    task: str | None = None
    family: str | None = None
    pair: str | None = None
    final_distance: float | None = None
    final_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("task", "family", "pair"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        if self.final_distance is not None:
            if isinstance(self.final_distance, bool) or not isinstance(
                self.final_distance, (int, float)
            ):
                raise TypeError("final_distance must be numeric or None")
            distance = float(self.final_distance)
            if not math.isfinite(distance) or distance < 0:
                raise ValueError("final_distance must be finite and non-negative")
            object.__setattr__(self, "final_distance", distance)
        object.__setattr__(
            self,
            "final_info",
            _json_safe_mapping(self.final_info, "final_info"),
        )


@dataclass(frozen=True)
class EvaluationResult:
    episodes: tuple[EpisodeResult, ...]
    execution_mode: str

    @property
    def success_rate(self) -> float:
        return sum(int(item.success) for item in self.episodes) / len(self.episodes)

    @property
    def mean_reward(self) -> float:
        return float(np.mean([item.total_reward for item in self.episodes]))

    @property
    def mean_steps(self) -> float:
        return float(np.mean([item.steps for item in self.episodes]))


class Engine:
    """One deterministic train/evaluate path for every registered v2 policy."""

    def __init__(
        self,
        config: EngineConfig | Mapping[str, Any] | None = None,
        *,
        registry: PolicyRegistry | None = None,
    ) -> None:
        if config is None:
            resolved = EngineConfig()
        elif isinstance(config, EngineConfig):
            resolved = config
        elif isinstance(config, Mapping):
            resolved = EngineConfig.from_mapping(config)
        else:
            raise TypeError("config must be EngineConfig, a mapping, or None")
        self.config = resolved
        self.registry = registry or default_policy_registry()

    def create_policy(
        self,
        policy_id: str,
        config: Mapping[str, Any] | None = None,
    ) -> VLAPolicy:
        policy_config = dict(config or {})
        policy_config.setdefault("device", self.config.device)
        policy_config.setdefault("seed", self.config.seed)
        policy = self.registry.create(policy_id, policy_config)
        self._validate_policy(policy)
        return policy

    def train(
        self,
        policy: VLAPolicy | str,
        source: DatasetSource,
        *,
        policy_config: Mapping[str, Any] | None = None,
    ) -> TrainingResult:
        resolved_policy = (
            self.create_policy(policy, policy_config) if isinstance(policy, str) else policy
        )
        self._validate_policy(resolved_policy)
        if not isinstance(source, DatasetSource):
            raise TypeError("source must implement DatasetSource.load()")
        transitions = tuple(source.load())
        batch = self._supervision_batch(transitions, resolved_policy)
        rng = np.random.default_rng(self.config.seed)
        losses: list[float] = []
        for _ in range(self.config.train_steps):
            replace = batch.batch_size < self.config.batch_size
            indices = rng.choice(
                batch.batch_size,
                size=self.config.batch_size,
                replace=replace,
            )
            step_batch = PolicyBatch(
                observations=tuple(batch.observations[int(index)] for index in indices),
                targets=batch.targets[indices],
                valid_mask=batch.valid_mask[indices],
                device=self.config.device,
            )
            loss = float(
                resolved_policy.train_batch(step_batch, learning_rate=self.config.learning_rate)
            )
            if not math.isfinite(loss) or loss < 0:
                raise ValueError("policy returned an invalid training loss")
            losses.append(loss)
        return TrainingResult(
            policy=resolved_policy,
            losses=tuple(losses),
            samples=batch.batch_size,
        )

    def evaluate(
        self,
        policy: VLAPolicy,
        env: TaskEnv,
        *,
        intervention: ObservationIntervention | None = None,
    ) -> EvaluationResult:
        if not isinstance(env, TaskEnv):
            raise TypeError("env must implement TaskEnv.reset(), step(), and close()")
        try:
            self._validate_policy(policy)
            if intervention is not None and not isinstance(
                intervention, ObservationIntervention
            ):
                raise TypeError("intervention must implement ObservationIntervention.apply()")
            episodes = tuple(
                self._evaluate_episode(policy, env, seed, intervention)
                for seed in self.config.evaluation_seeds
            )
            return EvaluationResult(episodes, self.config.execution_mode)
        finally:
            env.close()

    def save_checkpoint(
        self,
        policy: VLAPolicy,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        self._validate_policy(policy)
        return policy.save_checkpoint(Path(path), metadata=metadata)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        policy_id: str | None = None,
        policy_config: Mapping[str, Any] | None = None,
    ) -> VLAPolicy:
        load_config = dict(policy_config or {})
        load_config.setdefault("device", self.config.device)
        policy = self.registry.load_checkpoint(path, policy_id=policy_id, config=load_config)
        self._validate_policy(policy)
        return policy

    def _supervision_batch(
        self,
        transitions: tuple[Transition, ...],
        policy: VLAPolicy,
    ) -> PolicyBatch:
        if not transitions:
            raise ValueError("dataset source returned no transitions")
        if any(not isinstance(item, Transition) for item in transitions):
            raise TypeError("dataset source must return Transition instances")
        for index, transition in enumerate(transitions):
            if transition.action.shape != (policy.action_dim,):
                raise ValueError(
                    f"transition {index} action must have shape {(policy.action_dim,)}; "
                    f"got {transition.action.shape}"
                )

        targets = np.zeros(
            (len(transitions), policy.chunk_size, policy.action_dim), dtype=np.float32
        )
        valid_mask = np.zeros((len(transitions), policy.chunk_size), dtype=bool)
        for index in range(len(transitions)):
            for offset in range(policy.chunk_size):
                target_index = index + offset
                if target_index >= len(transitions):
                    break
                if offset and transitions[target_index - 1].terminated:
                    break
                transition = transitions[target_index]
                targets[index, offset] = transition.action
                valid_mask[index, offset] = True
                if transition.terminated:
                    break
        return PolicyBatch(
            observations=tuple(item.observation for item in transitions),
            targets=targets,
            valid_mask=valid_mask,
            device=self.config.device,
        )

    def _evaluate_episode(
        self,
        policy: VLAPolicy,
        env: TaskEnv,
        seed: int,
        intervention: ObservationIntervention | None = None,
    ) -> EpisodeResult:
        observation = env.reset(seed=seed)
        if not isinstance(observation, Observation):
            raise TypeError("TaskEnv.reset() must return Observation")
        total_reward = 0.0
        actions: list[tuple[float, ...]] = []
        terminated = False
        success = False
        final_info: dict[str, Any] = {}
        ensembler: Any | None = None
        if self.config.temporal_ensemble_decay is not None:
            from .temporal import TemporalEnsembler

            ensembler = TemporalEnsembler(
                decay=self.config.temporal_ensemble_decay,
                action_dim=policy.action_dim,
                chunk_size=policy.chunk_size,
            )

        while len(actions) < self.config.max_steps and not terminated:
            policy_observation = self._policy_observation(
                observation,
                intervention=intervention,
                episode_seed=seed,
                step_index=len(actions),
            )
            chunk = policy.predict_chunk(policy_observation)
            self._validate_chunk(chunk, policy)
            if ensembler is None:
                executable = [
                    chunk.values[int(index)] for index in np.flatnonzero(chunk.valid_mask)
                ]
                if self.config.execution_mode == "receding":
                    executable = executable[:1]
            else:
                executable = [ensembler.update(chunk)]
            for action in executable:
                if len(actions) >= self.config.max_steps:
                    break
                transition = env.step(action)
                if not isinstance(transition, Transition):
                    raise TypeError("TaskEnv.step() must return Transition")
                if transition.action.shape != (policy.action_dim,):
                    raise ValueError("TaskEnv returned an action with the wrong shape")
                total_reward += transition.reward
                actions.append(tuple(float(value) for value in transition.action))
                observation = transition.next_observation
                terminated = transition.terminated
                success = bool(transition.info.get("success", False))
                final_info = _json_safe_mapping(transition.info, "transition.info")
                if terminated:
                    break

        task = _optional_info_string(final_info, "task_id", "task")
        family = _optional_info_string(final_info, "task_family", "family")
        pair = _optional_info_string(final_info, "pair_id", "pair")
        final_distance = _optional_info_distance(final_info)

        return EpisodeResult(
            seed=seed,
            total_reward=float(total_reward),
            steps=len(actions),
            terminated=terminated,
            success=success,
            actions=tuple(actions),
            final_state=tuple(float(value) for value in observation.state),
            task=task,
            family=family,
            pair=pair,
            final_distance=final_distance,
            final_info=final_info,
        )

    @staticmethod
    def _policy_observation(
        observation: Observation,
        *,
        intervention: ObservationIntervention | None,
        episode_seed: int,
        step_index: int,
    ) -> Observation:
        state = np.array(observation.state, copy=True)
        image = None if observation.image is None else np.array(observation.image, copy=True)
        policy_observation = Observation(
            state=state,
            instruction=observation.instruction,
            image=image,
        )
        if intervention is None:
            return policy_observation
        intervened = intervention.apply(
            policy_observation,
            episode_seed=episode_seed,
            step_index=step_index,
        )
        if not isinstance(intervened, Observation):
            raise TypeError("ObservationIntervention.apply() must return Observation")
        if not np.array_equal(intervened.state, observation.state):
            raise ValueError("an observation intervention must not change state")
        return Observation(
            state=intervened.state,
            instruction=intervened.instruction,
            image=intervened.image,
        )

    def _validate_policy(self, policy: object) -> None:
        if not isinstance(policy, VLAPolicy):
            raise TypeError("policy must implement VLAPolicy")
        if normalize_device(policy.device) != self.config.device:
            raise ValueError(
                f"policy device {policy.device!r} does not match engine device "
                f"{self.config.device!r}"
            )
        if policy.action_dim <= 0 or policy.chunk_size <= 0:
            raise ValueError("policy action_dim and chunk_size must be positive")

    @staticmethod
    def _validate_chunk(chunk: object, policy: VLAPolicy) -> None:
        if not isinstance(chunk, ActionChunk):
            raise TypeError("VLAPolicy.predict_chunk() must return ActionChunk")
        expected = (policy.chunk_size, policy.action_dim)
        if chunk.values.shape != expected or chunk.valid_mask.shape != expected[:1]:
            raise ValueError(
                f"policy ActionChunk must have values shape {expected} and mask shape "
                f"{expected[:1]}"
            )
