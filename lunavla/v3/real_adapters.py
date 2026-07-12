from __future__ import annotations

import importlib
import math
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from .contracts import EpisodeRecordV3, FeatureSchema, ObservationV3, TransitionV3
from .integration_contracts import ExternalDatasetSpecV1, SimulationTaskSpecV1


Array = npt.NDArray[np.generic]


class DatasetFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Sequence[Mapping[str, Any]]: ...


def _lookup(mapping: Mapping[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    current: Any = mapping
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(key)
        current = current[part]
    return current


def _numpy(value: Any, *, name: str) -> Array:
    current = value
    for method in ("detach", "cpu"):
        operation = getattr(current, method, None)
        if callable(operation):
            current = operation()
    operation = getattr(current, "numpy", None)
    if callable(operation):
        current = operation()
    result = np.asarray(current)
    if result.dtype.kind == "O":
        raise TypeError(f"{name} cannot have object dtype")
    return result


def _scalar_int(value: Any, name: str) -> int:
    array = _numpy(value, name=name)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    item = array.reshape(-1)[0].item()
    if isinstance(item, bool) or not isinstance(item, int):
        raise TypeError(f"{name} must be an integer")
    if item < 0:
        raise ValueError(f"{name} must be non-negative")
    return item


def _text(value: Any, name: str) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _typed_image(value: Any, shape: tuple[int, ...], name: str) -> npt.NDArray[np.uint8]:
    image = _numpy(value, name=name)
    if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.dtype.kind == "f":
        if not np.all(np.isfinite(image)):
            raise ValueError(f"{name} contains non-finite values")
        if image.size and float(np.max(image)) <= 1.0:
            image = np.rint(image.astype(np.float32) * np.float32(255.0))
    if image.size and (float(np.min(image)) < 0 or float(np.max(image)) > 255):
        raise ValueError(f"{name} must lie in [0, 255]")
    result = image.astype(np.uint8, copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} shape {result.shape} does not match {shape}")
    return result


def _typed_vector(value: Any, shape: tuple[int, ...], name: str) -> npt.NDArray[np.float32]:
    vector = _numpy(value, name=name).astype(np.float32, copy=True)
    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]
    if vector.shape != shape:
        raise ValueError(f"{name} shape {vector.shape} does not match {shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


@dataclass(frozen=True)
class EpisodeSelectionV1:
    task_to_episode: Mapping[int, int]
    task_to_language: Mapping[int, str]

    def __post_init__(self) -> None:
        task_to_episode = dict(self.task_to_episode)
        task_to_language = dict(self.task_to_language)
        if set(task_to_episode) != set(task_to_language):
            raise ValueError("episode and language task indexes must match")
        if len(set(task_to_episode.values())) != len(task_to_episode):
            raise ValueError("selected episode IDs cannot repeat across tasks")
        for task, episode in task_to_episode.items():
            if isinstance(task, bool) or not isinstance(task, int) or task < 0:
                raise ValueError("task indexes must be non-negative integers")
            if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
                raise ValueError("episode indexes must be non-negative integers")
            _text(task_to_language[task], f"task language {task}")
        object.__setattr__(self, "task_to_episode", dict(sorted(task_to_episode.items())))
        object.__setattr__(self, "task_to_language", dict(sorted(task_to_language.items())))

    @property
    def episodes(self) -> tuple[int, ...]:
        return tuple(self.task_to_episode.values())


def select_minimum_episode_per_task(
    metadata: Sequence[Mapping[str, Any]],
    *,
    task_ids: Sequence[int],
    expected_languages: Mapping[int, str] | None = None,
) -> EpisodeSelectionV1:
    by_task: dict[int, list[tuple[int, str]]] = defaultdict(list)
    seen_episodes: set[int] = set()
    for row in metadata:
        task = _scalar_int(_lookup(row, "task_index"), "task_index")
        episode = _scalar_int(_lookup(row, "episode_index"), "episode_index")
        language = _text(_lookup(row, "task"), "task")
        if episode in seen_episodes:
            raise ValueError("metadata contains duplicate episode_index")
        seen_episodes.add(episode)
        by_task[task].append((episode, language))
    selection: dict[int, int] = {}
    languages: dict[int, str] = {}
    for task in task_ids:
        rows = sorted(by_task.get(task, ()))
        if not rows:
            raise ValueError(f"no episode metadata for task {task}")
        episode, language = rows[0]
        if expected_languages is not None and expected_languages.get(task) != language:
            raise ValueError(f"task-language mapping drift for task {task}")
        selection[task] = episode
        languages[task] = language
    return EpisodeSelectionV1(selection, languages)


class LeRobotDatasetSourceV3:
    """Strict LeRobot 0.6 frame adapter with no raw-object or metadata leakage."""

    def __init__(
        self,
        spec: ExternalDatasetSpecV1,
        schema: FeatureSchema,
        *,
        dataset_factory: DatasetFactory | None = None,
        root: str | Path | None = None,
        metadata: Sequence[Mapping[str, Any]] = (),
        expected_task_languages: Mapping[int, str] | None = None,
    ) -> None:
        spec.validate_supported_source()
        self.spec = spec
        self.schema = schema
        self._factory = dataset_factory
        self._root = None if root is None else Path(root)
        if spec.episode_selection == "explicit":
            self.selection = EpisodeSelectionV1(
                {0: spec.episodes[0]},
                {0: "PushT"},
            )
        else:
            self.selection = select_minimum_episode_per_task(
                metadata,
                task_ids=spec.task_ids,
                expected_languages=expected_task_languages,
            )

    @staticmethod
    def _default_factory(**kwargs: Any) -> Sequence[Mapping[str, Any]]:
        module = importlib.import_module("lerobot.datasets.lerobot_dataset")
        factory = getattr(module, "LeRobotDataset", None)
        if factory is None:
            raise ImportError("LeRobotDataset is unavailable in lerobot 0.6")
        return factory(**kwargs)

    def _observation(
        self,
        row: Mapping[str, Any],
        *,
        episode: int,
        step: int,
        instruction: str | None,
    ) -> ObservationV3:
        images: dict[str, Array] = {}
        state: dict[str, Array] = {}
        for feature in self.schema.features:
            if feature.role == "image":
                images[feature.name] = _typed_image(
                    _lookup(row, feature.source_key), feature.shape, feature.source_key
                )
            elif feature.role == "state":
                state[feature.name] = _typed_vector(
                    _lookup(row, feature.source_key), feature.shape, feature.source_key
                )
        observation = ObservationV3(
            images=images,
            state=state,
            instruction=instruction,
            timestamp_s=step / 10.0,
            episode_id=episode,
            step_index=step,
            metadata={},
        )
        self.schema.validate_observation(observation)
        return observation

    def load(self) -> tuple[EpisodeRecordV3, ...]:
        factory = self._factory or self._default_factory
        rows = factory(
            repo_id=self.spec.repo_id,
            root=self._root,
            revision=self.spec.revision,
            episodes=list(self.selection.episodes),
            video_backend=self.spec.video_backend,
            download_videos=True,
            return_uint8=self.spec.return_uint8,
        )
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        selected = set(self.selection.episodes)
        for row in rows:
            episode = _scalar_int(_lookup(row, "episode_index"), "episode_index")
            if episode not in selected:
                raise ValueError("dataset factory returned an unselected episode")
            grouped[episode].append(row)
        if set(grouped) != selected:
            raise ValueError("dataset factory did not return every selected episode")
        language_by_episode = {
            episode: self.selection.task_to_language[task]
            for task, episode in self.selection.task_to_episode.items()
        }
        records: list[EpisodeRecordV3] = []
        action_feature = self.schema.by_role("action")
        if len(action_feature) != 1:
            raise ValueError("real dataset mapping requires exactly one action feature")
        for episode in self.selection.episodes:
            episode_rows = sorted(
                grouped[episode],
                key=lambda row: _scalar_int(_lookup(row, "frame_index"), "frame_index"),
            )
            indices = [
                _scalar_int(_lookup(row, "frame_index"), "frame_index") for row in episode_rows
            ]
            if indices != list(range(len(indices))):
                raise ValueError("episode frame_index must be contiguous and start at zero")
            observations = [
                self._observation(
                    row,
                    episode=episode,
                    step=step,
                    instruction=language_by_episode[episode],
                )
                for step, row in enumerate(episode_rows)
            ]
            transitions: list[TransitionV3] = []
            for step, (row, observation) in enumerate(zip(episode_rows, observations)):
                action = _typed_vector(
                    _lookup(row, action_feature[0].source_key),
                    action_feature[0].shape,
                    action_feature[0].source_key,
                )
                self.schema.validate_action(action)
                final = step == len(observations) - 1
                next_observation = (
                    observations[step + 1]
                    if not final
                    else ObservationV3(
                        images=observation.images,
                        state=observation.state,
                        instruction=observation.instruction,
                        timestamp_s=(step + 1) / 10.0,
                        episode_id=episode,
                        step_index=step + 1,
                        metadata={},
                    )
                )
                transitions.append(
                    TransitionV3(
                        observation,
                        action,
                        0.0,
                        next_observation,
                        terminated=final,
                        truncated=False,
                        info={},
                    )
                )
            records.append(EpisodeRecordV3(episode, tuple(transitions), metadata={}))
        return tuple(records)


class _EnvV3Base:
    def __init__(
        self,
        env: Any,
        schema: FeatureSchema,
        spec: SimulationTaskSpecV1,
        *,
        observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self._env = env
        self._schema = schema
        self._spec = spec
        self._closed = False
        self._current: ObservationV3 | None = None
        self._step = 0
        self._episode_id: str | int = "unset"
        self._observation_processor = observation_processor or (lambda value: value)

    @property
    def close_count(self) -> int:
        return int(self._closed)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._env.close()

    def __enter__(self) -> "_EnvV3Base":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _map(self, raw: Mapping[str, Any], *, instruction: str | None) -> ObservationV3:
        processed = self._observation_processor(raw)
        if not isinstance(processed, Mapping):
            raise TypeError("observation processor must return a mapping")
        images: dict[str, Array] = {}
        state: dict[str, Array] = {}
        for source_key, target in self._spec.camera_mapping.items():
            feature = next(item for item in self._schema.by_role("image") if item.name == target)
            images[target] = _typed_image(
                _lookup(processed, source_key), feature.shape, source_key
            )
        for source_key, target in self._spec.state_mapping.items():
            feature = next(item for item in self._schema.by_role("state") if item.name == target)
            state[target] = _typed_vector(
                _lookup(processed, source_key), feature.shape, source_key
            )
        observation = ObservationV3(
            images,
            state,
            instruction,
            self._step / 10.0,
            self._episode_id,
            self._step,
            {},
        )
        self._schema.validate_observation(observation)
        return observation

    def _transition(self, result: Any, action: Array, *, instruction: str | None) -> TransitionV3:
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError("environment step must return Gymnasium's five-item tuple")
        raw, reward, terminated, truncated, _info = result
        if not isinstance(raw, Mapping):
            raise TypeError("environment observation must be a mapping")
        if self._current is None:
            raise RuntimeError("reset must be called before step")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("environment reward must be finite")
        self._step += 1
        next_observation = self._map(raw, instruction=instruction)
        transition = TransitionV3(
            self._current,
            action,
            float(reward),
            next_observation,
            bool(terminated),
            bool(truncated),
            info={},
        )
        self._current = next_observation
        return transition


class PushTEnvV3(_EnvV3Base):
    def __init__(
        self,
        schema: FeatureSchema,
        spec: SimulationTaskSpecV1,
        *,
        env_factory: Callable[..., Any] | None = None,
    ) -> None:
        if spec.environment_id != "gym_pusht/PushT-v0":
            raise ValueError("PushTEnvV3 requires gym_pusht/PushT-v0")
        if env_factory is None:
            gym = importlib.import_module("gymnasium")
            env_factory = gym.make
        super().__init__(env_factory(spec.environment_id, obs_type="pixels_agent_pos"), schema, spec)

    def reset(self, *, seed: int | None = None) -> ObservationV3:
        self._episode_id = 0 if seed is None else seed
        self._step = 0
        result = self._env.reset(seed=seed)
        raw = result[0] if isinstance(result, tuple) else result
        if not isinstance(raw, Mapping):
            raise TypeError("PushT reset observation must be a mapping")
        self._current = self._map(raw, instruction="Push the block to the target")
        return self._current

    def step(self, action: Array) -> TransitionV3:
        feature = self._schema.by_role("action")[0]
        typed = _typed_vector(action, feature.shape, "action")
        return self._transition(
            self._env.step(typed), typed, instruction="Push the block to the target"
        )


class LiberoSpatialEnvV3(_EnvV3Base):
    def __init__(
        self,
        schema: FeatureSchema,
        spec: SimulationTaskSpecV1,
        *,
        task_id: int,
        init_state_id: int,
        task_language: str,
        env_factory: Callable[..., Any],
        observation_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        if spec.environment_id != "libero" or spec.suite != "libero_spatial":
            raise ValueError("LiberoSpatialEnvV3 requires the pinned libero_spatial suite")
        if task_id not in spec.task_ids or init_state_id not in spec.init_state_ids:
            raise ValueError("task_id or init_state_id is outside the pinned subset")
        self.task_id = task_id
        self.init_state_id = init_state_id
        self.task_language = _text(task_language, "task_language")
        env = env_factory(
            suite=spec.suite,
            task_id=task_id,
            init_state_id=init_state_id,
            headless=spec.headless,
        )
        if observation_processor is None:
            observation_processor = self._official_observation_processor()
        super().__init__(env, schema, spec, observation_processor=observation_processor)

    @staticmethod
    def _official_observation_processor() -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
        utils = importlib.import_module("lerobot.envs.utils")
        processors = importlib.import_module("lerobot.processor.env_processor")
        pipeline_module = importlib.import_module("lerobot.processor.pipeline")
        pipeline = pipeline_module.PolicyProcessorPipeline(
            steps=[processors.LiberoProcessorStep()]
        )

        def process(raw: Mapping[str, Any]) -> Mapping[str, Any]:
            return pipeline(utils.preprocess_observation(dict(raw)))

        return process

    def reset(self, *, seed: int | None = None) -> ObservationV3:
        self._episode_id = f"task-{self.task_id}-init-{self.init_state_id}"
        self._step = 0
        result = self._env.reset(seed=seed, init_state_id=self.init_state_id)
        raw = result[0] if isinstance(result, tuple) else result
        if not isinstance(raw, Mapping):
            raise TypeError("LIBERO reset observation must be a mapping")
        self._current = self._map(raw, instruction=self.task_language)
        return self._current

    def step(self, action: Array) -> TransitionV3:
        feature = self._schema.by_role("action")[0]
        typed = _typed_vector(action, feature.shape, "action")
        if np.any(typed < -1.0) or np.any(typed > 1.0):
            raise ValueError("LIBERO relative actions must lie in [-1, 1]")
        return self._transition(self._env.step(typed), typed, instruction=self.task_language)


@contextmanager
def close_environments(environments: Sequence[_EnvV3Base]) -> Iterator[None]:
    """Yield once and close every environment exactly once, including partial failures."""

    try:
        yield None
    finally:
        for environment in environments:
            environment.close()
