from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from .contracts import Observation, Transition


LEROBOT_PUSHT_REPO_ID = "lerobot/pusht"
LEROBOT_PUSHT_REVISION = "b1c3ecbae7f244acc039a3dbc255a00dad1372b9"
LEROBOT_PUSHT_EPISODES = (0,)
LEROBOT_PUSHT_VIDEO_BACKEND = "pyav"
LEROBOT_PUSHT_IMAGE_SHAPE = (96, 96, 3)
LEROBOT_PUSHT_STATE_SHAPE = (2,)
LEROBOT_PUSHT_ACTION_SHAPE = (2,)


class LeRobotUnavailableError(ImportError):
    """Raised only when the optional LeRobot-backed path is requested."""


@dataclass(frozen=True)
class LeRobotFieldMap:
    """Map LeRobot 0.6 frame keys to the LunaVLA contracts."""

    state_key: str = "observation.state"
    image_keys: tuple[str, ...] = (
        "observation.images.top",
        "observation.images.front",
        "observation.image",
    )
    action_key: str = "action"
    instruction_keys: tuple[str, ...] = ("task", "language_instruction")
    reward_key: str = "next.reward"
    terminated_key: str = "next.done"
    episode_index_key: str = "episode_index"
    success_keys: tuple[str, ...] = ("next.success", "success", "is_success")


@dataclass(frozen=True)
class MappedLeRobotSample:
    observation: Observation
    action: NDArray[np.float32]
    metadata: Mapping[str, object]


def lerobot_installation_status() -> dict[str, object]:
    """Inspect availability without importing torch, datasets, codecs, or LeRobot."""

    installed = importlib.util.find_spec("lerobot") is not None
    version: str | None = None
    if installed:
        try:
            version = importlib.metadata.version("lerobot")
        except importlib.metadata.PackageNotFoundError:
            installed = False
    compatible = bool(version and re.match(r"^0\.6(?:\.|$)", version))
    return {"installed": installed, "version": version, "compatible_0_6": compatible}


def require_lerobot_06() -> str:
    status = lerobot_installation_status()
    if not status["installed"]:
        raise LeRobotUnavailableError(
            "LeRobot is optional. Install the v2 profile with lerobot[dataset]==0.6.* "
            "before loading a real repo_id."
        )
    version = str(status["version"])
    if not status["compatible_0_6"]:
        raise LeRobotUnavailableError(
            f"LunaVLA's adapter targets lerobot==0.6.*; found {version!r}."
        )
    return version


def load_lerobot_dataset_factory() -> Callable[..., Any]:
    """Resolve the official 0.6 dataset class only when a real dataset is requested."""

    require_lerobot_06()
    module = importlib.import_module("lerobot.datasets.lerobot_dataset")
    factory = getattr(module, "LeRobotDataset", None)
    if factory is None:
        raise LeRobotUnavailableError(
            "lerobot.datasets.lerobot_dataset.LeRobotDataset is unavailable in this installation"
        )
    return factory


def _lookup(mapping: Mapping[str, Any], path: str, *, default: Any = None) -> Any:
    if path in mapping:
        return mapping[path]
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _to_numpy(value: Any, *, name: str) -> NDArray[np.generic]:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    try:
        result = np.asarray(current)
    except Exception as exc:  # noqa: BLE001 - optional tensor backends vary.
        raise TypeError(f"cannot convert {name} to a NumPy array") from exc
    if result.dtype == np.dtype("O"):
        raise TypeError(f"{name} cannot have object dtype")
    return result


def _state_array(value: Any) -> NDArray[np.float32]:
    state = _to_numpy(value, name="state")
    if state.ndim == 2 and state.shape[0] == 1:
        state = state[0]
    if state.ndim != 1 or state.size == 0:
        raise ValueError(f"state must be a non-empty 1D array, got {state.shape}")
    state = cast(NDArray[np.float32], state.astype(np.float32, copy=True))
    if not np.all(np.isfinite(state)):
        raise ValueError("state contains NaN or Inf")
    return state


def _action_array(value: Any) -> NDArray[np.float32]:
    action = _to_numpy(value, name="action")
    if action.ndim == 2 and action.shape[0] == 1:
        action = action[0]
    if action.ndim != 1 or action.size == 0:
        raise ValueError(f"action must be a non-empty 1D array, got {action.shape}")
    action = cast(NDArray[np.float32], action.astype(np.float32, copy=True))
    if not np.all(np.isfinite(action)):
        raise ValueError("action contains NaN or Inf")
    return action


def _image_array(value: Any) -> NDArray[np.generic]:
    image = _to_numpy(value, name="image")
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim not in {2, 3}:
        raise ValueError(f"image must be grayscale or HWC/CHW, got {image.shape}")
    if image.ndim == 3 and image.shape[-1] not in {3, 4}:
        raise ValueError(f"image channel dimension must be 3 or 4, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32, copy=True)
        if not np.all(np.isfinite(image)):
            raise ValueError("image contains NaN or Inf")
        return image
    if not np.issubdtype(image.dtype, np.integer):
        raise TypeError(f"unsupported image dtype: {image.dtype}")
    if image.size and (int(np.min(image)) < 0 or int(np.max(image)) > 255):
        raise ValueError("integer image values must lie in [0, 255]")
    return image.astype(np.uint8, copy=True)


def _instruction(mapping: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = _lookup(mapping, key)
        if value is None:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        text = str(value).strip()
        if text:
            return text
    return None


def _optional_episode_index(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    array = _to_numpy(value, name="episode_index")
    if array.size != 1:
        raise ValueError("episode_index must be a scalar")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    if isinstance(scalar, bool) or not isinstance(scalar, int):
        raise TypeError("episode_index must be an integer or string")
    return scalar


def _episode_ids(values: Sequence[int]) -> tuple[int, ...]:
    episodes: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"episodes[{index}] must be an integer")
        episode = int(value)
        if episode < 0:
            raise ValueError(f"episodes[{index}] must be non-negative")
        episodes.append(episode)
    if not episodes:
        raise ValueError("episodes cannot be empty")
    if len(set(episodes)) != len(episodes):
        raise ValueError("episodes cannot contain duplicate episode IDs")
    return tuple(episodes)


def _boolean_scalar(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    array = _to_numpy(value, name=name)
    if array.size != 1 or array.dtype.kind != "b":
        raise TypeError(f"{name} must be a boolean scalar")
    return bool(array.reshape(-1)[0])


def _float_scalar(value: Any, *, name: str) -> float:
    array = _to_numpy(value, name=name)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar")
    try:
        result = float(array.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def map_lerobot_observation(
    sample: Mapping[str, Any] | NDArray[np.generic],
    *,
    field_map: LeRobotFieldMap = LeRobotFieldMap(),
    require_image: bool = False,
) -> Observation:
    if not isinstance(sample, Mapping):
        return Observation(state=_state_array(sample))
    state_value = _lookup(sample, field_map.state_key)
    if state_value is None:
        raise KeyError(f"LeRobot sample is missing state key {field_map.state_key!r}")
    image_value = None
    selected_image_key = None
    for key in field_map.image_keys:
        value = _lookup(sample, key)
        if value is not None:
            selected_image_key = key
            image_value = value
            break
    if require_image and image_value is None:
        raise KeyError(f"LeRobot sample is missing image keys {field_map.image_keys!r}")
    image = None if image_value is None else _image_array(image_value)
    observation = Observation(
        state=_state_array(state_value),
        instruction=_instruction(sample, field_map.instruction_keys),
        image=image,
    )
    # Keep selected_image_key discoverable through the sample mapper metadata,
    # while the common Observation remains backend-agnostic.
    del selected_image_key
    return observation


def map_lerobot_sample(
    sample: Mapping[str, Any],
    *,
    field_map: LeRobotFieldMap = LeRobotFieldMap(),
    require_image: bool = False,
    repo_id: str | None = None,
    index: int | None = None,
) -> MappedLeRobotSample:
    action_value = _lookup(sample, field_map.action_key)
    if action_value is None:
        raise KeyError(f"LeRobot sample is missing action key {field_map.action_key!r}")
    selected_image_key = next(
        (key for key in field_map.image_keys if _lookup(sample, key) is not None),
        None,
    )
    return MappedLeRobotSample(
        observation=map_lerobot_observation(
            sample,
            field_map=field_map,
            require_image=require_image,
        ),
        action=_action_array(action_value),
        metadata={
            "repo_id": repo_id,
            "index": index,
            "episode_index": _optional_episode_index(
                _lookup(sample, field_map.episode_index_key)
            ),
            "state_key": field_map.state_key,
            "image_key": selected_image_key,
            "action_key": field_map.action_key,
        },
    )


class LeRobotDatasetSource:
    """Lazy LeRobot 0.6 DatasetSource using an explicit Hub ``repo_id`` path.

    The real path is equivalent to the official construction
    ``LeRobotDataset(repo_id, episodes=...)``. Tests and downstream adapters can
    inject ``dataset_factory`` without installing torch or LeRobot.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        revision: str = LEROBOT_PUSHT_REVISION,
        episodes: Sequence[int] = LEROBOT_PUSHT_EPISODES,
        video_backend: str = LEROBOT_PUSHT_VIDEO_BACKEND,
        return_uint8: bool = True,
        max_samples: int | None = None,
        require_image: bool = True,
        field_map: LeRobotFieldMap = LeRobotFieldMap(),
        dataset_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(repo_id, str) or repo_id.count("/") != 1:
            raise ValueError("repo_id must be an explicit Hugging Face path such as 'lerobot/pusht'")
        repo_id = repo_id.strip()
        if any(not part for part in repo_id.split("/", maxsplit=1)):
            raise ValueError("repo_id must be an explicit Hugging Face path such as 'lerobot/pusht'")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be a non-empty string")
        if video_backend != LEROBOT_PUSHT_VIDEO_BACKEND:
            raise ValueError("video_backend must be pyav for reproducible CPU decoding")
        if not isinstance(return_uint8, bool):
            raise TypeError("return_uint8 must be boolean")
        if not return_uint8:
            raise ValueError("return_uint8 must be true for the LunaVLA image contract")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.repo_id = repo_id
        self.revision = revision.strip()
        self.episodes = _episode_ids(episodes)
        self.video_backend = video_backend
        self.return_uint8 = return_uint8
        self.max_samples = max_samples
        self.require_image = bool(require_image)
        self.field_map = field_map
        self._dataset_factory = dataset_factory

    @classmethod
    def from_repo_id(
        cls,
        repo_id: str = "lerobot/pusht",
        **kwargs: Any,
    ) -> "LeRobotDatasetSource":
        return cls(repo_id, **kwargs)

    def load_raw_dataset(self) -> Any:
        factory = self._dataset_factory or load_lerobot_dataset_factory()
        kwargs: dict[str, object] = {
            "episodes": list(self.episodes),
            "revision": self.revision,
            "video_backend": self.video_backend,
            "return_uint8": self.return_uint8,
        }
        return factory(self.repo_id, **kwargs)

    def _raw_samples(self) -> list[Mapping[str, Any]]:
        dataset = self.load_raw_dataset()
        length = len(dataset)
        if self.max_samples is not None:
            length = min(length, self.max_samples)
        samples = [dataset[index] for index in range(length)]
        if not all(isinstance(sample, Mapping) for sample in samples):
            raise TypeError("LeRobotDataset items must be mappings")
        return samples

    def load(self) -> Sequence[Transition]:
        samples = self._raw_samples()
        if not samples:
            return ()
        mapped = [
            map_lerobot_sample(
                sample,
                field_map=self.field_map,
                require_image=self.require_image,
                repo_id=self.repo_id,
                index=index,
            )
            for index, sample in enumerate(samples)
        ]
        requested_episodes = set(self.episodes)
        observed_episodes: set[int] = set()
        for index, item in enumerate(mapped):
            episode_index = item.metadata["episode_index"]
            if isinstance(episode_index, bool) or not isinstance(episode_index, int):
                raise TypeError(
                    "LeRobotDataset samples must expose an integer episode_index; "
                    f"sample {index} has {episode_index!r}"
                )
            if episode_index not in requested_episodes:
                raise ValueError(
                    f"LeRobotDataset returned unrequested episode {episode_index}; "
                    f"requested {sorted(requested_episodes)}"
                )
            observed_episodes.add(episode_index)
        if self.max_samples is None and observed_episodes != requested_episodes:
            missing = sorted(requested_episodes - observed_episodes)
            raise ValueError(
                "LeRobotDataset did not return every requested episode; "
                f"missing {missing}"
            )
        if self.repo_id == LEROBOT_PUSHT_REPO_ID:
            self._validate_pusht_contract(mapped)
        transitions: list[Transition] = []
        for index, (sample, current) in enumerate(zip(samples, mapped)):
            episode_index = current.metadata["episode_index"]
            next_index = index + 1
            same_episode = False
            if next_index < len(samples):
                next_episode = mapped[next_index].metadata["episode_index"]
                same_episode = episode_index is None or next_episode == episode_index
            raw_terminated = _lookup(sample, self.field_map.terminated_key)
            explicit_terminated = (
                False
                if raw_terminated is None
                else _boolean_scalar(raw_terminated, name=self.field_map.terminated_key)
            )
            terminated = explicit_terminated or not same_episode
            has_next_observation = same_episode and not explicit_terminated
            next_observation = (
                mapped[next_index].observation
                if has_next_observation
                else current.observation
            )
            reward_value = _lookup(sample, self.field_map.reward_key, default=0.0)
            reward = _float_scalar(reward_value, name=self.field_map.reward_key)
            info: dict[str, object] = {
                **dict(current.metadata),
                "source": "LeRobotDataset",
                "revision": self.revision,
                "video_backend": self.video_backend,
                "return_uint8": self.return_uint8,
                "next_observation_source": (
                    "next_frame" if has_next_observation else "terminal_self"
                ),
            }
            for success_key in self.field_map.success_keys:
                raw_success = _lookup(sample, success_key)
                if raw_success is not None:
                    info["success"] = _boolean_scalar(raw_success, name=success_key)
                    info["success_key"] = success_key
                    break
            transitions.append(
                Transition(
                    observation=current.observation,
                    action=current.action,
                    reward=reward,
                    next_observation=next_observation,
                    terminated=terminated,
                    info=info,
                )
            )
        return tuple(transitions)

    @staticmethod
    def _validate_pusht_contract(samples: Sequence[MappedLeRobotSample]) -> None:
        for index, sample in enumerate(samples):
            if sample.observation.state.shape != LEROBOT_PUSHT_STATE_SHAPE:
                raise ValueError(
                    "lerobot/pusht observation.state must have shape "
                    f"{LEROBOT_PUSHT_STATE_SHAPE}; sample {index} has "
                    f"{sample.observation.state.shape}"
                )
            if sample.action.shape != LEROBOT_PUSHT_ACTION_SHAPE:
                raise ValueError(
                    "lerobot/pusht action must have shape "
                    f"{LEROBOT_PUSHT_ACTION_SHAPE}; sample {index} has "
                    f"{sample.action.shape}"
                )
            image = sample.observation.image
            if image is None:
                raise ValueError(f"lerobot/pusht sample {index} is missing its RGB image")
            if image.shape != LEROBOT_PUSHT_IMAGE_SHAPE:
                raise ValueError(
                    "lerobot/pusht image must have shape "
                    f"{LEROBOT_PUSHT_IMAGE_SHAPE}; sample {index} has {image.shape}"
                )
            if image.dtype != np.uint8:
                raise TypeError(
                    f"lerobot/pusht image must be uint8; sample {index} has {image.dtype}"
                )


class LeRobotEnvAdapter:
    """Optional Gym-style environment wrapper implementing the LunaVLA TaskEnv protocol."""

    def __init__(
        self,
        env: Any,
        *,
        field_map: LeRobotFieldMap = LeRobotFieldMap(),
        require_image: bool = False,
    ) -> None:
        if not hasattr(env, "reset") or not hasattr(env, "step"):
            raise TypeError("env must provide reset() and step()")
        self.env = env
        self.field_map = field_map
        self.require_image = bool(require_image)
        self._observation: Observation | None = None

    def reset(self, *, seed: int | None = None) -> Observation:
        result = self.env.reset(seed=seed)
        raw_observation = result[0] if isinstance(result, tuple) and len(result) == 2 else result
        self._observation = map_lerobot_observation(
            raw_observation,
            field_map=self.field_map,
            require_image=self.require_image,
        )
        return self._observation

    def step(self, action: NDArray[np.generic]) -> Transition:
        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        action_array = _action_array(action)
        result = self.env.step(action_array)
        if not isinstance(result, tuple):
            raise TypeError("environment step must return a tuple")
        if len(result) == 5:
            raw_next, reward, terminated, truncated, raw_info = result
            done = bool(terminated or truncated)
        elif len(result) == 4:
            raw_next, reward, done, raw_info = result
            truncated = False
        else:
            raise ValueError("environment step must return 4 or 5 values")
        next_observation = map_lerobot_observation(
            raw_next,
            field_map=self.field_map,
            require_image=self.require_image,
        )
        info = dict(raw_info or {})
        info["truncated"] = bool(truncated)
        transition = Transition(
            observation=self._observation,
            action=action_array,
            reward=_float_scalar(reward, name="reward"),
            next_observation=next_observation,
            terminated=bool(done),
            info=info,
        )
        self._observation = next_observation
        return transition
