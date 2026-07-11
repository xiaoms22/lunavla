"""Lazy, headless Gymnasium adapter for the real PushT environment."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from .contracts import Observation, Transition


PUSHT_ENV_ID = "gym_pusht/PushT-v0"
PUSHT_OBS_TYPE = "pixels_agent_pos"
PUSHT_STATE_SHAPE = (2,)
PUSHT_IMAGE_SHAPE = (96, 96, 3)
PUSHT_ACTION_SHAPE = (2,)


class PushTUnavailableError(ImportError):
    """Raised only when the optional real PushT environment is requested."""


def load_pusht_env_factory() -> Callable[..., Any]:
    """Resolve the registered PushT factory without eager optional imports."""

    try:
        gymnasium = importlib.import_module("gymnasium")
        # ``gym_pusht`` registers ``gym_pusht/PushT-v0`` as an import side
        # effect.  Import it lazily here so ordinary LunaVLA imports remain
        # lightweight while the first real reset cannot fail with an unknown
        # environment ID merely because callers did not import the plugin.
        importlib.import_module("gym_pusht")
    except ImportError as exc:
        raise PushTUnavailableError(
            "The real PushT smoke requires gymnasium and gym-pusht; install the "
            "LeRobot integration profile before resetting this environment."
        ) from exc
    factory = getattr(gymnasium, "make", None)
    if not callable(factory):
        raise PushTUnavailableError("gymnasium.make is unavailable")
    return cast(Callable[..., Any], factory)


def _state(value: Any) -> NDArray[np.float32]:
    array = np.asarray(value)
    if array.shape != PUSHT_STATE_SHAPE:
        raise ValueError(
            f"agent_pos must have shape {PUSHT_STATE_SHAPE}; got {array.shape}"
        )
    if array.dtype.kind not in "fiu":
        raise TypeError("agent_pos must have a numeric dtype")
    result = np.asarray(array, dtype=np.float32).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("agent_pos contains NaN or Inf")
    return result


def _pixels(value: Any) -> NDArray[np.uint8]:
    array = np.asarray(value)
    if array.shape != PUSHT_IMAGE_SHAPE:
        raise ValueError(
            f"pixels must have HWC shape {PUSHT_IMAGE_SHAPE}; got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise TypeError(f"pixels must have dtype uint8; got {array.dtype}")
    return np.asarray(array, dtype=np.uint8).copy()


def _action(value: Any) -> NDArray[np.float32]:
    array = np.asarray(value)
    if array.shape != PUSHT_ACTION_SHAPE:
        raise ValueError(
            f"action must have shape {PUSHT_ACTION_SHAPE}; got {array.shape}"
        )
    if array.dtype.kind not in "fiu":
        raise TypeError("action must have a numeric dtype")
    result = np.asarray(array, dtype=np.float32).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("action contains NaN or Inf")
    return result


def _observation(value: Any) -> Observation:
    if not isinstance(value, Mapping):
        raise TypeError("PushT observation must be a mapping")
    missing = [key for key in ("agent_pos", "pixels") if key not in value]
    if missing:
        raise KeyError(f"PushT observation is missing field(s): {', '.join(missing)}")
    return Observation(state=_state(value["agent_pos"]), image=_pixels(value["pixels"]))


def _finite_reward(value: Any) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("PushT reward must be a scalar")
    try:
        reward = float(array.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise TypeError("PushT reward must be numeric") from exc
    if not np.isfinite(reward):
        raise ValueError("PushT reward must be finite")
    return reward


class PushTEnvAdapter:
    """TaskEnv wrapper around ``gym_pusht/PushT-v0``.

    Construction is side-effect free. Gymnasium is imported and the headless
    pixel environment is created only on the first ``reset`` call.
    """

    def __init__(self, *, env_factory: Callable[..., Any] | None = None) -> None:
        self._env_factory = env_factory
        self._env: Any | None = None
        self._observation: Observation | None = None

    def _environment(self) -> Any:
        if self._env is None:
            factory = self._env_factory or load_pusht_env_factory()
            self._env = factory(PUSHT_ENV_ID, obs_type=PUSHT_OBS_TYPE)
            if not hasattr(self._env, "reset") or not hasattr(self._env, "step"):
                raise TypeError("PushT environment must provide reset() and step()")
        return self._env

    def reset(self, *, seed: int | None = None) -> Observation:
        result = self._environment().reset(seed=seed)
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError("PushT reset must return observation or (observation, info)")
            raw_observation = result[0]
        else:
            raw_observation = result
        self._observation = _observation(raw_observation)
        return self._observation

    def step(self, action: NDArray[np.generic]) -> Transition:
        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        action_array = _action(action)
        result = self._environment().step(action_array)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError(
                "PushT step must return (observation, reward, terminated, truncated, info)"
            )
        raw_next, reward, terminated, truncated, raw_info = result
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
            truncated, (bool, np.bool_)
        ):
            raise TypeError("PushT terminated and truncated flags must be boolean")
        if not isinstance(raw_info, Mapping):
            raise TypeError("PushT step info must be a mapping")
        next_observation = _observation(raw_next)
        info = dict(raw_info)
        if "success" not in info and "is_success" in info:
            raw_success = info["is_success"]
            if not isinstance(raw_success, (bool, np.bool_)):
                raise TypeError("PushT is_success must be boolean")
            info["success"] = bool(raw_success)
        info["truncated"] = bool(truncated)
        transition = Transition(
            observation=self._observation,
            action=action_array,
            reward=_finite_reward(reward),
            next_observation=next_observation,
            terminated=bool(terminated or truncated),
            info=info,
        )
        self._observation = next_observation
        return transition

    def close(self) -> None:
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
            self._env = None
            self._observation = None
