from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .contracts import EpisodeRecordV3, FeatureSchema, ObservationV3, TransitionV3


Float32Array = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]


def _image(state: Float32Array, goal: Float32Array, size: int = 16) -> UInt8Array:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    point = np.clip(np.rint(state[:2] * (size - 1)).astype(int), 0, size - 1)
    target = np.clip(np.rint(goal * (size - 1)).astype(int), 0, size - 1)
    image[point[1], point[0], 0] = 255
    image[target[1], target[0], 1] = 255
    return image


def _observation(
    *, task_id: str, episode_id: str, step: int, state: Float32Array, goal: Float32Array,
    instruction_variant: str = "constant_v1",
) -> ObservationV3:
    images = {"camera.primary": _image(state, goal)} if task_id == "fake_libero" else {}
    instruction = None
    if task_id == "fake_libero":
        if instruction_variant == "constant_v1":
            instruction = "move the red point to the green target"
        elif instruction_variant == "region_instruction_v1":
            vertical = "lower" if float(goal[1]) < 0.75 else "upper"
            horizontal = "left" if float(goal[0]) < 0.75 else "right"
            instruction = (
                "move the red point to the green target in the "
                f"{vertical}-{horizontal} region"
            )
        else:
            raise ValueError("unsupported fake_libero instruction variant")
    return ObservationV3(
        images=images,
        state={"state.proprioception": np.concatenate([state[:2], goal]).astype(np.float32)},
        instruction=instruction,
        timestamp_s=step / 10,
        episode_id=episode_id,
        step_index=step,
        metadata={"task_id": task_id},
    )


def make_fake_episodes(
    *, task_id: str, seed: int, episode_count: int, steps: int,
    instruction_variant: str = "constant_v1",
) -> tuple[EpisodeRecordV3, ...]:
    if task_id not in {"fake_pusht", "fake_libero"}:
        raise ValueError("task_id must be fake_pusht or fake_libero")
    if episode_count < 3 or steps <= 0:
        raise ValueError("episode_count must be at least 3 and steps positive")
    rng = np.random.default_rng(seed)
    episodes: list[EpisodeRecordV3] = []
    for episode_index in range(episode_count):
        episode_id = f"{task_id}-{episode_index:04d}"
        state = rng.uniform(0.1, 0.4, size=2).astype(np.float32)
        goal = rng.uniform(0.6, 0.9, size=2).astype(np.float32)
        transitions: list[TransitionV3] = []
        for step in range(steps):
            observation = _observation(
                task_id=task_id, episode_id=episode_id, step=step, state=state, goal=goal
                , instruction_variant=instruction_variant
            )
            action = np.clip(goal - state, -0.1, 0.1).astype(np.float32)
            next_state = np.clip(state + action, 0, 1).astype(np.float32)
            terminated = step == steps - 1
            next_observation = _observation(
                task_id=task_id, episode_id=episode_id, step=step + 1,
                state=next_state, goal=goal, instruction_variant=instruction_variant,
            )
            distance = float(np.linalg.norm(next_state - goal))
            transitions.append(
                TransitionV3(
                    observation, action, -distance, next_observation, terminated, False,
                    {"success": distance < 0.12, "distance": distance},
                )
            )
            state = next_state
        episodes.append(EpisodeRecordV3(episode_id, tuple(transitions), {"task_id": task_id}))
    return tuple(episodes)


@dataclass
class FakePointEnvV3:
    task_id: str
    max_steps: int
    instruction_variant: str = "constant_v1"
    closed: bool = False
    _state: Float32Array = field(init=False, repr=False)
    _goal: Float32Array = field(init=False, repr=False)
    _step: int = field(init=False, repr=False)
    _episode_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.task_id not in {"fake_pusht", "fake_libero"}:
            raise ValueError("unsupported fake task")
        self._state = np.zeros(2, dtype=np.float32)
        self._goal = np.ones(2, dtype=np.float32)
        self._step = 0
        self._episode_id = "uninitialized"

    def reset(self, *, seed: int | None = None) -> ObservationV3:
        if self.closed:
            raise RuntimeError("environment is closed")
        actual_seed = 0 if seed is None else seed
        rng = np.random.default_rng(actual_seed)
        self._state = rng.uniform(0.1, 0.4, size=2).astype(np.float32)
        self._goal = rng.uniform(0.6, 0.9, size=2).astype(np.float32)
        self._step = 0
        self._episode_id = f"eval-{actual_seed}"
        return _observation(
            task_id=self.task_id, episode_id=self._episode_id, step=0,
            state=self._state, goal=self._goal, instruction_variant=self.instruction_variant,
        )

    def step(self, action: npt.NDArray[np.generic]) -> TransitionV3:
        if self.closed:
            raise RuntimeError("environment is closed")
        observation = _observation(
            task_id=self.task_id, episode_id=self._episode_id, step=self._step,
            state=self._state, goal=self._goal, instruction_variant=self.instruction_variant,
        )
        action_value = np.asarray(action, dtype=np.float32)
        if action_value.shape != (2,) or not np.all(np.isfinite(action_value)):
            raise ValueError("action must be finite shape (2,)")
        self._state = np.clip(self._state + np.clip(action_value, -0.1, 0.1), 0, 1)
        self._step += 1
        distance = float(np.linalg.norm(self._state - self._goal))
        terminated = distance < 0.12
        truncated = self._step >= self.max_steps and not terminated
        next_observation = _observation(
            task_id=self.task_id, episode_id=self._episode_id, step=self._step,
            state=self._state, goal=self._goal, instruction_variant=self.instruction_variant,
        )
        return TransitionV3(
            observation, action_value, -distance, next_observation, terminated, truncated,
            {"success": terminated, "distance": distance},
        )

    def close(self) -> None:
        self.closed = True


def fake_feature_schema(task_id: str) -> FeatureSchema:
    items = []
    if task_id == "fake_libero":
        items.append(
            {
                "name": "camera.primary", "role": "image", "dtype": "uint8", "shape": [16, 16, 3],
                "unit": "pixel", "frame": "sim_camera", "rate_hz": 10.0,
                "normalization": "none", "source_key": "image", "required_by": [],
            }
        )
    items.extend(
        [
            {
                "name": "state.proprioception", "role": "state", "dtype": "float32", "shape": [4],
                "unit": "unitless", "frame": "sim_world", "rate_hz": 10.0,
                "normalization": "none", "source_key": "state", "required_by": ["numpy_linear_chunk"],
            },
            {
                "name": "action.primary", "role": "action", "dtype": "float32", "shape": [2],
                "unit": "unitless", "frame": "sim_world", "rate_hz": 10.0,
                "normalization": "none", "source_key": "action", "required_by": ["numpy_linear_chunk"],
            },
        ]
    )
    return FeatureSchema.from_mapping({"schema_version": 1, "items": items})
