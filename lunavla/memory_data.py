from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .contracts import Array, Observation, Transition


Float32Array = npt.NDArray[np.float32]


@dataclass(frozen=True)
class InMemoryDatasetSource:
    """A deterministic DatasetSource for small tests, examples, and adapters."""

    transitions: tuple[Transition, ...]

    def __init__(self, transitions: Sequence[Transition]) -> None:
        values = tuple(transitions)
        if not values:
            raise ValueError("an in-memory dataset must contain at least one transition")
        if any(not isinstance(item, Transition) for item in values):
            raise TypeError("all dataset items must be Transition instances")
        object.__setattr__(self, "transitions", values)

    def load(self) -> tuple[Transition, ...]:
        return self.transitions

    @classmethod
    def from_arrays(
        cls,
        states: Array,
        actions: Array,
        next_states: Array,
        rewards: Array,
        terminated: Array,
        *,
        instructions: Sequence[str | None] | None = None,
        infos: Sequence[Mapping[str, Any]] | None = None,
    ) -> "InMemoryDatasetSource":
        state_array = np.asarray(states)
        action_array = np.asarray(actions)
        next_state_array = np.asarray(next_states)
        reward_array = np.asarray(rewards)
        terminated_array = np.asarray(terminated)
        if state_array.ndim != 2 or action_array.ndim != 2:
            raise ValueError("states and actions must be rank-2 arrays")
        if next_state_array.shape != state_array.shape:
            raise ValueError("next_states must have the same shape as states")
        count = len(state_array)
        if action_array.shape[0] != count:
            raise ValueError("actions and states must have the same row count")
        if reward_array.shape != (count,) or terminated_array.shape != (count,):
            raise ValueError("rewards and terminated must be rank-1 with one value per row")
        instruction_values = tuple(instructions or [None] * count)
        info_values = tuple(infos or [{} for _ in range(count)])
        if len(instruction_values) != count or len(info_values) != count:
            raise ValueError("instructions and infos must have one item per transition")
        return cls(
            [
                Transition(
                    observation=Observation(state_array[index], instruction_values[index]),
                    action=action_array[index],
                    reward=float(reward_array[index]),
                    next_observation=Observation(
                        next_state_array[index], instruction_values[index]
                    ),
                    terminated=bool(terminated_array[index]),
                    info=info_values[index],
                )
                for index in range(count)
            ]
        )


class PointReachTaskEnv:
    """Tiny deterministic state task used to exercise the unified engine on CPU."""

    def __init__(
        self,
        *,
        goal: Sequence[float] = (0.8, 0.2),
        start_low: float = 0.05,
        start_high: float = 0.95,
        action_clip: float = 0.12,
        success_distance: float = 0.10,
        instruction: str | None = "move the point to the goal",
    ) -> None:
        goal_array = np.asarray(goal, dtype=np.float32)
        if goal_array.shape != (2,) or not np.all(np.isfinite(goal_array)):
            raise ValueError("goal must contain two finite coordinates")
        if not 0 <= start_low < start_high <= 1:
            raise ValueError("start range must satisfy 0 <= start_low < start_high <= 1")
        if action_clip <= 0 or success_distance <= 0:
            raise ValueError("action_clip and success_distance must be positive")
        self.goal = goal_array
        self.start_low = float(start_low)
        self.start_high = float(start_high)
        self.action_clip = float(action_clip)
        self.success_distance = float(success_distance)
        self.instruction = instruction
        self._observation: Observation | None = None
        self._terminated = False

    def _make_observation(self, position: Float32Array) -> Observation:
        return Observation(
            np.concatenate([position, self.goal]).astype(np.float32),
            instruction=self.instruction,
        )

    def reset(self, *, seed: int | None = None) -> Observation:
        rng = np.random.default_rng(seed)
        position = rng.uniform(self.start_low, self.start_high, size=2).astype(np.float32)
        self._observation = self._make_observation(position)
        self._terminated = False
        return self._observation

    def step(self, action: Array) -> Transition:
        if self._observation is None:
            raise RuntimeError("reset must be called before step")
        if self._terminated:
            raise RuntimeError("cannot step a terminated episode; call reset")
        raw_action = np.asarray(action)
        if raw_action.dtype.kind not in "fiu":
            raise TypeError("action must have a numeric dtype")
        if raw_action.shape != (2,):
            raise ValueError(f"action must have shape (2,); got {raw_action.shape}")
        executable = raw_action.astype(np.float32, copy=False)
        if not np.all(np.isfinite(executable)):
            raise ValueError("action contains NaN or infinite values")
        executable = np.clip(executable, -self.action_clip, self.action_clip).astype(
            np.float32
        )

        previous = self._observation
        previous_state = np.asarray(previous.state, dtype=np.float32)
        position = np.clip(previous_state[:2] + executable, 0.0, 1.0).astype(np.float32)
        next_observation = self._make_observation(position)
        distance = float(np.linalg.norm(self.goal - position))
        success = distance <= self.success_distance
        self._observation = next_observation
        self._terminated = success
        return Transition(
            observation=previous,
            action=executable,
            reward=-distance,
            next_observation=next_observation,
            terminated=success,
            info={"success": success, "distance_to_goal": distance},
        )


def make_point_reach_demonstrations(
    *,
    episodes: int = 16,
    steps_per_episode: int = 24,
    seed: int = 0,
    action_gain: float = 0.35,
    env: PointReachTaskEnv | None = None,
) -> InMemoryDatasetSource:
    """Generate small ordered expert trajectories without depending on v1 records."""

    if episodes <= 0 or steps_per_episode <= 0 or action_gain <= 0:
        raise ValueError("episodes, steps_per_episode, and action_gain must be positive")
    task = env or PointReachTaskEnv()
    transitions: list[Transition] = []
    for episode in range(episodes):
        observation = task.reset(seed=seed + episode)
        for step in range(steps_per_episode):
            state = np.asarray(observation.state, dtype=np.float32)
            delta = task.goal - state[:2]
            action = np.clip(
                delta * action_gain, -task.action_clip, task.action_clip
            ).astype(np.float32)
            transition = task.step(action)
            transition = Transition(
                observation=transition.observation,
                action=transition.action,
                reward=transition.reward,
                next_observation=transition.next_observation,
                terminated=transition.terminated,
                info={
                    **transition.info,
                    "episode_id": episode,
                    "timestep": step,
                },
            )
            if step == steps_per_episode - 1 and not transition.terminated:
                transition = Transition(
                    observation=transition.observation,
                    action=transition.action,
                    reward=transition.reward,
                    next_observation=transition.next_observation,
                    terminated=True,
                    info={**transition.info, "success": False, "time_limit": True},
                )
            transitions.append(transition)
            observation = transition.next_observation
            if transition.terminated:
                break
    return InMemoryDatasetSource(transitions)
