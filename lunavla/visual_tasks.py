from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageDraw

from .contracts import Observation, Transition


VisualFamily = Literal["direct_reach", "waypoint_reach"]
ImageAblation = Literal["occlusion", "shuffle"]
ObservationMode = Literal["privileged", "vision_required"]
_OBSERVATION_MODES = {"privileged", "vision_required"}
_FAMILY_SEED_CODES: dict[VisualFamily, int] = {
    "direct_reach": 1,
    "waypoint_reach": 2,
}
_VISION_REQUIRED_START_BUCKETS = 16


def _vector2(value: ArrayLike, *, name: str) -> NDArray[np.float32]:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain two finite values")
    return result.copy()


@dataclass(frozen=True)
class VisualTaskSpec:
    family: VisualFamily
    goal: NDArray[np.float32]
    waypoint: NDArray[np.float32]

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal", _vector2(self.goal, name="goal"))
        object.__setattr__(self, "waypoint", _vector2(self.waypoint, name="waypoint"))


VISUAL_TASKS: tuple[VisualTaskSpec, ...] = (
    VisualTaskSpec(
        family="direct_reach",
        goal=np.asarray([0.82, 0.22], dtype=np.float32),
        waypoint=np.asarray([0.82, 0.22], dtype=np.float32),
    ),
    VisualTaskSpec(
        family="waypoint_reach",
        goal=np.asarray([0.82, 0.22], dtype=np.float32),
        waypoint=np.asarray([0.48, 0.78], dtype=np.float32),
    ),
)


def visual_task_specs() -> tuple[VisualTaskSpec, ...]:
    return VISUAL_TASKS


def _observation_mode(value: str) -> ObservationMode:
    if value not in _OBSERVATION_MODES:
        raise ValueError(
            "observation_mode must be 'privileged' or 'vision_required'"
        )
    return value  # type: ignore[return-value]


def _seeded_visual_task_spec(family: VisualFamily, seed: int) -> VisualTaskSpec:
    """Generate target geometry without placing it in the policy state."""

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), _FAMILY_SEED_CODES[family], 0x4C564C41])
    )
    goal = rng.uniform((0.66, 0.14), (0.90, 0.88), size=2).astype(np.float32)
    if family == "direct_reach":
        waypoint = goal.copy()
    else:
        waypoint = rng.uniform((0.36, 0.36), (0.58, 0.86), size=2).astype(
            np.float32
        )
        if float(np.linalg.norm(waypoint - goal)) < 0.24:
            waypoint[1] = np.float32(0.82 if goal[1] < 0.55 else 0.30)
    return VisualTaskSpec(family=family, goal=goal, waypoint=waypoint)


def _pixel(point: NDArray[np.float32], size: int) -> tuple[int, int]:
    x = int(round(float(point[0]) * (size - 1)))
    y = int(round((1.0 - float(point[1])) * (size - 1)))
    return x, y


def _ellipse(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, fill: str, outline: str) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)


def render_point_reach(
    position: ArrayLike,
    *,
    goal: ArrayLike,
    waypoint: ArrayLike | None = None,
    family: VisualFamily = "direct_reach",
    image_size: int = 64,
) -> NDArray[np.uint8]:
    """Render a point-reach observation with PIL and return HWC uint8 RGB."""

    if family not in {"direct_reach", "waypoint_reach"}:
        raise ValueError(f"unsupported visual task family: {family!r}")
    if image_size < 24:
        raise ValueError("image_size must be at least 24 pixels")
    position_array = _vector2(position, name="position")
    goal_array = _vector2(goal, name="goal")
    waypoint_array = goal_array if waypoint is None else _vector2(waypoint, name="waypoint")
    if any(np.any((value < 0.0) | (value > 1.0)) for value in (position_array, goal_array, waypoint_array)):
        raise ValueError("render coordinates must lie in [0, 1]")

    image = Image.new("RGB", (image_size, image_size), color="#f7f7f2")
    draw = ImageDraw.Draw(image)
    spacing = max(8, image_size // 8)
    for value in range(spacing, image_size, spacing):
        draw.line((value, 0, value, image_size), fill="#e7e7e1", width=1)
        draw.line((0, value, image_size, value), fill="#e7e7e1", width=1)

    _ellipse(draw, _pixel(goal_array, image_size), max(4, image_size // 12), "#6ac46a", "#286b28")
    if family == "waypoint_reach":
        waypoint_pixel = _pixel(waypoint_array, image_size)
        radius = max(4, image_size // 14)
        x, y = waypoint_pixel
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill="#f4b860", outline="#915c13", width=2)
        draw.line((*_pixel(waypoint_array, image_size), *_pixel(goal_array, image_size)), fill="#c58a35", width=2)
    _ellipse(draw, _pixel(position_array, image_size), max(4, image_size // 13), "#4c8de5", "#174d91")
    return np.asarray(image, dtype=np.uint8).copy()


@dataclass(frozen=True)
class VisualTaskExample:
    example_id: str
    family: VisualFamily
    seed: int
    observation: Observation
    action: NDArray[np.float32]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _vector2(self.action, name="action"))


@dataclass(frozen=True)
class ImageAblationPair:
    pair_id: str
    mode: ImageAblation
    control: VisualTaskExample
    ablated: VisualTaskExample
    metadata: Mapping[str, object]


def state_only_observation(observation: Observation) -> Observation:
    return Observation(
        state=np.asarray(observation.state).copy(),
        instruction=observation.instruction,
        image=None,
    )


def make_state_only_baseline(examples: Sequence[VisualTaskExample]) -> tuple[VisualTaskExample, ...]:
    """Remove pixels while preserving state/action pairing for a state-only control."""

    return tuple(
        replace(
            example,
            example_id=f"{example.example_id}:state-only",
            observation=state_only_observation(example.observation),
            metadata={
                **dict(example.metadata),
                "modality": "state_only",
                "paired_with": example.example_id,
                "interpretation": "baseline input only; no modality-effect claim",
            },
        )
        for example in examples
    )


class RenderedPointReachEnv:
    """Rendered control task with paired privileged and vision-required modes."""

    def __init__(
        self,
        family: VisualFamily = "direct_reach",
        *,
        state_only: bool = False,
        observation_mode: ObservationMode = "vision_required",
        image_size: int = 64,
        action_clip: float = 0.12,
        success_distance: float = 0.06,
        waypoint_distance: float = 0.07,
    ) -> None:
        specs = {spec.family: spec for spec in VISUAL_TASKS}
        if family not in specs:
            raise ValueError(f"unknown visual family: {family!r}")
        if action_clip <= 0 or success_distance <= 0 or waypoint_distance <= 0:
            raise ValueError("distance and action limits must be positive")
        if image_size < 24:
            raise ValueError("image_size must be at least 24")
        self.family = family
        self._privileged_spec = specs[family]
        self.observation_mode = _observation_mode(observation_mode)
        self.spec = (
            self._privileged_spec
            if self.observation_mode == "privileged"
            else _seeded_visual_task_spec(family, 0)
        )
        self.state_only = bool(state_only)
        self.image_size = int(image_size)
        self.action_clip = float(action_clip)
        self.success_distance = float(success_distance)
        self.waypoint_distance = float(waypoint_distance)
        self._position = np.asarray([0.18, 0.18], dtype=np.float32)
        self._phase = 0
        self._observation: Observation | None = None

    def _state(self) -> NDArray[np.float32]:
        if self.observation_mode == "vision_required":
            return np.asarray(
                [self._position[0], self._position[1], float(self._phase)],
                dtype=np.float32,
            )
        return np.asarray(
            [
                self._position[0],
                self._position[1],
                self.spec.goal[0],
                self.spec.goal[1],
                self.spec.waypoint[0],
                self.spec.waypoint[1],
                float(self._phase),
            ],
            dtype=np.float32,
        )

    def _make_observation(self) -> Observation:
        image = None
        if not self.state_only:
            image = render_point_reach(
                self._position,
                goal=self.spec.goal,
                waypoint=self.spec.waypoint,
                family=self.spec.family,
                image_size=self.image_size,
            )
        return Observation(
            state=self._state(),
            instruction=f"complete the {self.spec.family} task",
            image=image,
        )

    def active_target(self) -> NDArray[np.float32]:
        if self.spec.family == "waypoint_reach" and self._phase == 0:
            return self.spec.waypoint
        return self.spec.goal

    def expert_action(self) -> NDArray[np.float32]:
        return np.clip(
            self.active_target() - self._position,
            -self.action_clip,
            self.action_clip,
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None) -> Observation:
        resolved_seed = 0 if seed is None else int(seed)
        self.spec = (
            self._privileged_spec
            if self.observation_mode == "privileged"
            else _seeded_visual_task_spec(self.family, resolved_seed)
        )
        start_seed = (
            resolved_seed
            if self.observation_mode == "privileged"
            else resolved_seed % _VISION_REQUIRED_START_BUCKETS
        )
        rng = np.random.default_rng(start_seed)
        self._position = rng.uniform(0.10, 0.30, size=2).astype(np.float32)
        self._phase = 0
        self._observation = self._make_observation()
        return self._observation

    def step(self, action: NDArray[np.generic]) -> Transition:
        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        action_array = _vector2(action, name="action")
        applied = np.clip(action_array, -self.action_clip, self.action_clip).astype(np.float32)
        previous = self._observation
        self._position = np.clip(self._position + applied, 0.0, 1.0).astype(np.float32)

        reached_waypoint = False
        if self.spec.family == "waypoint_reach" and self._phase == 0:
            reached_waypoint = float(np.linalg.norm(self._position - self.spec.waypoint)) <= self.waypoint_distance
            if reached_waypoint:
                self._phase = 1
        goal_distance = float(np.linalg.norm(self._position - self.spec.goal))
        success = goal_distance <= self.success_distance and (
            self.spec.family == "direct_reach" or self._phase == 1
        )
        next_observation = self._make_observation()
        self._observation = next_observation
        return Transition(
            observation=previous,
            action=applied,
            reward=-goal_distance,
            next_observation=next_observation,
            terminated=success,
            info={
                "task_id": "rendered_visual_point_reach",
                "task_family": self.spec.family,
                "phase": self._phase,
                "reached_waypoint": reached_waypoint,
                "distance": goal_distance,
                "success": success,
                "modality": "state_only" if self.state_only else "state_and_image",
                "observation_mode": self.observation_mode,
            },
        )

    def close(self) -> None:
        """Release task resources (the NumPy renderer has none)."""

        return None


class RenderedVisualTaskSuiteEnv:
    """Select direct or waypoint reach from each configured evaluation seed."""

    def __init__(
        self,
        *,
        state_only: bool = False,
        observation_mode: ObservationMode = "vision_required",
        image_size: int = 64,
        families: Sequence[VisualFamily] = ("direct_reach", "waypoint_reach"),
    ) -> None:
        known = {spec.family for spec in VISUAL_TASKS}
        if not families or any(family not in known for family in families):
            raise ValueError(f"families must be a non-empty subset of {sorted(known)}")
        self.state_only = bool(state_only)
        self.observation_mode = _observation_mode(observation_mode)
        self.image_size = int(image_size)
        self.families = tuple(families)
        self._env: RenderedPointReachEnv | None = None

    def reset(self, *, seed: int | None = None) -> Observation:
        resolved_seed = 0 if seed is None else int(seed)
        family = self.families[resolved_seed % len(self.families)]
        self._env = RenderedPointReachEnv(
            family,
            state_only=self.state_only,
            observation_mode=self.observation_mode,
            image_size=self.image_size,
        )
        return self._env.reset(seed=resolved_seed)

    def step(self, action: NDArray[np.generic]) -> Transition:
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        return self._env.step(action)

    def close(self) -> None:
        """Release task resources (the rendered suite has none)."""

        return None


def build_visual_examples(
    *,
    seeds: Sequence[int] = (0, 1, 2),
    image_size: int = 64,
    observation_mode: ObservationMode = "vision_required",
) -> tuple[VisualTaskExample, ...]:
    resolved_mode = _observation_mode(observation_mode)
    examples: list[VisualTaskExample] = []
    for spec in VISUAL_TASKS:
        for seed in seeds:
            env = RenderedPointReachEnv(
                spec.family,
                image_size=image_size,
                observation_mode=resolved_mode,
            )
            observation = env.reset(seed=int(seed))
            examples.append(
                VisualTaskExample(
                    example_id=f"visual:{spec.family}:{int(seed)}",
                    family=spec.family,
                    seed=int(seed),
                    observation=observation,
                    action=env.expert_action(),
                    metadata={
                        "task_family": spec.family,
                        "seed": int(seed),
                        "modality": "state_and_image",
                        "observation_mode": resolved_mode,
                    },
                )
            )
    return tuple(examples)


def occlude_image(
    image: NDArray[np.generic],
    *,
    fraction: float = 0.40,
) -> tuple[NDArray[np.generic], tuple[int, int, int, int]]:
    array = np.asarray(image)
    if array.ndim not in {2, 3}:
        raise ValueError("image must be grayscale or HWC")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("occlusion fraction must lie in (0, 1]")
    height, width = array.shape[:2]
    occlusion_height = max(1, int(round(height * fraction)))
    occlusion_width = max(1, int(round(width * fraction)))
    top = (height - occlusion_height) // 2
    left = (width - occlusion_width) // 2
    bottom = top + occlusion_height
    right = left + occlusion_width
    result = array.copy()
    result[top:bottom, left:right, ...] = 0
    return result, (left, top, right, bottom)


def _clone_with_image(
    example: VisualTaskExample,
    image: NDArray[np.generic] | None,
    *,
    suffix: str,
) -> VisualTaskExample:
    observation = Observation(
        state=np.asarray(example.observation.state).copy(),
        instruction=example.observation.instruction,
        image=None if image is None else np.asarray(image).copy(),
    )
    return replace(example, example_id=f"{example.example_id}:{suffix}", observation=observation)


def _derangement(length: int, seed: int) -> list[int]:
    if length < 2:
        raise ValueError("image shuffle requires at least two examples")
    rng = np.random.default_rng(seed)
    indices = np.arange(length)
    for _ in range(256):
        candidate = rng.permutation(indices)
        if np.all(candidate != indices):
            return [int(value) for value in candidate]
    return [int((index + 1) % length) for index in indices]


def make_image_ablation_pairs(
    examples: Sequence[VisualTaskExample],
    mode: ImageAblation,
    *,
    seed: int = 0,
    occlusion_fraction: float = 0.40,
) -> tuple[ImageAblationPair, ...]:
    """Pair image interventions with unchanged state and action targets."""

    source = tuple(examples)
    if mode not in {"occlusion", "shuffle"}:
        raise ValueError(f"unsupported image ablation: {mode!r}")
    if any(example.observation.image is None for example in source):
        raise ValueError("image ablations require image-bearing examples")
    donors = _derangement(len(source), seed) if mode == "shuffle" and source else []
    pairs: list[ImageAblationPair] = []
    for index, control in enumerate(source):
        box: tuple[int, int, int, int] | None = None
        donor_example_id: str | None = None
        if mode == "occlusion":
            assert control.observation.image is not None
            image, box = occlude_image(control.observation.image, fraction=occlusion_fraction)
        else:
            donor = source[donors[index]]
            donor_example_id = donor.example_id
            assert donor.observation.image is not None
            image = np.asarray(donor.observation.image).copy()
        ablated = _clone_with_image(control, image, suffix=mode)
        pair_id = f"image:{mode}:{seed}:{control.example_id}"
        metadata: dict[str, object] = {
            "pair_id": pair_id,
            "paired": True,
            "mode": mode,
            "seed": int(seed),
            "source_example_id": control.example_id,
            "donor_example_id": donor_example_id,
            "occlusion_box_xyxy": box,
            "state_held_constant": True,
            "action_target_held_constant": True,
            "interpretation": "measurement input only; no visual-contribution claim",
        }
        pairs.append(
            ImageAblationPair(
                pair_id=pair_id,
                mode=mode,
                control=control,
                ablated=ablated,
                metadata=metadata,
            )
        )
    return tuple(pairs)


class RenderedVisualDatasetSource:
    """Generate expert transitions for both rendered task families or their state baseline."""

    def __init__(
        self,
        *,
        families: Sequence[VisualFamily] = ("direct_reach", "waypoint_reach"),
        seeds: Sequence[int] = (0, 1, 2),
        state_only: bool = False,
        observation_mode: ObservationMode = "vision_required",
        image_size: int = 64,
        max_steps: int = 16,
    ) -> None:
        known = {spec.family for spec in VISUAL_TASKS}
        if not families or any(family not in known for family in families):
            raise ValueError(f"families must be a non-empty subset of {sorted(known)}")
        if not seeds:
            raise ValueError("seeds cannot be empty")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.families = tuple(families)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.state_only = bool(state_only)
        self.observation_mode = _observation_mode(observation_mode)
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)

    def load(self) -> Sequence[Transition]:
        transitions: list[Transition] = []
        episode_id = 0
        for family in self.families:
            for seed in self.seeds:
                env = RenderedPointReachEnv(
                    family,
                    state_only=self.state_only,
                    observation_mode=self.observation_mode,
                    image_size=self.image_size,
                )
                env.reset(seed=seed)
                for step in range(self.max_steps):
                    transition = env.step(env.expert_action())
                    time_limit = step == self.max_steps - 1 and not transition.terminated
                    if time_limit:
                        transition = Transition(
                            observation=transition.observation,
                            action=transition.action,
                            reward=transition.reward,
                            next_observation=transition.next_observation,
                            terminated=True,
                            info={**transition.info, "success": False, "time_limit": True},
                        )
                    transition = Transition(
                        observation=transition.observation,
                        action=transition.action,
                        reward=transition.reward,
                        next_observation=transition.next_observation,
                        terminated=transition.terminated,
                        info={
                            **transition.info,
                            "episode_id": episode_id,
                            "seed": seed,
                            "step": step,
                        },
                    )
                    transitions.append(transition)
                    if transition.terminated:
                        break
                episode_id += 1
        return tuple(transitions)
