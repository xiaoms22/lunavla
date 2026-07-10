from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contracts import Observation, Transition


LanguageSplit = Literal["train", "heldout"]
InstructionAblation = Literal["mask", "shuffle", "counterfactual"]
MASK_INSTRUCTION = "[MASK]"


def _vector2(value: ArrayLike, *, name: str) -> NDArray[np.float32]:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain two finite values")
    return result.copy()


@dataclass(frozen=True)
class LanguageGoalSpec:
    """One target whose location is not present in the policy state."""

    task_id: str
    target: NDArray[np.float32]
    training_templates: tuple[str, ...]
    heldout_paraphrases: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _vector2(self.target, name="target"))
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        if not self.training_templates or not self.heldout_paraphrases:
            raise ValueError("every language goal needs train templates and held-out paraphrases")
        train = {text.strip().casefold() for text in self.training_templates}
        heldout = {text.strip().casefold() for text in self.heldout_paraphrases}
        if "" in train or "" in heldout:
            raise ValueError("instruction text cannot be empty")
        if train & heldout:
            raise ValueError("held-out paraphrases must not overlap training templates")


LANGUAGE_GOALS: tuple[LanguageGoalSpec, ...] = (
    LanguageGoalSpec(
        task_id="reach_red_left",
        target=np.asarray([0.15, 0.50], dtype=np.float32),
        training_templates=("move to the red goal", "reach the red marker"),
        heldout_paraphrases=("head toward the crimson target", "finish at the red dot"),
    ),
    LanguageGoalSpec(
        task_id="reach_blue_right",
        target=np.asarray([0.85, 0.50], dtype=np.float32),
        training_templates=("move to the blue goal", "reach the blue marker"),
        heldout_paraphrases=("head toward the azure target", "finish at the blue dot"),
    ),
    LanguageGoalSpec(
        task_id="reach_green_top",
        target=np.asarray([0.50, 0.85], dtype=np.float32),
        training_templates=("move to the green goal", "reach the green marker"),
        heldout_paraphrases=("head toward the emerald target", "finish at the green dot"),
    ),
)


@dataclass(frozen=True)
class LanguageTrainingTemplate:
    """Machine-readable description of the language-conditioned supervised example."""

    input_fields: tuple[str, ...] = ("observation.state", "observation.instruction")
    target_field: str = "action"
    train_split: str = "training_templates"
    evaluation_split: str = "heldout_paraphrases"
    pairing_key: str = "example_id"


@dataclass(frozen=True)
class LanguageTaskExample:
    example_id: str
    task_id: str
    split: LanguageSplit
    template_index: int
    observation: Observation
    target: NDArray[np.float32]
    action: NDArray[np.float32]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _vector2(self.target, name="target"))
        object.__setattr__(self, "action", _vector2(self.action, name="action"))


@dataclass(frozen=True)
class InstructionAblationPair:
    """Control and ablated observations sharing state, action target, and pair id."""

    pair_id: str
    mode: InstructionAblation
    control: LanguageTaskExample
    ablated: LanguageTaskExample
    metadata: Mapping[str, object]


def language_goal_specs() -> tuple[LanguageGoalSpec, ...]:
    return LANGUAGE_GOALS


def language_training_template() -> LanguageTrainingTemplate:
    return LanguageTrainingTemplate()


def build_language_examples(
    split: LanguageSplit = "train",
    *,
    initial_state: Sequence[float] = (0.5, 0.5),
    max_action: float = 0.20,
) -> tuple[LanguageTaskExample, ...]:
    """Build examples where state is identical and only instruction identifies the target."""

    if split not in {"train", "heldout"}:
        raise ValueError("split must be 'train' or 'heldout'")
    if not np.isfinite(max_action) or max_action <= 0:
        raise ValueError("max_action must be positive and finite")
    state = _vector2(initial_state, name="initial_state")
    examples: list[LanguageTaskExample] = []
    for spec in LANGUAGE_GOALS:
        templates = spec.training_templates if split == "train" else spec.heldout_paraphrases
        delta = np.clip(spec.target - state, -max_action, max_action).astype(np.float32)
        for index, instruction in enumerate(templates):
            examples.append(
                LanguageTaskExample(
                    example_id=f"{split}:{spec.task_id}:{index}",
                    task_id=spec.task_id,
                    split=split,
                    template_index=index,
                    observation=Observation(state=state.copy(), instruction=instruction),
                    target=spec.target.copy(),
                    action=delta.copy(),
                )
            )
    return tuple(examples)


def _clone_with_instruction(
    example: LanguageTaskExample,
    instruction: str | None,
    *,
    suffix: str,
) -> LanguageTaskExample:
    observation = Observation(
        state=np.asarray(example.observation.state).copy(),
        instruction=instruction,
        image=(
            None
            if example.observation.image is None
            else np.asarray(example.observation.image).copy()
        ),
    )
    return replace(example, example_id=f"{example.example_id}:{suffix}", observation=observation)


def _shuffle_donor_indices(examples: Sequence[LanguageTaskExample], seed: int) -> list[int]:
    if len({example.task_id for example in examples}) < 2:
        raise ValueError("instruction shuffle requires at least two task ids")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(examples))
    for _ in range(256):
        candidate = rng.permutation(indices)
        if all(examples[int(donor)].task_id != examples[index].task_id for index, donor in enumerate(candidate)):
            return [int(value) for value in candidate]
    # Deterministic fallback independent of Python hash randomization.
    return [
        next(
            donor
            for offset in range(1, len(examples) + 1)
            if (donor := (index + offset) % len(examples)) != index
            and examples[donor].task_id != example.task_id
        )
        for index, example in enumerate(examples)
    ]


def make_instruction_ablation_pairs(
    examples: Sequence[LanguageTaskExample],
    mode: InstructionAblation,
    *,
    seed: int = 0,
) -> tuple[InstructionAblationPair, ...]:
    """Create deterministic paired inputs without interpreting metric differences."""

    if mode not in {"mask", "shuffle", "counterfactual"}:
        raise ValueError(f"unsupported instruction ablation: {mode!r}")
    source = tuple(examples)
    if not source:
        return ()
    by_task = {spec.task_id: spec for spec in LANGUAGE_GOALS}
    task_order = [spec.task_id for spec in LANGUAGE_GOALS]
    shuffled = _shuffle_donor_indices(source, seed) if mode == "shuffle" else []
    pairs: list[InstructionAblationPair] = []

    for index, control in enumerate(source):
        donor_task_id: str | None = None
        applied_instruction: str | None
        if mode == "mask":
            applied_instruction = MASK_INSTRUCTION
        elif mode == "shuffle":
            donor = source[shuffled[index]]
            donor_task_id = donor.task_id
            applied_instruction = donor.observation.instruction
        else:
            if control.task_id not in by_task:
                raise ValueError(f"unknown task_id for counterfactual: {control.task_id}")
            task_index = task_order.index(control.task_id)
            offset = 1 + (abs(int(seed)) % (len(task_order) - 1))
            donor_task_id = task_order[(task_index + offset) % len(task_order)]
            donor_spec = by_task[donor_task_id]
            templates = (
                donor_spec.training_templates
                if control.split == "train"
                else donor_spec.heldout_paraphrases
            )
            applied_instruction = templates[control.template_index % len(templates)]

        ablated = _clone_with_instruction(control, applied_instruction, suffix=mode)
        pair_id = f"instruction:{mode}:{seed}:{control.example_id}"
        metadata: dict[str, object] = {
            "pair_id": pair_id,
            "paired": True,
            "mode": mode,
            "seed": int(seed),
            "source_task_id": control.task_id,
            "donor_task_id": donor_task_id,
            "original_instruction": control.observation.instruction,
            "applied_instruction": applied_instruction,
            "target_held_constant": True,
            "interpretation": "measurement input only; no effectiveness claim",
        }
        pairs.append(
            InstructionAblationPair(
                pair_id=pair_id,
                mode=mode,
                control=control,
                ablated=ablated,
                metadata=metadata,
            )
        )
    return tuple(pairs)


class InstructionConditionedPointReachEnv:
    """A tiny environment whose target is observable to a policy only through text."""

    def __init__(
        self,
        task_id: str,
        *,
        split: LanguageSplit = "train",
        template_index: int = 0,
        initial_state: ArrayLike = (0.5, 0.5),
        action_clip: float = 0.20,
        success_distance: float = 0.05,
    ) -> None:
        goals = {spec.task_id: spec for spec in LANGUAGE_GOALS}
        if task_id not in goals:
            raise ValueError(f"unknown language task: {task_id!r}")
        if split not in {"train", "heldout"}:
            raise ValueError("split must be 'train' or 'heldout'")
        if action_clip <= 0 or success_distance <= 0:
            raise ValueError("action_clip and success_distance must be positive")
        self.spec = goals[task_id]
        self.split = split
        self.template_index = int(template_index)
        self.initial_state = _vector2(initial_state, name="initial_state")
        self.action_clip = float(action_clip)
        self.success_distance = float(success_distance)
        self._state = self.initial_state.copy()
        self._observation: Observation | None = None

    @property
    def instruction(self) -> str:
        templates = (
            self.spec.training_templates
            if self.split == "train"
            else self.spec.heldout_paraphrases
        )
        return templates[self.template_index % len(templates)]

    def reset(self, *, seed: int | None = None) -> Observation:
        del seed  # The default reset is intentionally identical across language tasks.
        self._state = self.initial_state.copy()
        self._observation = Observation(state=self._state.copy(), instruction=self.instruction)
        return self._observation

    def step(self, action: NDArray[np.generic]) -> Transition:
        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        action_array = _vector2(action, name="action")
        applied = np.clip(action_array, -self.action_clip, self.action_clip).astype(np.float32)
        previous = self._observation
        self._state = np.clip(self._state + applied, 0.0, 1.0).astype(np.float32)
        distance = float(np.linalg.norm(self.spec.target - self._state))
        success = distance <= self.success_distance
        next_observation = Observation(state=self._state.copy(), instruction=self.instruction)
        self._observation = next_observation
        return Transition(
            observation=previous,
            action=applied,
            reward=-distance,
            next_observation=next_observation,
            terminated=success,
            info={
                "task_id": self.spec.task_id,
                "target": self.spec.target.tolist(),
                "distance": distance,
                "success": success,
                "language_split": self.split,
                "template_index": self.template_index,
            },
        )

    def close(self) -> None:
        """Release task resources (the synthetic task has none)."""

        return None


class LanguageTaskSuiteEnv:
    """Cycle all goals and held-out paraphrases from configured evaluation seeds."""

    def __init__(self, *, split: LanguageSplit = "heldout") -> None:
        if split not in {"train", "heldout"}:
            raise ValueError("split must be 'train' or 'heldout'")
        self.split = split
        self._env: InstructionConditionedPointReachEnv | None = None

    def reset(self, *, seed: int | None = None) -> Observation:
        resolved_seed = 0 if seed is None else int(seed)
        choices = [
            (spec.task_id, template_index)
            for spec in LANGUAGE_GOALS
            for template_index in range(
                len(
                    spec.training_templates
                    if self.split == "train"
                    else spec.heldout_paraphrases
                )
            )
        ]
        task_id, template_index = choices[resolved_seed % len(choices)]
        initial_state = np.random.default_rng(resolved_seed).uniform(
            0.35, 0.65, size=2
        ).astype(np.float32)
        self._env = InstructionConditionedPointReachEnv(
            task_id,
            split=self.split,
            template_index=template_index,
            initial_state=initial_state,
        )
        return self._env.reset(seed=resolved_seed)

    def step(self, action: NDArray[np.generic]) -> Transition:
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        return self._env.step(action)

    def close(self) -> None:
        """Release task resources (the synthetic suite has none)."""

        return None


class LanguageTemplateDatasetSource:
    """Generate train-template or held-out-paraphrase expert transitions."""

    def __init__(
        self,
        split: LanguageSplit = "train",
        *,
        max_steps: int = 8,
        initial_state: ArrayLike | None = (0.5, 0.5),
        seed: int = 0,
        episode_count: int | None = None,
    ) -> None:
        if split not in {"train", "heldout"}:
            raise ValueError("split must be 'train' or 'heldout'")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if episode_count is not None and episode_count <= 0:
            raise ValueError("episode_count must be positive when supplied")
        self.split = split
        self.max_steps = int(max_steps)
        self.initial_state = (
            None if initial_state is None else _vector2(initial_state, name="initial_state")
        )
        self.seed = int(seed)
        self.episode_count = episode_count

    def load(self) -> Sequence[Transition]:
        transitions: list[Transition] = []
        combinations: list[tuple[LanguageGoalSpec, int]] = []
        for spec in LANGUAGE_GOALS:
            templates = (
                spec.training_templates
                if self.split == "train"
                else spec.heldout_paraphrases
            )
            combinations.extend((spec, index) for index in range(len(templates)))
        count = self.episode_count or len(combinations)
        for episode_id in range(count):
            spec, template_index = combinations[episode_id % len(combinations)]
            cycle = episode_id // len(combinations)
            initial_state = self.initial_state
            if initial_state is None:
                initial_state = np.random.default_rng(self.seed + cycle).uniform(
                    0.35, 0.65, size=2
                ).astype(np.float32)
            env = InstructionConditionedPointReachEnv(
                spec.task_id,
                split=self.split,
                template_index=template_index,
                initial_state=initial_state,
            )
            observation = env.reset()
            for step in range(self.max_steps):
                transition = env.step(
                    spec.target - np.asarray(observation.state, dtype=np.float32)
                )
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
                        "step": step,
                        "data_seed": self.seed + cycle,
                        "language_split": self.split,
                        "template_index": template_index,
                    },
                )
                transitions.append(transition)
                observation = transition.next_observation
                if transition.terminated:
                    break
        return tuple(transitions)
