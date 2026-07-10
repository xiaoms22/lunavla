from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np
import pytest

from lunavla.contracts import DatasetSource, Observation, TaskEnv, Transition
from lunavla.language_tasks import (
    InstructionConditionedPointReachEnv,
    MASK_INSTRUCTION,
    LanguageTemplateDatasetSource,
    LanguageTaskSuiteEnv,
    build_language_examples,
    language_goal_specs,
    language_training_template,
    make_instruction_ablation_pairs,
)
from lunavla.lerobot_adapter import (
    LeRobotDatasetSource,
    LeRobotEnvAdapter,
    LeRobotFieldMap,
    LeRobotUnavailableError,
    lerobot_installation_status,
    load_lerobot_dataset_factory,
    map_lerobot_sample,
    require_lerobot_06,
)
from lunavla.visual_tasks import (
    RenderedPointReachEnv,
    RenderedVisualDatasetSource,
    RenderedVisualTaskSuiteEnv,
    build_visual_examples,
    make_image_ablation_pairs,
    make_state_only_baseline,
    render_point_reach,
    visual_task_specs,
)


def test_language_tasks_require_instruction_and_have_heldout_paraphrases() -> None:
    specs = language_goal_specs()
    train = build_language_examples("train")
    heldout = build_language_examples("heldout")

    assert len(specs) >= 3
    assert len({spec.task_id for spec in specs}) >= 3
    assert len({tuple(spec.target) for spec in specs}) >= 3
    assert all(np.array_equal(example.observation.state, train[0].observation.state) for example in train)
    assert len({tuple(example.action) for example in train}) >= 3
    assert all(example.observation.image is None for example in train)

    train_text = {str(example.observation.instruction).casefold() for example in train}
    heldout_text = {str(example.observation.instruction).casefold() for example in heldout}
    assert train_text.isdisjoint(heldout_text)
    template = language_training_template()
    assert template.input_fields == ("observation.state", "observation.instruction")
    assert template.evaluation_split == "heldout_paraphrases"


@pytest.mark.parametrize("mode", ["mask", "shuffle", "counterfactual"])
def test_instruction_ablation_pairs_are_deterministic_and_paired(mode: str) -> None:
    examples = build_language_examples("heldout")
    first = make_instruction_ablation_pairs(examples, mode, seed=19)  # type: ignore[arg-type]
    second = make_instruction_ablation_pairs(examples, mode, seed=19)  # type: ignore[arg-type]

    assert [pair.pair_id for pair in first] == [pair.pair_id for pair in second]
    assert [pair.ablated.observation.instruction for pair in first] == [
        pair.ablated.observation.instruction for pair in second
    ]
    for pair in first:
        assert pair.metadata["paired"] is True
        assert pair.metadata["target_held_constant"] is True
        assert "no effectiveness claim" in str(pair.metadata["interpretation"])
        assert np.array_equal(pair.control.observation.state, pair.ablated.observation.state)
        assert np.array_equal(pair.control.action, pair.ablated.action)
        assert np.array_equal(pair.control.target, pair.ablated.target)
        if mode == "mask":
            assert pair.ablated.observation.instruction == MASK_INSTRUCTION
        else:
            assert pair.metadata["donor_task_id"] != pair.control.task_id
            assert pair.ablated.observation.instruction != pair.control.observation.instruction


def test_language_environment_implements_task_protocol_with_same_reset_state() -> None:
    envs = [InstructionConditionedPointReachEnv(spec.task_id) for spec in language_goal_specs()]
    assert all(isinstance(env, TaskEnv) for env in envs)
    observations = [env.reset(seed=123) for env in envs]
    assert all(np.array_equal(observation.state, observations[0].state) for observation in observations)
    assert len({observation.instruction for observation in observations}) == len(envs)

    transitions = [env.step(env.spec.target - observation.state) for env, observation in zip(envs, observations)]
    assert all(isinstance(transition, Transition) for transition in transitions)
    assert all("success" in transition.info for transition in transitions)


def test_language_training_and_heldout_sources_use_distinct_instruction_splits() -> None:
    train_source = LanguageTemplateDatasetSource("train")
    heldout_source = LanguageTemplateDatasetSource("heldout")
    assert isinstance(train_source, DatasetSource)
    train = train_source.load()
    heldout = heldout_source.load()
    assert {transition.info["task_id"] for transition in train} == {
        spec.task_id for spec in language_goal_specs()
    }
    train_instructions = {transition.observation.instruction for transition in train}
    heldout_instructions = {transition.observation.instruction for transition in heldout}
    assert train_instructions.isdisjoint(heldout_instructions)
    assert all("success" in transition.info for transition in train + heldout)


def test_language_suite_seeds_cover_every_goal_and_heldout_paraphrase() -> None:
    env = LanguageTaskSuiteEnv(split="heldout")
    observations = [env.reset(seed=seed) for seed in range(6)]
    assert len({observation.instruction for observation in observations}) == 6
    assert len({env.reset(seed=seed).instruction for seed in range(6)}) == 6


def test_pil_visual_tasks_cover_two_families_and_state_only_baseline() -> None:
    assert {spec.family for spec in visual_task_specs()} == {"direct_reach", "waypoint_reach"}
    image = render_point_reach(
        (0.2, 0.2),
        goal=(0.8, 0.2),
        waypoint=(0.5, 0.8),
        family="waypoint_reach",
        image_size=48,
    )
    assert image.shape == (48, 48, 3)
    assert image.dtype == np.uint8
    assert np.unique(image.reshape(-1, 3), axis=0).shape[0] > 3

    examples = build_visual_examples(seeds=(2, 7), image_size=40)
    assert {example.family for example in examples} == {"direct_reach", "waypoint_reach"}
    assert all(example.observation.image is not None for example in examples)
    baseline = make_state_only_baseline(examples)
    for visual, state_only in zip(examples, baseline):
        assert state_only.observation.image is None
        assert np.array_equal(visual.observation.state, state_only.observation.state)
        assert np.array_equal(visual.action, state_only.action)
        assert state_only.metadata["paired_with"] == visual.example_id


@pytest.mark.parametrize("mode", ["occlusion", "shuffle"])
def test_image_ablation_pairs_are_deterministic_and_preserve_targets(mode: str) -> None:
    examples = build_visual_examples(seeds=(0, 1), image_size=36)
    first = make_image_ablation_pairs(examples, mode, seed=5)  # type: ignore[arg-type]
    second = make_image_ablation_pairs(examples, mode, seed=5)  # type: ignore[arg-type]
    assert [pair.pair_id for pair in first] == [pair.pair_id for pair in second]
    for left, right in zip(first, second):
        assert np.array_equal(left.ablated.observation.image, right.ablated.observation.image)
        assert np.array_equal(left.control.observation.state, left.ablated.observation.state)
        assert np.array_equal(left.control.action, left.ablated.action)
        assert left.metadata["state_held_constant"] is True
        assert left.metadata["action_target_held_constant"] is True
        if mode == "occlusion":
            assert left.metadata["occlusion_box_xyxy"] is not None
            assert np.any(np.asarray(left.ablated.observation.image) == 0)
        else:
            assert left.metadata["donor_example_id"] != left.control.example_id


@pytest.mark.parametrize("family", ["direct_reach", "waypoint_reach"])
def test_visual_environment_protocol_and_image_shape(family: str) -> None:
    env = RenderedPointReachEnv(family=family, image_size=32)  # type: ignore[arg-type]
    assert isinstance(env, TaskEnv)
    observation = env.reset(seed=8)
    assert observation.image is not None
    assert observation.image.shape == (32, 32, 3)
    transition = env.step(env.expert_action())
    assert transition.next_observation.image is not None
    assert transition.info["task_family"] == family
    assert "success" in transition.info


def test_rendered_visual_dataset_source_supports_visual_and_state_only_paths() -> None:
    visual_source = RenderedVisualDatasetSource(seeds=(3,), image_size=32)
    state_source = RenderedVisualDatasetSource(seeds=(3,), image_size=32, state_only=True)
    assert isinstance(visual_source, DatasetSource)
    visual = visual_source.load()
    state_only = state_source.load()
    assert {transition.info["task_family"] for transition in visual} == {
        "direct_reach",
        "waypoint_reach",
    }
    assert all(transition.observation.image is not None for transition in visual)
    assert all(transition.observation.image is None for transition in state_only)
    assert len(visual) == len(state_only)
    assert all(
        np.array_equal(left.observation.state, right.observation.state)
        and np.array_equal(left.action, right.action)
        for left, right in zip(visual, state_only)
    )


def test_visual_suite_seeds_cover_both_task_families() -> None:
    env = RenderedVisualTaskSuiteEnv(image_size=32)
    families: set[str] = set()
    for seed in (0, 1):
        env.reset(seed=seed)
        transition = env.step(np.zeros(2, dtype=np.float32))
        families.add(str(transition.info["task_family"]))
    assert families == {"direct_reach", "waypoint_reach"}


class _FakeTensor:
    def __init__(self, value: Any) -> None:
        self.value = np.asarray(value)

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.value


class _FakeLeRobotDataset:
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def _fake_lerobot_samples() -> list[dict[str, Any]]:
    return [
        {
            "observation.state": _FakeTensor([0.1 + 0.1 * index, 0.2]),
            "observation.images.top": _FakeTensor(
                np.full((3, 8, 10), fill_value=index / 10.0, dtype=np.float32)
            ),
            "action": _FakeTensor([0.02, -0.01]),
            "task": "push the block",
            "episode_index": 0 if index < 2 else 1,
            "next.reward": float(index),
            "next.done": index in {1, 2},
        }
        for index in range(3)
    ]


def test_lerobot_import_is_lazy_and_missing_dependency_error_is_lightweight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden_import(name: str, package: str | None = None) -> Any:
        nonlocal called
        called = True
        raise AssertionError(f"unexpected import: {name}, {package}")

    monkeypatch.setattr(importlib, "import_module", forbidden_import)
    status = lerobot_installation_status()
    assert "installed" in status
    assert called is False

    monkeypatch.setattr(
        "lunavla.lerobot_adapter.lerobot_installation_status",
        lambda: {"installed": False, "version": None, "compatible_0_6": False},
    )
    with pytest.raises(LeRobotUnavailableError, match=r"lerobot\[dataset\]==0\.6\.\*"):
        require_lerobot_06()


@pytest.mark.lerobot
def test_installed_lerobot_dataset_profile_resolves_official_factory() -> None:
    status = lerobot_installation_status()
    if not status["installed"]:
        if os.environ.get("LUNAVLA_REQUIRE_LEROBOT") == "1":
            pytest.fail("full v2 LeRobot profile was required but is not installed")
        pytest.skip("full v2 LeRobot profile is not installed")
    assert require_lerobot_06().startswith("0.6.")
    factory = load_lerobot_dataset_factory()
    assert factory.__name__ == "LeRobotDataset"


def test_fake_lerobot_dataset_maps_real_field_shapes_without_heavy_imports() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    samples = _fake_lerobot_samples()

    def factory(repo_id: str, **kwargs: object) -> _FakeLeRobotDataset:
        calls.append((repo_id, kwargs))
        return _FakeLeRobotDataset(samples)

    source = LeRobotDatasetSource.from_repo_id(
        "lerobot/pusht",
        episodes=[0, 1],
        dataset_factory=factory,
    )
    assert isinstance(source, DatasetSource)
    transitions = source.load()
    assert calls == [("lerobot/pusht", {"episodes": [0, 1]})]
    assert len(transitions) == 3
    assert transitions[0].observation.state.shape == (2,)
    assert transitions[0].observation.state.dtype == np.float32
    assert transitions[0].observation.image is not None
    assert transitions[0].observation.image.shape == (8, 10, 3)
    assert transitions[0].observation.image.dtype == np.float32
    assert transitions[0].action.shape == (2,)
    assert transitions[0].terminated is False
    assert transitions[1].terminated is True
    assert transitions[2].terminated is True
    assert np.array_equal(
        transitions[0].next_observation.state,
        transitions[1].observation.state,
    )
    assert transitions[0].info["repo_id"] == "lerobot/pusht"
    assert transitions[0].info["image_key"] == "observation.images.top"
    assert transitions[0].info["success"] is False


def test_sample_mapping_supports_nested_keys_and_uint8_hwc() -> None:
    sample = {
        "observation": {
            "state": np.asarray([1.0, 2.0], dtype=np.float64),
            "camera": np.full((6, 7, 3), 127, dtype=np.uint8),
        },
        "control": {"action": np.asarray([0.1, -0.2], dtype=np.float64)},
        "language": "pick the cube",
    }
    field_map = LeRobotFieldMap(
        state_key="observation.state",
        image_keys=("observation.camera",),
        action_key="control.action",
        instruction_keys=("language",),
    )
    mapped = map_lerobot_sample(sample, field_map=field_map, require_image=True)
    assert mapped.observation.image is not None
    assert mapped.observation.image.shape == (6, 7, 3)
    assert mapped.observation.image.dtype == np.uint8
    assert mapped.observation.instruction == "pick the cube"
    assert mapped.action.dtype == np.float32


class _FakeGymEnv:
    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        value = float(seed or 0)
        return {"observation.state": np.asarray([value, 0.0], dtype=np.float32)}, {}

    def step(
        self, action: np.ndarray[Any, Any]
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        return (
            {"observation.state": np.asarray(action, dtype=np.float32)},
            1.0,
            True,
            False,
            {"success": True},
        )


def test_optional_lerobot_env_adapter_implements_task_protocol() -> None:
    env = LeRobotEnvAdapter(_FakeGymEnv())
    assert isinstance(env, TaskEnv)
    observation = env.reset(seed=3)
    transition = env.step(np.asarray([0.2, -0.1], dtype=np.float32))
    assert isinstance(observation, Observation)
    assert isinstance(transition, Transition)
    assert transition.terminated is True
    assert transition.info["success"] is True
    assert np.allclose(transition.next_observation.state, [0.2, -0.1])
