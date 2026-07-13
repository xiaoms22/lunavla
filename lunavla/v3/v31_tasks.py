from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import numpy.typing as npt

from .contracts import EpisodeRecordV3, FeatureSchema, ObservationV3, TransitionV3
from .data import DataAuditManifest, DatasetBundle, audit_episodes
from .v31_contracts import TaskSuiteSpecV1


Float32Array = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]

V31_TASK_IDS = ("direct_pick_place", "waypoint_sequence", "failure_recovery")
V31_HELD_OUT_STRATA = ("composition", "paraphrase")
_SPLITS = ("train", "validation", "test")
_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("red", (235, 64, 52)),
    ("blue", (52, 105, 235)),
    ("green", (48, 180, 92)),
    ("yellow", (235, 196, 52)),
)
_SHAPES = ("circle", "square", "diamond")
_HELD_OUT_COMBINATIONS = {("red", "diamond"), ("blue", "square"), ("green", "circle")}
_TRAIN_TEMPLATES = {
    "direct_pick_place": "pick the {color} {shape} and place it in the marked zone",
    "waypoint_sequence": "move the {color} {shape} through the shown waypoints in order",
    "failure_recovery": "recover the displaced {color} {shape} and finish in the marked zone",
}
_PARAPHRASES = {
    "direct_pick_place": "put the {shape} that is {color} into the outlined destination",
    "waypoint_sequence": "guide the {color} {shape} along each marker without changing order",
    "failure_recovery": "correct the offset of the {color} {shape}, then complete its placement",
}
_FORBIDDEN_OBSERVATION_KEYS = {
    "goal", "goal_coordinates", "waypoints", "oracle_action", "answer_key", "target_xy",
}


def task_suite_spec_v1() -> TaskSuiteSpecV1:
    return TaskSuiteSpecV1(
        suite_id="synthetic_vlm_v1",
        task_ids=V31_TASK_IDS,
        geometry_generator="lunavla.seeded_geometry/v1",
        visible_modalities=("camera.primary", "instruction", "state.proprioception"),
        instruction_generator="lunavla.compositional_instruction/v1",
        held_out_strata=V31_HELD_OUT_STRATA,
        success_conditions={
            "direct_pick_place": "target_distance<=0.045 and gripper_closed",
            "waypoint_sequence": "ordered_waypoints_complete and target_distance<=0.045",
            "failure_recovery": "recovery_phase_complete and target_distance<=0.045",
        },
        image_shape=(96, 96, 3),
        state_fields=("x", "y", "gripper", "phase"),
        action_fields=("dx", "dy", "gripper"),
        action_min=-1,
        action_max=1,
        control_rate_hz=10,
        max_steps=64,
        oracle_excluded_fields=tuple(sorted(_FORBIDDEN_OBSERVATION_KEYS)),
    )


def v31_feature_schema() -> FeatureSchema:
    return FeatureSchema.from_mapping(
        {
            "schema_version": 1,
            "items": [
                {
                    "name": "camera.primary", "role": "image", "dtype": "uint8",
                    "shape": [96, 96, 3], "unit": "pixel", "frame": "synthetic_camera",
                    "rate_hz": 10.0, "normalization": "none", "source_key": "image",
                    "required_by": ["act_v3"],
                },
                {
                    "name": "state.proprioception", "role": "state", "dtype": "float32",
                    "shape": [4], "unit": "unitless", "frame": "synthetic_world",
                    "rate_hz": 10.0, "normalization": "standard", "source_key": "state",
                    "required_by": ["act_v3"],
                },
                {
                    "name": "action.primary", "role": "action", "dtype": "float32",
                    "shape": [3], "unit": "unitless", "frame": "synthetic_world",
                    "rate_hz": 10.0, "normalization": "standard", "source_key": "action",
                    "required_by": ["act_v3"],
                },
            ],
        }
    )


def _seed(task_id: str, split: str, stratum: str, data_seed: int, index: int) -> int:
    payload = f"{task_id}|{split}|{stratum}|{data_seed}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _combination(rng: np.random.Generator, *, held_out: bool) -> tuple[str, str, tuple[int, int, int]]:
    allowed = [
        (name, shape, rgb)
        for name, rgb in _COLORS
        for shape in _SHAPES
        if ((name, shape) in _HELD_OUT_COMBINATIONS) is held_out
    ]
    name, shape, rgb = allowed[int(rng.integers(0, len(allowed)))]
    return name, shape, rgb


def _pixel(point: Float32Array) -> tuple[int, int]:
    xy = np.clip(np.rint(point * 91 + 2).astype(int), 2, 93)
    return int(xy[0]), int(xy[1])


def _draw_marker(image: UInt8Array, point: Float32Array, color: tuple[int, int, int], shape: str) -> None:
    x, y = _pixel(point)
    yy, xx = np.ogrid[:96, :96]
    if shape == "circle":
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= 16
    elif shape == "square":
        mask = (np.abs(xx - x) <= 4) & (np.abs(yy - y) <= 4)
    elif shape == "diamond":
        mask = np.abs(xx - x) + np.abs(yy - y) <= 5
    else:
        raise ValueError(f"unsupported shape {shape!r}")
    image[mask] = np.asarray(color, dtype=np.uint8)


def _render(
    position: Float32Array,
    target: Float32Array,
    waypoints: tuple[Float32Array, ...],
    rgb: tuple[int, int, int],
    shape: str,
) -> UInt8Array:
    image = np.full((96, 96, 3), 244, dtype=np.uint8)
    image[::8, :, :] = 232
    image[:, ::8, :] = 232
    _draw_marker(image, target, (35, 35, 35), "square")
    for waypoint in waypoints:
        _draw_marker(image, waypoint, (155, 80, 190), "diamond")
    _draw_marker(image, position, rgb, shape)
    return image


def _instruction(task_id: str, color: str, shape: str, stratum: str) -> str:
    template = _PARAPHRASES[task_id] if stratum == "paraphrase" else _TRAIN_TEMPLATES[task_id]
    return template.format(color=color, shape=shape)


def _observation(
    *, episode_id: str, step: int, task_id: str, split: str, stratum: str,
    color: str, shape: str, rgb: tuple[int, int, int], position: Float32Array,
    target: Float32Array, waypoints: tuple[Float32Array, ...], gripper: float, phase: float,
) -> ObservationV3:
    metadata = {
        "task_id": task_id,
        "split": split,
        "held_out_stratum": stratum,
        "visible_attributes": {"color": color, "shape": shape},
        "generator": "lunavla.seeded_geometry/v1",
    }
    if _FORBIDDEN_OBSERVATION_KEYS & set(metadata):
        raise AssertionError("oracle field leaked into observation metadata")
    return ObservationV3(
        images={"camera.primary": _render(position, target, waypoints, rgb, shape)},
        state={
            "state.proprioception": np.asarray(
                [position[0], position[1], gripper, phase], dtype=np.float32
            )
        },
        instruction=_instruction(task_id, color, shape, stratum),
        timestamp_s=step / 10,
        episode_id=episode_id,
        step_index=step,
        metadata=metadata,
    )


def make_v31_episode(
    *, task_id: str, split: str, stratum: str, data_seed: int, index: int,
) -> EpisodeRecordV3:
    if task_id not in V31_TASK_IDS:
        raise ValueError("unsupported v3.1 task_id")
    if split not in _SPLITS:
        raise ValueError("split must be train, validation, or test")
    if split == "train" and stratum != "train":
        raise ValueError("training episodes must use stratum=train")
    if split != "train" and stratum not in V31_HELD_OUT_STRATA:
        raise ValueError("evaluation episodes require a registered held-out stratum")
    if isinstance(data_seed, bool) or not isinstance(data_seed, int) or data_seed < 0:
        raise ValueError("data_seed must be a non-negative integer")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer")
    episode_seed = _seed(task_id, split, stratum, data_seed, index)
    rng = np.random.default_rng(episode_seed)
    color, shape, rgb = _combination(rng, held_out=stratum == "composition")
    position = rng.uniform(0.12, 0.32, size=2).astype(np.float32)
    target = rng.uniform(0.68, 0.88, size=2).astype(np.float32)
    waypoints: tuple[Float32Array, ...]
    route: tuple[Float32Array, ...]
    if task_id == "waypoint_sequence":
        waypoints = (
            np.asarray([0.42, 0.30 + 0.25 * rng.random()], dtype=np.float32),
            np.asarray([0.58, 0.50 + 0.20 * rng.random()], dtype=np.float32),
        )
        route = (*waypoints, target)
    elif task_id == "failure_recovery":
        disturbance = rng.uniform(-0.18, 0.18, size=2).astype(np.float32)
        recovery = np.clip(position + disturbance, 0.08, 0.92).astype(np.float32)
        route = (recovery, target)
        waypoints = (recovery,)
    else:
        route = (target,)
        waypoints = ()
    episode_id = f"v31-{split}-{task_id}-{stratum}-{episode_seed:016x}"
    transitions: list[TransitionV3] = []
    step = 0
    gripper_state = 0.0
    for phase_index, destination in enumerate(route):
        for _ in range(8):
            gripper_command = 1.0 if phase_index == len(route) - 1 else 0.0
            phase = phase_index / len(route)
            observation = _observation(
                episode_id=episode_id, step=step, task_id=task_id, split=split,
                stratum=stratum, color=color, shape=shape, rgb=rgb, position=position,
                target=target, waypoints=waypoints, gripper=gripper_state, phase=phase,
            )
            delta = np.clip(destination - position, -0.14, 0.14).astype(np.float32)
            action = np.asarray([delta[0], delta[1], gripper_command], dtype=np.float32)
            position = np.clip(position + delta, 0, 1).astype(np.float32)
            gripper_state = gripper_command
            step += 1
            reached = bool(np.linalg.norm(position - destination) <= 0.045)
            final = phase_index == len(route) - 1 and reached
            next_phase = phase_index + int(reached)
            next_observation = _observation(
                episode_id=episode_id, step=step, task_id=task_id, split=split,
                stratum=stratum, color=color, shape=shape, rgb=rgb, position=position,
                target=target, waypoints=waypoints, gripper=gripper_state,
                phase=next_phase / len(route),
            )
            distance = float(np.linalg.norm(position - destination))
            transitions.append(
                TransitionV3(
                    observation=observation,
                    action=action,
                    reward=-distance,
                    next_observation=next_observation,
                    terminated=bool(final),
                    truncated=False,
                    info={"success": bool(final), "phase_index": phase_index},
                )
            )
            if final:
                break
            if reached:
                break
        if transitions[-1].terminated:
            break
    if not transitions[-1].terminated:
        last = transitions[-1]
        transitions[-1] = TransitionV3(
            last.observation, last.action, last.reward, last.next_observation, False, True,
            {"success": False, "phase_index": len(route) - 1},
        )
    return EpisodeRecordV3(
        episode_id,
        tuple(transitions),
        {
            "task_id": task_id, "split": split, "held_out_stratum": stratum,
            "episode_seed": episode_seed, "combination": f"{color}:{shape}",
            "instruction_family": "paraphrase" if stratum == "paraphrase" else "canonical",
        },
    )


@dataclass(frozen=True)
class V31TaskDataset:
    bundle: DatasetBundle
    suite_spec: TaskSuiteSpecV1

    def __post_init__(self) -> None:
        split_sets = {name: set(self.bundle.split[name]) for name in _SPLITS}
        if any(split_sets[left] & split_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError("v3.1 episode splits overlap")
        for episode in self.bundle.episodes:
            expected_split = str(episode.metadata["split"])
            if episode.episode_id not in split_sets[expected_split]:
                raise ValueError("episode metadata split does not match dataset split")
            for transition in episode.transitions:
                metadata = set(transition.observation.metadata)
                if metadata & _FORBIDDEN_OBSERVATION_KEYS:
                    raise ValueError("oracle information leaked into policy observation")

    @property
    def audit(self) -> DataAuditManifest:
        return self.bundle.audit


def make_v31_task_dataset(
    *, data_seed: int = 42, train_per_task: int = 6, held_out_per_cell: int = 2,
) -> V31TaskDataset:
    if train_per_task <= 0 or held_out_per_cell <= 0:
        raise ValueError("episode counts must be positive")
    episodes: list[EpisodeRecordV3] = []
    split: dict[str, list[str | int]] = {name: [] for name in _SPLITS}
    for task_id in V31_TASK_IDS:
        for index in range(train_per_task):
            episode = make_v31_episode(
                task_id=task_id, split="train", stratum="train",
                data_seed=data_seed, index=index,
            )
            episodes.append(episode)
            split["train"].append(episode.episode_id)
        for split_name in ("validation", "test"):
            for stratum in V31_HELD_OUT_STRATA:
                for index in range(held_out_per_cell):
                    episode = make_v31_episode(
                        task_id=task_id, split=split_name, stratum=stratum,
                        data_seed=data_seed, index=index,
                    )
                    episodes.append(episode)
                    split[split_name].append(episode.episode_id)
    records = tuple(episodes)
    normalized_split: Mapping[str, tuple[str | int, ...]] = MappingProxyType(
        {name: tuple(split[name]) for name in _SPLITS}
    )
    schema = v31_feature_schema()
    audit = audit_episodes(records, feature_schema=schema, split=normalized_split)
    bundle = DatasetBundle(records, normalized_split, audit)
    return V31TaskDataset(bundle=bundle, suite_spec=task_suite_spec_v1())


def task_dataset_sha256(dataset: V31TaskDataset) -> str:
    payload = {
        "suite_spec_sha256": dataset.suite_spec.sha256(),
        "audit_sha256": dataset.audit.sha256(),
        "split": {name: list(dataset.bundle.split[name]) for name in _SPLITS},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()
