from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from lunavla.config import ExperimentConfig as ExperimentConfigV2
from lunavla.contracts import Observation

from .config import ExperimentConfig
from .contracts import ObservationV3


def _feature(
    *, name: str, role: str, dtype: str, shape: list[int], unit: str, frame: str, source_key: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "dtype": dtype,
        "shape": shape,
        "unit": unit,
        "frame": frame,
        "rate_hz": None,
        "normalization": "none",
        "source_key": source_key,
        "required_by": [],
    }


def migrate_v2_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise TypeError("v2 config must be a mapping")
    version = source.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError("schema_version must be an integer")
    if version != 2:
        raise ValueError("migrate_v2_mapping requires schema_version 2")
    v2 = ExperimentConfigV2.from_mapping(copy.deepcopy(dict(source)))
    resolved_v2 = v2.to_dict()
    policy = resolved_v2["policy"]
    state_dim = int(policy["state_dim"])
    action_dim = int(policy["action_dim"])
    features = [
        _feature(
            name="state.proprioception", role="state", dtype="float32", shape=[state_dim],
            unit="unspecified_v2", frame="unspecified_v2", source_key="state",
        )
    ]
    image_shape = policy.get("image_shape")
    camera_mapping: dict[str, str] = {}
    if image_shape is not None:
        features.append(
            _feature(
                name="camera.primary", role="image", dtype="uint8", shape=list(image_shape),
                unit="pixel", frame="unspecified_v2", source_key="image",
            )
        )
        camera_mapping["image"] = "camera.primary"
    features.append(
        _feature(
            name="action.primary", role="action", dtype="float32", shape=[action_dim],
            unit="unspecified_v2", frame="unspecified_v2", source_key="action",
        )
    )
    task = resolved_v2["task"]
    dataset = resolved_v2["dataset"]
    training = resolved_v2["training"]
    evaluation = resolved_v2["evaluation"]
    artifacts = resolved_v2["artifacts"]
    evaluation_seeds = list(evaluation.get("seeds", []))
    evaluation_episodes = int(evaluation.get("episodes", 5))
    evaluation_seed = int(evaluation.get("seed", training.get("seed", 0)))
    if not evaluation_seeds:
        evaluation_seeds = list(range(evaluation_seed, evaluation_seed + evaluation_episodes))
    migrated: dict[str, Any] = {
        "schema_version": 3,
        "contract_revision": 2,
        "project_name": v2.project_name,
        "engine": "lunavla_v3",
        "policy": {
            "type": policy.pop("type"),
            "parameters": {"legacy": policy, "compat_read_only": True},
        },
        "task": {"id": task.pop("id"), "parameters": {"legacy": task}},
        "dataset": {
            "type": "v2_compat",
            "split": dataset.get("split", "train"),
            "seed": int(dataset.get("seed", training.get("seed", 0))),
            "parameters": {"legacy": dataset},
        },
        "embodiment": {
            "id": f"v2_compat/{v2.task['id']}",
            "task_id": v2.task["id"],
            "control_rate_hz": None,
            "camera_mapping": camera_mapping,
            "state_mapping": {"state": "state.proprioception"},
            "action_mapping": {"action": "action.primary"},
        },
        "features": {"schema_version": 1, "items": features},
        "training": {
            "device": training.get("device", policy.get("device", "cpu")),
            "seed": int(training.get("seed", 0)),
            "batch_size": int(training.get("batch_size", 32)),
            "steps": int(training.get("steps", 100)),
            "learning_rate": float(training.get("learning_rate", 0.04)),
        },
        "evaluation": {
            "execution_mode": evaluation.get("execution_mode", "receding_horizon"),
            "episodes": evaluation_episodes,
            "seed": evaluation_seed,
            "seeds": evaluation_seeds,
            "max_steps": int(v2.task.get("max_steps", 40)),
        },
        "diagnostics": {"enabled": False},
        "prompt": {
            "enabled": False,
            "renderer_id": "lunavla.canonical_json",
            "renderer_version": 1,
            "assistant_target": "action_chunk",
            "neutral_token": "[MASKED]",
            "camera_order": list(camera_mapping.values()),
            "public_slots": {},
        },
        "routing": {
            "mode": "expert_only",
            "state_features": ["state.proprioception"],
        },
        "artifacts": {
            "output_dir": artifacts["output_dir"],
            "checkpoint_name": artifacts.get("checkpoint_name", "checkpoint.json"),
        },
    }
    ExperimentConfig.from_mapping(migrated)
    return migrated


def migrate_v2_file(source: str | Path, destination: str | Path) -> Path:
    payload = yaml.safe_load(Path(source).read_text(encoding="utf-8-sig"))
    migrated = migrate_v2_mapping(payload)
    output = Path(destination)
    output.write_text(yaml.safe_dump(migrated, sort_keys=False), encoding="utf-8")
    return output


def observation_from_v2(
    observation: Observation,
    *,
    episode_id: str | int,
    step_index: int,
    timestamp_s: float,
) -> ObservationV3:
    images = {} if observation.image is None else {"camera.primary": observation.image}
    return ObservationV3(
        images=images,
        state={"state.proprioception": observation.state},
        instruction=observation.instruction,
        timestamp_s=timestamp_s,
        episode_id=episode_id,
        step_index=step_index,
    )


def observation_to_v2(observation: ObservationV3) -> Observation:
    if tuple(observation.state) != ("state.proprioception",):
        raise ValueError("v2 conversion requires exactly state.proprioception")
    if tuple(observation.images) not in ((), ("camera.primary",)):
        raise ValueError("v2 conversion supports only camera.primary")
    image = observation.images.get("camera.primary")
    return Observation(
        np.asarray(observation.state["state.proprioception"]),
        instruction=observation.instruction,
        image=None if image is None else np.asarray(image),
    )
