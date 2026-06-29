from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    subtask_id: str
    phase: str
    instruction: str | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_pusht_phase(distance_to_goal: float, success_distance: float = 0.10) -> str:
    if distance_to_goal <= success_distance:
        return "settle"
    if distance_to_goal <= max(success_distance * 2.2, 0.22):
        return "push_to_goal"
    if distance_to_goal <= 0.45:
        return "align_push"
    return "approach_block"


def build_pusht_task_context(
    position: list[float] | np.ndarray,
    goal: list[float] | np.ndarray,
    instruction: str | None,
    success_distance: float = 0.10,
) -> TaskContext:
    position_array = np.asarray(position, dtype=np.float32)
    goal_array = np.asarray(goal, dtype=np.float32)
    distance = float(np.linalg.norm(goal_array - position_array))
    phase = classify_pusht_phase(distance, success_distance=success_distance)
    return TaskContext(
        task_id="pusht_mock",
        subtask_id=phase,
        phase=phase,
        instruction=instruction,
        metadata={"distance_to_goal": distance, "success_distance": success_distance},
    )
