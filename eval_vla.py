from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.pusht_dataset import _instruction_features
from dataset.task_context import build_pusht_task_context
from model import MiniVLAPolicy
from trainer.trainer_utils import append_jsonl, ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LunaVLA checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of eval episodes.")
    parser.add_argument("--save-rollouts", action="store_true", help="Save rollout JSON files.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    return parser.parse_args()


def make_input(position: np.ndarray, goal: np.ndarray, instruction: str | None) -> np.ndarray:
    observation = np.asarray([position[0], position[1], goal[0], goal[1]], dtype=np.float32)
    return np.concatenate([observation, _instruction_features(instruction)]).astype(np.float32)


def mean_action_norm(frames: list[dict[str, Any]]) -> float:
    if not frames:
        return 0.0
    norms = [float(np.linalg.norm(np.asarray(frame["action"], dtype=np.float32))) for frame in frames]
    return float(np.mean(norms))


def count_frame_subtasks(rollouts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rollout in rollouts:
        for frame in rollout.get("frames", []):
            context = frame.get("task_context", {})
            subtask_id = str(context.get("subtask_id", "unknown"))
            counts[subtask_id] = counts.get(subtask_id, 0) + 1
    return counts


def final_task_context(rollout: dict[str, Any]) -> dict[str, Any]:
    frames = rollout.get("frames", [])
    if not frames:
        return {}
    return dict(frames[-1].get("task_context", {}))


def classify_failure(rollout: dict[str, Any], success_distance: float) -> dict[str, Any]:
    frames = rollout.get("frames", [])
    distances = [float(frame["distance_to_goal"]) for frame in frames]
    if not distances:
        return {
            "category": "did_not_reach_goal",
            "note": "No rollout frames were recorded.",
            "next_minimal_fix": "Check evaluation rollout_steps and checkpoint loading.",
        }

    initial_distance = float(rollout.get("initial_distance", distances[0]))
    final_distance = float(rollout["final_distance"])
    min_distance = min(distances)
    early_window = distances[: min(3, len(distances))]
    early_distance_increase = bool(early_window and early_window[-1] > initial_distance + 0.02)
    last_action_norm = mean_action_norm(frames[-min(5, len(frames)) :])
    clipped_actions = sum(
        int(max(abs(float(value)) for value in frame["action"]) >= 0.119)
        for frame in frames
    )
    clipped_ratio = clipped_actions / max(len(frames), 1)

    if early_distance_increase or final_distance > initial_distance + 0.02:
        return {
            "category": "wrong_direction",
            "note": "The rollout moved away from the goal instead of closing distance.",
            "next_minimal_fix": "Inspect observation/action alignment and the sign of action targets.",
        }
    if last_action_norm < 0.015 and final_distance > success_distance:
        return {
            "category": "stuck",
            "note": "The policy produced very small late actions before reaching the goal.",
            "next_minimal_fix": "Check demonstration coverage near the goal and whether the model underfits.",
        }
    if min_distance <= success_distance * 1.8 and rollout["action_smoothness"] > 0.04:
        return {
            "category": "oscillation",
            "note": "The rollout approached the goal but did not settle smoothly.",
            "next_minimal_fix": "Compare action smoothness and try a different action chunk size.",
        }
    if clipped_ratio > 0.5:
        return {
            "category": "action_clipping",
            "note": "Most actions hit the evaluation clip limit.",
            "next_minimal_fix": "Inspect action scaling and the range of demonstration actions.",
        }
    return {
        "category": "did_not_reach_goal",
        "note": "The rollout reduced distance but did not cross the success threshold in time.",
        "next_minimal_fix": "Increase eval episodes, inspect rollout horizon, then tune model capacity or chunk size.",
    }


def rollout_episode(
    policy: MiniVLAPolicy,
    seed: int,
    rollout_steps: int,
    success_distance: float,
    instruction: str | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    goal = np.array([0.80, 0.20], dtype=np.float32)
    position = rng.uniform(low=0.05, high=0.95, size=2).astype(np.float32)
    initial_position = position.copy()
    initial_distance = float(np.linalg.norm(goal - initial_position))
    initial_context = build_pusht_task_context(
        position=initial_position,
        goal=goal,
        instruction=instruction,
        success_distance=success_distance,
    )
    frames: list[dict[str, Any]] = []

    for timestep in range(rollout_steps):
        model_input = make_input(position, goal, instruction)
        action_chunk = policy.predict(model_input)[0]
        action = np.clip(action_chunk[:2], -0.12, 0.12)
        position = np.clip(position + action, 0.0, 1.0)
        distance = float(np.linalg.norm(goal - position))
        task_context = build_pusht_task_context(
            position=position,
            goal=goal,
            instruction=instruction,
            success_distance=success_distance,
        )
        frames.append(
            {
                "timestep": timestep,
                "position": [float(position[0]), float(position[1])],
                "action": [float(action[0]), float(action[1])],
                "distance_to_goal": distance,
                "task_context": task_context.to_dict(),
                "task_id": task_context.task_id,
                "subtask_id": task_context.subtask_id,
                "phase": task_context.phase,
            }
        )
        if distance <= success_distance:
            break

    final_distance = frames[-1]["distance_to_goal"]
    final_context = final_task_context({"frames": frames})
    action_deltas = []
    for prev, cur in zip(frames, frames[1:]):
        prev_action = np.asarray(prev["action"], dtype=np.float32)
        cur_action = np.asarray(cur["action"], dtype=np.float32)
        action_deltas.append(float(np.linalg.norm(cur_action - prev_action)))
    return {
        "success": final_distance <= success_distance,
        "task_id": initial_context.task_id,
        "instruction": instruction,
        "initial_position": [float(initial_position[0]), float(initial_position[1])],
        "goal": [float(goal[0]), float(goal[1])],
        "initial_task_context": initial_context.to_dict(),
        "final_task_context": final_context,
        "initial_distance": initial_distance,
        "min_distance": min(float(frame["distance_to_goal"]) for frame in frames),
        "final_distance": final_distance,
        "steps": len(frames),
        "action_smoothness": float(np.mean(action_deltas)) if action_deltas else 0.0,
        "subtask_frame_counts": count_frame_subtasks([{"frames": frames}]),
        "frames": frames,
    }


def main() -> int:
    args = parse_args()
    checkpoint_path = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    policy, metadata = MiniVLAPolicy.load(checkpoint_path)
    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent)
    failure_path = output_dir / "failure_cases.jsonl"
    if failure_path.exists():
        failure_path.unlink()

    eval_config = metadata.get("eval", {})
    dataset_config = metadata.get("dataset", {})
    episodes = int(args.episodes or eval_config.get("episodes", 10))
    rollout_steps = int(eval_config.get("rollout_steps", 40))
    success_distance = float(eval_config.get("success_distance", 0.10))
    instruction = dataset_config.get("language_instruction")

    rollouts: list[dict[str, Any]] = []
    success_count = 0
    failure_category_counts: dict[str, int] = {}
    failure_subtask_counts: dict[str, int] = {}
    for episode_id in range(episodes):
        rollout = rollout_episode(
            policy=policy,
            seed=1000 + episode_id,
            rollout_steps=rollout_steps,
            success_distance=success_distance,
            instruction=instruction,
        )
        rollout["episode_id"] = episode_id
        rollouts.append(rollout)
        success_count += int(rollout["success"])
        if not rollout["success"]:
            failure = classify_failure(rollout, success_distance)
            task_context = final_task_context(rollout)
            failure_subtask = str(task_context.get("subtask_id", "unknown"))
            failure_category_counts[failure["category"]] = failure_category_counts.get(failure["category"], 0) + 1
            failure_subtask_counts[failure_subtask] = failure_subtask_counts.get(failure_subtask, 0) + 1
            append_jsonl(
                failure_path,
                {
                    "episode_id": episode_id,
                    "task_id": task_context.get("task_id", "unknown"),
                    "subtask_id": failure_subtask,
                    "phase": task_context.get("phase", "unknown"),
                    "final_distance": rollout["final_distance"],
                    "initial_distance": rollout["initial_distance"],
                    "min_distance": rollout["min_distance"],
                    "steps": rollout["steps"],
                    "action_smoothness": rollout["action_smoothness"],
                    **failure,
                },
            )
        if args.save_rollouts:
            rollout_dir = ensure_dir(output_dir / "rollouts")
            write_json(rollout_dir / f"episode_{episode_id:03d}.json", rollout)

    summary = {
        "checkpoint": str(checkpoint_path),
        "episodes": episodes,
        "success_count": success_count,
        "success_rate": round(success_count / max(episodes, 1), 4),
        "mean_final_distance": round(float(np.mean([r["final_distance"] for r in rollouts])), 6),
        "mean_rollout_length": round(float(np.mean([r["steps"] for r in rollouts])), 4),
        "mean_action_smoothness": round(float(np.mean([r["action_smoothness"] for r in rollouts])), 6),
        "failure_count": episodes - success_count,
        "failure_category_counts": failure_category_counts,
        "subtask_frame_counts": count_frame_subtasks(rollouts),
        "failure_subtask_counts": failure_subtask_counts,
        "rollout_steps": rollout_steps,
        "success_distance": success_distance,
    }
    write_json(output_dir / "eval_summary.json", summary)
    print(f"success_rate: {summary['success_rate']}")
    print(f"mean_final_distance: {summary['mean_final_distance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
