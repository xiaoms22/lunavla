from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import instruction_features
from dataset.task_context import build_pusht_task_context
from model import ActionChunk, MiniVLAPolicyBase, load_policy
from trainer.artifacts import RunManifest
from trainer.trainer_utils import (
    append_jsonl,
    ensure_dir,
    remove_stale_rollouts,
    reset_rollout_dir,
    write_json,
)


EXECUTION_MODES = {"open_loop_chunk", "receding_horizon"}
Array = npt.NDArray[np.generic]
Float32Array = npt.NDArray[np.float32]


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LunaVLA NumPy checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.json.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of eval episodes.")
    parser.add_argument("--seed", type=int, default=None, help="First evaluation seed.")
    parser.add_argument(
        "--execution-mode",
        choices=sorted(EXECUTION_MODES),
        default=None,
        help="Override the configured chunk execution mode.",
    )
    parser.add_argument("--save-rollouts", action="store_true", help="Save rollout JSON files.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    return parser.parse_args()


def make_input(
    position: Array,
    goal: Array,
    instruction: str | None,
    instruction_dim: int = 8,
) -> Float32Array:
    observation = np.asarray([position[0], position[1], goal[0], goal[1]], dtype=np.float32)
    return np.concatenate(
        [observation, instruction_features(instruction, instruction_dim)]
    ).astype(np.float32)


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
    action_clip = float(rollout.get("action_clip", 0.12))
    clipped_actions = sum(
        int(max(abs(float(value)) for value in frame["action"]) >= action_clip - 1e-6)
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
        "next_minimal_fix": "Inspect rollout coverage, model capacity, or chunk size.",
    }


def _policy_chunk(policy: Any, model_input: Array, action_dim: int) -> ActionChunk:
    if hasattr(policy, "predict_chunk"):
        chunk = policy.predict_chunk(model_input)
        if not isinstance(chunk, ActionChunk):
            raise TypeError("predict_chunk must return ActionChunk")
        if chunk.values.shape[1] != action_dim:
            raise ValueError(
                f"policy action dimension is {chunk.values.shape[1]}, expected {action_dim}"
            )
        return chunk
    # Compatibility for small v1.0 diagnostic policies outside the public contract.
    flat = np.asarray(policy.predict_action(model_input), dtype=np.float32)
    if flat.ndim != 1 or flat.size == 0 or flat.size % action_dim:
        raise ValueError("legacy predict_action output cannot be reshaped into an action chunk")
    values = flat.reshape(-1, action_dim)
    return ActionChunk(values, np.ones(len(values), dtype=bool))


def rollout_episode(
    policy: MiniVLAPolicyBase,
    seed: int,
    rollout_steps: int,
    success_distance: float,
    instruction: str | None,
    *,
    execution_mode: str = "receding_horizon",
    goal: list[float] | tuple[float, float] = (0.80, 0.20),
    start_low: float = 0.05,
    start_high: float = 0.95,
    action_clip: float = 0.12,
    instruction_dim: int = 8,
) -> dict[str, Any]:
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(f"unsupported execution_mode: {execution_mode!r}")
    if rollout_steps <= 0 or success_distance <= 0 or action_clip <= 0:
        raise ValueError("rollout_steps, success_distance, and action_clip must be positive")
    if not 0 <= start_low < start_high <= 1:
        raise ValueError("start range must satisfy 0 <= start_low < start_high <= 1")
    goal_array = np.asarray(goal, dtype=np.float32)
    if goal_array.shape != (2,) or not np.all(np.isfinite(goal_array)):
        raise ValueError("goal must contain two finite coordinates")

    rng = np.random.default_rng(seed)
    position = rng.uniform(low=start_low, high=start_high, size=2).astype(np.float32)
    initial_position = position.copy()
    initial_distance = float(np.linalg.norm(goal_array - initial_position))
    initial_context = build_pusht_task_context(
        position=initial_position,
        goal=goal_array,
        instruction=instruction,
        success_distance=success_distance,
    )
    frames: list[dict[str, Any]] = []
    while len(frames) < rollout_steps:
        model_input = make_input(position, goal_array, instruction, instruction_dim)
        chunk = _policy_chunk(policy, model_input, action_dim=2)
        action_indices = np.flatnonzero(chunk.valid_mask)
        if execution_mode == "receding_horizon":
            action_indices = action_indices[:1]
        reached_goal = False
        for chunk_index in action_indices:
            if len(frames) >= rollout_steps:
                break
            action = np.clip(chunk.values[int(chunk_index)], -action_clip, action_clip)
            position = np.clip(position + action, 0.0, 1.0)
            distance = float(np.linalg.norm(goal_array - position))
            task_context = build_pusht_task_context(
                position=position,
                goal=goal_array,
                instruction=instruction,
                success_distance=success_distance,
            )
            frames.append(
                {
                    "timestep": len(frames),
                    "chunk_index": int(chunk_index),
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
                reached_goal = True
                break
        if reached_goal:
            break

    final_distance = float(frames[-1]["distance_to_goal"])
    action_deltas = [
        float(
            np.linalg.norm(
                np.asarray(current["action"], dtype=np.float32)
                - np.asarray(previous["action"], dtype=np.float32)
            )
        )
        for previous, current in zip(frames, frames[1:])
    ]
    return {
        "success": final_distance <= success_distance,
        "task_id": initial_context.task_id,
        "instruction": instruction,
        "execution_mode": execution_mode,
        "action_clip": action_clip,
        "initial_position": [float(initial_position[0]), float(initial_position[1])],
        "goal": [float(goal_array[0]), float(goal_array[1])],
        "initial_task_context": initial_context.to_dict(),
        "final_task_context": final_task_context({"frames": frames}),
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
    requested_path = Path(args.checkpoint)
    checkpoint_path = requested_path if requested_path.is_absolute() else ROOT / requested_path
    policy, metadata = load_policy(checkpoint_path)
    if not checkpoint_path.exists() and checkpoint_path.name == "checkpoint.pt":
        checkpoint_path = checkpoint_path.with_name("checkpoint.json")
    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent)
    failure_path = output_dir / "failure_cases.jsonl"
    if failure_path.exists():
        failure_path.unlink()
    rollout_path = output_dir / "rollouts"
    if args.save_rollouts:
        rollout_dir: Path | None = reset_rollout_dir(rollout_path)
    else:
        remove_stale_rollouts(rollout_path)
        rollout_dir = None

    eval_config = dict(metadata.get("eval", {}))
    dataset_config = dict(metadata.get("dataset", {}))
    policy_config = dict(metadata.get("policy", {}))
    action_stats = dict(metadata.get("action_stats", {}))
    episodes = int(args.episodes if args.episodes is not None else eval_config.get("episodes", 10))
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    rollout_steps = int(eval_config.get("rollout_steps", 40))
    success_distance = float(eval_config.get("success_distance", 0.10))
    instruction = dataset_config.get("language_instruction")
    execution_mode = str(
        args.execution_mode or eval_config.get("execution_mode", "receding_horizon")
    )
    eval_seed = int(args.seed if args.seed is not None else eval_config.get("seed", 1000))
    configured_seeds = [int(seed) for seed in eval_config.get("seeds", [])]
    seeds = configured_seeds[:episodes]
    if len(seeds) < episodes:
        seeds.extend(range(eval_seed + len(seeds), eval_seed + episodes))

    rollouts: list[dict[str, Any]] = []
    success_count = 0
    failure_category_counts: dict[str, int] = {}
    failure_subtask_counts: dict[str, int] = {}
    goal_value = eval_config.get("goal") or dataset_config.get("goal") or [0.80, 0.20]
    start_low_value = eval_config.get("start_low")
    if start_low_value is None:
        start_low_value = dataset_config.get("start_low", 0.05)
    start_high_value = eval_config.get("start_high")
    if start_high_value is None:
        start_high_value = dataset_config.get("start_high", 0.95)
    action_clip_value = eval_config.get("action_clip")
    if action_clip_value is None:
        action_clip_value = dataset_config.get("action_clip", 0.12)
    for episode_id, seed in enumerate(seeds):
        rollout = rollout_episode(
            policy=policy,
            seed=seed,
            rollout_steps=rollout_steps,
            success_distance=success_distance,
            instruction=instruction,
            execution_mode=execution_mode,
            goal=goal_value,
            start_low=float(start_low_value),
            start_high=float(start_high_value),
            action_clip=float(action_clip_value),
            instruction_dim=int(policy_config.get("instruction_dim", 8)),
        )
        rollout["episode_id"] = episode_id
        rollout["seed"] = seed
        rollouts.append(rollout)
        success_count += int(rollout["success"])
        if not rollout["success"]:
            failure = classify_failure(rollout, success_distance)
            task_context = final_task_context(rollout)
            failure_subtask = str(task_context.get("subtask_id", "unknown"))
            failure_category_counts[failure["category"]] = (
                failure_category_counts.get(failure["category"], 0) + 1
            )
            failure_subtask_counts[failure_subtask] = (
                failure_subtask_counts.get(failure_subtask, 0) + 1
            )
            append_jsonl(
                failure_path,
                {
                    "episode_id": episode_id,
                    "seed": seed,
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
        if rollout_dir is not None:
            write_json(rollout_dir / f"episode_{episode_id:03d}.json", rollout)

    summary = {
        "checkpoint": portable_path(checkpoint_path),
        "policy_name": getattr(policy, "policy_name", "unknown"),
        "policy_interface": metadata.get("policy_interface", {}),
        "execution_mode": execution_mode,
        "eval_seeds": seeds,
        "action_stats_source": action_stats.get("source", "missing"),
        "action_stats_path": action_stats.get("path", "n/a"),
        "action_mean": action_stats.get("mean", []),
        "action_std": action_stats.get("std", []),
        "episodes": episodes,
        "success_count": success_count,
        "success_rate": round(success_count / episodes, 4),
        "mean_final_distance": round(float(np.mean([r["final_distance"] for r in rollouts])), 6),
        "mean_rollout_length": round(float(np.mean([r["steps"] for r in rollouts])), 4),
        "mean_action_smoothness": round(
            float(np.mean([r["action_smoothness"] for r in rollouts])), 6
        ),
        "failure_count": episodes - success_count,
        "failure_category_counts": failure_category_counts,
        "subtask_frame_counts": count_frame_subtasks(rollouts),
        "failure_subtask_counts": failure_subtask_counts,
        "rollout_steps": rollout_steps,
        "success_distance": success_distance,
    }
    write_json(output_dir / "eval_summary.json", summary)
    manifest_path = checkpoint_path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = RunManifest.load(manifest_path)
        manifest.add_metrics({"evaluation": summary})
        manifest.eval_seeds = seeds
        manifest.write(manifest_path)
    print(f"success_rate: {summary['success_rate']}")
    print(f"mean_final_distance: {summary['mean_final_distance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
