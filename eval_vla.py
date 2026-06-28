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
from model import MiniVLAPolicy
from trainer.trainer_utils import append_jsonl, ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a MiniMind-VLA checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of eval episodes.")
    parser.add_argument("--save-rollouts", action="store_true", help="Save rollout JSON files.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    return parser.parse_args()


def make_input(position: np.ndarray, goal: np.ndarray, instruction: str | None) -> np.ndarray:
    observation = np.asarray([position[0], position[1], goal[0], goal[1]], dtype=np.float32)
    return np.concatenate([observation, _instruction_features(instruction)]).astype(np.float32)


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
    frames: list[dict[str, Any]] = []

    for timestep in range(rollout_steps):
        model_input = make_input(position, goal, instruction)
        action_chunk = policy.predict(model_input)[0]
        action = np.clip(action_chunk[:2], -0.12, 0.12)
        position = np.clip(position + action, 0.0, 1.0)
        distance = float(np.linalg.norm(goal - position))
        frames.append(
            {
                "timestep": timestep,
                "position": [float(position[0]), float(position[1])],
                "action": [float(action[0]), float(action[1])],
                "distance_to_goal": distance,
            }
        )
        if distance <= success_distance:
            break

    final_distance = frames[-1]["distance_to_goal"]
    action_deltas = []
    for prev, cur in zip(frames, frames[1:]):
        prev_action = np.asarray(prev["action"], dtype=np.float32)
        cur_action = np.asarray(cur["action"], dtype=np.float32)
        action_deltas.append(float(np.linalg.norm(cur_action - prev_action)))
    return {
        "success": final_distance <= success_distance,
        "final_distance": final_distance,
        "steps": len(frames),
        "action_smoothness": float(np.mean(action_deltas)) if action_deltas else 0.0,
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
            append_jsonl(
                failure_path,
                {
                    "episode_id": episode_id,
                    "final_distance": rollout["final_distance"],
                    "category": "did_not_reach_goal",
                    "note": "The predicted first action chunk did not close the distance enough.",
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
        "rollout_steps": rollout_steps,
        "success_distance": success_distance,
    }
    write_json(output_dir / "eval_summary.json", summary)
    print(f"success_rate: {summary['success_rate']}")
    print(f"mean_final_distance: {summary['mean_final_distance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
