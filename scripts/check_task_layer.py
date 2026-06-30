from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import load_dataset_from_config
from eval_vla import rollout_episode
from trainer.trainer_utils import load_yaml


EXPECTED_CONTEXT_KEYS = {"task_id", "subtask_id", "phase", "instruction"}


class GoalSeekingPolicy:
    def predict(self, model_input: np.ndarray) -> np.ndarray:
        position = np.asarray(model_input[:2], dtype=np.float32)
        goal = np.asarray(model_input[2:4], dtype=np.float32)
        action = np.clip((goal - position) * 0.25, -0.08, 0.08)
        return action.reshape(1, 2).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LunaVLA Task Layer records, rollouts, and reports.")
    parser.add_argument("--config", default="configs/act_pusht_cpu_smoke.yaml", help="Config used for source-level checks.")
    parser.add_argument("--run-dir", default="outputs/cpu_smoke", help="Generated run directory to inspect.")
    parser.add_argument(
        "--require-generated",
        action="store_true",
        help="Fail when generated run artifacts are missing instead of only checking source behavior.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"task layer check failed: {message}")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def require_context(context: dict[str, Any], label: str) -> None:
    missing = sorted(EXPECTED_CONTEXT_KEYS - set(context))
    require(not missing, f"{label} is missing task context keys: {', '.join(missing)}")
    require(str(context.get("task_id", "unknown")) != "unknown", f"{label} has unknown task_id")
    require(str(context.get("subtask_id", "unknown")) != "unknown", f"{label} has unknown subtask_id")
    require(str(context.get("phase", "unknown")) != "unknown", f"{label} has unknown phase")


def check_dataset_records(config_path: Path) -> None:
    config = load_yaml(config_path)
    records = load_dataset_from_config(config["dataset"])
    require(bool(records), f"{config_path} produced no records")

    phases = {record.phase for record in records}
    require("unknown" not in phases, "generated records include unknown phase")
    require(any(record.subtask_id == record.phase for record in records), "records do not expose subtask_id/phase")
    for idx, record in enumerate(records[:10]):
        require(record.task_id != "unknown", f"record {idx} has unknown task_id")
        require(record.subtask_id != "unknown", f"record {idx} has unknown subtask_id")
        require(record.phase != "unknown", f"record {idx} has unknown phase")
        context = record.metadata.get("task_context")
        require(isinstance(context, dict), f"record {idx} metadata lacks task_context")
        require_context(context, f"record {idx} metadata.task_context")


def check_rollout_source(config_path: Path) -> None:
    config = load_yaml(config_path)
    dataset_config = config.get("dataset", {})
    eval_config = config.get("eval", {})
    rollout = rollout_episode(
        policy=GoalSeekingPolicy(),
        seed=123,
        rollout_steps=int(eval_config.get("rollout_steps", 8)),
        success_distance=float(eval_config.get("success_distance", 0.10)),
        instruction=dataset_config.get("language_instruction"),
    )
    require_context(rollout.get("initial_task_context", {}), "rollout initial_task_context")
    require_context(rollout.get("final_task_context", {}), "rollout final_task_context")
    require(isinstance(rollout.get("subtask_frame_counts"), dict), "rollout lacks subtask_frame_counts")
    frames = rollout.get("frames", [])
    require(bool(frames), "rollout produced no frames")
    for idx, frame in enumerate(frames[:5]):
        context = frame.get("task_context")
        require(isinstance(context, dict), f"frame {idx} lacks task_context")
        require_context(context, f"frame {idx} task_context")
        for key in ["task_id", "subtask_id", "phase"]:
            require(frame.get(key) == context.get(key), f"frame {idx} {key} does not mirror task_context")


def require_text(path: Path, phrases: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for phrase in phrases:
        require(phrase in text, f"{path.relative_to(ROOT).as_posix()} is missing `{phrase}`")


def first_rollout_file(run_dir: Path) -> Path | None:
    rollout_dir = run_dir / "rollouts"
    files = sorted(rollout_dir.glob("episode_*.json"))
    return files[0] if files else None


def check_generated_run(run_dir: Path, require_generated: bool) -> None:
    required = [
        run_dir / "eval_summary.json",
        run_dir / "summary_report.md",
        run_dir / "project_report.md",
        run_dir / "web_demo.html",
    ]
    missing = [path for path in required if not path.exists()]
    if missing and require_generated:
        fail("missing generated task-layer artifacts: " + ", ".join(path.relative_to(ROOT).as_posix() for path in missing))
    if missing:
        print("generated task-layer artifacts not found; source-level checks passed")
        return

    summary = read_json(run_dir / "eval_summary.json")
    require(isinstance(summary.get("subtask_frame_counts"), dict), "eval_summary.json lacks subtask_frame_counts")
    require(bool(summary.get("subtask_frame_counts")), "eval_summary.json has empty subtask_frame_counts")
    require("failure_subtask_counts" in summary, "eval_summary.json lacks failure_subtask_counts")
    require(isinstance(summary.get("failure_subtask_counts"), dict), "failure_subtask_counts should be a dict")

    rollout_path = first_rollout_file(run_dir)
    if rollout_path is not None:
        rollout = read_json(rollout_path)
        require_context(rollout.get("initial_task_context", {}), f"{rollout_path.name} initial_task_context")
        require_context(rollout.get("final_task_context", {}), f"{rollout_path.name} final_task_context")
        frames = rollout.get("frames", [])
        require(bool(frames), f"{rollout_path.name} has no frames")
        require_context(frames[0].get("task_context", {}), f"{rollout_path.name} first frame task_context")
    elif require_generated:
        fail(f"{run_dir.relative_to(ROOT).as_posix()} has no saved rollout JSON")

    require_text(run_dir / "summary_report.md", ["failure_subtasks", "subtask_frames"])
    require_text(run_dir / "project_report.md", ["Subtask Summary", "failure subtask counts"])
    require_text(run_dir / "web_demo.html", ["final subtask"])


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    run_dir = resolve(args.run_dir)
    check_dataset_records(config_path)
    check_rollout_source(config_path)
    check_generated_run(run_dir, require_generated=args.require_generated)
    print("task layer check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
