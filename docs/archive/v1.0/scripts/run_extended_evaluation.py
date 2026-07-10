from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = [
    "outputs/cpu_smoke/checkpoint.pt",
    "outputs/bc_pusht_cpu_smoke/checkpoint.pt",
    "outputs/act_pusht_baseline/checkpoint.pt",
    "outputs/act_pusht_jsonl_noisy_smoke/checkpoint.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extended LunaVLA rollout evaluation across checkpoints.")
    parser.add_argument("--checkpoints", nargs="*", default=DEFAULT_CHECKPOINTS, help="Checkpoint paths to evaluate.")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes per checkpoint.")
    parser.add_argument("--out-dir", default="outputs/extended_evaluation", help="Directory for extended eval runs.")
    parser.add_argument("--report", default="outputs/extended_evaluation_report.md", help="Markdown report output.")
    parser.add_argument("--csv", default="outputs/extended_evaluation_report.csv", help="Machine-readable report output.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def rollout_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "rollouts").glob("episode_*.json"))


def clean_run_dir(run_dir: Path, root: Path) -> None:
    root = root.resolve()
    run_dir = run_dir.resolve()
    if run_dir.exists():
        if root not in run_dir.parents and run_dir != root:
            raise ValueError(f"refusing to remove output outside {root}: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def run_name_from_checkpoint(checkpoint: Path) -> str:
    return checkpoint.parent.name


def example_rollout(run_dir: Path, success: bool) -> str:
    for path in rollout_files(run_dir):
        rollout = read_json(path)
        if bool(rollout.get("success")) == success:
            final_distance = rollout.get("final_distance", "n/a")
            steps = rollout.get("steps", "n/a")
            return f"{relative(path)} (final_distance={final_distance}, steps={steps})"
    return "none"


def conclusion(summary: dict[str, Any]) -> str:
    success_rate = float(summary.get("success_rate", 0.0))
    mean_final_distance = float(summary.get("mean_final_distance", 999.0))
    failures = int(summary.get("failure_count", 0))
    if success_rate >= 0.8 and mean_final_distance <= 0.14:
        return "strong for this teaching eval; still inspect saved rollouts before claiming robustness"
    if success_rate >= 0.5:
        return "mixed result; compare success rate with final distance and inspect failure cases"
    if failures > 0 and mean_final_distance <= 0.20:
        return "often gets near the goal but does not reliably satisfy the success threshold"
    return "weak extended eval; inspect failure examples before tuning policy or data"


def row_from_run(checkpoint: Path, run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "eval_summary.json")
    return {
        "run": run_name_from_checkpoint(checkpoint),
        "checkpoint": relative(checkpoint),
        "extended_eval_dir": relative(run_dir),
        "episodes": summary.get("episodes", "n/a"),
        "success_rate": summary.get("success_rate", "n/a"),
        "mean_final_distance": summary.get("mean_final_distance", "n/a"),
        "mean_rollout_length": summary.get("mean_rollout_length", "n/a"),
        "mean_action_smoothness": summary.get("mean_action_smoothness", "n/a"),
        "failure_count": summary.get("failure_count", "n/a"),
        "failure_categories": summary.get("failure_category_counts", {}),
        "failure_subtasks": summary.get("failure_subtask_counts", {}),
        "saved_rollouts": len(rollout_files(run_dir)),
        "success_example": example_rollout(run_dir, True),
        "failure_example": example_rollout(run_dir, False),
        "conclusion": conclusion(summary),
    }


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{key}:{item}" for key, item in sorted(value.items()))
    return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows: list[dict[str, Any]], csv_path: Path) -> str:
    lines: list[str] = [
        "# LunaVLA Extended Evaluation Report",
        "",
        "This report reruns rollout evaluation with more episodes and saved rollout JSON.",
        "",
        "It is designed for beginner project evidence: compare success rate with mean final distance, then inspect at least one success and one failure before writing a conclusion.",
        "",
        "## Results",
        "",
    ]
    lines.extend(markdown_table(rows))
    lines.extend(
        [
            "",
            f"CSV: `{relative(csv_path)}`",
            "",
            "## How To Read It",
            "",
            "- `success_rate` tells how often the rollout crosses the success threshold.",
            "- `mean_final_distance` shows whether failed runs still move toward the goal.",
            "- `failure_example` should be opened before changing data, policy capacity, or chunk size.",
            "- `success_example` helps avoid writing conclusions from failure cases only.",
            "- Treat this as teaching-scale evidence, not a real-robot benchmark.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/run_extended_evaluation.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_root = resolve(args.out_dir)
    rows: list[dict[str, Any]] = []
    for checkpoint_arg in args.checkpoints:
        checkpoint = resolve(checkpoint_arg)
        if not checkpoint.exists():
            print(f"skip missing checkpoint: {relative(checkpoint)}")
            continue
        run_name = f"{run_name_from_checkpoint(checkpoint)}_{args.episodes}ep"
        run_dir = output_root / run_name
        clean_run_dir(run_dir, output_root)
        run(
            [
                sys.executable,
                "eval_vla.py",
                "--checkpoint",
                relative(checkpoint),
                "--episodes",
                str(args.episodes),
                "--save-rollouts",
                "--output-dir",
                relative(run_dir),
            ]
        )
        rows.append(row_from_run(checkpoint, run_dir))

    if not rows:
        raise FileNotFoundError("No checkpoints were found for extended evaluation.")
    report_path = resolve(args.report)
    csv_path = resolve(args.csv)
    write_csv(csv_path, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(rows, csv_path), encoding="utf-8")
    print(f"extended evaluation report: {report_path}")
    print(f"extended evaluation csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
