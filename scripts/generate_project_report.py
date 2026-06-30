from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a beginner-friendly LunaVLA project report.")
    parser.add_argument("--run-dir", required=True, help="Run directory under outputs/ or an absolute path.")
    parser.add_argument("--out", default=None, help="Markdown output path. Defaults to <run-dir>/project_report.md.")
    parser.add_argument("--title", default="LunaVLA Project Report", help="Report title.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{key}:{value}" for key, value in sorted(value.items()))
    return str(value)


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def first_rollout_path(run_dir: Path) -> Path | None:
    rollout_dir = run_dir / "rollouts"
    files = sorted(rollout_dir.glob("episode_*.json"))
    return files[0] if files else None


def metric_table(rows: list[tuple[str, Any]]) -> list[str]:
    lines = ["| metric | value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | `{format_value(value)}` |")
    return lines


def failure_table(failures: list[dict[str, Any]]) -> list[str]:
    if not failures:
        return [
            "No failure cases were logged for this run. For a stronger report, evaluate more episodes and inspect whether rare failures appear.",
        ]

    lines = [
        "| episode | subtask | category | final distance | note | next minimal fix |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for failure in failures[:5]:
        lines.append(
            "| "
            + " | ".join(
                [
                    format_value(failure.get("episode_id")),
                    format_value(failure.get("subtask_id", "unknown")),
                    format_value(failure.get("category", "unknown")),
                    format_value(failure.get("final_distance")),
                    format_value(failure.get("note", "")),
                    format_value(failure.get("next_minimal_fix", "")),
                ]
            )
            + " |"
        )
    if len(failures) > 5:
        lines.append(f"\nShowing 5 of {len(failures)} logged failures.")
    return lines


def failure_category_table(evaluation: dict[str, Any], failures: list[dict[str, Any]]) -> list[str]:
    counts = evaluation.get("failure_category_counts") or {}
    if not counts and failures:
        for failure in failures:
            category = str(failure.get("category", "unknown"))
            counts[category] = counts.get(category, 0) + 1
    if not counts:
        return ["No failure category counts were recorded for this run."]

    rows = ["| category | count |", "| --- | --- |"]
    for category, count in sorted(counts.items()):
        rows.append(f"| `{category}` | `{count}` |")
    return rows


def build_report(run_dir: Path, title: str) -> str:
    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    failures = read_jsonl(run_dir / "failure_cases.jsonl")
    rollout_path = first_rollout_path(run_dir)

    project_name = training.get("project_name", run_dir.name)
    success_rate = evaluation.get("success_rate", "n/a")
    final_loss = training.get("final_loss", "n/a")
    mean_distance = evaluation.get("mean_final_distance", "n/a")

    lines: list[str] = [
        f"# {title}",
        "",
        "## Abstract",
        "",
        (
            f"This report summarizes `{project_name}`, a small ACT-style imitation-learning run for a "
            "PushT-style observation-to-action task. The run trains from generated demonstrations, "
            "evaluates behavior through rollouts, and records metrics that can be inspected before making "
            "any resume or interview claim."
        ),
        "",
        "## Experiment Setup",
        "",
    ]
    lines.extend(
        metric_table(
            [
                ("run directory", relative(run_dir)),
                ("policy name", training.get("policy_name", evaluation.get("policy_name", "n/a"))),
                ("policy interface", training.get("policy_interface", "n/a")),
                ("checkpoint", training.get("checkpoint", "n/a")),
                ("records", training.get("records", "n/a")),
                ("input dim", training.get("input_dim", "n/a")),
                ("target dim", training.get("target_dim", "n/a")),
                ("chunk size", training.get("chunk_size", "n/a")),
                ("training steps", training.get("num_steps", "n/a")),
                ("eval episodes", evaluation.get("episodes", "n/a")),
                ("rollout steps", evaluation.get("rollout_steps", "n/a")),
                ("success distance", evaluation.get("success_distance", "n/a")),
            ]
        )
    )

    lines.extend(
        [
            "",
            "## Results",
            "",
        ]
    )
    lines.extend(
        metric_table(
            [
                ("final loss", final_loss),
                ("success rate", success_rate),
                ("success count", evaluation.get("success_count", "n/a")),
                ("mean final distance", mean_distance),
                ("mean rollout length", evaluation.get("mean_rollout_length", "n/a")),
                ("mean action smoothness", evaluation.get("mean_action_smoothness", "n/a")),
                ("failure cases", len(failures)),
                ("failure subtasks", evaluation.get("failure_subtask_counts", {})),
            ]
        )
    )

    lines.extend(
        [
            "",
            "## Rollout Evidence",
            "",
            f"- Loss curve CSV: `{relative(run_dir / 'loss_curve.csv')}`",
            f"- Eval summary JSON: `{relative(run_dir / 'eval_summary.json')}`",
            f"- Static rollout browser: `{relative(run_dir / 'web_demo.html')}`",
            f"- First saved rollout: `{relative(rollout_path) if rollout_path else 'run eval with --save-rollouts'}`",
            "",
            "## Failure Analysis",
            "",
            "### Category Summary",
            "",
        ]
    )
    lines.extend(failure_category_table(evaluation, failures))
    lines.extend(["", "### Subtask Summary", ""])
    lines.extend(
        metric_table(
            [
                ("subtask frame counts", evaluation.get("subtask_frame_counts", {})),
                ("failure subtask counts", evaluation.get("failure_subtask_counts", {})),
            ]
        )
    )
    lines.extend(
        [
            "",
            "### Logged Cases",
            "",
        ]
    )
    lines.extend(failure_table(failures))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"The final training loss is `{format_value(final_loss)}`, while rollout success rate is "
                f"`{format_value(success_rate)}` with mean final distance `{format_value(mean_distance)}`. "
                "Training loss checks whether the imitation objective fits the demonstrations; rollout metrics "
                "check whether repeated predicted actions actually move the state toward the goal."
            ),
            "",
            "## Resume-Safe Claim",
            "",
            (
                "A safe claim should mention only the verified loop: generated demonstration data, trained an "
                "ACT-style action-chunk policy, evaluated saved rollouts, and reported success rate, distance, "
                "smoothness, and failure cases."
            ),
            "",
            "## Honest Boundaries",
            "",
            "- This run uses a teaching-scale PushT-style mock environment.",
            "- It is useful for learning the data, policy, rollout, evaluation, and reporting loop.",
            "- It does not demonstrate real-robot deployment or state-of-the-art robotics performance.",
            "",
            "## Next Reproducible Step",
            "",
            "Run the chunk-size ablation and compare the generated report against this baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    out_path = resolve(args.out) if args.out else run_dir / "project_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(run_dir, args.title), encoding="utf-8")
    print(f"project report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
