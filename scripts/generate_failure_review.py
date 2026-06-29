from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    "outputs/cpu_smoke",
    "outputs/act_pusht_baseline",
    "outputs/act_pusht_ablation_chunk_size",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a MiniMind-VLA failure review across public run directories.")
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS, help="Run directories to include.")
    parser.add_argument("--out", default="outputs/failure_review.md", help="Markdown report path.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


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


def category_counts(evaluation: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, int]:
    counts = dict(evaluation.get("failure_category_counts") or {})
    if counts:
        return {str(key): int(value) for key, value in counts.items()}
    for failure in failures:
        category = str(failure.get("category", "unknown"))
        counts[category] = int(counts.get(category, 0)) + 1
    return counts


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def run_summary(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    failures = read_jsonl(run_dir / "failure_cases.jsonl")
    counts = category_counts(evaluation, failures)
    row = {
        "run": run_dir.name,
        "exists": "yes" if run_dir.exists() else "no",
        "episodes": evaluation.get("episodes", "n/a"),
        "success_rate": evaluation.get("success_rate", "n/a"),
        "mean_final_distance": evaluation.get("mean_final_distance", "n/a"),
        "failure_count": evaluation.get("failure_count", len(failures) if run_dir.exists() else "n/a"),
        "failure_categories": counts,
        "diagnostic": relative(run_dir / "run_diagnostic.md"),
        "browser": relative(run_dir / "web_demo.html"),
        "project_name": training.get("project_name", run_dir.name),
    }
    return row, failures, counts


def logged_failure_rows(run_dir: Path, failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for failure in failures:
        rows.append(
            {
                "run": run_dir.name,
                "episode": failure.get("episode_id", "n/a"),
                "category": failure.get("category", "unknown"),
                "final_distance": failure.get("final_distance", "n/a"),
                "min_distance": failure.get("min_distance", "n/a"),
                "note": failure.get("note", ""),
                "next_check": failure.get("next_minimal_fix", ""),
            }
        )
    return rows


def build_report(run_dirs: list[Path]) -> str:
    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    total_counts: dict[str, int] = {}

    for run_dir in run_dirs:
        summary, failures, counts = run_summary(run_dir)
        summary_rows.append(summary)
        failure_rows.extend(logged_failure_rows(run_dir, failures))
        for category, count in counts.items():
            total_counts[category] = total_counts.get(category, 0) + int(count)

    category_rows = [
        {"category": category, "count": count}
        for category, count in sorted(total_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    if not category_rows:
        category_rows = [{"category": "none", "count": 0}]

    lines: list[str] = [
        "# MiniMind-VLA Failure Review",
        "",
        "This report collects first-pass failure evidence across the public MiniMind-VLA run directories.",
        "",
        "Failure labels are a teaching aid. Inspect saved rollouts before writing a final conclusion.",
        "",
        "## Run Summary",
        "",
    ]
    lines.extend(markdown_table(summary_rows))
    lines.extend(["", "## Category Counts", ""])
    lines.extend(markdown_table(category_rows))
    lines.extend(["", "## Logged Failure Cases", ""])
    if failure_rows:
        lines.extend(markdown_table(failure_rows))
    else:
        lines.append("No failure cases were logged in the selected runs. For a stronger report, rerun evaluation with more episodes and inspect the rollout browser.")
    lines.extend(
        [
            "",
            "## Inspection Checklist",
            "",
            "- If `wrong_direction` appears, inspect observation/action alignment and action sign.",
            "- If `stuck` appears, inspect action magnitude near the goal and training coverage.",
            "- If `oscillation` appears, compare action smoothness and chunk size.",
            "- If `action_clipping` appears, inspect action scaling and target range.",
            "- If `did_not_reach_goal` appears, inspect rollout horizon, final distance, and progress toward the goal.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_failure_review.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dirs = [resolve(path) for path in args.runs]
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(run_dirs), encoding="utf-8")
    print(f"failure review: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
