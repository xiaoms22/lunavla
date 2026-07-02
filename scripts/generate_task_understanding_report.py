from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    "outputs/cpu_smoke",
    "outputs/bc_pusht_cpu_smoke",
    "outputs/act_pusht_baseline",
    "outputs/act_pusht_jsonl_noisy_smoke",
]
PHASE_ORDER = {
    "approach_block": 0,
    "align_push": 1,
    "push_to_goal": 2,
    "settle": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect rollout task phases and first-pass failure labels.")
    parser.add_argument("--run-dir", default=None, help="Single run directory with saved rollout JSON files.")
    parser.add_argument("--runs", nargs="*", default=None, help="Run directories with saved rollout JSON files.")
    parser.add_argument("--out", default="outputs/task_understanding_report.md", help="Markdown report output.")
    parser.add_argument("--csv", default="outputs/task_understanding_report.csv", help="Machine-readable rollout table.")
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
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def rollout_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "rollouts").glob("episode_*.json"))


def frame_phase(frame: dict[str, Any]) -> str:
    context = frame.get("task_context", {})
    return str(frame.get("phase") or context.get("phase") or context.get("subtask_id") or "unknown")


def compressed_path(phases: list[str]) -> list[str]:
    compact: list[str] = []
    for phase in phases:
        if not compact or compact[-1] != phase:
            compact.append(phase)
    return compact


def count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def most_common(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    key, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return f"{key} ({count})"


def first_pass_label(success: bool, phases: list[str], distances: list[float]) -> str:
    if success:
        return "success"
    if not phases or not distances:
        return "missing_trace"

    orders = [PHASE_ORDER.get(phase, -1) for phase in phases]
    max_order = max(orders)
    final_order = orders[-1]
    final_distance = distances[-1]
    min_distance = min(distances)

    if max_order > final_order or final_distance > min_distance + 0.04:
        return "phase_regression"
    if phases[-1] == "approach_block":
        return "failed_in_approach"
    if phases[-1] == "align_push":
        return "failed_in_align"
    if phases[-1] == "push_to_goal":
        return "failed_near_goal"
    return "unlabeled_failure"


def row_from_rollout(run_dir: Path, path: Path) -> dict[str, Any]:
    rollout = read_json(path)
    frames = rollout.get("frames", [])
    phases = [frame_phase(frame) for frame in frames]
    distances = [float(frame.get("distance_to_goal", 0.0)) for frame in frames]
    counts = count_values(phases)
    success = bool(rollout.get("success", False))
    final_phase = phases[-1] if phases else str(rollout.get("final_task_context", {}).get("phase", "unknown"))
    return {
        "run": run_dir.name,
        "episode": rollout.get("episode_id", path.stem.replace("episode_", "")),
        "success": success,
        "frames": len(frames),
        "initial_phase": phases[0] if phases else "unknown",
        "final_phase": final_phase,
        "most_seen_phase": most_common(counts),
        "phase_path": " -> ".join(compressed_path(phases)) if phases else "missing",
        "initial_distance": rollout.get("initial_distance", "n/a"),
        "min_distance": min(distances) if distances else rollout.get("min_distance", "n/a"),
        "final_distance": distances[-1] if distances else rollout.get("final_distance", "n/a"),
        "task_understanding_label": first_pass_label(success, phases, distances),
        "source": relative(path),
    }


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
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


def count_rows(rows: list[dict[str, Any]], field: str, failures_only: bool = False) -> list[dict[str, Any]]:
    values: list[str] = []
    for row in rows:
        if failures_only and bool(row.get("success")):
            continue
        values.append(str(row.get(field, "unknown")))
    counts = count_values(values)
    return [
        {field: key, "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ] or [{field: "none", "count": 0}]


def source_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        files = rollout_files(run_dir)
        eval_summary = read_json(run_dir / "eval_summary.json")
        rows.append(
            {
                "run": run_dir.name,
                "exists": run_dir.exists(),
                "saved_rollouts": len(files),
                "success_rate": eval_summary.get("success_rate", "n/a"),
                "failure_count": eval_summary.get("failure_count", "n/a"),
            }
        )
    return rows


def build_report(run_dirs: list[Path], rows: list[dict[str, Any]], csv_path: Path) -> str:
    failure_phase_rows = count_rows(rows, "final_phase", failures_only=True)
    label_rows = count_rows(rows, "task_understanding_label")
    frame_phase_counts: dict[str, int] = {}
    for row in rows:
        phase = str(row.get("most_seen_phase", "unknown")).split(" (", 1)[0]
        frame_phase_counts[phase] = frame_phase_counts.get(phase, 0) + 1
    frame_phase_rows = [
        {"dominant_phase": key, "episodes": count}
        for key, count in sorted(frame_phase_counts.items(), key=lambda item: (-item[1], item[0]))
    ] or [{"dominant_phase": "none", "episodes": 0}]

    lines: list[str] = [
        "# LunaVLA Task Understanding Report",
        "",
        "This report reads saved rollout JSON files and summarizes how phase/subtask labels appear over time.",
        "",
        "It keeps the Task Layer rule-based. The first-pass label `phase_regression` is assigned only from saved rollout traces: a failed episode reached a later/closer phase or distance, then ended worse.",
        "",
        "## Source",
        "",
        f"- Saved rollout files: `{len(rows)}`",
        f"- CSV: `{relative(csv_path)}`",
        "",
    ]
    lines.extend(markdown_table(source_rows(run_dirs)))
    lines.extend(
        [
            "",
            "Missing or zero-rollout runs are skipped in the rollout table. Run the matching evidence command first if you want them included.",
            "",
        ]
    )
    lines.extend(
        [
            "## Failed Final Phase Counts",
            "",
        ]
    )
    lines.extend(markdown_table(failure_phase_rows))
    lines.extend(["", "## First-Pass Label Counts", ""])
    lines.extend(markdown_table(label_rows))
    lines.extend(["", "## Dominant Phase Per Episode", ""])
    lines.extend(markdown_table(frame_phase_rows))
    lines.extend(["", "## Rollout Trace Table", ""])
    lines.extend(markdown_table(rows))
    lines.extend(
        [
            "",
            "## How To Use This",
            "",
            "- If failures end in `approach_block`, improve early movement and initial-state coverage.",
            "- If failures end in `align_push`, inspect whether the policy approaches the goal with the right direction.",
            "- If failures end in `push_to_goal`, inspect action smoothness and whether the policy settles near the target.",
            "- If `phase_regression` appears, inspect the saved rollout frames before tuning; the policy may get close and then drift away.",
            "- Do not turn these labels into hard benchmark claims. They are first-pass debugging evidence for a teaching-scale loop.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_task_understanding_report.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.runs is not None:
        run_dirs = [resolve(path) for path in args.runs]
    elif args.run_dir is not None:
        run_dirs = [resolve(args.run_dir)]
    else:
        run_dirs = [resolve(path) for path in DEFAULT_RUNS]

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.extend(row_from_rollout(run_dir, path) for path in rollout_files(run_dir))
    if not rows:
        searched = ", ".join(relative(run_dir / "rollouts") for run_dir in run_dirs)
        raise FileNotFoundError(f"No saved rollout JSON files found under: {searched}")
    out_path = resolve(args.out)
    csv_path = resolve(args.csv)
    write_csv(csv_path, rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(run_dirs, rows, csv_path), encoding="utf-8")
    print(f"task understanding report: {out_path}")
    print(f"task understanding csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
