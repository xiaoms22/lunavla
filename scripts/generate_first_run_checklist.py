from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a first-run checklist after the CPU smoke path.")
    parser.add_argument("--run-dir", default="outputs/cpu_smoke", help="CPU smoke run directory.")
    parser.add_argument("--out", default="outputs/first_run_checklist.md", help="Markdown checklist path.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any core first-run artifact is missing.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def core_artifacts(run_dir: Path) -> list[tuple[str, str, str]]:
    return [
        ("environment", "outputs/environment_check.md", "Python, dependency, path, and write checks"),
        ("checkpoint", relative(run_dir / "checkpoint.pt"), "saved tiny policy weights"),
        ("training", relative(run_dir / "training_summary.json"), "training config and final loss"),
        ("evaluation", relative(run_dir / "eval_summary.json"), "rollout metrics"),
        ("summary", relative(run_dir / "summary_report.md"), "human-readable metric summary"),
        ("report", relative(run_dir / "project_report.md"), "smallest project report"),
        ("diagnostic", relative(run_dir / "run_diagnostic.md"), "claim-safety notes"),
        ("rollout browser", relative(run_dir / "web_demo.html"), "static rollout viewer"),
    ]


def optional_artifacts(run_dir: Path) -> list[tuple[str, str, str]]:
    return [
        ("dataset inspection", "outputs/dataset_inspection.md", "one VLA sample and action-chunk target"),
        ("resume pack", relative(run_dir / "resume_pack.md"), "smallest resume and interview draft"),
    ]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{key}:{item}" for key, item in sorted(value.items()))
    return str(value)


def exists(path_text: str | Path) -> bool:
    return resolve(path_text).exists()


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def artifact_rows(run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group, artifacts in [("core", core_artifacts(run_dir)), ("optional", optional_artifacts(run_dir))]:
        for name, path, purpose in artifacts:
            rows.append(
                {
                    "group": group,
                    "artifact": name,
                    "path": path,
                    "exists": "yes" if exists(path) else "no",
                    "purpose": purpose,
                }
            )
    return rows


def metric_rows(run_dir: Path) -> list[dict[str, Any]]:
    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    return [
        {"metric": "records", "value": training.get("records", "n/a")},
        {"metric": "chunk_size", "value": training.get("chunk_size", "n/a")},
        {"metric": "final_loss", "value": training.get("final_loss", "n/a")},
        {"metric": "episodes", "value": evaluation.get("episodes", "n/a")},
        {"metric": "success_rate", "value": evaluation.get("success_rate", "n/a")},
        {"metric": "mean_final_distance", "value": evaluation.get("mean_final_distance", "n/a")},
        {"metric": "failure_count", "value": evaluation.get("failure_count", "n/a")},
        {"metric": "failure_categories", "value": evaluation.get("failure_category_counts", {})},
    ]


def status(run_dir: Path) -> str:
    missing_core = [path for _, path, _ in core_artifacts(run_dir) if not exists(path)]
    if missing_core:
        return "needs attention"
    missing_optional = [path for _, path, _ in optional_artifacts(run_dir) if not exists(path)]
    return "ready" if not missing_optional else "ready with optional gaps"


def next_step(run_dir: Path) -> str:
    if status(run_dir) == "needs attention":
        return "Rerun `python scripts/check_environment.py` and `python scripts/run_cpu_smoke.py`, then rebuild this checklist."
    return "Run `python scripts/run_baseline_evidence.py` when you want stronger metrics, README assets, and report-ready evidence."


def build_report(run_dir: Path) -> tuple[str, str]:
    run_status = status(run_dir)
    lines: list[str] = [
        "# MiniMind-VLA First Run Checklist",
        "",
        "Run this after the CPU smoke path to confirm the smallest runnable loop produced the files a beginner should inspect first.",
        "",
        f"Status: `{run_status}`",
        "",
        "## Smoke Metrics",
        "",
    ]
    lines.extend(markdown_table(metric_rows(run_dir)))
    lines.extend(["", "## Artifact Checklist", ""])
    lines.extend(markdown_table(artifact_rows(run_dir)))
    lines.extend(
        [
            "",
            "## Open In This Order",
            "",
            "1. `outputs/environment_check.md` to confirm the local setup.",
            "2. `outputs/cpu_smoke/summary_report.md` to read the headline metrics.",
            "3. `outputs/cpu_smoke/web_demo.html` to inspect rollout behavior.",
            "4. `outputs/cpu_smoke/run_diagnostic.md` to decide which claims are safe.",
            "5. `outputs/dataset_inspection.md` if you want to explain one training sample.",
            "",
            "## What This First Run Proves",
            "",
            "- The repo can train a tiny policy, save a checkpoint, evaluate rollouts, and build a rollout browser.",
            "- The result is useful as a learning check before running the stronger baseline path.",
            "- The safe claim is still small: a reproducible teaching-scale imitation-learning loop.",
            "",
            "## Next Step",
            "",
            next_step(run_dir),
            "",
            "## Rebuild",
            "",
            "```bash",
            f"python scripts/generate_first_run_checklist.py --run-dir {relative(run_dir)}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n", run_status


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report, run_status = build_report(run_dir)
    out_path.write_text(report, encoding="utf-8")
    print(f"first run checklist: {out_path}")
    print(f"first run status: {run_status}")
    return 1 if args.strict and run_status == "needs attention" else 0


if __name__ == "__main__":
    raise SystemExit(main())
