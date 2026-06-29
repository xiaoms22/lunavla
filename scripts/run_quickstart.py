from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_ARTIFACTS = [
    ("environment check", "outputs/environment_check.md", "machine readiness"),
    ("dataset inspection", "outputs/dataset_inspection.md", "one VLA sample"),
    ("checkpoint", "outputs/cpu_smoke/checkpoint.pt", "tiny policy weights"),
    ("smoke summary", "outputs/cpu_smoke/summary_report.md", "headline smoke metrics"),
    ("rollout browser", "outputs/cpu_smoke/web_demo.html", "static rollout inspection"),
    ("first-run checklist", "outputs/first_run_checklist.md", "what to open first"),
    ("troubleshooting guide", "outputs/troubleshooting_guide.md", "recovery commands"),
    ("code walkthrough", "outputs/code_walkthrough.md", "guided code reading"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the one-command MiniMind-VLA beginner quickstart.")
    parser.add_argument("--out", default="outputs/quickstart_summary.md", help="Markdown quickstart summary path.")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not rerun smoke commands; only refresh checklist, troubleshooting, and summary from existing artifacts.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any expected quickstart artifact is missing.")
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


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def artifact_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, path, purpose in EXPECTED_ARTIFACTS:
        rows.append(
            {
                "artifact": name,
                "path": path,
                "exists": "yes" if resolve(path).exists() else "no",
                "purpose": purpose,
            }
        )
    return rows


def metric_rows() -> list[dict[str, Any]]:
    training = read_json(ROOT / "outputs/cpu_smoke/training_summary.json")
    evaluation = read_json(ROOT / "outputs/cpu_smoke/eval_summary.json")
    return [
        {"metric": "records", "value": training.get("records", "n/a")},
        {"metric": "chunk_size", "value": training.get("chunk_size", "n/a")},
        {"metric": "final_loss", "value": training.get("final_loss", "n/a")},
        {"metric": "episodes", "value": evaluation.get("episodes", "n/a")},
        {"metric": "success_rate", "value": evaluation.get("success_rate", "n/a")},
        {"metric": "mean_final_distance", "value": evaluation.get("mean_final_distance", "n/a")},
        {"metric": "failure_count", "value": evaluation.get("failure_count", "n/a")},
    ]


def quickstart_status(rows: list[dict[str, str]]) -> str:
    missing = [row for row in rows if row["exists"] != "yes"]
    return "ready" if not missing else "needs attention"


def build_summary() -> tuple[str, str]:
    artifacts = artifact_rows()
    status = quickstart_status(artifacts)
    lines: list[str] = [
        "# MiniMind-VLA Quickstart Summary",
        "",
        "This file summarizes the one-command beginner path for running the smallest MiniMind-VLA loop.",
        "",
        f"Status: `{status}`",
        "",
        "## Commands Covered",
        "",
        "```bash",
        "python scripts/check_environment.py",
        "python scripts/inspect_dataset.py",
        "python scripts/run_cpu_smoke.py",
        "python scripts/generate_first_run_checklist.py",
        "python scripts/generate_troubleshooting_guide.py",
        "python scripts/generate_code_walkthrough.py",
        "```",
        "",
        "## Smoke Metrics",
        "",
    ]
    lines.extend(markdown_table(metric_rows()))
    lines.extend(["", "## Generated Artifacts", ""])
    lines.extend(markdown_table(artifacts))
    lines.extend(
        [
            "",
            "## Open Next",
            "",
            "1. `outputs/first_run_checklist.md` for the beginner review order.",
            "2. `outputs/cpu_smoke/summary_report.md` for metrics.",
            "3. `outputs/cpu_smoke/web_demo.html` for rollout behavior.",
            "4. `outputs/code_walkthrough.md` for the code reading order.",
            "5. `outputs/troubleshooting_guide.md` if something is missing.",
            "",
            "## Next Step",
            "",
            "Run `python scripts/run_baseline_evidence.py` when the quickstart is ready and you want stronger project evidence.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/run_quickstart.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n", status


def main() -> int:
    args = parse_args()
    python = sys.executable
    if not args.skip_run:
        run([python, "scripts/check_environment.py"])
        run([python, "scripts/inspect_dataset.py"])
        run([python, "scripts/run_cpu_smoke.py"])
    run([python, "scripts/generate_first_run_checklist.py"])
    run([python, "scripts/generate_troubleshooting_guide.py"])
    run([python, "scripts/generate_code_walkthrough.py"])

    report, status = build_summary()
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"quickstart summary: {out_path}")
    print(f"quickstart status: {status}")
    return 1 if args.strict and status != "ready" else 0


if __name__ == "__main__":
    raise SystemExit(main())
