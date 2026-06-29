from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = "outputs/act_pusht_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a copyable LunaVLA learner showcase issue draft.")
    parser.add_argument("--run-dir", default=BASELINE_DIR, help="Primary run directory.")
    parser.add_argument("--out", default="outputs/learner_showcase.md", help="Markdown showcase draft path.")
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


def completion_level() -> str:
    checks = [
        ("Used it for interview preparation", ROOT / "outputs/learning_checkpoint.md"),
        ("Wrote project report", ROOT / "outputs/project_card.md"),
        ("Finished ablation", ROOT / "outputs/run_comparison.md"),
        ("Finished baseline", ROOT / BASELINE_DIR / "project_report.md"),
        ("Ran CPU smoke", ROOT / "outputs/cpu_smoke/summary_report.md"),
    ]
    for label, path in checks:
        if path.exists():
            return label
    return "Ran CPU smoke"


def evidence_rows(run_dir: Path) -> list[dict[str, str]]:
    evidence = [
        ("Project card", "outputs/project_card.md"),
        ("Learning checkpoint", "outputs/learning_checkpoint.md"),
        ("Baseline report", relative(run_dir / "project_report.md")),
        ("Baseline diagnostic", relative(run_dir / "run_diagnostic.md")),
        ("Failure review", "outputs/failure_review.md"),
        ("Ablation comparison", "outputs/run_comparison.md"),
        ("Rollout browser", relative(run_dir / "web_demo.html")),
        ("README rollout GIF", "images/pusht_rollout.gif"),
        ("README result table", "images/result_table.svg"),
    ]
    return [
        {
            "evidence": name,
            "file": path,
            "exists": "yes" if resolve(path).exists() else "no",
        }
        for name, path in evidence
    ]


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def metric_lines(run_dir: Path) -> list[str]:
    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    rows = [
        {"metric": "records", "value": training.get("records", "n/a")},
        {"metric": "chunk_size", "value": training.get("chunk_size", "n/a")},
        {"metric": "final_loss", "value": training.get("final_loss", "n/a")},
        {"metric": "episodes", "value": evaluation.get("episodes", "n/a")},
        {"metric": "success_rate", "value": evaluation.get("success_rate", "n/a")},
        {"metric": "mean_final_distance", "value": evaluation.get("mean_final_distance", "n/a")},
        {"metric": "mean_action_smoothness", "value": evaluation.get("mean_action_smoothness", "n/a")},
        {"metric": "failure_count", "value": evaluation.get("failure_count", "n/a")},
    ]
    return markdown_table(rows)


def build_showcase(run_dir: Path) -> str:
    lines: list[str] = [
        "# Learner Showcase Draft",
        "",
        "Copy the sections below into a LunaVLA learner showcase issue or your own project README.",
        "",
        "## Suggested Issue Title",
        "",
        "[Showcase] Finished LunaVLA baseline evidence pack",
        "",
        "## Completion Level",
        "",
        completion_level(),
        "",
        "## Commands I Ran",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python scripts/check_environment.py",
        "python scripts/run_cpu_smoke.py",
        "python scripts/run_baseline_evidence.py",
        "python scripts/run_ablation_evidence.py --skip-baseline",
        "python scripts/build_evidence_pack.py --skip-runs",
        "```",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(metric_lines(run_dir))
    lines.extend(["", "## Evidence", ""])
    lines.extend(markdown_table(evidence_rows(run_dir)))
    lines.extend(
        [
            "",
            "## What I Learned",
            "",
            "- How a VLA-shaped record connects observation, optional instruction features, action, episode id, timestep, success, and metadata.",
            "- Why behavior cloning trains from demonstrations instead of reward.",
            "- Why an ACT-style policy predicts an action chunk instead of a single action.",
            "- Why rollout evaluation is needed in addition to training loss.",
            "- How failure labels and rollout browsers help keep conclusions grounded.",
            "",
            "## Next Step",
            "",
            "Inspect more rollout episodes, rerun the baseline with more evaluation episodes, and compare one controlled ablation before changing multiple variables.",
            "",
            "## Honesty Check",
            "",
            "- I am sharing a learning result, not claiming production robot capability.",
            "- I can explain the commands and metrics I used.",
            "- I keep the boundary to a teaching-scale PushT-style imitation-learning setup.",
            "",
            "## Rebuild",
            "",
            "```bash",
            f"python scripts/generate_showcase_issue.py --run-dir {relative(run_dir)}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_showcase(run_dir), encoding="utf-8")
    print(f"learner showcase: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
