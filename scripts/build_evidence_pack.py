from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MiniMind-VLA evidence index for a completed run set.")
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes when running evidence commands.")
    parser.add_argument("--out", default="outputs/evidence_index.md", help="Markdown evidence index path.")
    parser.add_argument("--skip-runs", action="store_true", help="Only build the index from existing artifacts.")
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
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{key}:{value}" for key, value in sorted(value.items()))
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def artifact_row(path: str, purpose: str) -> dict[str, str]:
    artifact = resolve(path)
    return {
        "artifact": path,
        "exists": "yes" if artifact.exists() else "no",
        "purpose": purpose,
    }


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def run_metrics(run_dir: str) -> dict[str, Any]:
    path = resolve(run_dir)
    train = read_json(path / "training_summary.json")
    eval_summary = read_json(path / "eval_summary.json")
    return {
        "run": path.name,
        "records": train.get("records", "n/a"),
        "chunk_size": train.get("chunk_size", "n/a"),
        "final_loss": train.get("final_loss", "n/a"),
        "success_rate": eval_summary.get("success_rate", "n/a"),
        "mean_final_distance": eval_summary.get("mean_final_distance", "n/a"),
        "mean_action_smoothness": eval_summary.get("mean_action_smoothness", "n/a"),
        "failure_count": eval_summary.get("failure_count", "n/a"),
        "failure_categories": eval_summary.get("failure_category_counts", {}),
    }


def required_artifacts() -> list[dict[str, str]]:
    return [
        artifact_row("outputs/environment_check.md", "Confirm the local environment can run public commands."),
        artifact_row("outputs/readme_asset_check.md", "Confirm README images and animations are renderable."),
        artifact_row("outputs/project_progress.md", "Show which public project evidence stages are complete."),
        artifact_row("outputs/dataset_inspection.md", "Understand one VLA sample and action chunk target."),
        artifact_row("outputs/cpu_smoke/summary_report.md", "Confirm the one-command smoke loop works."),
        artifact_row("outputs/cpu_smoke/project_report.md", "Smallest report a learner can inspect."),
        artifact_row("outputs/cpu_smoke/resume_pack.md", "Smallest resume and interview pack a learner can inspect."),
        artifact_row("outputs/cpu_smoke/run_diagnostic.md", "Smallest run diagnostic and claim-safety check."),
        artifact_row("outputs/cpu_smoke/web_demo.html", "Static rollout browser from the CPU smoke path."),
        artifact_row("outputs/act_pusht_baseline/summary_report.md", "Baseline metric summary."),
        artifact_row("outputs/act_pusht_baseline/project_report.md", "Baseline project report."),
        artifact_row("outputs/act_pusht_baseline/resume_pack.md", "Baseline resume and interview pack."),
        artifact_row("outputs/act_pusht_baseline/run_diagnostic.md", "Baseline run diagnostic and claim-safety check."),
        artifact_row("outputs/act_pusht_baseline/web_demo.html", "Baseline rollout browser."),
        artifact_row("outputs/act_pusht_ablation_chunk_size/summary_report.md", "Ablation metric summary."),
        artifact_row("outputs/act_pusht_ablation_chunk_size/project_report.md", "Ablation project report."),
        artifact_row("outputs/act_pusht_ablation_chunk_size/resume_pack.md", "Ablation resume and interview pack."),
        artifact_row("outputs/act_pusht_ablation_chunk_size/run_diagnostic.md", "Ablation run diagnostic and claim-safety check."),
        artifact_row("outputs/act_pusht_ablation_chunk_size/web_demo.html", "Ablation rollout browser."),
        artifact_row("outputs/run_comparison.md", "Baseline vs ablation comparison report."),
        artifact_row("outputs/run_comparison.csv", "Machine-readable comparison table."),
        artifact_row("outputs/run_comparison_deltas.csv", "Machine-readable metric deltas."),
        artifact_row("images/pusht_rollout.gif", "README-visible rollout animation."),
        artifact_row("images/act_action_chunk.gif", "README-visible action chunk animation."),
        artifact_row("images/loss_curve.gif", "README-visible loss curve animation."),
    ]


def build_index() -> str:
    artifacts = required_artifacts()
    missing = [row["artifact"] for row in artifacts if row["exists"] != "yes"]
    metrics = [
        run_metrics("outputs/cpu_smoke"),
        run_metrics("outputs/act_pusht_baseline"),
        run_metrics("outputs/act_pusht_ablation_chunk_size"),
    ]

    lines: list[str] = [
        "# MiniMind-VLA Evidence Index",
        "",
        "This file is the public-facing evidence map for a completed local MiniMind-VLA run.",
        "",
        "## What This Pack Proves",
        "",
        "- The dataset path can be inspected before training.",
        "- The local environment passes the public command readiness check.",
        "- The README-visible assets pass image and animation checks.",
        "- The project progress report maps generated artifacts to report-ready stages.",
        "- The CPU smoke loop trains, evaluates, summarizes, and exports a demo.",
        "- The baseline path produces rollout metrics, reports, and README assets.",
        "- The chunk-size ablation produces a comparison report and metric deltas.",
        "- The claims remain limited to a teaching-scale PushT-style imitation-learning setup.",
        "",
        "## Run Metrics",
        "",
    ]
    lines.extend(markdown_table(metrics))
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    lines.extend(markdown_table(artifacts))
    lines.extend(
        [
            "",
            "## How To Use This In A Project Report",
            "",
            "1. Start with `outputs/environment_check.md` to show the run environment was ready.",
            "2. Use `outputs/readme_asset_check.md` to confirm the visual assets are intact.",
            "3. Use `outputs/project_progress.md` to check which evidence stages are complete.",
            "4. Use `outputs/dataset_inspection.md` to explain the sample format.",
            "5. Use `outputs/act_pusht_baseline/project_report.md` for the baseline story.",
            "6. Use `outputs/act_pusht_baseline/run_diagnostic.md` to decide which claims are safe.",
            "7. Use `outputs/run_comparison.md` for the ablation story.",
            "8. Use `outputs/act_pusht_baseline/resume_pack.md` for the resume bullet and interview pitch.",
            "9. Use the README GIFs and rollout demo as visual evidence.",
            "10. Keep the boundary honest: this is a small reproducible learning loop, not a real-robot deployment claim.",
        ]
    )
    if missing:
        lines.extend(["", "## Missing Artifacts", ""])
        lines.extend(f"- `{path}`" for path in missing)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    python = sys.executable
    run([python, "scripts/check_environment.py"])
    if not args.skip_runs:
        run([python, "scripts/validate_configs.py"])
        run([python, "scripts/inspect_dataset.py"])
        run([python, "scripts/run_cpu_smoke.py"])
        run([python, "scripts/run_baseline_evidence.py", "--episodes", str(args.episodes)])
        run([python, "scripts/run_ablation_evidence.py", "--episodes", str(args.episodes), "--skip-baseline"])
    run([python, "scripts/check_readme_assets.py"])

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_index(), encoding="utf-8")
    run([python, "scripts/check_project_progress.py"])
    missing = [row["artifact"] for row in required_artifacts() if row["exists"] != "yes"]
    if missing:
        raise FileNotFoundError("Missing evidence artifacts: " + ", ".join(missing))
    run([python, "scripts/build_submission_pack.py", "--evidence-index", relative(out_path)])
    print(f"evidence index: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
