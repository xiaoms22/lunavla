from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


STAGES = [
    {
        "stage": "environment and data",
        "goal": "The repo can run locally and one VLA sample can be inspected.",
        "command": "python scripts/check_environment.py; python scripts/inspect_dataset.py",
        "artifacts": [
            "outputs/environment_check.md",
            "outputs/dataset_inspection.md",
            "outputs/quickstart_summary.md",
            "outputs/command_reference.md",
            "outputs/code_walkthrough.md",
            "outputs/action_chunk_lesson.md",
        ],
    },
    {
        "stage": "CPU smoke loop",
        "goal": "A tiny train/eval/report/demo loop completes.",
        "command": "python scripts/run_cpu_smoke.py",
        "artifacts": [
            "outputs/cpu_smoke/checkpoint.pt",
            "outputs/cpu_smoke/summary_report.md",
            "outputs/cpu_smoke/project_report.md",
            "outputs/cpu_smoke/resume_pack.md",
            "outputs/cpu_smoke/run_diagnostic.md",
            "outputs/cpu_smoke/web_demo.html",
            "outputs/first_run_checklist.md",
            "outputs/troubleshooting_guide.md",
            "outputs/command_reference.md",
            "outputs/code_walkthrough.md",
            "outputs/action_chunk_lesson.md",
        ],
    },
    {
        "stage": "baseline evidence",
        "goal": "The ACT + PushT-style baseline has metrics, reports, demo, and README assets.",
        "command": "python scripts/run_baseline_evidence.py",
        "artifacts": [
            "outputs/act_pusht_baseline/checkpoint.pt",
            "outputs/act_pusht_baseline/summary_report.md",
            "outputs/act_pusht_baseline/project_report.md",
            "outputs/act_pusht_baseline/resume_pack.md",
            "outputs/act_pusht_baseline/run_diagnostic.md",
            "outputs/act_pusht_baseline/web_demo.html",
            "images/pusht_act_eval.gif",
            "images/pusht_diffusion_policy_eval.gif",
            "outputs/readme_asset_check.md",
            "outputs/action_chunk_lesson.md",
            "outputs/project_card.md",
            "outputs/experiment_ledger.md",
            "outputs/experiment_ledger.json",
            "outputs/learning_checkpoint.md",
            "outputs/interview_flashcards.md",
            "outputs/skill_evidence_map.md",
            "outputs/learner_showcase.md",
            "outputs/failure_review.md",
        ],
    },
    {
        "stage": "ablation evidence",
        "goal": "The chunk-size ablation can be compared against the baseline.",
        "command": "python scripts/run_ablation_evidence.py",
        "artifacts": [
            "outputs/act_pusht_ablation_chunk_size/checkpoint.pt",
            "outputs/act_pusht_ablation_chunk_size/summary_report.md",
            "outputs/act_pusht_ablation_chunk_size/project_report.md",
            "outputs/act_pusht_ablation_chunk_size/resume_pack.md",
            "outputs/act_pusht_ablation_chunk_size/run_diagnostic.md",
            "outputs/act_pusht_ablation_chunk_size/web_demo.html",
            "outputs/run_comparison.md",
            "outputs/run_comparison.csv",
            "outputs/run_comparison_deltas.csv",
            "outputs/config_diff.md",
            "outputs/config_diff.json",
        ],
    },
    {
        "stage": "evidence index",
        "goal": "Generated files are mapped into one report-ready evidence index.",
        "command": "python scripts/build_evidence_pack.py --skip-runs",
        "artifacts": [
            "outputs/evidence_index.md",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LunaVLA project evidence progress.")
    parser.add_argument("--out", default="outputs/project_progress.md", help="Markdown report path.")
    parser.add_argument("--strict", action="store_true", help="Exit with a non-zero status if any stage is incomplete.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def stage_status(found: int, total: int) -> str:
    if found == total:
        return "complete"
    if found > 0:
        return "partial"
    return "missing"


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    stage_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, str]] = []
    for stage in STAGES:
        artifacts = [str(path) for path in stage["artifacts"]]
        found = [path for path in artifacts if resolve(path).exists()]
        missing = [path for path in artifacts if not resolve(path).exists()]
        status = stage_status(len(found), len(artifacts))
        stage_rows.append(
            {
                "stage": stage["stage"],
                "status": status,
                "artifacts": f"{len(found)}/{len(artifacts)}",
                "command": stage["command"] if missing else "done",
            }
        )
        for artifact in artifacts:
            artifact_rows.append(
                {
                    "stage": str(stage["stage"]),
                    "artifact": artifact,
                    "exists": "yes" if resolve(artifact).exists() else "no",
                }
            )
    return stage_rows, artifact_rows


def overall(stage_rows: list[dict[str, Any]]) -> str:
    statuses = {str(row["status"]) for row in stage_rows}
    if statuses == {"complete"}:
        return "complete"
    if "partial" in statuses or "complete" in statuses:
        return "partial"
    return "missing"


def build_report() -> tuple[str, str]:
    stage_rows, artifact_rows = build_rows()
    status = overall(stage_rows)
    lines: list[str] = [
        "# LunaVLA Project Progress",
        "",
        f"Overall: `{status}`",
        "",
        "This report turns generated LunaVLA artifacts into a beginner-friendly project evidence checklist.",
        "",
        "## Stage Summary",
        "",
    ]
    lines.extend(markdown_table(stage_rows))
    lines.extend(
        [
            "",
            "## Artifact Coverage",
            "",
        ]
    )
    lines.extend(markdown_table(artifact_rows))
    lines.extend(
        [
            "",
            "## How To Use This",
            "",
            "- Use `complete` stages as material for your project report.",
            "- Use `partial` stages to find the public command that will generate the missing artifacts.",
            "- Keep claims tied to generated reports, diagnostics, rollout evidence, and comparison files.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/check_project_progress.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n", status


def main() -> int:
    args = parse_args()
    report, status = build_report()
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"project progress: {status}")
    print(f"project progress report: {out_path}")
    return 1 if args.strict and status != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
