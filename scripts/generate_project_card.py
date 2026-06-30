from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("CPU smoke", "outputs/cpu_smoke"),
    ("Baseline", "outputs/act_pusht_baseline"),
    ("Chunk-size ablation", "outputs/act_pusht_ablation_chunk_size"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a one-page LunaVLA project evidence card.")
    parser.add_argument("--out", default="outputs/project_card.md", help="Markdown project card path.")
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


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, run_dir_text in RUNS:
        run_dir = resolve(run_dir_text)
        training = read_json(run_dir / "training_summary.json")
        evaluation = read_json(run_dir / "eval_summary.json")
        rows.append(
            {
                "run": label,
                "records": training.get("records", "n/a"),
                "chunk_size": training.get("chunk_size", "n/a"),
                "final_loss": training.get("final_loss", "n/a"),
                "episodes": evaluation.get("episodes", "n/a"),
                "success_rate": evaluation.get("success_rate", "n/a"),
                "mean_final_distance": evaluation.get("mean_final_distance", "n/a"),
                "failure_count": evaluation.get("failure_count", "n/a"),
            }
        )
    return rows


def evidence_rows() -> list[dict[str, str]]:
    evidence = [
        ("quickstart", "outputs/quickstart_summary.md", "one-command beginner path summary"),
        ("environment", "outputs/environment_check.md", "local readiness"),
        ("first run", "outputs/first_run_checklist.md", "smallest runnable loop checklist"),
        ("troubleshooting", "outputs/troubleshooting_guide.md", "symptom-to-command recovery guide"),
        ("command reference", "outputs/command_reference.md", "public command map"),
        ("code walkthrough", "outputs/code_walkthrough.md", "guided code reading order"),
        ("dataset", "outputs/dataset_inspection.md", "one VLA sample and action chunk"),
        ("action statistics", "outputs/action_statistics.md", "action scale and normalization notes"),
        ("action analysis", "outputs/action_analysis_report.md", "train-target vs eval-executable action analysis"),
        ("extended evaluation", "outputs/extended_evaluation_report.md", "more-episode rollout evidence with success/failure examples"),
        ("homepage summary", "outputs/homepage_summary.md", "README-facing result claims tied to checked metrics"),
        ("baseline report", "outputs/act_pusht_baseline/project_report.md", "main technical story"),
        ("baseline diagnostic", "outputs/act_pusht_baseline/run_diagnostic.md", "claim-safety check"),
        ("learning checkpoint", "outputs/learning_checkpoint.md", "concept-to-evidence review"),
        ("interview flashcards", "outputs/interview_flashcards.md", "evidence-backed interview answers"),
        ("skill evidence map", "outputs/skill_evidence_map.md", "skills mapped to code and artifacts"),
        ("learner showcase", "outputs/learner_showcase.md", "copyable public sharing draft"),
        ("failure review", "outputs/failure_review.md", "cross-run failure behavior"),
        ("ablation", "outputs/run_comparison.md", "chunk-size comparison"),
        ("config diff", "outputs/config_diff.md", "ablation setup audit"),
        ("README assets", "outputs/readme_asset_check.md", "visual evidence quality"),
        ("experiment ledger", "outputs/experiment_ledger.md", "commands, config hashes, metrics, and artifacts"),
        ("rollout browser", "outputs/act_pusht_baseline/web_demo.html", "saved rollout inspection"),
        ("submission pack", "outputs/submission_pack/SUBMISSION_README.md", "compact review folder"),
    ]
    return [
        {
            "evidence": name,
            "file": path,
            "exists": "yes" if resolve(path).exists() else "no",
            "purpose": purpose,
        }
        for name, path, purpose in evidence
    ]


def build_card() -> str:
    lines: list[str] = [
        "# LunaVLA Project Card",
        "",
        "LunaVLA is a tiny, reproducible VLA project starter for learning the `observation -> action -> rollout -> evaluation` loop.",
        "",
        "## What This Project Demonstrates",
        "",
        "- Generated PushT-style demonstration data with observation, action, episode, timestep, success, and metadata fields.",
        "- Recorded action statistics for scale, clipping, checkpoint provenance, and normalization explanation.",
        "- Trained an ACT-style action-chunk policy from the generated data.",
        "- Evaluated rollout behavior with success rate, final distance, rollout length, action smoothness, and failure cases.",
        "- Exported reports, diagnostics, README assets, rollout browser, and a compact submission pack.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "python scripts/check_environment.py",
        "python scripts/run_cpu_smoke.py",
        "python scripts/run_baseline_evidence.py",
        "python scripts/generate_action_statistics.py",
        "python scripts/run_ablation_evidence.py --skip-baseline",
        "python scripts/build_evidence_pack.py --skip-runs",
        "```",
        "",
        "## Headline Metrics",
        "",
    ]
    lines.extend(markdown_table(metric_rows()))
    lines.extend(["", "## Evidence Files", ""])
    lines.extend(markdown_table(evidence_rows()))
    lines.extend(
        [
            "",
            "## Safe One-Sentence Claim",
            "",
            "Built a lightweight ACT-style imitation-learning VLA loop with generated demonstrations, rollout evaluation, failure review, README assets, and reproducible project reports.",
            "",
            "## Boundaries",
            "",
            "- This is a teaching-scale PushT-style imitation-learning project.",
            "- It is useful for learning data, policy, rollout, evaluation, reporting, and interview explanation.",
            "- It is not a real-robot deployment benchmark and does not claim state-of-the-art robotics performance.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_project_card.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_card(), encoding="utf-8")
    print(f"project card: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
