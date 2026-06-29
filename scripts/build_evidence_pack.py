from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LunaVLA evidence index for a completed run set.")
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
        artifact_row("outputs/quickstart_summary.md", "Summarize the one-command beginner quickstart."),
        artifact_row("outputs/environment_check.md", "Confirm the local environment can run public commands."),
        artifact_row("outputs/first_run_checklist.md", "Confirm the CPU smoke loop generated the first files to inspect."),
        artifact_row("outputs/troubleshooting_guide.md", "Map common run symptoms to files and recovery commands."),
        artifact_row("outputs/command_reference.md", "Map each public command to its purpose and generated artifacts."),
        artifact_row("outputs/code_walkthrough.md", "Guide beginners through the runnable code path."),
        artifact_row("outputs/action_chunk_lesson.md", "Explain ACT-style future-action chunks with a real sample."),
        artifact_row("outputs/readme_asset_check.md", "Confirm README images and animations are renderable."),
        artifact_row("outputs/project_progress.md", "Show which public project evidence stages are complete."),
        artifact_row("outputs/project_card.md", "One-page project evidence card."),
        artifact_row("outputs/experiment_ledger.md", "Audit commands, configs, metrics, and run artifacts."),
        artifact_row("outputs/experiment_ledger.json", "Machine-readable experiment ledger."),
        artifact_row("outputs/learning_checkpoint.md", "Concept-to-evidence learning checkpoint."),
        artifact_row("outputs/interview_flashcards.md", "Interview flashcards tied to code and run evidence."),
        artifact_row("outputs/skill_evidence_map.md", "Map beginner-facing VLA skills to code and run evidence."),
        artifact_row("outputs/learner_showcase.md", "Copyable learner showcase draft."),
        artifact_row("outputs/failure_review.md", "Summarize failure cases across public runs."),
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
        artifact_row("outputs/config_diff.md", "Config-level ablation audit."),
        artifact_row("outputs/config_diff.json", "Machine-readable config diff."),
        artifact_row("images/pusht_act_eval.gif", "README-visible ACT PushT evaluation animation."),
        artifact_row("images/pusht_diffusion_policy_eval.gif", "README-visible Diffusion Policy PushT evaluation animation."),
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
        "# LunaVLA Evidence Index",
        "",
        "This file is the public-facing evidence map for a completed local LunaVLA run.",
        "",
        "## What This Pack Proves",
        "",
        "- The dataset path can be inspected before training.",
        "- The quickstart path runs the smallest beginner loop with one command.",
        "- The local environment passes the public command readiness check.",
        "- The first-run checklist confirms the CPU smoke artifacts are ready to inspect.",
        "- The troubleshooting guide maps missing artifacts and weak runs to recovery commands.",
        "- The command reference maps every public command to the artifacts learners should inspect.",
        "- The code walkthrough shows the recommended reading order for the runnable implementation.",
        "- The action chunk lesson explains ACT-style targets using a concrete sample.",
        "- The README-visible assets pass image and animation checks.",
        "- The project progress report maps generated artifacts to report-ready stages.",
        "- The project card compresses commands, metrics, evidence links, and boundaries into one page.",
        "- The experiment ledger ties commands, config hashes, metrics, and artifacts together.",
        "- The learning checkpoint maps VLA concepts to code, reports, and self-check questions.",
        "- The interview flashcards help learners practice evidence-backed answers.",
        "- The skill evidence map connects VLA skills to code files, commands, and generated artifacts.",
        "- The learner showcase draft turns generated evidence into a shareable public story.",
        "- The failure review summarizes logged rollout failure cases.",
        "- The CPU smoke loop trains, evaluates, summarizes, and exports a demo.",
        "- The baseline path produces rollout metrics, reports, and README assets.",
        "- The chunk-size ablation produces a comparison report and metric deltas.",
        "- The config diff verifies which settings changed for the ablation.",
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
            "1. Start with `outputs/quickstart_summary.md` to show the one-command beginner path.",
            "2. Use `outputs/environment_check.md` to show the run environment was ready.",
            "3. Use `outputs/first_run_checklist.md` to confirm the smallest loop produced the expected files.",
            "4. Use `outputs/troubleshooting_guide.md` if any artifact is missing or a run needs debugging.",
            "5. Use `outputs/command_reference.md` to explain what each public command generates.",
            "6. Use `outputs/code_walkthrough.md` to explain how the code path fits together.",
            "7. Use `outputs/action_chunk_lesson.md` to explain ACT-style future-action targets.",
            "8. Use `outputs/readme_asset_check.md` to confirm the visual assets are intact.",
            "9. Use `outputs/project_progress.md` to check which evidence stages are complete.",
            "10. Use `outputs/project_card.md` as the one-page overview.",
            "11. Use `outputs/experiment_ledger.md` to audit commands, configs, metrics, and artifacts.",
            "12. Use `outputs/learning_checkpoint.md` to practice the core explanation.",
            "13. Use `outputs/interview_flashcards.md` for quick interview practice.",
            "14. Use `outputs/skill_evidence_map.md` to connect skills to code and run evidence.",
            "15. Use `outputs/learner_showcase.md` for a copyable public sharing draft.",
            "16. Use `outputs/failure_review.md` to explain failure behavior.",
            "17. Use `outputs/dataset_inspection.md` to explain the sample format.",
            "18. Use `outputs/act_pusht_baseline/project_report.md` for the baseline story.",
            "19. Use `outputs/act_pusht_baseline/run_diagnostic.md` to decide which claims are safe.",
            "20. Use `outputs/run_comparison.md` for the ablation story.",
            "21. Use `outputs/config_diff.md` to confirm what changed in the ablation.",
            "22. Use `outputs/act_pusht_baseline/resume_pack.md` for the resume bullet and interview pitch.",
            "23. Use the README GIFs and rollout browser as visual evidence.",
            "24. Keep the boundary honest: this is a small reproducible learning loop, not a real-robot deployment claim.",
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
        run([python, "scripts/run_quickstart.py"])
        run([python, "scripts/run_baseline_evidence.py", "--episodes", str(args.episodes)])
        run([python, "scripts/run_ablation_evidence.py", "--episodes", str(args.episodes), "--skip-baseline"])
    else:
        run([python, "scripts/run_quickstart.py", "--skip-run"])
        run([python, "scripts/generate_first_run_checklist.py"])
    run([python, "scripts/generate_failure_review.py"])
    run([python, "scripts/generate_command_reference.py"])
    run([python, "scripts/generate_code_walkthrough.py"])
    run([python, "scripts/generate_action_chunk_lesson.py"])
    run([python, "scripts/check_readme_assets.py"])
    run([python, "scripts/generate_learning_checkpoint.py"])
    run([python, "scripts/generate_interview_flashcards.py"])
    run([python, "scripts/generate_skill_evidence_map.py"])
    run([python, "scripts/generate_project_card.py"])
    run([python, "scripts/generate_experiment_ledger.py"])
    run([python, "scripts/generate_showcase_issue.py"])

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run([python, "scripts/generate_troubleshooting_guide.py"])
    run([python, "scripts/generate_command_reference.py"])
    run([python, "scripts/generate_code_walkthrough.py"])
    run([python, "scripts/generate_experiment_ledger.py"])
    run([python, "scripts/check_project_progress.py"])
    run([python, "scripts/generate_troubleshooting_guide.py"])
    out_path.write_text(build_index(), encoding="utf-8")
    missing = [row["artifact"] for row in required_artifacts() if row["exists"] != "yes"]
    if missing:
        raise FileNotFoundError("Missing evidence artifacts: " + ", ".join(missing))
    run([python, "scripts/build_submission_pack.py", "--evidence-index", relative(out_path)])
    print(f"evidence index: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
