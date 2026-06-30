from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact LunaVLA submission pack from generated evidence.")
    parser.add_argument("--run-dir", default="outputs/act_pusht_baseline", help="Primary run directory.")
    parser.add_argument("--comparison", default="outputs/run_comparison.md", help="Ablation comparison report path.")
    parser.add_argument("--evidence-index", default="outputs/evidence_index.md", help="Evidence index path.")
    parser.add_argument("--out-dir", default="outputs/submission_pack", help="Submission pack output directory.")
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


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {relative(path)}")


def copy_file(src: Path, dst: Path) -> dict[str, str]:
    require(src, "submission source")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source": relative(src), "pack_path": relative(dst)}


def metric_rows(run_dir: Path) -> list[tuple[str, Any]]:
    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    return [
        ("project name", training.get("project_name", run_dir.name)),
        ("records", training.get("records", "n/a")),
        ("chunk size", training.get("chunk_size", "n/a")),
        ("final loss", training.get("final_loss", "n/a")),
        ("success rate", evaluation.get("success_rate", "n/a")),
        ("mean final distance", evaluation.get("mean_final_distance", "n/a")),
        ("mean rollout length", evaluation.get("mean_rollout_length", "n/a")),
        ("mean action smoothness", evaluation.get("mean_action_smoothness", "n/a")),
        ("failure count", evaluation.get("failure_count", "n/a")),
        ("failure categories", evaluation.get("failure_category_counts", {})),
    ]


def markdown_table(rows: list[tuple[str, Any]]) -> list[str]:
    lines = ["| item | value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | `{format_value(value)}` |")
    return lines


def pack_relative(path_text: str, out_dir: Path) -> str:
    path = Path(path_text)
    try:
        return path.relative_to(relative(out_dir)).as_posix()
    except ValueError:
        return path.as_posix()


def build_summary(run_dir: Path, comparison: Path, evidence_index: Path, out_dir: Path, copied: list[dict[str, str]]) -> str:
    lines: list[str] = [
        "# LunaVLA Submission Pack",
        "",
        "This folder collects the generated evidence from one completed LunaVLA run.",
        "",
        "## What This Pack Contains",
        "",
        "- baseline metrics and project report;",
        "- quickstart summary for the one-command beginner path;",
        "- environment check for reproducibility context;",
        "- first-run checklist for the smallest runnable loop;",
        "- troubleshooting guide for missing artifacts and weak runs;",
        "- command reference for choosing the next public command;",
        "- code walkthrough for reading the runnable implementation;",
        "- action chunk lesson for explaining the ACT-style target;",
        "- action statistics for explaining scale and normalization boundaries;",
        "- policy ladder for explaining BC vs ACT with rollout evidence;",
        "- policy tuning comparison for explaining hidden-size changes with rollout evidence;",
        "- README asset check for visual evidence quality;",
        "- project progress report for artifact coverage;",
        "- one-page project card for quick review;",
        "- experiment ledger for commands, configs, metrics, and artifacts;",
        "- learning checkpoint for concept-to-evidence review;",
        "- interview flashcards for evidence-backed answers;",
        "- skill evidence map for connecting abilities to code and artifacts;",
        "- learner showcase draft for public sharing;",
        "- failure review for rollout failure analysis;",
        "- run diagnostic for checking which claims are safe;",
        "- resume-safe bullet and two-minute interview pitch;",
        "- rollout browser for inspecting saved episodes;",
        "- chunk-size ablation comparison;",
        "- config diff for checking the ablation setup;",
        "- README-ready PushT comparison media.",
        "",
        "## Metrics To Cite",
        "",
    ]
    lines.extend(markdown_table(metric_rows(run_dir)))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "| file | source |",
            "| --- | --- |",
        ]
    )
    for item in copied:
        lines.append(f"| `{pack_relative(item['pack_path'], out_dir)}` | `{item['source']}` |")
    lines.extend(
        [
            "",
            "## How To Present It",
            "",
            "1. Open `quickstart_summary.md` to confirm the one-command beginner path.",
            "2. Open `environment_check.md` to confirm the run context.",
            "3. Open `first_run_checklist.md` to confirm the smallest loop is ready.",
            "4. Open `troubleshooting_guide.md` if a file is missing or a metric looks weak.",
            "5. Open `command_reference.md` to choose the next public command.",
            "6. Open `code_walkthrough.md` to read the implementation in order.",
            "7. Open `action_chunk_lesson.md` to explain the ACT-style target.",
            "8. Open `action_statistics.md` to explain action scale and normalization.",
            "9. Open `policy_ladder.md` to explain why rollout evidence matters beyond BC loss.",
            "10. Open `policy_tuning_comparison.md` to explain the BC hidden-size comparison.",
            "11. Open `readme_asset_check.md` to confirm visual evidence is intact.",
            "12. Open `project_progress.md` to check artifact coverage.",
            "13. Open `project_card.md` for the one-page overview.",
            "14. Open `experiment_ledger.md` to audit commands, config hashes, metrics, and artifacts.",
            "15. Open `learning_checkpoint.md` to practice the explanation.",
            "16. Open `interview_flashcards.md` for evidence-backed answers.",
            "17. Open `skill_evidence_map.md` to connect skills to files and artifacts.",
            "18. Open `learner_showcase.md` for a copyable public sharing draft.",
            "19. Open `failure_review.md` for failure behavior and inspection notes.",
            "20. Open `project_report.md` for the technical story.",
            "21. Open `run_diagnostic.md` before deciding what the run proves.",
            "22. Open `resume_pack.md` for the resume bullet and interview pitch.",
            "23. Open `rollout_browser.html` to inspect rollout behavior.",
            "24. Use `ablation_comparison.md` only for claims about the chunk-size ablation.",
            "25. Use `config_diff.md` to confirm the ablation setup.",
            "26. Use files under `assets/` for README screenshots or a project page.",
            "",
            "## Honest Boundary",
            "",
            "This pack demonstrates a tiny PushT-style imitation-learning loop. It is not a real-robot deployment claim.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/build_submission_pack.py "
            f"--run-dir {relative(run_dir)} "
            f"--comparison {relative(comparison)} "
            f"--evidence-index {relative(evidence_index)} "
            f"--out-dir {relative(out_dir)}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    comparison = resolve(args.comparison)
    evidence_index = resolve(args.evidence_index)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required_sources = [
        (ROOT / "outputs/quickstart_summary.md", out_dir / "quickstart_summary.md"),
        (ROOT / "outputs/environment_check.md", out_dir / "environment_check.md"),
        (ROOT / "outputs/first_run_checklist.md", out_dir / "first_run_checklist.md"),
        (ROOT / "outputs/troubleshooting_guide.md", out_dir / "troubleshooting_guide.md"),
        (ROOT / "outputs/command_reference.md", out_dir / "command_reference.md"),
        (ROOT / "outputs/code_walkthrough.md", out_dir / "code_walkthrough.md"),
        (ROOT / "outputs/action_chunk_lesson.md", out_dir / "action_chunk_lesson.md"),
        (ROOT / "outputs/action_statistics.json", out_dir / "action_statistics.json"),
        (ROOT / "outputs/action_statistics.md", out_dir / "action_statistics.md"),
        (ROOT / "outputs/policy_ladder.md", out_dir / "policy_ladder.md"),
        (ROOT / "outputs/policy_ladder.csv", out_dir / "policy_ladder.csv"),
        (ROOT / "outputs/policy_tuning_comparison.md", out_dir / "policy_tuning_comparison.md"),
        (ROOT / "outputs/policy_tuning_comparison.csv", out_dir / "policy_tuning_comparison.csv"),
        (ROOT / "outputs/policy_tuning_comparison_deltas.csv", out_dir / "policy_tuning_comparison_deltas.csv"),
        (ROOT / "outputs/policy_tuning_config_diff.md", out_dir / "policy_tuning_config_diff.md"),
        (ROOT / "outputs/policy_tuning_config_diff.json", out_dir / "policy_tuning_config_diff.json"),
        (ROOT / "outputs/readme_asset_check.md", out_dir / "readme_asset_check.md"),
        (ROOT / "outputs/project_progress.md", out_dir / "project_progress.md"),
        (ROOT / "outputs/project_card.md", out_dir / "project_card.md"),
        (ROOT / "outputs/experiment_ledger.md", out_dir / "experiment_ledger.md"),
        (ROOT / "outputs/experiment_ledger.json", out_dir / "experiment_ledger.json"),
        (ROOT / "outputs/learning_checkpoint.md", out_dir / "learning_checkpoint.md"),
        (ROOT / "outputs/interview_flashcards.md", out_dir / "interview_flashcards.md"),
        (ROOT / "outputs/skill_evidence_map.md", out_dir / "skill_evidence_map.md"),
        (ROOT / "outputs/learner_showcase.md", out_dir / "learner_showcase.md"),
        (ROOT / "outputs/failure_review.md", out_dir / "failure_review.md"),
        (run_dir / "summary_report.md", out_dir / "run_summary.md"),
        (run_dir / "project_report.md", out_dir / "project_report.md"),
        (run_dir / "resume_pack.md", out_dir / "resume_pack.md"),
        (run_dir / "run_diagnostic.md", out_dir / "run_diagnostic.md"),
        (run_dir / "web_demo.html", out_dir / "rollout_browser.html"),
        (comparison, out_dir / "ablation_comparison.md"),
        (ROOT / "outputs/config_diff.md", out_dir / "config_diff.md"),
        (ROOT / "outputs/config_diff.json", out_dir / "config_diff.json"),
        (evidence_index, out_dir / "evidence_index.md"),
        (ROOT / "images/pusht_act_eval.gif", out_dir / "assets/pusht_act_eval.gif"),
        (ROOT / "images/pusht_diffusion_policy_eval.gif", out_dir / "assets/pusht_diffusion_policy_eval.gif"),
        (ROOT / "images/policy_ladder.svg", out_dir / "assets/policy_ladder.svg"),
    ]
    copied = [copy_file(src, dst) for src, dst in required_sources]

    summary = build_summary(run_dir, comparison, evidence_index, out_dir, copied)
    (out_dir / "SUBMISSION_README.md").write_text(summary, encoding="utf-8")
    manifest = {
        "run_dir": relative(run_dir),
        "comparison": relative(comparison),
        "evidence_index": relative(evidence_index),
        "out_dir": relative(out_dir),
        "files": copied,
        "boundary": "teaching-scale PushT-style imitation-learning loop, not real-robot deployment",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"submission pack: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
