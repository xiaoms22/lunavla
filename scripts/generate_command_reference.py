from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


COMMANDS: list[dict[str, Any]] = [
    {
        "stage": "start",
        "command": "python scripts/run_quickstart.py",
        "purpose": "Run the shortest beginner path.",
        "outputs": [
            "outputs/quickstart_summary.md",
            "outputs/environment_check.md",
            "outputs/dataset_inspection.md",
            "outputs/cpu_smoke/summary_report.md",
            "outputs/first_run_checklist.md",
            "outputs/troubleshooting_guide.md",
        ],
        "next": "Open the quickstart summary and first-run checklist.",
    },
    {
        "stage": "start",
        "command": "python scripts/run_quickstart.py --skip-run",
        "purpose": "Refresh quickstart guidance without rerunning the smoke loop.",
        "outputs": [
            "outputs/quickstart_summary.md",
            "outputs/first_run_checklist.md",
            "outputs/troubleshooting_guide.md",
        ],
        "next": "Use this after generated run artifacts already exist.",
    },
    {
        "stage": "check",
        "command": "python scripts/check_environment.py",
        "purpose": "Check Python, dependencies, key files, and output write access.",
        "outputs": ["outputs/environment_check.md"],
        "next": "Fix missing dependencies before training.",
    },
    {
        "stage": "validate",
        "command": "python scripts/validate_configs.py",
        "purpose": "Check runnable config structure before training.",
        "outputs": [],
        "next": "Fix config errors before launching evidence commands.",
    },
    {
        "stage": "release guard",
        "command": "python scripts/check_negative_paths.py",
        "purpose": "Confirm release utilities reject malformed or incomplete inputs clearly.",
        "outputs": [],
        "next": "Fix validation errors before running release readiness.",
    },
    {
        "stage": "task layer",
        "command": "python scripts/check_task_layer.py",
        "purpose": "Verify generated records, rollout frames, summaries, reports, and the browser expose Task Layer context.",
        "outputs": [],
        "next": "Regenerate smoke or baseline artifacts if generated Task Layer evidence is missing.",
    },
    {
        "stage": "policy interface",
        "command": "python scripts/check_policy_interface.py",
        "purpose": "Verify forward, predict_action, save_pretrained, and from_pretrained on the tiny policy contract.",
        "outputs": [],
        "next": "Fix policy interface compatibility before adding BC, ACT, or future policy variants.",
    },
    {
        "stage": "bc smoke",
        "command": "python scripts/run_bc_smoke.py",
        "purpose": "Train and evaluate the from-scratch behavior cloning MLP smoke baseline.",
        "outputs": [
            "outputs/bc_pusht_cpu_smoke/checkpoint.pt",
            "outputs/bc_pusht_cpu_smoke/summary_report.md",
            "outputs/bc_pusht_cpu_smoke/project_report.md",
            "outputs/bc_pusht_cpu_smoke/run_diagnostic.md",
            "outputs/bc_pusht_cpu_smoke/web_demo.html",
        ],
        "next": "Compare BC's one-step supervised objective against ACT-style action chunking.",
    },
    {
        "stage": "policy ladder",
        "command": "python scripts/generate_policy_ladder.py",
        "purpose": "Compare BC and ACT-style policies as a beginner-readable policy ladder.",
        "outputs": [
            "outputs/policy_ladder.md",
            "outputs/policy_ladder.csv",
            "images/policy_ladder.svg",
        ],
        "next": "Use the ladder to explain why rollout evidence matters beyond supervised loss.",
    },
    {
        "stage": "find",
        "command": "python scripts/generate_command_reference.py",
        "purpose": "Generate the public command-to-artifact map.",
        "outputs": ["outputs/command_reference.md"],
        "next": "Use the reference to choose the next runnable command.",
    },
    {
        "stage": "read data",
        "command": "python scripts/inspect_dataset.py",
        "purpose": "Inspect one observation, action, episode id, and action chunk target.",
        "outputs": ["outputs/dataset_inspection.md"],
        "next": "Read the dataset and policy files with the sample in mind.",
    },
    {
        "stage": "read code",
        "command": "python scripts/generate_code_walkthrough.py",
        "purpose": "Build a guided reading order for the runnable code path.",
        "outputs": ["outputs/code_walkthrough.md"],
        "next": "Read files in the order shown by the walkthrough.",
    },
    {
        "stage": "learn ACT",
        "command": "python scripts/generate_action_chunk_lesson.py",
        "purpose": "Explain how one sample becomes an ACT-style future-action chunk.",
        "outputs": ["outputs/action_chunk_lesson.md"],
        "next": "Use the lesson to explain chunk size, target shape, and rollout metrics.",
    },
    {
        "stage": "action stats",
        "command": "python scripts/generate_action_statistics.py",
        "purpose": "Compute demonstration action scale and explain normalization boundaries.",
        "outputs": [
            "outputs/action_statistics.json",
            "outputs/action_statistics.md",
            "outputs/act_pusht_baseline/action_statistics.json",
        ],
        "next": "Use mean/std and clip fraction to explain why action scale matters in rollout.",
    },
    {
        "stage": "smoke",
        "command": "python scripts/run_cpu_smoke.py",
        "purpose": "Train, evaluate, summarize, and export the smallest local loop.",
        "outputs": [
            "outputs/cpu_smoke/checkpoint.pt",
            "outputs/cpu_smoke/summary_report.md",
            "outputs/cpu_smoke/project_report.md",
            "outputs/cpu_smoke/resume_pack.md",
            "outputs/cpu_smoke/run_diagnostic.md",
            "outputs/cpu_smoke/web_demo.html",
        ],
        "next": "Use the smoke report to confirm the pipeline works.",
    },
    {
        "stage": "first run",
        "command": "python scripts/generate_first_run_checklist.py",
        "purpose": "Turn smoke artifacts into a small readiness checklist.",
        "outputs": ["outputs/first_run_checklist.md"],
        "next": "Use missing checklist items to choose the next command.",
    },
    {
        "stage": "debug",
        "command": "python scripts/generate_troubleshooting_guide.py",
        "purpose": "Map common missing-file and weak-run symptoms to recovery commands.",
        "outputs": ["outputs/troubleshooting_guide.md"],
        "next": "Run the recovery command that matches the symptom.",
    },
    {
        "stage": "baseline",
        "command": "python scripts/run_baseline_evidence.py",
        "purpose": "Build the main train, eval, report, resume, and README asset path.",
        "outputs": [
            "outputs/act_pusht_baseline/checkpoint.pt",
            "outputs/act_pusht_baseline/summary_report.md",
            "outputs/act_pusht_baseline/project_report.md",
            "outputs/act_pusht_baseline/resume_pack.md",
            "outputs/act_pusht_baseline/run_diagnostic.md",
            "outputs/act_pusht_baseline/web_demo.html",
            "images/pusht_act_eval.gif",
            "images/pusht_diffusion_policy_eval.gif",
            "images/asset_manifest.json",
        ],
        "next": "Open the project report, diagnostic, and rollout browser.",
    },
    {
        "stage": "baseline train",
        "command": "python trainer/train_act_pusht.py --config configs/act_pusht_baseline.yaml",
        "purpose": "Train the documented ACT PushT-style baseline.",
        "outputs": [
            "outputs/act_pusht_baseline/checkpoint.pt",
            "outputs/act_pusht_baseline/train_summary.json",
        ],
        "next": "Run baseline evaluation with the saved checkpoint.",
    },
    {
        "stage": "baseline eval",
        "command": "python eval_vla.py --checkpoint outputs/act_pusht_baseline/checkpoint.pt --episodes 50 --save-rollouts",
        "purpose": "Evaluate the baseline checkpoint and save rollout evidence.",
        "outputs": [
            "outputs/act_pusht_baseline/eval_summary.json",
            "outputs/act_pusht_baseline/rollouts",
        ],
        "next": "Summarize the run and generate report artifacts.",
    },
    {
        "stage": "baseline summary",
        "command": "python scripts/summarize_results.py --run-dir outputs/act_pusht_baseline",
        "purpose": "Turn baseline metrics into a compact Markdown summary.",
        "outputs": ["outputs/act_pusht_baseline/summary_report.md"],
        "next": "Use the summary in the project report and diagnostics.",
    },
    {
        "stage": "baseline report",
        "command": "python scripts/generate_project_report.py --run-dir outputs/act_pusht_baseline",
        "purpose": "Generate a report draft from baseline metrics and artifacts.",
        "outputs": ["outputs/act_pusht_baseline/project_report.md"],
        "next": "Review the report boundaries before citing metrics.",
    },
    {
        "stage": "baseline resume",
        "command": "python scripts/generate_resume_pack.py --run-dir outputs/act_pusht_baseline",
        "purpose": "Generate resume-safe bullets from baseline evidence.",
        "outputs": ["outputs/act_pusht_baseline/resume_pack.md"],
        "next": "Use only bullets backed by generated artifacts.",
    },
    {
        "stage": "baseline diagnose",
        "command": "python scripts/diagnose_run.py --run-dir outputs/act_pusht_baseline",
        "purpose": "Check baseline artifacts, metrics, and claim boundaries.",
        "outputs": ["outputs/act_pusht_baseline/run_diagnostic.md"],
        "next": "Fix missing artifacts before packaging the run.",
    },
    {
        "stage": "baseline assets",
        "command": "python scripts/export_readme_assets.py --run-dir outputs/act_pusht_baseline --out-dir images",
        "purpose": "Refresh README-visible run assets from baseline evidence.",
        "outputs": [
            "images/pusht_act_eval.gif",
            "images/asset_manifest.json",
        ],
        "next": "Run README asset checks after export.",
    },
    {
        "stage": "baseline lesson",
        "command": "python scripts/generate_action_chunk_lesson.py --config configs/act_pusht_baseline.yaml --run-dir outputs/act_pusht_baseline",
        "purpose": "Regenerate the ACT/action-chunk lesson from the baseline config and checkpoint.",
        "outputs": ["outputs/action_chunk_lesson.md"],
        "next": "Use the lesson to explain action horizon and rollout behavior.",
    },
    {
        "stage": "assets",
        "command": "python scripts/prepare_homepage_media.py",
        "purpose": "Refresh homepage ecosystem media and the source manifest.",
        "outputs": [
            "images/ecosystem/source_manifest.json",
            "images/ecosystem/lerobot_control_demo.webp",
            "images/ecosystem/lerobot_so100_demo.webp",
            "images/ecosystem/lerobot_vla_architecture.jpg",
            "images/ecosystem/libero_sim_overview.jpg",
        ],
        "next": "Check visual attribution and README assets before release.",
    },
    {
        "stage": "assets",
        "command": "python scripts/check_readme_assets.py",
        "purpose": "Check that README images and animations exist and are usable.",
        "outputs": ["outputs/readme_asset_check.md"],
        "next": "Regenerate assets if any README visual is missing.",
    },
    {
        "stage": "progress",
        "command": "python scripts/check_project_progress.py",
        "purpose": "Check which generated artifacts are ready for project reporting.",
        "outputs": ["outputs/project_progress.md"],
        "next": "Generate any missing evidence before building the pack.",
    },
    {
        "stage": "ablation",
        "command": "python scripts/run_ablation_evidence.py",
        "purpose": "Run a chunk-size comparison against the baseline.",
        "outputs": [
            "outputs/act_pusht_ablation_chunk_size/summary_report.md",
            "outputs/act_pusht_ablation_chunk_size/project_report.md",
            "outputs/act_pusht_ablation_chunk_size/resume_pack.md",
            "outputs/act_pusht_ablation_chunk_size/run_diagnostic.md",
            "outputs/run_comparison.md",
        ],
        "next": "Use only the comparison report for ablation claims.",
    },
    {
        "stage": "ablation train",
        "command": "python trainer/train_act_pusht.py --config configs/act_pusht_ablation_chunk_size.yaml",
        "purpose": "Train the documented chunk-size ablation.",
        "outputs": [
            "outputs/act_pusht_ablation_chunk_size/checkpoint.pt",
            "outputs/act_pusht_ablation_chunk_size/train_summary.json",
        ],
        "next": "Evaluate the ablation checkpoint with saved rollouts.",
    },
    {
        "stage": "ablation eval",
        "command": "python eval_vla.py --checkpoint outputs/act_pusht_ablation_chunk_size/checkpoint.pt --episodes 50 --save-rollouts",
        "purpose": "Evaluate the ablation checkpoint and save rollout evidence.",
        "outputs": [
            "outputs/act_pusht_ablation_chunk_size/eval_summary.json",
            "outputs/act_pusht_ablation_chunk_size/rollouts",
        ],
        "next": "Summarize the ablation run before comparison.",
    },
    {
        "stage": "ablation summary",
        "command": "python scripts/summarize_results.py --run-dir outputs/act_pusht_ablation_chunk_size",
        "purpose": "Turn ablation metrics into a compact Markdown summary.",
        "outputs": ["outputs/act_pusht_ablation_chunk_size/summary_report.md"],
        "next": "Compare the ablation against the baseline.",
    },
    {
        "stage": "ablation report",
        "command": "python scripts/generate_project_report.py --run-dir outputs/act_pusht_ablation_chunk_size",
        "purpose": "Generate a report draft from ablation metrics and artifacts.",
        "outputs": ["outputs/act_pusht_ablation_chunk_size/project_report.md"],
        "next": "Use the comparison report for cross-run claims.",
    },
    {
        "stage": "ablation diagnose",
        "command": "python scripts/diagnose_run.py --run-dir outputs/act_pusht_ablation_chunk_size",
        "purpose": "Check ablation artifacts, metrics, and claim boundaries.",
        "outputs": ["outputs/act_pusht_ablation_chunk_size/run_diagnostic.md"],
        "next": "Fix missing ablation evidence before comparing runs.",
    },
    {
        "stage": "ablation compare",
        "command": "python scripts/compare_runs.py --runs outputs/act_pusht_baseline outputs/act_pusht_ablation_chunk_size --out outputs/run_comparison.md",
        "purpose": "Compare baseline and ablation metrics side by side.",
        "outputs": [
            "outputs/run_comparison.md",
            "outputs/run_comparison.csv",
            "outputs/run_comparison_deltas.csv",
        ],
        "next": "Use this file, not raw guesses, for ablation claims.",
    },
    {
        "stage": "config diff",
        "command": "python scripts/generate_config_diff.py",
        "purpose": "Check which config fields changed between baseline and ablation.",
        "outputs": ["outputs/config_diff.md", "outputs/config_diff.json"],
        "next": "Confirm the ablation changed only the intended experiment variable.",
    },
    {
        "stage": "ablation resume",
        "command": "python scripts/generate_resume_pack.py --run-dir outputs/act_pusht_baseline --comparison outputs/run_comparison.md",
        "purpose": "Regenerate resume-safe bullets with comparison evidence included.",
        "outputs": ["outputs/act_pusht_baseline/resume_pack.md"],
        "next": "Keep claims tied to the generated comparison.",
    },
    {
        "stage": "review",
        "command": "python scripts/generate_failure_review.py",
        "purpose": "Summarize rollout failure cases across generated runs.",
        "outputs": ["outputs/failure_review.md"],
        "next": "Use failure behavior to write a more honest report.",
    },
    {
        "stage": "learn",
        "command": "python scripts/generate_learning_checkpoint.py",
        "purpose": "Map core VLA concepts to code, reports, and self-check questions.",
        "outputs": ["outputs/learning_checkpoint.md"],
        "next": "Answer the self-check questions before writing the report.",
    },
    {
        "stage": "practice",
        "command": "python scripts/generate_interview_flashcards.py",
        "purpose": "Generate evidence-backed interview flashcards.",
        "outputs": ["outputs/interview_flashcards.md"],
        "next": "Practice answers using file and metric evidence.",
    },
    {
        "stage": "map skills",
        "command": "python scripts/generate_skill_evidence_map.py",
        "purpose": "Connect project skills to code files, commands, and artifacts.",
        "outputs": ["outputs/skill_evidence_map.md"],
        "next": "Use the map to choose resume-safe evidence.",
    },
    {
        "stage": "card",
        "command": "python scripts/generate_project_card.py",
        "purpose": "Compress commands, metrics, files, and boundaries into one page.",
        "outputs": ["outputs/project_card.md"],
        "next": "Use the card as the fastest review artifact.",
    },
    {
        "stage": "audit",
        "command": "python scripts/generate_experiment_ledger.py",
        "purpose": "Record commands, config hashes, metrics, and artifact coverage.",
        "outputs": ["outputs/experiment_ledger.md", "outputs/experiment_ledger.json"],
        "next": "Use the ledger to support report and interview claims.",
    },
    {
        "stage": "share",
        "command": "python scripts/generate_showcase_issue.py",
        "purpose": "Generate a copyable learner showcase draft.",
        "outputs": ["outputs/learner_showcase.md"],
        "next": "Edit the draft with your own metrics and screenshots.",
    },
    {
        "stage": "package",
        "command": "python scripts/build_evidence_pack.py --skip-runs",
        "purpose": "Create a single evidence index after core artifacts exist.",
        "outputs": ["outputs/evidence_index.md"],
        "next": "Use the index to find report-ready files.",
    },
    {
        "stage": "submit",
        "command": "python scripts/build_submission_pack.py",
        "purpose": "Collect key generated files into one review folder.",
        "outputs": [
            "outputs/submission_pack/SUBMISSION_README.md",
            "outputs/submission_pack/manifest.json",
        ],
        "next": "Open the submission README for final review.",
    },
    {
        "stage": "release check",
        "command": "python scripts/check_release_readiness.py",
        "purpose": "Check public files, generated evidence, links, and required commands.",
        "outputs": [],
        "next": "Fix missing files or links before sharing the repo.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a LunaVLA public command reference.")
    parser.add_argument("--out", default="outputs/command_reference.md", help="Markdown command reference path.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def exists_label(paths: list[str]) -> str:
    if not paths:
        return "n/a"
    found = sum(1 for path in paths if resolve(path).exists())
    return f"{found}/{len(paths)}"


def format_outputs(paths: list[str]) -> str:
    if not paths:
        return "No file output."
    return "<br>".join(f"`{path}`" for path in paths)


def markdown_table(rows: list[dict[str, str]]) -> list[str]:
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    return lines


def build_reference() -> str:
    rows = [
        {
            "stage": item["stage"],
            "command": f"`{item['command']}`",
            "why run it": item["purpose"],
            "main output": format_outputs(item["outputs"]),
            "ready": exists_label(item["outputs"]),
        }
        for item in COMMANDS
    ]

    lines: list[str] = [
        "# LunaVLA Command Reference",
        "",
        "This file maps public commands to the learning loop they support and the artifacts they generate.",
        "",
        "## Suggested Order",
        "",
        "1. Start with `python scripts/run_quickstart.py`.",
        "2. Open `outputs/quickstart_summary.md` and `outputs/first_run_checklist.md`.",
        "3. Read `outputs/action_chunk_lesson.md` to understand the ACT-style target.",
        "4. Run `python scripts/run_baseline_evidence.py` when the smoke loop is clear.",
        "5. Run `python scripts/generate_action_statistics.py` to inspect action scale.",
        "6. Run `python scripts/generate_policy_ladder.py` after BC smoke and baseline evidence exist.",
        "7. Run `python scripts/generate_action_chunk_lesson.py` again after the baseline checkpoint exists.",
        "8. Run `python scripts/run_ablation_evidence.py` after the baseline report exists.",
        "9. Finish with `python scripts/build_evidence_pack.py --skip-runs` and `python scripts/build_submission_pack.py`.",
        "",
        "## Command Map",
        "",
    ]
    lines.extend(markdown_table(rows))
    lines.extend(
        [
            "",
            "## Report-Friendly Reading Path",
            "",
            "- To explain data: read `outputs/dataset_inspection.md`.",
            "- To read code in order: read `outputs/code_walkthrough.md`.",
            "- To explain ACT/action chunking: read `outputs/action_chunk_lesson.md`.",
            "- To explain action scale and normalization: read `outputs/action_statistics.md`.",
            "- To explain BC vs ACT: read `outputs/policy_ladder.md`.",
            "- To explain training and evaluation: read `outputs/act_pusht_baseline/project_report.md`.",
            "- To explain what the run proves: read `outputs/act_pusht_baseline/run_diagnostic.md`.",
            "- To audit configs and metrics: read `outputs/experiment_ledger.md`.",
            "- To explain comparison results: read `outputs/run_comparison.md`.",
            "- To explain ablation setup: read `outputs/config_diff.md`.",
            "- To prepare for interviews: read `outputs/interview_flashcards.md` and `outputs/skill_evidence_map.md`.",
            "- To package the work: read `outputs/project_card.md` and `outputs/submission_pack/SUBMISSION_README.md`.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_command_reference.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_reference(), encoding="utf-8")
    print(f"command reference: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
