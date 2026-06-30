from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CORE_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "DATA_CARD.md",
    "MODEL_CARD.md",
    "RELEASE_NOTES.md",
    "docs/visual_attributions.md",
    "docs/tutorials/action_chunking_act.md",
    "docs/tutorials/behavior_cloning_from_scratch.md",
    "docs/tutorials/action_normalization.md",
    "images/ecosystem/source_manifest.json",
    "configs/act_pusht_cpu_smoke.yaml",
    "configs/bc_pusht_cpu_smoke.yaml",
    "configs/act_pusht_baseline.yaml",
    "configs/act_pusht_ablation_chunk_size.yaml",
    "dataset/action_stats.py",
    "scripts/run_quickstart.py",
    "scripts/run_cpu_smoke.py",
    "scripts/run_bc_smoke.py",
    "scripts/run_baseline_evidence.py",
    "scripts/run_ablation_evidence.py",
    "scripts/generate_config_diff.py",
    "scripts/build_evidence_pack.py",
    "scripts/build_submission_pack.py",
    "scripts/generate_first_run_checklist.py",
    "scripts/generate_troubleshooting_guide.py",
    "scripts/generate_command_reference.py",
    "scripts/generate_code_walkthrough.py",
    "scripts/generate_action_chunk_lesson.py",
    "scripts/generate_action_statistics.py",
    "scripts/generate_policy_ladder.py",
    "scripts/prepare_homepage_media.py",
    "scripts/check_negative_paths.py",
    "scripts/check_task_layer.py",
    "scripts/check_policy_interface.py",
    "scripts/check_repo_quality.py",
    "scripts/check_environment.py",
    "scripts/check_readme_assets.py",
    "scripts/check_project_progress.py",
    "scripts/generate_project_report.py",
    "scripts/generate_project_card.py",
    "scripts/generate_experiment_ledger.py",
    "scripts/generate_learning_checkpoint.py",
    "scripts/generate_interview_flashcards.py",
    "scripts/generate_skill_evidence_map.py",
    "scripts/generate_showcase_issue.py",
    "scripts/generate_failure_review.py",
    "scripts/generate_resume_pack.py",
    "scripts/diagnose_run.py",
    "scripts/validate_configs.py",
    "scripts/web_demo_vla.py",
    "eval_vla.py",
]

README_ASSETS = [
    "images/lunavla-architecture.svg",
    "images/pusht_act_eval.gif",
    "images/pusht_diffusion_policy_eval.gif",
    "images/policy_ladder.svg",
    "images/ecosystem/lerobot_control_demo.webp",
    "images/ecosystem/lerobot_so100_demo.webp",
    "images/ecosystem/lerobot_vla_architecture.jpg",
    "images/ecosystem/libero_sim_overview.jpg",
    "images/asset_manifest.json",
]

GENERATED_EVIDENCE = [
    "outputs/environment_check.md",
    "outputs/quickstart_summary.md",
    "outputs/first_run_checklist.md",
    "outputs/troubleshooting_guide.md",
    "outputs/command_reference.md",
    "outputs/code_walkthrough.md",
    "outputs/action_chunk_lesson.md",
    "outputs/action_statistics.json",
    "outputs/action_statistics.md",
    "outputs/policy_ladder.md",
    "outputs/policy_ladder.csv",
    "outputs/readme_asset_check.md",
    "outputs/project_progress.md",
    "outputs/project_card.md",
    "outputs/experiment_ledger.md",
    "outputs/experiment_ledger.json",
    "outputs/learning_checkpoint.md",
    "outputs/interview_flashcards.md",
    "outputs/skill_evidence_map.md",
    "outputs/learner_showcase.md",
    "outputs/failure_review.md",
    "outputs/dataset_inspection.md",
    "outputs/cpu_smoke/summary_report.md",
    "outputs/cpu_smoke/action_statistics.json",
    "outputs/cpu_smoke/project_report.md",
    "outputs/cpu_smoke/resume_pack.md",
    "outputs/cpu_smoke/run_diagnostic.md",
    "outputs/cpu_smoke/web_demo.html",
    "outputs/bc_pusht_cpu_smoke/summary_report.md",
    "outputs/bc_pusht_cpu_smoke/action_statistics.json",
    "outputs/bc_pusht_cpu_smoke/project_report.md",
    "outputs/bc_pusht_cpu_smoke/run_diagnostic.md",
    "outputs/bc_pusht_cpu_smoke/web_demo.html",
    "outputs/act_pusht_baseline/summary_report.md",
    "outputs/act_pusht_baseline/action_statistics.json",
    "outputs/act_pusht_baseline/project_report.md",
    "outputs/act_pusht_baseline/resume_pack.md",
    "outputs/act_pusht_baseline/run_diagnostic.md",
    "outputs/act_pusht_baseline/web_demo.html",
    "outputs/act_pusht_ablation_chunk_size/summary_report.md",
    "outputs/act_pusht_ablation_chunk_size/action_statistics.json",
    "outputs/act_pusht_ablation_chunk_size/project_report.md",
    "outputs/act_pusht_ablation_chunk_size/resume_pack.md",
    "outputs/act_pusht_ablation_chunk_size/run_diagnostic.md",
    "outputs/act_pusht_ablation_chunk_size/web_demo.html",
    "outputs/run_comparison.md",
    "outputs/run_comparison.csv",
    "outputs/run_comparison_deltas.csv",
    "outputs/config_diff.md",
    "outputs/config_diff.json",
    "outputs/evidence_index.md",
    "outputs/submission_pack/SUBMISSION_README.md",
    "outputs/submission_pack/manifest.json",
    "outputs/submission_pack/quickstart_summary.md",
    "outputs/submission_pack/environment_check.md",
    "outputs/submission_pack/first_run_checklist.md",
    "outputs/submission_pack/troubleshooting_guide.md",
    "outputs/submission_pack/command_reference.md",
    "outputs/submission_pack/code_walkthrough.md",
    "outputs/submission_pack/action_chunk_lesson.md",
    "outputs/submission_pack/action_statistics.json",
    "outputs/submission_pack/action_statistics.md",
    "outputs/submission_pack/policy_ladder.md",
    "outputs/submission_pack/policy_ladder.csv",
    "outputs/submission_pack/readme_asset_check.md",
    "outputs/submission_pack/project_progress.md",
    "outputs/submission_pack/project_card.md",
    "outputs/submission_pack/experiment_ledger.md",
    "outputs/submission_pack/experiment_ledger.json",
    "outputs/submission_pack/learning_checkpoint.md",
    "outputs/submission_pack/interview_flashcards.md",
    "outputs/submission_pack/skill_evidence_map.md",
    "outputs/submission_pack/learner_showcase.md",
    "outputs/submission_pack/failure_review.md",
    "outputs/submission_pack/project_report.md",
    "outputs/submission_pack/resume_pack.md",
    "outputs/submission_pack/run_diagnostic.md",
    "outputs/submission_pack/rollout_browser.html",
    "outputs/submission_pack/ablation_comparison.md",
    "outputs/submission_pack/config_diff.md",
    "outputs/submission_pack/config_diff.json",
    "outputs/submission_pack/evidence_index.md",
    "outputs/submission_pack/assets/pusht_act_eval.gif",
    "outputs/submission_pack/assets/pusht_diffusion_policy_eval.gif",
    "outputs/submission_pack/assets/policy_ladder.svg",
]

PUBLIC_COMMANDS = [
    "python scripts/run_cpu_smoke.py",
    "python scripts/run_bc_smoke.py",
    "python scripts/run_quickstart.py",
    "python scripts/generate_first_run_checklist.py",
    "python scripts/generate_troubleshooting_guide.py",
    "python scripts/generate_command_reference.py",
    "python scripts/generate_code_walkthrough.py",
    "python scripts/generate_action_chunk_lesson.py",
    "python scripts/generate_action_statistics.py",
    "python scripts/generate_policy_ladder.py",
    "python scripts/prepare_homepage_media.py",
    "python scripts/check_negative_paths.py",
    "python scripts/check_task_layer.py",
    "python scripts/check_policy_interface.py",
    "python scripts/check_environment.py",
    "python scripts/check_readme_assets.py",
    "python scripts/check_project_progress.py",
    "python scripts/generate_project_card.py",
    "python scripts/generate_experiment_ledger.py",
    "python scripts/generate_learning_checkpoint.py",
    "python scripts/generate_interview_flashcards.py",
    "python scripts/generate_skill_evidence_map.py",
    "python scripts/generate_showcase_issue.py",
    "python scripts/generate_failure_review.py",
    "python scripts/inspect_dataset.py",
    "python scripts/validate_configs.py",
    "python scripts/run_baseline_evidence.py",
    "python trainer/train_act_pusht.py --config configs/act_pusht_baseline.yaml",
    "python eval_vla.py --checkpoint outputs/act_pusht_baseline/checkpoint.pt --episodes 50 --save-rollouts",
    "python scripts/summarize_results.py --run-dir outputs/act_pusht_baseline",
    "python scripts/generate_project_report.py --run-dir outputs/act_pusht_baseline",
    "python scripts/generate_resume_pack.py --run-dir outputs/act_pusht_baseline",
    "python scripts/diagnose_run.py --run-dir outputs/act_pusht_baseline",
    "python scripts/export_readme_assets.py --run-dir outputs/act_pusht_baseline --out-dir images",
    "python scripts/run_ablation_evidence.py",
    "python scripts/generate_config_diff.py",
    "python scripts/build_evidence_pack.py --skip-runs",
    "python scripts/build_submission_pack.py",
]

README_REQUIRED_PHRASES = [
    "observation -> action -> rollout -> evaluation",
    "teaching-scale action-learning baseline",
    "Mock PushT is the low-cost teaching layer",
    "It is not a real-robot deployment benchmark",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LunaVLA public release readiness.")
    parser.add_argument(
        "--skip-generated",
        action="store_true",
        help="Only check tracked release materials, not generated outputs under outputs/.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"release readiness check failed: {message}")


def require_paths(paths: list[str], label: str) -> None:
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        fail(f"missing {label}: " + ", ".join(missing))


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require_text(path: str, phrases: list[str]) -> None:
    text = read_text(path)
    for phrase in phrases:
        if phrase not in text:
            fail(f"{path} is missing `{phrase}`")


def readme_python_commands() -> list[str]:
    text = read_text("README.md")
    commands = set(re.findall(r"`(python [^`]+)`", text))
    for block in re.findall(r"```(?:bash)?\n(.*?)```", text, re.S):
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if line.startswith("python "):
                commands.add(line)
    return sorted(commands)


def command_reference_commands() -> set[str]:
    path = ROOT / "scripts" / "generate_command_reference.py"
    spec = importlib.util.spec_from_file_location("generate_command_reference", path)
    if spec is None or spec.loader is None:
        fail("could not load scripts/generate_command_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = getattr(module, "COMMANDS", None)
    if not isinstance(commands, list):
        fail("scripts/generate_command_reference.py does not expose COMMANDS")
    return {str(item.get("command", "")) for item in commands if isinstance(item, dict)}


def check_command_reference_covers_readme() -> None:
    reference_commands = command_reference_commands()
    required_commands = sorted(set(PUBLIC_COMMANDS) | set(readme_python_commands()))
    missing = [command for command in required_commands if command not in reference_commands]
    if missing:
        fail("command reference missing public command(s): " + ", ".join(f"`{command}`" for command in missing))


def iter_markdown_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    return [link.split("#", 1)[0].strip() for link in links]


def check_markdown_links() -> None:
    markdown_files = [ROOT / "README.md", ROOT / "DATA_CARD.md", ROOT / "MODEL_CARD.md", ROOT / "RELEASE_NOTES.md"]
    markdown_files.extend(sorted((ROOT / "docs").glob("**/*.md")))
    for path in markdown_files:
        if not path.exists():
            continue
        for target in iter_markdown_links(path):
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (ROOT / target).exists():
                fail(f"{path.relative_to(ROOT).as_posix()} links to missing path `{target}`")


def check_cards_are_linked() -> None:
    readme = read_text("README.md")
    for card in ["DATA_CARD.md", "MODEL_CARD.md", "RELEASE_NOTES.md"]:
        if f"]({card})" not in readme and f"`{card}`" not in readme:
            fail(f"README.md should link or mention {card}")


def check_readme_assets() -> None:
    readme = read_text("README.md")
    for asset in README_ASSETS[:-1]:
        if asset not in readme:
            fail(f"README.md should reference {asset}")


def check_negative_paths() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_negative_paths.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
        fail("negative path checks failed" + (f": {details}" if details else ""))


def check_task_layer() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_task_layer.py",
            "--run-dir",
            "outputs/act_pusht_baseline",
            "--require-generated",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
        fail("task layer checks failed" + (f": {details}" if details else ""))


def check_policy_interface() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_policy_interface.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
        fail("policy interface checks failed" + (f": {details}" if details else ""))


def main() -> int:
    args = parse_args()
    require_paths(CORE_FILES, "core release files")
    require_paths(README_ASSETS, "README assets")
    if not args.skip_generated:
        require_paths(GENERATED_EVIDENCE, "generated evidence artifacts")
    require_text("README.md", PUBLIC_COMMANDS)
    require_text("README.md", README_REQUIRED_PHRASES)
    check_command_reference_covers_readme()
    check_readme_assets()
    check_cards_are_linked()
    check_markdown_links()
    check_negative_paths()
    check_task_layer()
    check_policy_interface()
    print("release readiness check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
