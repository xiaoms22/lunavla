from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CORE_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "DATA_CARD.md",
    "MODEL_CARD.md",
    "RELEASE_NOTES.md",
    "configs/act_pusht_cpu_smoke.yaml",
    "configs/act_pusht_baseline.yaml",
    "configs/act_pusht_ablation_chunk_size.yaml",
    "scripts/run_cpu_smoke.py",
    "scripts/run_baseline_evidence.py",
    "scripts/run_ablation_evidence.py",
    "scripts/build_evidence_pack.py",
    "scripts/generate_project_report.py",
    "scripts/generate_resume_pack.py",
    "scripts/validate_configs.py",
    "scripts/web_demo_vla.py",
    "eval_vla.py",
]

README_ASSETS = [
    "images/minimind-vla-architecture.svg",
    "images/pusht_rollout.gif",
    "images/act_action_chunk.gif",
    "images/loss_curve.gif",
    "images/rollout_demo.png",
    "images/loss_curve_baseline.png",
    "images/result_table.svg",
    "images/asset_manifest.json",
]

GENERATED_EVIDENCE = [
    "outputs/dataset_inspection.md",
    "outputs/cpu_smoke/summary_report.md",
    "outputs/cpu_smoke/project_report.md",
    "outputs/cpu_smoke/resume_pack.md",
    "outputs/cpu_smoke/web_demo.html",
    "outputs/act_pusht_baseline/summary_report.md",
    "outputs/act_pusht_baseline/project_report.md",
    "outputs/act_pusht_baseline/resume_pack.md",
    "outputs/act_pusht_baseline/web_demo.html",
    "outputs/act_pusht_ablation_chunk_size/summary_report.md",
    "outputs/act_pusht_ablation_chunk_size/project_report.md",
    "outputs/act_pusht_ablation_chunk_size/resume_pack.md",
    "outputs/act_pusht_ablation_chunk_size/web_demo.html",
    "outputs/run_comparison.md",
    "outputs/run_comparison.csv",
    "outputs/run_comparison_deltas.csv",
    "outputs/evidence_index.md",
]

PUBLIC_COMMANDS = [
    "python scripts/run_cpu_smoke.py",
    "python scripts/inspect_dataset.py",
    "python scripts/validate_configs.py",
    "python scripts/run_baseline_evidence.py",
    "python trainer/train_act_pusht.py --config configs/act_pusht_baseline.yaml",
    "python eval_vla.py --checkpoint outputs/act_pusht_baseline/checkpoint.pt --episodes 50 --save-rollouts",
    "python scripts/summarize_results.py --run-dir outputs/act_pusht_baseline",
    "python scripts/generate_project_report.py --run-dir outputs/act_pusht_baseline",
    "python scripts/generate_resume_pack.py --run-dir outputs/act_pusht_baseline",
    "python scripts/export_readme_assets.py --run-dir outputs/act_pusht_baseline --out-dir images",
    "python scripts/run_ablation_evidence.py",
    "python scripts/build_evidence_pack.py --skip-runs",
]

README_REQUIRED_PHRASES = [
    "observation -> action -> rollout -> evaluation",
    "teaching-scale action-learning baseline",
    "Mock PushT is the low-cost teaching layer",
    "It is not a real-robot deployment benchmark",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MiniMind-VLA public release readiness.")
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


def main() -> int:
    args = parse_args()
    require_paths(CORE_FILES, "core release files")
    require_paths(README_ASSETS, "README assets")
    if not args.skip_generated:
        require_paths(GENERATED_EVIDENCE, "generated evidence artifacts")
    require_text("README.md", PUBLIC_COMMANDS)
    require_text("README.md", README_REQUIRED_PHRASES)
    check_readme_assets()
    check_cards_are_linked()
    check_markdown_links()
    print("release readiness check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
