from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


STATUS_FILES = [
    ("environment", "outputs/environment_check.md", "python scripts/check_environment.py"),
    ("first run", "outputs/first_run_checklist.md", "python scripts/generate_first_run_checklist.py"),
    ("project progress", "outputs/project_progress.md", "python scripts/check_project_progress.py"),
    ("CPU smoke diagnostic", "outputs/cpu_smoke/run_diagnostic.md", "python scripts/run_cpu_smoke.py"),
    (
        "baseline diagnostic",
        "outputs/act_pusht_baseline/run_diagnostic.md",
        "python scripts/run_baseline_evidence.py",
    ),
    (
        "ablation diagnostic",
        "outputs/act_pusht_ablation_chunk_size/run_diagnostic.md",
        "python scripts/run_ablation_evidence.py",
    ),
    ("README assets", "outputs/readme_asset_check.md", "python scripts/check_readme_assets.py"),
]


TROUBLESHOOTING_ROWS = [
    {
        "symptom": "install or import fails",
        "first file to open": "outputs/environment_check.md",
        "likely cause": "missing Python package, wrong interpreter, or non-writable output folder",
        "fix command": "pip install -r requirements.txt; python scripts/check_environment.py",
    },
    {
        "symptom": "CPU smoke did not create reports",
        "first file to open": "outputs/first_run_checklist.md",
        "likely cause": "smoke command stopped before training, eval, summary, or web demo completed",
        "fix command": "python scripts/run_cpu_smoke.py; python scripts/generate_first_run_checklist.py",
    },
    {
        "symptom": "README image or GIF is missing",
        "first file to open": "outputs/readme_asset_check.md",
        "likely cause": "README assets were not exported from the baseline run",
        "fix command": "python scripts/run_baseline_evidence.py; python scripts/check_readme_assets.py",
    },
    {
        "symptom": "baseline result looks weak",
        "first file to open": "outputs/act_pusht_baseline/run_diagnostic.md",
        "likely cause": "short eval, failed rollouts, high final distance, or incomplete run artifacts",
        "fix command": "python scripts/diagnose_run.py --run-dir outputs/act_pusht_baseline",
    },
    {
        "symptom": "ablation conclusion feels vague",
        "first file to open": "outputs/run_comparison.md",
        "likely cause": "comparison report was not generated or metrics were not cited",
        "fix command": "python scripts/run_ablation_evidence.py",
    },
    {
        "symptom": "too many generated files to review",
        "first file to open": "outputs/evidence_index.md",
        "likely cause": "the run produced evidence, but the review order is unclear",
        "fix command": "python scripts/build_evidence_pack.py --skip-runs",
    },
    {
        "symptom": "not sure what to say in a report or interview",
        "first file to open": "outputs/skill_evidence_map.md",
        "likely cause": "claims are not yet tied to code files and run artifacts",
        "fix command": "python scripts/generate_skill_evidence_map.py",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a public LunaVLA troubleshooting guide.")
    parser.add_argument("--out", default="outputs/troubleshooting_guide.md", help="Markdown troubleshooting path.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_text(path_text: str) -> str:
    path = resolve(path_text)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def extract_status(text: str) -> str:
    if not text:
        return "missing"
    patterns = [
        r"Overall:\s*`([^`]+)`",
        r"Status:\s*`([^`]+)`",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    if "| fail |" in text:
        return "fail"
    if "| warn |" in text:
        return "warn"
    if "| pass |" in text:
        return "pass"
    return "present"


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines


def status_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, path, command in STATUS_FILES:
        artifact = resolve(path)
        rows.append(
            {
                "check": name,
                "status": extract_status(read_text(path)),
                "artifact": path,
                "exists": "yes" if artifact.exists() else "no",
                "rebuild command": f"`{command}`",
            }
        )
    return rows


def overall_status(rows: list[dict[str, str]]) -> str:
    statuses = {row["status"] for row in rows}
    if "missing" in statuses or "fail" in statuses or "needs attention" in statuses:
        return "needs attention"
    if "warn" in statuses or "partial" in statuses or "ready with optional gaps" in statuses:
        return "usable with warnings"
    return "ready"


def build_report() -> str:
    rows = status_rows()
    lines: list[str] = [
        "# LunaVLA Troubleshooting Guide",
        "",
        "Use this guide when a public command fails, an artifact is missing, or you are not sure which generated file to inspect first.",
        "",
        f"Overall: `{overall_status(rows)}`",
        "",
        "## Current Status",
        "",
    ]
    lines.extend(markdown_table(rows))
    lines.extend(["", "## Common Symptoms", ""])
    lines.extend(markdown_table(TROUBLESHOOTING_ROWS))
    lines.extend(
        [
            "",
            "## Recovery Order",
            "",
            "1. Run `python scripts/check_environment.py`.",
            "2. Run `python scripts/run_cpu_smoke.py`.",
            "3. Run `python scripts/generate_first_run_checklist.py`.",
            "4. Run `python scripts/run_baseline_evidence.py`.",
            "5. Run `python scripts/build_evidence_pack.py --skip-runs`.",
            "6. Run `python scripts/check_release_readiness.py`.",
            "",
            "## Claim Boundary",
            "",
            "Troubleshooting should improve reproducibility and explanation quality. It should not turn a weak run into a stronger claim. Keep public claims tied to generated metrics, rollout artifacts, and diagnostics.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_troubleshooting_guide.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(), encoding="utf-8")
    print(f"troubleshooting guide: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
