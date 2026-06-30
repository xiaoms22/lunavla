from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


COMMAND_CHECKS = [
    {
        "check": "one-command starter",
        "command": "python scripts/run_quickstart.py",
        "artifact": "outputs/quickstart_summary.md",
        "why": "A new learner can start without choosing scripts manually.",
    },
    {
        "check": "CPU smoke loop",
        "command": "python scripts/run_cpu_smoke.py",
        "artifact": "outputs/cpu_smoke/summary_report.md",
        "why": "The smallest train/eval/report loop is runnable.",
    },
    {
        "check": "baseline evidence",
        "command": "python scripts/run_baseline_evidence.py",
        "artifact": "outputs/act_pusht_baseline/project_report.md",
        "why": "The main ACT-style PushT path produces a report.",
    },
    {
        "check": "ablation evidence",
        "command": "python scripts/run_ablation_evidence.py",
        "artifact": "outputs/run_comparison.md",
        "why": "The project can explain one controlled variable change.",
    },
    {
        "check": "homepage result card",
        "command": "python scripts/generate_homepage_summary.py",
        "artifact": "outputs/homepage_summary.md",
        "why": "README metrics are tied to generated evidence.",
    },
    {
        "check": "submission pack",
        "command": "python scripts/build_submission_pack.py",
        "artifact": "outputs/submission_pack/SUBMISSION_README.md",
        "why": "A reviewer can inspect one compact evidence folder.",
    },
]


ARTIFACT_CHECKS = [
    ("README result card", "images/homepage_results.svg"),
    ("ACT PushT media", "images/pusht_act_eval.gif"),
    ("Diffusion Policy PushT media", "images/pusht_diffusion_policy_eval.gif"),
    ("policy ladder visual", "images/policy_ladder.svg"),
    ("command reference", "outputs/command_reference.md"),
    ("project progress", "outputs/project_progress.md"),
    ("project card", "outputs/project_card.md"),
    ("experiment ledger", "outputs/experiment_ledger.md"),
    ("evidence index", "outputs/evidence_index.md"),
    ("submission pack manifest", "outputs/submission_pack/manifest.json"),
    ("run diagnostic", "outputs/act_pusht_baseline/run_diagnostic.md"),
    ("resume pack", "outputs/act_pusht_baseline/resume_pack.md"),
    ("rollout browser", "outputs/act_pusht_baseline/web_demo.html"),
]


BOUNDARY_CHECKS = [
    {
        "check": "README scope boundary",
        "path": "README.md",
        "phrase": "It is not a real-robot deployment benchmark",
    },
    {
        "check": "README project identity",
        "path": "README.md",
        "phrase": "independent educational project",
    },
    {
        "check": "interview claim hygiene",
        "path": "docs/internship_pack/03_interview_qa.md",
        "phrase": "Tie every claim to a command, config, metric, and artifact.",
    },
    {
        "check": "completion rule",
        "path": "docs/internship_pack/06_4_week_project_path.md",
        "phrase": "every resume/interview claim is supported by a file",
    },
]


LEAK_PATTERNS = [
    "star_" + "playbook",
    "code_" + "analysis",
    "sync_" + "upstreams",
    "lunavla-" + "internal",
    "internal-" + "ai-guide",
    "future " + "adapter",
    "Not" + "Implemented",
]


LEAK_SCAN_PATHS = [
    "README.md",
    "DATA_CARD.md",
    "MODEL_CARD.md",
    "RELEASE_NOTES.md",
    "docs",
    "configs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether LunaVLA is ready for an external reviewer.")
    parser.add_argument("--out", default="outputs/reviewer_readiness.md", help="Markdown readiness report path.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any check fails.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_text(path: str | Path) -> str:
    resolved = resolve(path)
    return resolved.read_text(encoding="utf-8") if resolved.exists() else ""


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines


def status_label(ok: bool) -> str:
    return "pass" if ok else "fail"


def command_rows() -> list[dict[str, str]]:
    readme = read_text("README.md")
    command_reference = read_text("outputs/command_reference.md")
    rows: list[dict[str, str]] = []
    for item in COMMAND_CHECKS:
        artifact = str(item["artifact"])
        command = str(item["command"])
        artifact_exists = resolve(artifact).exists()
        command_listed = command in readme or command in command_reference
        rows.append(
            {
                "check": str(item["check"]),
                "status": status_label(artifact_exists and command_listed),
                "command": f"`{command}`",
                "artifact": f"`{artifact}`",
                "why": str(item["why"]),
            }
        )
    return rows


def artifact_rows() -> list[dict[str, str]]:
    return [
        {
            "artifact": name,
            "status": status_label(resolve(path).exists()),
            "path": f"`{path}`",
        }
        for name, path in ARTIFACT_CHECKS
    ]


def boundary_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in BOUNDARY_CHECKS:
        text = read_text(str(item["path"]))
        phrase = str(item["phrase"])
        rows.append(
            {
                "check": str(item["check"]),
                "status": status_label(phrase in text),
                "file": f"`{item['path']}`",
                "required phrase": f"`{phrase}`",
            }
        )
    return rows


def iter_scan_files() -> list[Path]:
    files: list[Path] = []
    for path_text in LEAK_SCAN_PATHS:
        path = resolve(path_text)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(candidate for candidate in path.rglob("*") if candidate.is_file()))
    return files


def leak_rows() -> list[dict[str, str]]:
    pattern = re.compile("|".join(re.escape(item) for item in LEAK_PATTERNS), re.IGNORECASE)
    hits: list[dict[str, str]] = []
    for path in iter_scan_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            match = pattern.search(line)
            if match:
                hits.append(
                    {
                        "status": "fail",
                        "file": f"`{relative(path)}:{line_no}`",
                        "match": f"`{match.group(0)}`",
                    }
                )
    return hits or [{"status": "pass", "file": "public docs/configs", "match": "none"}]


def all_pass(rows: list[dict[str, str]]) -> bool:
    return all(str(row.get("status")) == "pass" for row in rows)


def build_report() -> tuple[str, bool]:
    commands = command_rows()
    artifacts = artifact_rows()
    boundaries = boundary_rows()
    leaks = leak_rows()
    ok = all_pass(commands) and all_pass(artifacts) and all_pass(boundaries) and all_pass(leaks)
    lines: list[str] = [
        "# LunaVLA Reviewer Readiness",
        "",
        f"Overall: `{status_label(ok)}`",
        "",
        "This report operationalizes the completion rule: a reviewer should be able to open the repo, run the public commands, inspect generated artifacts, and trace each resume/interview claim to a file.",
        "",
        "## Command Evidence",
        "",
    ]
    lines.extend(markdown_table(commands))
    lines.extend(["", "## Artifact Evidence", ""])
    lines.extend(markdown_table(artifacts))
    lines.extend(["", "## Boundary Evidence", ""])
    lines.extend(markdown_table(boundaries))
    lines.extend(["", "## Public Leak Scan", ""])
    lines.extend(markdown_table(leaks))
    lines.extend(
        [
            "",
            "## Recommended Review Order",
            "",
            "1. Open `README.md` and run the Quick Start.",
            "2. Open `outputs/command_reference.md` to choose the next command.",
            "3. Open `outputs/homepage_summary.md` before copying any homepage result claim.",
            "4. Open `outputs/project_card.md` for the one-page summary.",
            "5. Open `outputs/experiment_ledger.md` to audit commands, config hashes, metrics, and artifacts.",
            "6. Open `outputs/submission_pack/SUBMISSION_README.md` for the compact review folder.",
            "7. Open `outputs/act_pusht_baseline/run_diagnostic.md` before writing resume or interview claims.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/check_reviewer_readiness.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n", ok


def main() -> int:
    args = parse_args()
    report, ok = build_report()
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"reviewer readiness: {status_label(ok)}")
    print(f"reviewer readiness report: {out_path}")
    return 1 if args.strict and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
