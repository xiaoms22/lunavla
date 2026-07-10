from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DELIVERABLES = [
    {
        "deliverable": "GitHub repo",
        "paths": ["README.md", "DATA_CARD.md", "MODEL_CARD.md", "RELEASE_NOTES.md"],
        "proof": "Public materials describe a runnable beginner project with honest boundaries.",
    },
    {
        "deliverable": "baseline report",
        "paths": [
            "outputs/act_pusht_baseline/project_report.md",
            "outputs/act_pusht_baseline/summary_report.md",
            "outputs/act_pusht_baseline/run_diagnostic.md",
        ],
        "proof": "The ACT PushT-style baseline has metrics, report text, and claim-safety diagnostics.",
    },
    {
        "deliverable": "ablation report",
        "paths": ["outputs/run_comparison.md", "outputs/config_diff.md", "outputs/config_diff.json"],
        "proof": "The ablation changes one documented variable and has comparison evidence.",
    },
    {
        "deliverable": "rollout demo",
        "paths": [
            "outputs/act_pusht_baseline/web_demo.html",
            "images/pusht_act_eval.gif",
            "images/pusht_diffusion_policy_eval.gif",
        ],
        "proof": "A reviewer can inspect behavior visually instead of reading only loss.",
    },
    {
        "deliverable": "failure taxonomy",
        "paths": ["docs/failure_taxonomy.md", "outputs/failure_review.md"],
        "proof": "Failure cases are categorized and connected to next minimal checks.",
    },
    {
        "deliverable": "resume bullet",
        "paths": ["outputs/act_pusht_baseline/resume_pack.md", "docs/internship_pack/02_resume_bullets.md"],
        "proof": "Resume claims are grounded in generated metrics and boundaries.",
    },
    {
        "deliverable": "two-minute explanation",
        "paths": ["docs/interview_pitch.md", "outputs/interview_flashcards.md", "outputs/skill_evidence_map.md"],
        "proof": "The user can explain data, policy, rollout, metrics, failure analysis, and limits.",
    },
    {
        "deliverable": "submission pack",
        "paths": [
            "outputs/submission_pack/SUBMISSION_README.md",
            "outputs/submission_pack/manifest.json",
            "outputs/submission_pack/final_package_checklist.md",
        ],
        "proof": "The final review folder collects key evidence in one place.",
    },
]


BOUNDARY_PHRASES = [
    ("README.md", "It is not a real-robot deployment benchmark"),
    ("docs/internship_pack/08_final_package_checklist.md", "not a real-robot benchmark or frontier VLA reproduction"),
    ("outputs/act_pusht_baseline/run_diagnostic.md", "claims about real-robot deployment"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LunaVLA final package deliverables.")
    parser.add_argument("--out", default="outputs/final_package_check.md", help="Markdown report path.")
    parser.add_argument(
        "--copy-to-submission-pack",
        dest="copy_to_submission_pack",
        action="store_true",
        default=True,
        help="Copy the report to outputs/submission_pack/final_package_check.md when the folder exists.",
    )
    parser.add_argument(
        "--no-copy-to-submission-pack",
        dest="copy_to_submission_pack",
        action="store_false",
        help="Only write the main report and do not copy it into outputs/submission_pack.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any deliverable is incomplete.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def status_label(ok: bool) -> str:
    return "pass" if ok else "fail"


def read_text(path: str | Path) -> str:
    resolved = resolve(path)
    return resolved.read_text(encoding="utf-8") if resolved.exists() else ""


def deliverable_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in DELIVERABLES:
        paths = [str(path) for path in item["paths"]]
        missing = [path for path in paths if not resolve(path).exists()]
        rows.append(
            {
                "deliverable": item["deliverable"],
                "status": status_label(not missing),
                "evidence": "<br>".join(f"`{path}`" for path in paths),
                "missing": "none" if not missing else "<br>".join(f"`{path}`" for path in missing),
                "proof": item["proof"],
            }
        )
    return rows


def boundary_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path, phrase in BOUNDARY_PHRASES:
        ok = phrase in read_text(path)
        rows.append(
            {
                "file": f"`{path}`",
                "status": status_label(ok),
                "required phrase": f"`{phrase}`",
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines


def all_pass(rows: list[dict[str, Any]]) -> bool:
    return all(str(row.get("status")) == "pass" for row in rows)


def build_report() -> tuple[str, bool]:
    deliverables = deliverable_rows()
    boundaries = boundary_rows()
    ok = all_pass(deliverables) and all_pass(boundaries)
    lines: list[str] = [
        "# LunaVLA Final Package Check",
        "",
        f"Overall: `{status_label(ok)}`",
        "",
        "This report checks whether a LunaVLA run has the final materials needed for a GitHub project, report, resume bullet, and interview explanation.",
        "",
        "## Deliverables",
        "",
    ]
    lines.extend(markdown_table(deliverables))
    lines.extend(["", "## Boundary Checks", ""])
    lines.extend(markdown_table(boundaries))
    lines.extend(
        [
            "",
            "## Reviewer Order",
            "",
            "1. Read `README.md`.",
            "2. Open `outputs/submission_pack/SUBMISSION_README.md`.",
            "3. Open `outputs/act_pusht_baseline/project_report.md`.",
            "4. Open `outputs/run_comparison.md` and `outputs/config_diff.md`.",
            "5. Open `outputs/act_pusht_baseline/web_demo.html`.",
            "6. Open `outputs/failure_review.md`.",
            "7. Open `outputs/act_pusht_baseline/resume_pack.md`.",
            "8. Practice with `outputs/interview_flashcards.md`.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/check_final_package.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n", ok


def maybe_copy_to_submission_pack(out_path: Path, enabled: bool) -> Path | None:
    if not enabled:
        return None
    pack_dir = ROOT / "outputs" / "submission_pack"
    if not pack_dir.exists():
        return None
    target = pack_dir / "final_package_check.md"
    shutil.copy2(out_path, target)
    return target


def main() -> int:
    args = parse_args()
    report, ok = build_report()
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    copied = maybe_copy_to_submission_pack(out_path, args.copy_to_submission_pack)
    print(f"final package: {status_label(ok)}")
    print(f"final package report: {out_path}")
    if copied:
        print(f"submission pack copy: {copied}")
    return 1 if args.strict and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
