from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a beginner-friendly LunaVLA skill evidence map.")
    parser.add_argument("--run-dir", default="outputs/act_pusht_baseline", help="Primary run directory.")
    parser.add_argument("--out", default="outputs/skill_evidence_map.md", help="Markdown skill map path.")
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


def metric_text(run_dir: Path) -> str:
    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    final_loss = training.get("final_loss", "n/a")
    success_rate = evaluation.get("success_rate", "n/a")
    final_distance = evaluation.get("mean_final_distance", "n/a")
    chunk_size = training.get("chunk_size", "n/a")
    return f"chunk_size={chunk_size}, final_loss={final_loss}, success_rate={success_rate}, mean_final_distance={final_distance}"


def exists_label(path_text: str | Path) -> str:
    return "yes" if resolve(path_text).exists() else "no"


def markdown_table(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(header, "") for header in headers) + " |")
    return lines


def rows(run_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "skill": "VLA data contract",
            "code evidence": "`dataset/vla_dataset.py`, `dataset/pusht_dataset.py`",
            "run evidence": "`outputs/dataset_inspection.md`",
            "command": "`python scripts/inspect_dataset.py`",
            "how to explain": "A sample becomes observation features plus an action-chunk target.",
            "exists": exists_label("outputs/dataset_inspection.md"),
        },
        {
            "skill": "Demonstration generation",
            "code evidence": "`dataset/pusht_dataset.py`",
            "run evidence": f"`{relative(run_dir / 'train_records.jsonl')}`",
            "command": "`python trainer/train_act_pusht.py --config configs/act_pusht_baseline.yaml`",
            "how to explain": "The baseline learns from generated PushT-style expert actions.",
            "exists": exists_label(run_dir / "train_records.jsonl"),
        },
        {
            "skill": "Behavior cloning training",
            "code evidence": "`trainer/train_act_pusht.py`, `model/minivla_policy.py`",
            "run evidence": f"`{relative(run_dir / 'training_summary.json')}`",
            "command": "`python scripts/run_baseline_evidence.py`",
            "how to explain": "The policy minimizes supervised action prediction loss on demonstrations.",
            "exists": exists_label(run_dir / "training_summary.json"),
        },
        {
            "skill": "ACT-style action chunks",
            "code evidence": "`model/act_wrapper.py`, `configs/act_pusht_baseline.yaml`",
            "run evidence": "`outputs/run_comparison.md`",
            "command": "`python scripts/run_ablation_evidence.py`",
            "how to explain": "Chunk size is a controlled action-horizon variable that can be ablated.",
            "exists": exists_label("outputs/run_comparison.md"),
        },
        {
            "skill": "Rollout evaluation",
            "code evidence": "`eval_vla.py`",
            "run evidence": f"`{relative(run_dir / 'summary_report.md')}`",
            "command": f"`python eval_vla.py --checkpoint {relative(run_dir / 'checkpoint.pt')} --episodes 50 --save-rollouts`",
            "how to explain": "Closed-loop rollouts test whether predicted actions still work over time.",
            "exists": exists_label(run_dir / "summary_report.md"),
        },
        {
            "skill": "Metrics and reporting",
            "code evidence": "`scripts/summarize_results.py`, `scripts/generate_project_report.py`",
            "run evidence": f"`{relative(run_dir / 'project_report.md')}`",
            "command": f"`python scripts/generate_project_report.py --run-dir {relative(run_dir)}`",
            "how to explain": metric_text(run_dir),
            "exists": exists_label(run_dir / "project_report.md"),
        },
        {
            "skill": "Failure analysis",
            "code evidence": "`scripts/generate_failure_review.py`, `scripts/diagnose_run.py`",
            "run evidence": "`outputs/failure_review.md`",
            "command": "`python scripts/generate_failure_review.py`",
            "how to explain": "Discuss failed behavior with saved rollout evidence and a small debugging hypothesis.",
            "exists": exists_label("outputs/failure_review.md"),
        },
        {
            "skill": "Visual evidence",
            "code evidence": "`scripts/export_readme_assets.py`, `scripts/web_demo_vla.py`",
            "run evidence": "`images/pusht_act_eval.gif`, `images/pusht_diffusion_policy_eval.gif`, `images/local_rollout.gif`, `images/result_table.svg`",
            "command": "`python scripts/export_readme_assets.py --run-dir outputs/act_pusht_baseline --out-dir images`",
            "how to explain": "The README assets compare ACT and Diffusion Policy PushT behavior, then separate that visual context from the local rollout trace, action chunks, and loss curve.",
            "exists": exists_label("images/pusht_act_eval.gif")
            + " / "
            + exists_label("images/pusht_diffusion_policy_eval.gif")
            + " / "
            + exists_label("images/local_rollout.gif"),
        },
        {
            "skill": "Claim safety",
            "code evidence": "`scripts/diagnose_run.py`, `DATA_CARD.md`, `MODEL_CARD.md`",
            "run evidence": f"`{relative(run_dir / 'run_diagnostic.md')}`",
            "command": f"`python scripts/diagnose_run.py --run-dir {relative(run_dir)}`",
            "how to explain": "Safe claims stay tied to a small reproducible imitation-learning loop.",
            "exists": exists_label(run_dir / "run_diagnostic.md"),
        },
        {
            "skill": "Project communication",
            "code evidence": "`scripts/generate_resume_pack.py`, `scripts/generate_interview_flashcards.py`",
            "run evidence": "`outputs/interview_flashcards.md`, `outputs/submission_pack/SUBMISSION_README.md`",
            "command": "`python scripts/build_submission_pack.py`",
            "how to explain": "The final pack turns code, metrics, visuals, and boundaries into review-ready evidence.",
            "exists": exists_label("outputs/submission_pack/SUBMISSION_README.md"),
        },
    ]


def build_report(run_dir: Path) -> str:
    lines: list[str] = [
        "# LunaVLA Skill Evidence Map",
        "",
        "This map connects beginner-facing VLA skills to the code, commands, and generated artifacts that support them.",
        "",
        "## Skill Map",
        "",
    ]
    lines.extend(markdown_table(rows(run_dir)))
    lines.extend(
        [
            "",
            "## How To Use This Map",
            "",
            "1. Pick one skill you want to claim.",
            "2. Run the command in that row.",
            "3. Open the code evidence and run evidence before writing the claim.",
            "4. Keep the wording tied to generated artifacts and the teaching-scale project boundary.",
            "",
            "## Rebuild",
            "",
            "```bash",
            f"python scripts/generate_skill_evidence_map.py --run-dir {relative(run_dir)}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(run_dir), encoding="utf-8")
    print(f"skill evidence map: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
