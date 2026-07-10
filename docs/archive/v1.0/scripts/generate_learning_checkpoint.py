from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a LunaVLA learning checkpoint for beginners.")
    parser.add_argument("--run-dir", default="outputs/act_pusht_baseline", help="Primary run directory.")
    parser.add_argument("--out", default="outputs/learning_checkpoint.md", help="Markdown checkpoint path.")
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


def concept_rows(run_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "concept": "VLA record",
            "what to explain": "`observation`, optional instruction features, `action`, `episode_id`, `timestep`, `success`, `metadata`",
            "evidence": "outputs/dataset_inspection.md",
        },
        {
            "concept": "imitation learning",
            "what to explain": "the policy fits generated demonstration actions instead of learning from reward",
            "evidence": "trainer/train_act_pusht.py",
        },
        {
            "concept": "ACT-style action chunk",
            "what to explain": "the target is a short flattened sequence of future 2D actions",
            "evidence": "outputs/action_chunk_lesson.md",
        },
        {
            "concept": "action normalization",
            "what to explain": "mean/std describe action scale; normalized training actions and executable rollout actions are different",
            "evidence": "outputs/action_statistics.md",
        },
        {
            "concept": "rollout evaluation",
            "what to explain": "predicted actions are fed back into state updates to test closed-loop behavior",
            "evidence": "eval_vla.py",
        },
        {
            "concept": "failure analysis",
            "what to explain": "failed episodes are labeled and inspected before writing conclusions",
            "evidence": "outputs/failure_review.md",
        },
        {
            "concept": "safe project claim",
            "what to explain": "claims are tied to metrics, diagnostics, reports, and the teaching-scale boundary",
            "evidence": relative(run_dir / "run_diagnostic.md"),
        },
    ]


def metric_rows(run_dir: Path) -> list[dict[str, Any]]:
    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    return [
        {"metric": "records", "value": training.get("records", "n/a"), "why it matters": "amount of demonstration data"},
        {"metric": "chunk_size", "value": training.get("chunk_size", "n/a"), "why it matters": "action horizon predicted by the policy"},
        {"metric": "action_mean", "value": training.get("action_mean", "n/a"), "why it matters": "center of demonstration action scale"},
        {"metric": "action_std", "value": training.get("action_std", "n/a"), "why it matters": "scale used to explain normalization"},
        {"metric": "final_loss", "value": training.get("final_loss", "n/a"), "why it matters": "imitation objective fit"},
        {"metric": "episodes", "value": evaluation.get("episodes", "n/a"), "why it matters": "rollout sample count"},
        {"metric": "success_rate", "value": evaluation.get("success_rate", "n/a"), "why it matters": "headline behavior metric"},
        {
            "metric": "mean_final_distance",
            "value": evaluation.get("mean_final_distance", "n/a"),
            "why it matters": "how close rollouts end to the goal",
        },
        {
            "metric": "mean_action_smoothness",
            "value": evaluation.get("mean_action_smoothness", "n/a"),
            "why it matters": "whether actions change abruptly",
        },
        {
            "metric": "failure_count",
            "value": evaluation.get("failure_count", "n/a"),
            "why it matters": "cases that need inspection before reporting",
        },
    ]


def question_rows(run_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "question": "How does one raw record become a training sample?",
            "look at": "outputs/dataset_inspection.md and outputs/action_chunk_lesson.md",
            "good answer should mention": "observation vector, instruction features, and flattened action chunk target",
        },
        {
            "question": "Why is training loss not enough?",
            "look at": relative(run_dir / "project_report.md"),
            "good answer should mention": "closed-loop rollout drift and success-rate/final-distance metrics",
        },
        {
            "question": "What does action chunk size change?",
            "look at": "outputs/run_comparison.md",
            "good answer should mention": "temporal action prediction horizon and ablation deltas",
        },
        {
            "question": "Why do action statistics belong in the run artifacts?",
            "look at": "outputs/action_statistics.md",
            "good answer should mention": "action scale, normalization formulas, checkpoint provenance, and executable rollout actions",
        },
        {
            "question": "What failed, and how would you inspect it?",
            "look at": "outputs/failure_review.md",
            "good answer should mention": "failure category, rollout browser, and next minimal check",
        },
        {
            "question": "What can you safely claim in a resume or interview?",
            "look at": relative(run_dir / "resume_pack.md"),
            "good answer should mention": "generated demonstrations, ACT-style policy, rollout metrics, and honest boundary",
        },
    ]


def build_checkpoint(run_dir: Path) -> str:
    lines: list[str] = [
        "# LunaVLA Learning Checkpoint",
        "",
        "Use this file after running the public evidence commands. It checks whether you can explain the project, not just execute it.",
        "",
        "## Concepts To Explain",
        "",
    ]
    lines.extend(markdown_table(concept_rows(run_dir)))
    lines.extend(["", "## Metrics To Understand", ""])
    lines.extend(markdown_table(metric_rows(run_dir)))
    lines.extend(["", "## Self-Check Questions", ""])
    lines.extend(markdown_table(question_rows(run_dir)))
    lines.extend(
        [
            "",
            "## Two-Minute Structure",
            "",
            "1. State the goal: a tiny observation-to-action learning loop for VLA beginners.",
            "2. Explain the data: generated PushT-style demonstrations become observation and action-chunk targets.",
            "3. Explain the scale: action stats record mean/std and the normalization boundary.",
            "4. Explain the policy: an ACT-style model predicts a short sequence of actions.",
            "5. Explain evaluation: rollouts report success rate, final distance, smoothness, and failure cases.",
            "6. Explain the boundary: teaching-scale imitation-learning evidence, not real-robot deployment.",
            "",
            "## Rebuild",
            "",
            "```bash",
            f"python scripts/generate_learning_checkpoint.py --run-dir {relative(run_dir)}",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_checkpoint(run_dir), encoding="utf-8")
    print(f"learning checkpoint: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
