from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LunaVLA interview flashcards from public run evidence.")
    parser.add_argument("--run-dir", default="outputs/act_pusht_baseline", help="Primary run directory.")
    parser.add_argument("--out", default="outputs/interview_flashcards.md", help="Markdown flashcard path.")
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


def value(data: dict[str, Any], key: str) -> str:
    item = data.get(key, "n/a")
    if isinstance(item, float):
        return f"{item:.6g}"
    if isinstance(item, dict):
        if not item:
            return "none"
        return ", ".join(f"{k}:{v}" for k, v in sorted(item.items()))
    return str(item)


def markdown_table(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(header, "") for header in headers) + " |")
    return lines


def flashcards(run_dir: Path) -> list[dict[str, str]]:
    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    return [
        {
            "topic": "VLA record",
            "question": "What is the input/output contract in this project?",
            "answer": "A record stores observation, optional instruction features, action, episode id, timestep, success, and metadata. The model input is observation plus instruction features; the target is a flattened future action chunk.",
            "evidence": "outputs/dataset_inspection.md",
        },
        {
            "topic": "Behavior cloning",
            "question": "What does the policy learn from?",
            "answer": "It learns to imitate generated PushT-style demonstration actions. There is no reward optimization in the baseline path.",
            "evidence": "trainer/train_act_pusht.py",
        },
        {
            "topic": "ACT-style chunk",
            "question": "Why predict a chunk instead of one action?",
            "answer": f"The baseline predicts chunk_size={value(training, 'chunk_size')} future 2D actions, so the policy models short-horizon action structure and gives a clear ablation variable.",
            "evidence": "configs/act_pusht_baseline.yaml",
        },
        {
            "topic": "Rollout evaluation",
            "question": "Why is rollout evaluation necessary?",
            "answer": "Low training loss can still fail after predictions are fed back into the state. Rollouts measure closed-loop behavior with success rate, final distance, rollout length, and action smoothness.",
            "evidence": "eval_vla.py",
        },
        {
            "topic": "Metrics",
            "question": "What numbers should you cite?",
            "answer": f"For this checked run: success_rate={value(evaluation, 'success_rate')}, mean_final_distance={value(evaluation, 'mean_final_distance')}, mean_action_smoothness={value(evaluation, 'mean_action_smoothness')}, failure_count={value(evaluation, 'failure_count')}.",
            "evidence": relative(run_dir / "summary_report.md"),
        },
        {
            "topic": "Failure analysis",
            "question": "How do you discuss a failed rollout?",
            "answer": "Name the failure category, inspect the saved rollout or browser, and propose one minimal check such as action sign, action magnitude, horizon, smoothness, or data coverage.",
            "evidence": "outputs/failure_review.md",
        },
        {
            "topic": "Ablation",
            "question": "What does the chunk-size ablation prove?",
            "answer": "It compares one controlled variable, chunk size, while keeping the rest of the learning loop fixed. The conclusion should cite the comparison report, not a vague impression.",
            "evidence": "outputs/run_comparison.md",
        },
        {
            "topic": "Honest boundary",
            "question": "What should you avoid claiming?",
            "answer": "Do not claim real-robot deployment or state-of-the-art robotics performance. The safe claim is a small reproducible PushT-style imitation-learning loop with rollout evaluation and reportable evidence.",
            "evidence": relative(run_dir / "run_diagnostic.md"),
        },
    ]


def build_report(run_dir: Path) -> str:
    lines: list[str] = [
        "# LunaVLA Interview Flashcards",
        "",
        "Use these cards after running the public evidence commands. Each answer points to a code file or generated report that supports the claim.",
        "",
        "## Flashcards",
        "",
    ]
    lines.extend(markdown_table(flashcards(run_dir)))
    lines.extend(
        [
            "",
            "## Practice Loop",
            "",
            "1. Read one card aloud.",
            "2. Open the evidence file listed in that row.",
            "3. Replace vague words with the exact metric or artifact path.",
            "4. End with the honest boundary when describing the project externally.",
            "",
            "## Rebuild",
            "",
            "```bash",
            f"python scripts/generate_interview_flashcards.py --run-dir {relative(run_dir)}",
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
    print(f"interview flashcards: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
