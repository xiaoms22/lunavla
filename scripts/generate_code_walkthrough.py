from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


READING_STEPS: list[dict[str, Any]] = [
    {
        "step": "data record",
        "files": ["dataset/vla_dataset.py", "dataset/pusht_dataset.py", "configs/schema.md"],
        "question": "What fields does one VLA record contain?",
        "output": "outputs/dataset_inspection.md",
    },
    {
        "step": "policy input",
        "files": ["model/minivla_policy.py", "model/act_wrapper.py"],
        "question": "How do observation and instruction features become an action chunk prediction?",
        "output": "outputs/learning_checkpoint.md",
    },
    {
        "step": "training loop",
        "files": ["trainer/train_act_pusht.py", "trainer/trainer_utils.py", "configs/act_pusht_baseline.yaml"],
        "question": "Which config values control dataset size, chunk size, epochs, and artifacts?",
        "output": "outputs/act_pusht_baseline/summary_report.md",
    },
    {
        "step": "rollout eval",
        "files": ["eval_vla.py", "scripts/summarize_results.py", "scripts/diagnose_run.py"],
        "question": "Why does the project report rollout metrics instead of only reporting training loss?",
        "output": "outputs/act_pusht_baseline/run_diagnostic.md",
    },
    {
        "step": "visual evidence",
        "files": ["scripts/export_readme_assets.py", "scripts/web_demo_vla.py", "scripts/check_readme_assets.py"],
        "question": "How are rollout behavior, action chunks, loss curves, and result tables shown?",
        "output": "outputs/readme_asset_check.md",
    },
    {
        "step": "project evidence",
        "files": [
            "scripts/generate_project_report.py",
            "scripts/generate_resume_pack.py",
            "scripts/build_evidence_pack.py",
            "scripts/build_submission_pack.py",
        ],
        "question": "How do run artifacts become a report, interview pitch, and review folder?",
        "output": "outputs/submission_pack/SUBMISSION_README.md",
    },
]


TRACE_STEPS = [
    ("1", "A generated PushT-style record stores observation, action, timestep, success, and metadata."),
    ("2", "The dataset turns the current observation plus instruction features into one model input vector."),
    ("3", "The training target is a flattened action chunk made from short future expert actions."),
    ("4", "The policy predicts an action chunk, and the evaluator feeds predicted actions back into rollout state updates."),
    ("5", "Reports combine loss, success rate, final distance, smoothness, failure cases, and visual assets."),
]


EXERCISES = [
    {
        "task": "Trace one sample",
        "command": "python scripts/inspect_dataset.py",
        "evidence": "outputs/dataset_inspection.md",
        "check": "Point to the observation vector and flattened action chunk target.",
    },
    {
        "task": "Trace one run",
        "command": "python scripts/run_cpu_smoke.py",
        "evidence": "outputs/cpu_smoke/project_report.md",
        "check": "Explain how a checkpoint becomes rollout metrics and a browser artifact.",
    },
    {
        "task": "Trace one baseline",
        "command": "python scripts/run_baseline_evidence.py",
        "evidence": "outputs/act_pusht_baseline/run_diagnostic.md",
        "check": "List the claims that are safe to make from generated evidence.",
    },
    {
        "task": "Trace one comparison",
        "command": "python scripts/run_ablation_evidence.py",
        "evidence": "outputs/run_comparison.md",
        "check": "Describe what changed and which metric supports the conclusion.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a beginner-friendly MiniMind-VLA code walkthrough.")
    parser.add_argument("--out", default="outputs/code_walkthrough.md", help="Markdown walkthrough path.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def exists_label(paths: list[str]) -> str:
    found = sum(1 for path in paths if resolve(path).exists())
    return f"{found}/{len(paths)}"


def format_files(paths: list[str]) -> str:
    return "<br>".join(f"`{path}`" for path in paths)


def format_output(path: str) -> str:
    marker = "ready" if resolve(path).exists() else "missing"
    return f"`{path}` ({marker})"


def markdown_table(rows: list[dict[str, str]]) -> list[str]:
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    return lines


def reading_rows() -> list[dict[str, str]]:
    return [
        {
            "step": item["step"],
            "files": format_files(item["files"]),
            "ready": exists_label(item["files"]),
            "question": item["question"],
            "evidence": format_output(item["output"]),
        }
        for item in READING_STEPS
    ]


def trace_rows() -> list[dict[str, str]]:
    return [{"order": order, "what happens": text} for order, text in TRACE_STEPS]


def exercise_rows() -> list[dict[str, str]]:
    return [
        {
            "task": item["task"],
            "command": f"`{item['command']}`",
            "evidence": f"`{item['evidence']}`",
            "self-check": item["check"],
        }
        for item in EXERCISES
    ]


def build_walkthrough() -> str:
    lines: list[str] = [
        "# MiniMind-VLA Code Walkthrough",
        "",
        "This guide shows a beginner how to read the runnable code after the first smoke run.",
        "",
        "## Reading Order",
        "",
    ]
    lines.extend(markdown_table(reading_rows()))
    lines.extend(["", "## One Record Through The Loop", ""])
    lines.extend(markdown_table(trace_rows()))
    lines.extend(
        [
            "",
            "## Small Exercises",
            "",
        ]
    )
    lines.extend(markdown_table(exercise_rows()))
    lines.extend(
        [
            "",
            "## What A Good Explanation Should Include",
            "",
            "- The project uses generated PushT-style demonstrations as a low-cost teaching dataset.",
            "- The policy learns action chunks from observation and instruction features.",
            "- Rollout evaluation is needed because closed-loop behavior can differ from training loss.",
            "- The project evidence is strongest when metrics, rollout visuals, failure cases, and reports agree.",
            "- The boundary stays honest: this is teaching-scale imitation-learning evidence, not a real-robot deployment claim.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_code_walkthrough.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_walkthrough(), encoding="utf-8")
    print(f"code walkthrough: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
