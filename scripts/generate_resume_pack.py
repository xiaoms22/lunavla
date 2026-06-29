from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a resume and interview pack from a LunaVLA run.")
    parser.add_argument("--run-dir", required=True, help="Run directory under outputs/ or an absolute path.")
    parser.add_argument("--out", default=None, help="Markdown output path. Defaults to <run-dir>/resume_pack.md.")
    parser.add_argument(
        "--comparison",
        default=None,
        help="Optional comparison report path to reference when it exists.",
    )
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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
        return []
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def artifact_rows(run_dir: Path, comparison_path: Path | None) -> list[dict[str, str]]:
    artifacts = [
        (run_dir / "checkpoint.pt", "trained policy checkpoint"),
        (run_dir / "eval_summary.json", "rollout metrics"),
        (run_dir / "summary_report.md", "compact run summary"),
        (run_dir / "project_report.md", "long-form project report"),
        (run_dir / "web_demo.html", "static rollout browser"),
        (run_dir / "rollouts", "saved rollout JSON files"),
        (ROOT / "outputs/evidence_index.md", "complete evidence map"),
        (ROOT / "images/pusht_act_eval.gif", "README ACT PushT evaluation animation"),
        (ROOT / "images/pusht_diffusion_policy_eval.gif", "README Diffusion Policy PushT evaluation animation"),
    ]
    if comparison_path is not None:
        artifacts.insert(7, (comparison_path, "baseline vs ablation report"))
    return [
        {
            "artifact": relative(path),
            "exists": "yes" if path.exists() else "no",
            "use": purpose,
        }
        for path, purpose in artifacts
    ]


def metric_rows(training: dict[str, Any], evaluation: dict[str, Any], failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"metric": "project_name", "value": training.get("project_name", "unknown")},
        {"metric": "records", "value": training.get("records", "n/a")},
        {"metric": "chunk_size", "value": training.get("chunk_size", "n/a")},
        {"metric": "final_loss", "value": training.get("final_loss", "n/a")},
        {"metric": "success_rate", "value": evaluation.get("success_rate", "n/a")},
        {"metric": "mean_final_distance", "value": evaluation.get("mean_final_distance", "n/a")},
        {"metric": "mean_rollout_length", "value": evaluation.get("mean_rollout_length", "n/a")},
        {"metric": "mean_action_smoothness", "value": evaluation.get("mean_action_smoothness", "n/a")},
        {"metric": "failure_cases", "value": evaluation.get("failure_count", len(failures))},
        {"metric": "failure_categories", "value": evaluation.get("failure_category_counts", {})},
    ]


def best_resume_bullet(training: dict[str, Any], evaluation: dict[str, Any], comparison_exists: bool) -> str:
    success_rate = format_value(evaluation.get("success_rate", "n/a"))
    final_distance = format_value(evaluation.get("mean_final_distance", "n/a"))
    chunk_size = format_value(training.get("chunk_size", "n/a"))
    if comparison_exists:
        return (
            "Implemented a tiny ACT-style imitation-learning project on a PushT-style task, then ran a "
            f"chunk-size ablation against a `chunk_size={chunk_size}` baseline and reported rollout success "
            f"rate `{success_rate}`, mean final distance `{final_distance}`, action smoothness, and failure categories."
        )
    return (
        "Implemented a tiny ACT-style imitation-learning baseline on a PushT-style task, including generated "
        f"demonstrations, action-chunk prediction, checkpoint export, rollout evaluation with success rate "
        f"`{success_rate}`, mean final distance `{final_distance}`, and failure-case analysis."
    )


def build_pitch(training: dict[str, Any], evaluation: dict[str, Any]) -> list[str]:
    project_name = training.get("project_name", "LunaVLA")
    success_rate = format_value(evaluation.get("success_rate", "n/a"))
    final_distance = format_value(evaluation.get("mean_final_distance", "n/a"))
    chunk_size = format_value(training.get("chunk_size", "n/a"))
    return [
        (
            f"I built `{project_name}` as a small VLA-style project starter. The goal was to make the "
            "observation-to-action learning loop runnable and explainable for beginners."
        ),
        (
            "The system generates PushT-style demonstrations, converts each record into observation and "
            f"instruction features, trains an ACT-style policy to predict action chunks of size `{chunk_size}`, "
            "then evaluates the policy through saved rollouts."
        ),
        (
            f"In this checked run, rollout success rate was `{success_rate}` and mean final distance was "
            f"`{final_distance}`. I report loss, rollout metrics, action smoothness, and failure categories "
            "because training loss alone does not prove closed-loop behavior."
        ),
        (
            "The honest boundary is that this is a teaching-scale PushT-style imitation-learning loop. It is "
            "useful for learning data, policy, evaluation, and reporting, but it is not a real-robot deployment claim."
        ),
    ]


def build_pack(run_dir: Path, comparison_path: Path | None) -> str:
    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    failures = read_jsonl(run_dir / "failure_cases.jsonl")
    comparison_exists = comparison_path is not None and comparison_path.exists()

    lines: list[str] = [
        "# LunaVLA Resume And Interview Pack",
        "",
        "This pack turns one verified local run into resume-safe project evidence.",
        "",
        "## Evidence Checklist",
        "",
    ]
    lines.extend(markdown_table(artifact_rows(run_dir, comparison_path)))
    lines.extend(["", "## Metrics To Cite", ""])
    lines.extend(markdown_table(metric_rows(training, evaluation, failures)))
    lines.extend(
        [
            "",
            "## Resume-Safe Bullet",
            "",
            "> " + best_resume_bullet(training, evaluation, comparison_exists),
            "",
            "## Two-Minute Pitch",
            "",
        ]
    )
    for paragraph in build_pitch(training, evaluation):
        lines.append(paragraph)
        lines.append("")

    lines.extend(
        [
            "## Interview Anchors",
            "",
            "- Dataset: explain `observation`, optional instruction features, action, episode id, timestep, and metadata.",
            "- Policy: explain why the ACT-style target is a flattened action chunk instead of a single action.",
            "- Evaluation: explain why rollout success rate and final distance matter more than training loss alone.",
            "- Failure analysis: use `failure_category_counts` and saved rollout JSON to discuss behavior, not just numbers.",
            "- Ablation: compare exactly one controlled variable before making a conclusion.",
            "",
            "## Boundary Statement",
            "",
            (
                "This is a tiny, reproducible, PushT-style imitation-learning project for learning and internship "
                "evidence. Do not describe it as a real-robot deployment, a frontier robot foundation model, or a "
                "state-of-the-art robotics benchmark."
            ),
            "",
            "## Reproduce This Pack",
            "",
            "```bash",
            "python scripts/generate_resume_pack.py --run-dir "
            + relative(run_dir)
            + (f" --comparison {relative(comparison_path)}" if comparison_path is not None else ""),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    comparison_path = resolve(args.comparison) if args.comparison else None
    out_path = resolve(args.out) if args.out else run_dir / "resume_pack.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_pack(run_dir, comparison_path), encoding="utf-8")
    print(f"resume pack: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
