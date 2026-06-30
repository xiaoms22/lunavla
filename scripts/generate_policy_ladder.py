from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    "outputs/bc_pusht_cpu_smoke",
    "outputs/cpu_smoke",
    "outputs/act_pusht_baseline",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a BC -> ACT policy ladder report.")
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="Run directories to compare.")
    parser.add_argument("--out", default="outputs/policy_ladder.md", help="Markdown output path.")
    parser.add_argument("--csv", default="outputs/policy_ladder.csv", help="CSV output path.")
    parser.add_argument("--image", default="images/policy_ladder.svg", help="SVG ladder output path.")
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


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_value(value: Any) -> str:
    numeric = number(value)
    if numeric is not None:
        return f"{numeric:.6g}"
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{key}:{item}" for key, item in sorted(value.items()))
    return str(value)


def infer_policy_label(run_dir: Path, train: dict[str, Any], eval_summary: dict[str, Any]) -> str:
    policy = str(train.get("policy_name") or eval_summary.get("policy_name") or "unknown")
    chunk = train.get("chunk_size", "n/a")
    if policy == "bc_mlp":
        return "BC MLP"
    if policy == "act":
        return f"ACT chunk-{chunk}"
    return policy


def infer_lesson(policy_name: str, chunk_size: Any, run_name: str) -> str:
    if policy_name == "bc_mlp":
        return "Next-action BC is easy to supervise, but rollout can drift after small errors."
    if policy_name == "act" and str(chunk_size) in {"1", "2"}:
        return "A tiny ACT path introduces future-action chunks while staying CPU-friendly."
    if policy_name == "act":
        return "Longer action chunks let the policy predict a short plan, not just the next move."
    return f"Inspect {run_name} before making a policy claim."


def run_row(run_dir: Path) -> dict[str, Any]:
    train = read_json(run_dir / "training_summary.json")
    eval_summary = read_json(run_dir / "eval_summary.json")
    policy_name = str(train.get("policy_name") or eval_summary.get("policy_name") or "missing")
    chunk_size = train.get("chunk_size", "n/a")
    failure_cases = eval_summary.get("failure_count")
    if failure_cases is None:
        failure_cases = count_jsonl(run_dir / "failure_cases.jsonl")
    exists = train != {} or eval_summary != {}
    return {
        "policy": infer_policy_label(run_dir, train, eval_summary) if exists else run_dir.name,
        "run": run_dir.name,
        "chunk_size": chunk_size,
        "records": train.get("records", "n/a"),
        "final_loss": train.get("final_loss", "n/a"),
        "success_rate": eval_summary.get("success_rate", "n/a"),
        "mean_final_distance": eval_summary.get("mean_final_distance", "n/a"),
        "mean_action_smoothness": eval_summary.get("mean_action_smoothness", "n/a"),
        "failure_cases": failure_cases,
        "lesson": infer_lesson(policy_name, chunk_size, run_dir.name) if exists else "Run this command path first.",
    }


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(rows: list[dict[str, Any]], csv_path: Path, image_path: Path) -> str:
    lines: list[str] = [
        "# LunaVLA Policy Ladder",
        "",
        "This report compares the teaching-scale policy path from next-action behavior cloning to ACT-style action chunking.",
        "",
        "It is meant to explain the IL/VA core of LunaVLA. It is not a real-robot benchmark or a frontier VLA claim.",
        "",
        "## Runs",
        "",
    ]
    lines.extend(markdown_table(rows))
    lines.extend(
        [
            "",
            f"CSV: `{relative(csv_path)}`",
            f"SVG: `{relative(image_path)}`",
            "",
            "## What To Learn",
            "",
            "- BC turns demonstrations into a supervised next-action problem.",
            "- Low BC loss does not guarantee rollout success because small prediction errors can compound.",
            "- ACT changes the target from one next action to a short future action chunk.",
            "- Rollout metrics matter because action-learning policies are judged by behavior, not only loss.",
            "- Compare success rate, final distance, smoothness, and failure subtasks before writing a claim.",
            "",
            "## Resume-Safe Claim",
            "",
            (
                "I compared a from-scratch BC MLP smoke baseline with ACT-style action-chunk policies on a "
                "teaching-scale PushT-style loop, using rollout success, final distance, action smoothness, "
                "and failure cases to explain why loss alone is not enough."
            ),
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_policy_ladder.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def success_bar_width(value: Any, max_width: int = 260) -> int:
    numeric = number(value)
    if numeric is None:
        return 0
    return max(0, min(max_width, int(round(numeric * max_width))))


def write_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 980
    row_height = 86
    height = 120 + row_height * max(1, len(rows))
    colors = ["#2f6fed", "#1f9d55", "#e67e22", "#8e44ad"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcff"/>',
        '<text x="32" y="48" font-family="Arial, sans-serif" font-size="28" font-weight="700" fill="#172033">LunaVLA policy ladder</text>',
        '<text x="32" y="78" font-family="Arial, sans-serif" font-size="15" fill="#4c5870">BC next action -> ACT action chunks -> rollout evidence</text>',
    ]
    y = 116
    for index, row in enumerate(rows):
        color = colors[index % len(colors)]
        bar_width = success_bar_width(row.get("success_rate"))
        policy = html.escape(str(row.get("policy", "unknown")))
        lesson = html.escape(str(row.get("lesson", ""))[:92])
        success = html.escape(format_value(row.get("success_rate", "n/a")))
        loss = html.escape(format_value(row.get("final_loss", "n/a")))
        distance = html.escape(format_value(row.get("mean_final_distance", "n/a")))
        parts.extend(
            [
                f'<rect x="32" y="{y - 28}" width="916" height="66" rx="10" fill="#ffffff" stroke="#d8deea"/>',
                f'<circle cx="66" cy="{y + 4}" r="14" fill="{color}"/>',
                f'<text x="58" y="{y + 10}" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#ffffff">{index + 1}</text>',
                f'<text x="92" y="{y - 2}" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#172033">{policy}</text>',
                f'<text x="92" y="{y + 22}" font-family="Arial, sans-serif" font-size="13" fill="#4c5870">{lesson}</text>',
                f'<rect x="566" y="{y - 12}" width="260" height="16" rx="8" fill="#e9edf5"/>',
                f'<rect x="566" y="{y - 12}" width="{bar_width}" height="16" rx="8" fill="{color}"/>',
                f'<text x="842" y="{y + 1}" font-family="Arial, sans-serif" font-size="13" fill="#172033">success {success}</text>',
                f'<text x="566" y="{y + 24}" font-family="Arial, sans-serif" font-size="12" fill="#4c5870">loss {loss} | final distance {distance}</text>',
            ]
        )
        y += row_height
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = [run_row(resolve(run)) for run in args.runs]
    out_path = resolve(args.out)
    csv_path = resolve(args.csv)
    image_path = resolve(args.image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, rows)
    write_svg(image_path, rows)
    out_path.write_text(build_markdown(rows, csv_path, image_path), encoding="utf-8")
    print(f"policy ladder: {out_path}")
    print(f"policy ladder csv: {csv_path}")
    print(f"policy ladder svg: {image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
