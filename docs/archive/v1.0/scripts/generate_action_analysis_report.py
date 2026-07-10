from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    "outputs/cpu_smoke",
    "outputs/bc_pusht_cpu_smoke",
    "outputs/bc_pusht_hidden64_smoke",
    "outputs/act_pusht_baseline",
    "outputs/act_pusht_ablation_chunk_size",
    "outputs/act_pusht_jsonl_smoke",
    "outputs/act_pusht_jsonl_noisy_smoke",
]
CLIP_LIMIT = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare train-time action targets with eval-time executable actions.")
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS, help="Run directories to include.")
    parser.add_argument("--out", default="outputs/action_analysis_report.md", help="Markdown report output.")
    parser.add_argument("--csv", default="outputs/action_analysis_report.csv", help="Machine-readable comparison table.")
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
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def rollout_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "rollouts").glob("episode_*.json"))


def rollout_actions(run_dir: Path) -> np.ndarray | None:
    actions: list[list[float]] = []
    for path in rollout_files(run_dir):
        rollout = read_json(path)
        for frame in rollout.get("frames", []):
            action = frame.get("action")
            if isinstance(action, list) and len(action) == 2:
                actions.append([float(action[0]), float(action[1])])
    if not actions:
        return None
    return np.asarray(actions, dtype=np.float32)


def round_list(values: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values.tolist()]


def action_summary(actions: np.ndarray | None) -> dict[str, Any]:
    if actions is None or actions.size == 0:
        return {
            "count": 0,
            "mean": "n/a",
            "std": "n/a",
            "min": "n/a",
            "max": "n/a",
            "max_abs": "n/a",
            "clipped_fraction": "n/a",
        }
    clipped = np.abs(actions) >= CLIP_LIMIT
    return {
        "count": int(actions.shape[0]),
        "mean": round_list(np.mean(actions, axis=0)),
        "std": round_list(np.std(actions, axis=0)),
        "min": round_list(np.min(actions, axis=0)),
        "max": round_list(np.max(actions, axis=0)),
        "max_abs": round(float(np.max(np.abs(actions))), 6),
        "clipped_fraction": round(float(np.mean(clipped)), 6),
    }


def train_action_summary(run_dir: Path) -> dict[str, Any]:
    stats = read_json(run_dir / "action_statistics.json")
    action = stats.get("action", {})
    return {
        "count": action.get("count", "n/a"),
        "mean": action.get("mean", "n/a"),
        "std": action.get("std", "n/a"),
        "min": action.get("min", "n/a"),
        "max": action.get("max", "n/a"),
        "max_abs": action.get("max_abs", "n/a"),
        "clipped_fraction": action.get("clipped_fraction", "n/a"),
        "normalization": stats.get("normalization", {}),
        "source": stats.get("source", "missing"),
    }


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def action_scale_note(train_clip: Any, eval_clip: Any, eval_max_abs: Any) -> str:
    train_clip_num = number(train_clip)
    eval_clip_num = number(eval_clip)
    eval_max = number(eval_max_abs)
    if eval_clip_num is None:
        return "missing eval rollouts"
    if eval_clip_num > 0.25:
        return "many eval actions hit the clip limit; inspect action scale before tuning policy capacity"
    if train_clip_num is not None and abs(eval_clip_num - train_clip_num) > 0.10:
        return "train/eval clipped fractions differ noticeably; compare target scale and rollout behavior"
    if eval_max is not None and eval_max < 0.02:
        return "eval actions are very small; inspect stuck or underfit behavior"
    return "scale looks usable for teaching analysis; still inspect rollouts before making claims"


def run_row(run_dir: Path) -> dict[str, Any]:
    train_summary = read_json(run_dir / "training_summary.json")
    eval_summary = read_json(run_dir / "eval_summary.json")
    train_action = train_action_summary(run_dir)
    eval_action = action_summary(rollout_actions(run_dir))
    return {
        "run": run_dir.name,
        "exists": run_dir.exists(),
        "policy": train_summary.get("policy_name", eval_summary.get("policy_name", "n/a")),
        "chunk_size": train_summary.get("chunk_size", "n/a"),
        "success_rate": eval_summary.get("success_rate", "n/a"),
        "train_action_count": train_action["count"],
        "train_action_mean": train_action["mean"],
        "train_action_std": train_action["std"],
        "train_action_min": train_action["min"],
        "train_action_max": train_action["max"],
        "train_clipped_fraction": train_action["clipped_fraction"],
        "eval_action_count": eval_action["count"],
        "eval_action_mean": eval_action["mean"],
        "eval_action_std": eval_action["std"],
        "eval_action_min": eval_action["min"],
        "eval_action_max": eval_action["max"],
        "eval_clipped_fraction": eval_action["clipped_fraction"],
        "eval_max_abs": eval_action["max_abs"],
        "note": action_scale_note(train_action["clipped_fraction"], eval_action["clipped_fraction"], eval_action["max_abs"]),
    }


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
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


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "run",
        "policy",
        "chunk_size",
        "success_rate",
        "train_clipped_fraction",
        "eval_clipped_fraction",
        "eval_max_abs",
        "note",
    ]
    return [{key: row.get(key, "n/a") for key in keys} for row in rows]


def build_report(rows: list[dict[str, Any]], csv_path: Path) -> str:
    lines: list[str] = [
        "# LunaVLA Action Analysis Report",
        "",
        "This report compares train-time action targets with eval-time executable rollout actions.",
        "",
        "The goal is to make action scale visible before adding heavier policies or action-normalization ablations.",
        "",
        "## Key Idea",
        "",
        "- Train-time actions are supervised targets from demonstration records.",
        "- Eval-time executable actions are the policy outputs after clipping inside rollout evaluation.",
        f"- LunaVLA uses `{CLIP_LIMIT}` as the current teaching-scale eval clip limit.",
        "- A low training loss can still be misleading if executable actions saturate, shrink, or differ strongly from the demonstration scale.",
        "",
        "## Compact Comparison",
        "",
    ]
    lines.extend(markdown_table(compact_rows(rows)))
    lines.extend(["", "## Full Action Distribution", ""])
    lines.extend(markdown_table(rows))
    lines.extend(
        [
            "",
            f"CSV: `{relative(csv_path)}`",
            "",
            "## Normalization Boundary",
            "",
            "The saved action statistics include a z-score formula:",
            "",
            "```text",
            "normalized_action = (action - mean) / std",
            "action = normalized_action * std + mean",
            "```",
            "",
            "The current public path records these statistics for diagnostics and explanation. Do not claim a real-robot normalization pipeline until a verified real-data adapter and eval path exist.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_action_analysis_report.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dirs = [resolve(path) for path in args.runs]
    rows = [run_row(run_dir) for run_dir in run_dirs if run_dir.exists()]
    if not rows:
        raise FileNotFoundError("No run directories found for action analysis.")
    out_path = resolve(args.out)
    csv_path = resolve(args.csv)
    write_csv(csv_path, rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(rows, csv_path), encoding="utf-8")
    print(f"action analysis report: {out_path}")
    print(f"action analysis csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
