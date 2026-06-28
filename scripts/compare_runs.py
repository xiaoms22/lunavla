from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MiniMind-VLA run directories.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories under outputs/ or absolute paths.")
    parser.add_argument("--out", default="outputs/run_comparison.md", help="Markdown output path.")
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run_row(run_dir: Path) -> dict[str, Any]:
    train = read_json(run_dir / "training_summary.json")
    eval_summary = read_json(run_dir / "eval_summary.json")
    return {
        "run": run_dir.name,
        "records": train.get("records", "n/a"),
        "chunk_size": train.get("chunk_size", "n/a"),
        "final_loss": train.get("final_loss", "n/a"),
        "success_rate": eval_summary.get("success_rate", "n/a"),
        "mean_final_distance": eval_summary.get("mean_final_distance", "n/a"),
        "mean_rollout_length": eval_summary.get("mean_rollout_length", "n/a"),
        "mean_action_smoothness": eval_summary.get("mean_action_smoothness", "n/a"),
        "failure_cases": count_jsonl(run_dir / "failure_cases.jsonl"),
    }


def main() -> int:
    args = parse_args()
    rows = [run_row(resolve(run)) for run in args.runs]
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = list(rows[0].keys()) if rows else []
    lines = ["# MiniMind-VLA Run Comparison", ""]
    if rows:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

        csv_path = out_path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        lines.extend(["", f"CSV: `{csv_path.relative_to(ROOT).as_posix()}`"])

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
