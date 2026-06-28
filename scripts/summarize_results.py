from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a MiniMind-VLA run directory.")
    parser.add_argument("--run-dir", required=True, help="Run directory under outputs/ or an absolute path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    training = read_json(run_dir / "training_summary.json")
    evaluation = read_json(run_dir / "eval_summary.json")
    failure_count = count_jsonl(run_dir / "failure_cases.jsonl")

    table = {
        "project_name": training.get("project_name", "unknown"),
        "final_loss": training.get("final_loss", "n/a"),
        "chunk_size": training.get("chunk_size", "n/a"),
        "records": training.get("records", "n/a"),
        "success_rate": evaluation.get("success_rate", "n/a"),
        "mean_final_distance": evaluation.get("mean_final_distance", "n/a"),
        "mean_rollout_length": evaluation.get("mean_rollout_length", "n/a"),
        "mean_action_smoothness": evaluation.get("mean_action_smoothness", "n/a"),
        "failure_cases": failure_count,
    }

    csv_path = run_dir / "result_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table.keys()))
        writer.writeheader()
        writer.writerow(table)

    report = [
        "# MiniMind-VLA Run Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in table.items():
        report.append(f"| `{key}` | `{value}` |")
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `final_loss` checks whether the imitation objective is numerically moving.",
            "- `success_rate` is the headline rollout metric for internship project evidence.",
            "- `failure_cases` should be read with `docs/failure_taxonomy.md` before adding claims to a resume.",
        ]
    )
    report_path = run_dir / "summary_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"summary: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
