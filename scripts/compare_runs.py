from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRIC_DIRECTIONS = {
    "final_loss": "lower",
    "success_rate": "higher",
    "mean_final_distance": "lower",
    "mean_rollout_length": "context",
    "mean_action_smoothness": "lower",
    "failure_cases": "lower",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LunaVLA run directories.")
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


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_value(value: Any) -> str:
    number = as_number(value)
    if number is None:
        return str(value)
    return f"{number:.6g}"


def format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.6g}"


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


def changed_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for key in ["records", "chunk_size"]:
        values = {str(row.get(key, "n/a")) for row in rows}
        if len(values) > 1:
            fields.append(key)
    return fields


def metric_interpretation(metric: str, baseline: Any, candidate: Any) -> str:
    base = as_number(baseline)
    value = as_number(candidate)
    if base is None or value is None:
        return "not numeric"
    delta = value - base
    if abs(delta) < 1e-12:
        return "unchanged"
    direction = METRIC_DIRECTIONS.get(metric, "context")
    if direction == "higher":
        return "improved" if delta > 0 else "worse"
    if direction == "lower":
        return "improved" if delta < 0 else "worse"
    return "changed; inspect rollout behavior"


def delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return []
    baseline = rows[0]
    deltas: list[dict[str, Any]] = []
    for row in rows[1:]:
        for metric in METRIC_DIRECTIONS:
            base = as_number(baseline.get(metric))
            value = as_number(row.get(metric))
            delta = None if base is None or value is None else value - base
            deltas.append(
                {
                    "run": row["run"],
                    "metric": metric,
                    "baseline": baseline.get(metric, "n/a"),
                    "candidate": row.get(metric, "n/a"),
                    "delta": "n/a" if delta is None else round(delta, 8),
                    "interpretation": metric_interpretation(metric, baseline.get(metric), row.get(metric)),
                }
            )
    return deltas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row[h]) for h in headers) + " |")
    return lines


def summary_bullets(rows: list[dict[str, Any]], deltas: list[dict[str, Any]]) -> list[str]:
    if len(rows) < 2:
        return ["- Add at least two runs to generate an ablation interpretation."]

    changed = changed_fields(rows)
    baseline = rows[0]
    candidate = rows[1]
    bullets = []
    if changed:
        changes = ", ".join(f"`{field}` `{baseline.get(field)}` -> `{candidate.get(field)}`" for field in changed)
        bullets.append(f"- The checked ablation changes {changes}.")
    else:
        bullets.append("- No setup field changed in the summary; verify that the intended config was used.")

    success_delta = next((row for row in deltas if row["run"] == candidate["run"] and row["metric"] == "success_rate"), None)
    distance_delta = next((row for row in deltas if row["run"] == candidate["run"] and row["metric"] == "mean_final_distance"), None)
    smooth_delta = next((row for row in deltas if row["run"] == candidate["run"] and row["metric"] == "mean_action_smoothness"), None)
    if success_delta:
        bullets.append(
            "- Success rate changed from "
            f"`{format_value(success_delta['baseline'])}` to `{format_value(success_delta['candidate'])}` "
            f"({format_delta(as_number(success_delta['candidate']) - as_number(success_delta['baseline']))})."
        )
    if distance_delta:
        bullets.append(
            "- Mean final distance changed by "
            f"`{format_delta(as_number(distance_delta['candidate']) - as_number(distance_delta['baseline']))}`; "
            "lower is better for this task."
        )
    if smooth_delta:
        bullets.append(
            "- Action smoothness changed by "
            f"`{format_delta(as_number(smooth_delta['candidate']) - as_number(smooth_delta['baseline']))}`; "
            "use rollout inspection before calling this better or worse."
        )
    bullets.append("- Treat this as a teaching-scale ablation. Strong claims need more eval episodes or repeated seeds.")
    return bullets


def main() -> int:
    args = parse_args()
    rows = [run_row(resolve(run)) for run in args.runs]
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    deltas = delta_rows(rows)
    lines = ["# LunaVLA Ablation Comparison", ""]
    if rows:
        lines.extend(
            [
                "## Question",
                "",
                "What behavior changes when one policy or training setting changes?",
                "",
                "Config audit: `outputs/config_diff.md`",
                "",
                "## Runs",
                "",
            ]
        )
        lines.extend(markdown_table(rows))

        csv_path = out_path.with_suffix(".csv")
        write_csv(csv_path, rows)
        lines.extend(["", f"Run CSV: `{csv_path.relative_to(ROOT).as_posix()}`"])

        if deltas:
            delta_csv_path = out_path.with_name(out_path.stem + "_deltas.csv")
            write_csv(delta_csv_path, deltas)
            lines.extend(
                [
                    "",
                    "## Metric Deltas Vs Baseline",
                    "",
                ]
            )
            lines.extend(markdown_table(deltas))
            lines.extend(["", f"Delta CSV: `{delta_csv_path.relative_to(ROOT).as_posix()}`"])

        lines.extend(
            [
                "",
                "## Auto Interpretation",
                "",
            ]
        )
        lines.extend(summary_bullets(rows, deltas))
        lines.extend(
            [
                "",
                "## Resume-Safe Claim",
                "",
                (
                    "I ran a controlled chunk-size ablation for a tiny ACT-style policy and compared final loss, "
                    "rollout success rate, final distance, rollout length, action smoothness, and failure cases. "
                    "The claim is limited to this teaching-scale PushT-style setup."
                ),
                "",
                "## Next Check",
                "",
                "Inspect the saved rollout JSON or web demo for each run before turning the numeric delta into a conclusion.",
            ]
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
