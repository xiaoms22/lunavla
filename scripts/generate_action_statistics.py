from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import compute_action_statistics, load_dataset_from_config, load_jsonl, write_action_statistics
from trainer.trainer_utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LunaVLA action statistics.")
    parser.add_argument("--config", default="configs/act_pusht_baseline.yaml", help="Config used if train records are missing.")
    parser.add_argument("--run-dir", default="outputs/act_pusht_baseline", help="Run directory with train_records.jsonl.")
    parser.add_argument("--out", default="outputs/action_statistics.json", help="Public action statistics output path.")
    parser.add_argument("--report", default="outputs/action_statistics.md", help="Markdown action statistics report.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_records(config_path: Path, run_dir: Path) -> tuple[list[Any], str]:
    train_records = run_dir / "train_records.jsonl"
    if train_records.exists():
        return load_jsonl(train_records), relative(train_records)
    config = load_yaml(config_path)
    return load_dataset_from_config(config["dataset"]), relative(config_path)


def markdown_table(rows: list[tuple[str, Any]]) -> list[str]:
    lines = ["| item | value |", "| --- | --- |"]
    for key, value in rows:
        if isinstance(value, (list, dict)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        lines.append(f"| {key} | `{text}` |")
    return lines


def build_report(stats: dict[str, Any], out_path: Path, run_dir: Path, records_source: str) -> str:
    action = stats["action"]
    normalization = stats["normalization"]
    lines: list[str] = [
        "# LunaVLA Action Statistics",
        "",
        "This report explains the action scale used by the teaching PushT-style demonstrations.",
        "",
        "## Source",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                ("records source", records_source),
                ("run directory", relative(run_dir)),
                ("json", relative(out_path)),
                ("count", action["count"]),
                ("action dim", action["dim"]),
                ("unit", stats["unit"]),
            ]
        )
    )
    lines.extend(["", "## Distribution", ""])
    lines.extend(
        markdown_table(
            [
                ("mean", action["mean"]),
                ("std", action["std"]),
                ("min", action["min"]),
                ("max", action["max"]),
                ("p01", action["p01"]),
                ("p99", action["p99"]),
                ("max abs", action["max_abs"]),
                ("clip limit", action.get("clip_limit", "n/a")),
                ("clipped fraction", action.get("clipped_fraction", "n/a")),
            ]
        )
    )
    lines.extend(
        [
            "",
            "## Normalization",
            "",
            f"- Train-time formula: `{normalization['train_formula']}`.",
            f"- Eval-time formula: `{normalization['eval_formula']}`.",
            "- LunaVLA stores executable rollout actions in original action units so reports match environment behavior.",
            "- The current public baseline records stats for teaching and diagnostics; it does not claim real-robot calibration.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_action_statistics.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    run_dir = resolve(args.run_dir)
    out_path = resolve(args.out)
    report_path = resolve(args.report)
    records, records_source = load_records(config_path, run_dir)
    stats = compute_action_statistics(records, source=records_source, action_dim=2)
    write_action_statistics(out_path, stats)
    run_stats_path = run_dir / "action_statistics.json"
    if run_dir.exists() and not run_stats_path.exists():
        write_action_statistics(run_stats_path, stats)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(stats, out_path, run_dir, records_source), encoding="utf-8")
    print(f"action statistics: {out_path}")
    print(f"action statistics report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
