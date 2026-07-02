from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import generate_mock_pusht_records, save_jsonl
from trainer.trainer_utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a small PushT-style demonstration set as JSONL.")
    parser.add_argument("--config", default="configs/act_pusht_jsonl_smoke.yaml", help="Config with JSONL dataset settings.")
    parser.add_argument("--out", default=None, help="JSONL output path. Defaults to dataset.path in the config.")
    parser.add_argument("--report", default="outputs/jsonl_dataset_export.md", help="Markdown export report.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def markdown_table(rows: list[tuple[str, Any]]) -> list[str]:
    lines = ["| field | value |", "| --- | --- |"]
    for key, value in rows:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        lines.append(f"| {key} | `{text}` |")
    return lines


def build_report(config_path: Path, out_path: Path, records: list[Any], dataset_config: dict[str, Any]) -> str:
    first = records[0]
    lines: list[str] = [
        "# LunaVLA JSONL Dataset Export",
        "",
        "This report shows a local PushT-style demonstration export that can be reloaded with `dataset.source: jsonl`.",
        "",
        "## Export",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                ("config", relative(config_path)),
                ("jsonl path", relative(out_path)),
                ("records", len(records)),
                ("num episodes", dataset_config.get("num_episodes", "n/a")),
                ("steps per episode", dataset_config.get("steps_per_episode", "n/a")),
                ("seed", dataset_config.get("seed", "n/a")),
                ("start low", dataset_config.get("start_low", 0.05)),
                ("start high", dataset_config.get("start_high", 0.95)),
                ("goal", dataset_config.get("goal", [0.80, 0.20])),
                ("action noise std", dataset_config.get("action_noise_std", 0.004)),
                ("language instruction", dataset_config.get("language_instruction", "n/a")),
            ]
        )
    )
    lines.extend(
        [
            "",
            "## First Record",
            "",
            "```json",
            json.dumps(first.__dict__, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Rebuild",
            "",
            "```bash",
            f"python scripts/export_pusht_jsonl_dataset.py --config {relative(config_path)}",
            "```",
            "",
            "## Boundary",
            "",
            "This is still teaching-scale PushT-style demonstration data. The purpose is to learn the local file-data path before moving to heavier robotics datasets.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_yaml(config_path)
    dataset_config = config["dataset"]
    out_path = resolve(args.out or dataset_config["path"])
    report_path = resolve(args.report)

    records = generate_mock_pusht_records(
        num_episodes=int(dataset_config.get("num_episodes", 12)),
        steps_per_episode=int(dataset_config.get("steps_per_episode", 16)),
        seed=int(dataset_config.get("seed", 123)),
        language_instruction=dataset_config.get("language_instruction", "push the T block to the goal"),
        goal=dataset_config.get("goal", [0.80, 0.20]),
        start_low=float(dataset_config.get("start_low", 0.05)),
        start_high=float(dataset_config.get("start_high", 0.95)),
        action_gain=float(dataset_config.get("action_gain", 0.35)),
        action_clip=float(dataset_config.get("action_clip", 0.12)),
        action_noise_std=float(dataset_config.get("action_noise_std", 0.004)),
        success_distance=float(dataset_config.get("success_distance", 0.08)),
    )
    save_jsonl(records, out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(config_path, out_path, records, dataset_config), encoding="utf-8")
    print(f"jsonl dataset: {out_path}")
    print(f"jsonl export report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
