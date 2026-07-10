from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import build_training_batch, load_dataset_from_config
from trainer.trainer_utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a LunaVLA dataset sample and action chunk target.")
    parser.add_argument("--config", default="configs/act_pusht_cpu_smoke.yaml", help="Config file to inspect.")
    parser.add_argument("--index", type=int, default=0, help="Training sample index to inspect.")
    parser.add_argument("--out", default="outputs/dataset_inspection.md", help="Markdown report path.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def rounded(values: np.ndarray | list[float], digits: int = 4) -> list[float]:
    array = np.asarray(values, dtype=np.float32)
    return [round(float(value), digits) for value in array.tolist()]


def action_chunk_pairs(target: np.ndarray, action_dim: int) -> list[list[float]]:
    if target.size % action_dim != 0:
        raise ValueError(f"target dim {target.size} is not divisible by action_dim {action_dim}")
    return [rounded(target[idx : idx + action_dim]) for idx in range(0, target.size, action_dim)]


def markdown_table(rows: list[tuple[str, Any]]) -> list[str]:
    lines = ["| field | value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | `{value}` |")
    return lines


def build_report(config_path: Path, sample_index: int) -> str:
    config = load_yaml(config_path)
    chunk_size = int(config["policy"].get("chunk_size", config.get("model", {}).get("chunk_size", 1)))
    action_dim = int(config.get("model", {}).get("action_dim", 2))
    instruction_dim = int(config.get("model", {}).get("instruction_dim", 8))
    records = load_dataset_from_config(config["dataset"])
    inputs, targets = build_training_batch(records, chunk_size=chunk_size, instruction_dim=instruction_dim)

    if not records:
        raise ValueError("dataset has no records")
    if sample_index < 0 or sample_index >= len(inputs):
        raise IndexError(f"sample index {sample_index} is outside [0, {len(inputs) - 1}]")

    record = records[sample_index]
    model_input = inputs[sample_index]
    target = targets[sample_index]
    observation = model_input[: len(record.observation)]
    instruction = model_input[len(record.observation) :]
    chunk_pairs = action_chunk_pairs(target, action_dim)

    lines: list[str] = [
        "# LunaVLA Dataset Inspection",
        "",
        "This report shows how one VLA record becomes one training sample.",
        "",
        "## Dataset Summary",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                ("config", relative(config_path)),
                ("dataset source", config["dataset"].get("source", "unknown")),
                ("records", len(records)),
                ("input shape", tuple(inputs.shape)),
                ("target shape", tuple(targets.shape)),
                ("chunk size", chunk_size),
                ("action dim", action_dim),
                ("instruction dim", instruction_dim),
            ]
        )
    )
    lines.extend(
        [
            "",
            "## Raw VLA Record",
            "",
            "```json",
            json.dumps(
                {
                    "observation": rounded(record.observation),
                    "action": rounded(record.action),
                    "episode_id": record.episode_id,
                    "timestep": record.timestep,
                    "success": record.success,
                    "language_instruction": record.language_instruction,
                    "metadata": record.metadata,
                },
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            "## Model Input",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                ("observation part", rounded(observation)),
                ("instruction feature part", rounded(instruction)),
                ("full input dim", model_input.size),
            ]
        )
    )
    lines.extend(
        [
            "",
            "## Action Chunk Target",
            "",
            "The target is a flattened sequence of future expert actions. For ACT-style training, the policy predicts the whole chunk from the current input.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                ("flattened target", rounded(target)),
                ("action pairs", chunk_pairs),
                ("target dim", target.size),
            ]
        )
    )
    lines.extend(
        [
            "",
            "## How To Explain It",
            "",
            (
                f"At sample `{sample_index}`, the policy sees the current observation plus an instruction feature vector "
                f"and learns to predict `{chunk_size}` future 2D actions. This is the small-scale version of the "
                "`observation -> action chunk` idea used in this repo."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    report = build_report(config_path, args.index)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"dataset inspection: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
