from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import build_training_batch, load_dataset_from_config
from model.minivla_policy import MiniVLAPolicy
from trainer.trainer_utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a beginner ACT/action-chunk lesson.")
    parser.add_argument("--config", default="configs/act_pusht_baseline.yaml", help="Config file to inspect.")
    parser.add_argument("--run-dir", default="outputs/act_pusht_baseline", help="Run directory with checkpoint.")
    parser.add_argument("--index", type=int, default=0, help="Training sample index to explain.")
    parser.add_argument("--out", default="outputs/action_chunk_lesson.md", help="Markdown lesson path.")
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


def action_pairs(values: np.ndarray, action_dim: int) -> list[list[float]]:
    if values.size % action_dim != 0:
        raise ValueError(f"action vector dim {values.size} is not divisible by action_dim {action_dim}")
    return [rounded(values[idx : idx + action_dim]) for idx in range(0, values.size, action_dim)]


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines


def sample_rows(records: list[Any], sample_index: int, chunk_size: int, action_dim: int) -> tuple[list[dict[str, Any]], int]:
    if sample_index < 0 or sample_index >= len(records):
        raise IndexError(f"sample index {sample_index} is outside [0, {len(records) - 1}]")
    sample = records[sample_index]
    episode_records = [record for record in records if record.episode_id == sample.episode_id]
    episode_records.sort(key=lambda item: item.timestep)
    local_index = next(idx for idx, record in enumerate(episode_records) if record.timestep == sample.timestep)
    rows: list[dict[str, Any]] = []
    for offset in range(chunk_size):
        action_index = min(local_index + offset, len(episode_records) - 1)
        action_record = episode_records[action_index]
        rows.append(
            {
                "offset": offset,
                "episode timestep": action_record.timestep,
                "expert action": rounded(action_record.action),
                "clamped at episode end": "yes" if local_index + offset >= len(episode_records) else "no",
            }
        )
    return rows, local_index


def prediction_rows(
    checkpoint_path: Path,
    model_input: np.ndarray,
    target: np.ndarray,
    action_dim: int,
) -> list[dict[str, Any]]:
    if not checkpoint_path.exists():
        return [
            {
                "item": "checkpoint",
                "value": "missing",
                "lesson": f"Run training first to create `{relative(checkpoint_path)}`.",
            }
        ]
    policy, _metadata = MiniVLAPolicy.load(checkpoint_path)
    prediction = policy.predict(model_input)[0]
    error = prediction - target
    return [
        {
            "item": "target chunk",
            "value": action_pairs(target, action_dim),
            "lesson": "Future expert actions flattened into one supervised-learning label.",
        },
        {
            "item": "predicted chunk",
            "value": action_pairs(prediction, action_dim),
            "lesson": "The policy predicts the full chunk from the current observation and instruction features.",
        },
        {
            "item": "mean absolute error",
            "value": round(float(np.mean(np.abs(error))), 6),
            "lesson": "Small supervised error helps, but rollout metrics decide whether behavior is useful.",
        },
    ]


def build_lesson(config_path: Path, run_dir: Path, sample_index: int) -> str:
    config = load_yaml(config_path)
    chunk_size = int(config["policy"].get("chunk_size", config.get("model", {}).get("chunk_size", 1)))
    action_dim = int(config.get("model", {}).get("action_dim", 2))
    instruction_dim = int(config.get("model", {}).get("instruction_dim", 8))
    records = load_dataset_from_config(config["dataset"])
    inputs, targets = build_training_batch(records, chunk_size=chunk_size, instruction_dim=instruction_dim)
    if sample_index >= len(inputs):
        raise IndexError(f"sample index {sample_index} is outside [0, {len(inputs) - 1}]")

    sample = records[sample_index]
    model_input = inputs[sample_index]
    target = targets[sample_index]
    obs_dim = len(sample.observation)
    sample_chunk_rows, local_index = sample_rows(records, sample_index, chunk_size, action_dim)
    checkpoint_path = run_dir / "checkpoint.pt"

    overview_rows = [
        {"field": "config", "value": relative(config_path), "why it matters": "Defines chunk size and dataset size."},
        {"field": "dataset source", "value": config["dataset"].get("source", "unknown"), "why it matters": "Shows this is the teaching data layer."},
        {"field": "records", "value": len(records), "why it matters": "Number of demonstration steps."},
        {"field": "chunk size", "value": chunk_size, "why it matters": "Number of future actions predicted at once."},
        {"field": "input dim", "value": int(inputs.shape[1]), "why it matters": "Observation plus instruction features."},
        {"field": "target dim", "value": int(targets.shape[1]), "why it matters": "chunk_size multiplied by action_dim."},
    ]

    input_rows = [
        {
            "part": "observation",
            "value": rounded(model_input[:obs_dim]),
            "meaning": "Current object/goal state seen by the tiny policy.",
        },
        {
            "part": "instruction features",
            "value": rounded(model_input[obs_dim:]),
            "meaning": "Small hashed text feature vector for the language field.",
        },
        {
            "part": "raw instruction",
            "value": sample.language_instruction,
            "meaning": "Kept optional by the sample protocol while the current lesson stays focused on imitation learning.",
        },
    ]

    lesson_lines: list[str] = [
        "# LunaVLA Action Chunk Lesson",
        "",
        "This lesson explains the core ACT-style idea used by LunaVLA: predict a short chunk of future actions from the current observation.",
        "",
        "## Why This Matters",
        "",
        "A next-action behavior cloning policy predicts only `action_t`. An ACT-style policy predicts `action_t, action_t+1, ... action_t+K-1`. That small change gives beginners a concrete way to discuss temporal action representation, chunk-size ablations, and rollout behavior.",
        "",
        "## Config And Shapes",
        "",
    ]
    lesson_lines.extend(markdown_table(overview_rows))
    lesson_lines.extend(
        [
            "",
            "## One Sample",
            "",
            f"Sample `{sample_index}` belongs to episode `{sample.episode_id}` and local timestep `{local_index}`.",
            "",
        ]
    )
    lesson_lines.extend(markdown_table(input_rows))
    lesson_lines.extend(
        [
            "",
            "## How The Action Chunk Is Built",
            "",
            "For each training sample, LunaVLA looks ahead inside the same demonstration episode. Near the episode end, the last action is repeated so every target has the same shape.",
            "",
        ]
    )
    lesson_lines.extend(markdown_table(sample_chunk_rows))
    lesson_lines.extend(
        [
            "",
            "## Target Versus Prediction",
            "",
        ]
    )
    lesson_lines.extend(markdown_table(prediction_rows(checkpoint_path, model_input, target, action_dim)))
    lesson_lines.extend(
        [
            "",
            "## How To Explain ACT In This Repo",
            "",
            "1. Demonstrations provide ordered expert actions.",
            "2. The dataset converts each timestep into one input vector and one flattened future-action chunk.",
            "3. The policy learns this chunk with supervised loss.",
            "4. During rollout, the evaluator executes predicted actions in a closed loop.",
            "5. The ablation changes `chunk_size`, then compares success rate, final distance, action smoothness, and failure cases.",
            "",
            "## Common Beginner Trap",
            "",
            "Low training loss only says the policy matched the demonstration labels on sampled inputs. Rollout metrics are still required because predictions change the next state, and small mistakes can compound over time.",
            "",
            "## Files To Read Next",
            "",
            "- `dataset/pusht_dataset.py` for the action chunk construction.",
            "- `trainer/train_act_pusht.py` for the supervised training loop.",
            "- `eval_vla.py` for closed-loop rollout evaluation.",
            "- `outputs/run_comparison.md` for the chunk-size ablation result.",
            "- `docs/tutorials/action_chunking_act.md` for the static tutorial version.",
            "",
            "## Rebuild",
            "",
            "```bash",
            f"python scripts/generate_action_chunk_lesson.py --config {relative(config_path)} --run-dir {relative(run_dir)}",
            "```",
        ]
    )
    return "\n".join(lesson_lines) + "\n"


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    run_dir = resolve(args.run_dir)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_lesson(config_path, run_dir, args.index), encoding="utf-8")
    print(f"action chunk lesson: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
