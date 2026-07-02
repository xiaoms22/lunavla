from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import (
    build_training_batch,
    compact_action_statistics,
    compute_action_statistics,
    load_dataset_from_config,
    save_jsonl,
    write_action_statistics,
)
from model import ACTPolicyWrapper
from trainer.trainer_utils import append_jsonl, ensure_dir, load_yaml, write_csv, write_json, write_run_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny ACT-style PushT policy.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)

    artifact_config = config["artifacts"]
    output_dir = ensure_dir(ROOT / artifact_config["output_dir"])
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    chunk_size = int(config["policy"].get("chunk_size", 1))
    records = load_dataset_from_config(config["dataset"])
    save_jsonl(records, output_dir / "train_records.jsonl")
    inputs, targets = build_training_batch(records, chunk_size=chunk_size)
    action_stats = compute_action_statistics(records, source="training_records", action_dim=2)
    action_stats_path = output_dir / "action_statistics.json"
    write_action_statistics(action_stats_path, action_stats)
    action_stats_ref = action_stats_path.relative_to(ROOT).as_posix()

    train_config = config["training"]
    rng = np.random.default_rng(int(train_config.get("seed", 42)))
    policy = ACTPolicyWrapper(
        input_dim=inputs.shape[1],
        action_dim=2,
        chunk_size=chunk_size,
        seed=int(train_config.get("seed", 42)),
    )

    num_steps = int(train_config.get("num_steps", 100))
    batch_size = min(int(train_config.get("batch_size", 32)), len(inputs))
    learning_rate = float(train_config.get("learning_rate", 0.04))
    log_interval = max(1, int(train_config.get("log_interval", 10)))
    loss_rows: list[dict[str, float | int]] = []

    for step in range(1, num_steps + 1):
        indices = rng.choice(len(inputs), size=batch_size, replace=len(inputs) < batch_size)
        loss = policy.train_step(inputs[indices], targets[indices], learning_rate)
        if step == 1 or step % log_interval == 0 or step == num_steps:
            row = {"step": step, "loss": round(loss, 8)}
            loss_rows.append(row)
            append_jsonl(metrics_path, row)

    checkpoint_path = output_dir / artifact_config.get("checkpoint_name", "checkpoint.pt")
    metadata = {
        "project_name": config["project_name"],
        "framework": config["framework"],
        "task": config["task"],
        "policy": config["policy"],
        "dataset": config["dataset"],
        "training": config["training"],
        "eval": config.get("eval", {}),
        "config_path": str(config_path.as_posix()),
        "policy_interface": {
            "policy_class": policy.__class__.__name__,
            "policy_name": getattr(policy, "policy_name", "unknown"),
            "contract": "forward(batch)->losses; predict_action(sample)->action_chunk",
        },
        "action_stats": compact_action_statistics(action_stats, path=action_stats_ref),
    }
    policy.save(checkpoint_path, metadata=metadata)

    summary = {
        "project_name": config["project_name"],
        "policy_name": getattr(policy, "policy_name", "unknown"),
        "policy_interface": "MiniVLAPolicyBase",
        "dataset_source": config["dataset"].get("source", "unknown"),
        "dataset_path": config["dataset"].get("path", "n/a"),
        "dataset_split": config["dataset"].get("split", "n/a"),
        "checkpoint": str(checkpoint_path.relative_to(ROOT).as_posix()),
        "records": len(records),
        "input_dim": int(inputs.shape[1]),
        "target_dim": int(targets.shape[1]),
        "chunk_size": chunk_size,
        "action_stats_path": action_stats_ref,
        "action_mean": action_stats["action"]["mean"],
        "action_std": action_stats["action"]["std"],
        "action_min": action_stats["action"]["min"],
        "action_max": action_stats["action"]["max"],
        "num_steps": num_steps,
        "learning_rate": learning_rate,
        "final_loss": loss_rows[-1]["loss"],
    }
    write_csv(output_dir / "loss_curve.csv", loss_rows)
    write_json(output_dir / "training_summary.json", summary)
    write_run_card(output_dir / "run_card.md", "LunaVLA Training Run", summary)

    print(f"trained: {summary['project_name']}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"final_loss: {summary['final_loss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
