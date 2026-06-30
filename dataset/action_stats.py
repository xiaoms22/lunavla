from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


EPSILON = 1e-6


def _record_action(record: Any) -> Sequence[float]:
    if isinstance(record, dict):
        return record["action"]
    return record.action


def actions_to_array(records: Iterable[Any], action_dim: int | None = None) -> np.ndarray:
    actions = [np.asarray(_record_action(record), dtype=np.float32) for record in records]
    if not actions:
        raise ValueError("cannot compute action statistics from an empty record set")
    values = np.vstack(actions).astype(np.float32)
    if action_dim is not None and values.shape[1] != action_dim:
        raise ValueError(f"expected action_dim={action_dim}, got {values.shape[1]}")
    return values


def _round_list(values: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values.tolist()]


def summarize_actions(actions: np.ndarray, clip_limit: float | None = 0.12) -> dict[str, Any]:
    stats = {
        "count": int(actions.shape[0]),
        "dim": int(actions.shape[1]),
        "mean": _round_list(np.mean(actions, axis=0)),
        "std": _round_list(np.std(actions, axis=0) + EPSILON),
        "min": _round_list(np.min(actions, axis=0)),
        "max": _round_list(np.max(actions, axis=0)),
        "p01": _round_list(np.percentile(actions, 1, axis=0)),
        "p99": _round_list(np.percentile(actions, 99, axis=0)),
        "max_abs": round(float(np.max(np.abs(actions))), 6),
    }
    if clip_limit is not None:
        clipped = np.abs(actions) >= float(clip_limit)
        stats["clip_limit"] = float(clip_limit)
        stats["clipped_fraction"] = round(float(np.mean(clipped)), 6)
    return stats


def compute_action_statistics(
    records: Iterable[Any],
    *,
    source: str,
    action_dim: int | None = 2,
    clip_limit: float | None = 0.12,
    unit: str = "delta_xy_per_step",
) -> dict[str, Any]:
    actions = actions_to_array(records, action_dim=action_dim)
    action_summary = summarize_actions(actions, clip_limit=clip_limit)
    return {
        "schema_version": 1,
        "source": source,
        "unit": unit,
        "action": action_summary,
        "normalization": {
            "type": "z_score",
            "epsilon": EPSILON,
            "mean": action_summary["mean"],
            "std": action_summary["std"],
            "train_formula": "normalized_action = (action - mean) / std",
            "eval_formula": "action = normalized_action * std + mean",
        },
        "boundary": (
            "These statistics describe teaching-scale PushT-style demonstration actions. "
            "They are useful for checking scale and normalization, not for real-robot deployment claims."
        ),
    }


def normalize_actions(actions: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(stats["normalization"]["std"], dtype=np.float32)
    return (np.asarray(actions, dtype=np.float32) - mean) / np.maximum(std, EPSILON)


def unnormalize_actions(normalized_actions: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(stats["normalization"]["std"], dtype=np.float32)
    return np.asarray(normalized_actions, dtype=np.float32) * np.maximum(std, EPSILON) + mean


def compact_action_statistics(stats: dict[str, Any], path: str | None = None) -> dict[str, Any]:
    action = stats.get("action", {})
    compact = {
        "source": stats.get("source", "unknown"),
        "count": action.get("count", "n/a"),
        "dim": action.get("dim", "n/a"),
        "mean": action.get("mean", []),
        "std": action.get("std", []),
        "min": action.get("min", []),
        "max": action.get("max", []),
        "normalization": stats.get("normalization", {}).get("type", "none"),
    }
    if path is not None:
        compact["path"] = path
    return compact


def write_action_statistics(path: str | Path, stats: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return target

