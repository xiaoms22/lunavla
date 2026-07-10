from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def prepare_run_dir(path: str | Path, *, overwrite: bool = False) -> Path:
    """Create an experiment directory without silently mixing runs."""

    target = Path(path)
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"experiment directory is not empty: {target}; pass --overwrite to rebuild it"
            )
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def reset_rollout_dir(path: str | Path) -> Path:
    """Remove stale rollout files before saving a new evaluation run."""

    target = Path(path)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def remove_stale_rollouts(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_metric(path: str | Path, step: int, metrics: dict[str, Any]) -> None:
    payload = {"step": step, **metrics}
    append_jsonl(path, payload)


def checkpoint_path(output_dir: str | Path, checkpoint_name: str = "checkpoint.json") -> Path:
    return Path(output_dir) / checkpoint_name


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_run_card(path: str | Path, title: str, summary: dict[str, Any]) -> None:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in summary.items():
        lines.append(f"| `{key}` | `{value}` |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
