from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest

from dataset import (
    compact_action_statistics,
    compute_action_statistics,
    normalize_actions,
    unnormalize_actions,
    write_action_statistics,
)
from dataset.action_stats import actions_to_array, summarize_actions
from trainer.trainer_utils import (
    append_jsonl,
    checkpoint_path,
    ensure_dir,
    load_yaml,
    prepare_run_dir,
    read_json,
    reset_rollout_dir,
    setup_seed,
    write_csv,
    write_json,
    write_metric,
    write_run_card,
)


def test_action_statistics_and_normalization_round_trip(tmp_path: Path) -> None:
    records = [
        {"action": [-0.12, 0.04]},
        {"action": [0.00, 0.08]},
        {"action": [0.12, -0.04]},
    ]
    stats = compute_action_statistics(records, source="unit-test", action_dim=2)
    actions = actions_to_array(records)
    restored = unnormalize_actions(normalize_actions(actions, stats), stats)
    compact = compact_action_statistics(stats, "stats.json")
    path = write_action_statistics(tmp_path / "stats.json", stats)

    np.testing.assert_allclose(restored, actions, atol=1e-6)
    assert stats["action"]["count"] == 3
    assert stats["action"]["clipped_fraction"] > 0
    assert compact["path"] == "stats.json"
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_action_statistics_error_and_no_clip_paths() -> None:
    with pytest.raises(ValueError, match="empty"):
        actions_to_array([])
    with pytest.raises(ValueError, match="action_dim"):
        actions_to_array([{"action": [1.0]}], action_dim=2)
    summary = summarize_actions(np.array([[1.0, 2.0]], dtype=np.float32), clip_limit=None)
    assert "clipped_fraction" not in summary
    assert "path" not in compact_action_statistics({})


def test_run_directory_helpers_prevent_stale_artifacts(tmp_path: Path) -> None:
    run_dir = prepare_run_dir(tmp_path / "run")
    (run_dir / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare_run_dir(run_dir)
    rebuilt = prepare_run_dir(run_dir, overwrite=True)
    assert list(rebuilt.iterdir()) == []

    rollout_dir = ensure_dir(rebuilt / "rollouts")
    (rollout_dir / "old.json").write_text("{}", encoding="utf-8")
    reset = reset_rollout_dir(rollout_dir)
    assert list(reset.iterdir()) == []


def test_serialization_and_report_helpers(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("value: 3\n", encoding="utf-8")
    assert load_yaml(yaml_path) == {"value": 3}

    json_path = tmp_path / "nested" / "value.json"
    write_json(json_path, {"value": 3})
    assert read_json(json_path) == {"value": 3}

    metrics_path = tmp_path / "metrics.jsonl"
    append_jsonl(metrics_path, {"kind": "header"})
    write_metric(metrics_path, 2, {"loss": 0.5})
    assert [json.loads(line) for line in metrics_path.read_text().splitlines()] == [
        {"kind": "header"},
        {"step": 2, "loss": 0.5},
    ]

    csv_path = tmp_path / "metrics.csv"
    write_csv(csv_path, [])
    assert not csv_path.exists()
    write_csv(csv_path, [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.2}])
    with csv_path.open(newline="", encoding="utf-8") as file:
        assert list(csv.DictReader(file))[1]["loss"] == "0.2"

    card = tmp_path / "run-card.md"
    write_run_card(card, "Test Run", {"status": "ok"})
    assert "# Test Run" in card.read_text(encoding="utf-8")
    assert checkpoint_path(tmp_path).name == "checkpoint.json"


def test_setup_seed_controls_python_and_numpy() -> None:
    setup_seed(55)
    first = (random.random(), np.random.random())
    setup_seed(55)
    second = (random.random(), np.random.random())
    assert first == second
