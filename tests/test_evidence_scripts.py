from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.build_v11_evidence_snapshot import main as snapshot_main
from scripts.build_v11_evidence_snapshot import snapshot_analysis
from scripts.render_readme_results import render


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_snapshot_rejects_duplicate_declared_run_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for run_dir in (first, second):
        write_json(run_dir / "manifest.json", {"config": {"project_name": "same-run"}})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_v11_evidence_snapshot.py",
            "--runs",
            str(first),
            str(second),
            "--out-dir",
            str(tmp_path / "snapshot"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate run_id"):
        snapshot_main()
    assert not (tmp_path / "snapshot").exists()


def test_analysis_snapshot_and_renderer_use_multi_seed_aggregates(tmp_path: Path) -> None:
    source = tmp_path / "controlled"
    write_json(
        source / "design.json",
        {
            "controlled": True,
            "train_seeds": [11, 22, 33, 44, 55],
            "eval_seeds": list(range(1000, 1020)),
            "families": {"chunk": {"treatments": ["chunk-1", "chunk-2"]}},
        },
    )
    (source / "per_episode.csv").write_text("family,treatment\n", encoding="utf-8")
    summary = {
        "family": "chunk",
        "controlled": True,
        "aggregates": [
            {
                "treatment": "chunk-1",
                "successes": 50,
                "trials": 100,
                "success_rate": 0.5,
                "success_wilson_95": [0.4038, 0.5962],
                "mean_final_distance": 0.12,
                "mean_action_smoothness": 0.02,
            },
            {
                "treatment": "chunk-2",
                "successes": 60,
                "trials": 100,
                "success_rate": 0.6,
                "success_wilson_95": [0.5020, 0.6906],
                "mean_final_distance": 0.10,
                "mean_action_smoothness": 0.03,
            },
        ],
        "contrasts": [
            {
                "reference": "chunk-1",
                "treatment": "chunk-2",
                "metric": "final_distance",
                "paired_n": 100,
                "mean_difference": -0.02,
                "paired_bootstrap_95": [-0.03, -0.01],
            }
        ],
    }
    write_json(source / "chunk" / "summary.json", summary)
    output = tmp_path / "release" / "results" / "v1.1"

    analysis = snapshot_analysis(source, output, allow_observational=False)
    write_json(
        output / "index.json",
        {
            "schema_version": "1.0",
            "release": "v1.1",
            "runs": [{"run_id": "placeholder"}],
            "analysis": analysis,
        },
    )
    readme_path = tmp_path / "release" / "README.md"
    readme_path.write_text("placeholder\n", encoding="utf-8")
    rendered = render(output / "index.json", readme_path)

    assert analysis["train_seed_count"] == 5
    assert analysis["eval_episode_count"] == 20
    assert "5 training seeds × 20 fixed evaluation episodes" in rendered
    assert "50.0% (40.4%–59.6%)" in rendered
    assert "`chunk-2` − `chunk-1`" in rendered
    assert "[-0.03, -0.01]" in rendered
    assert "(results/v1.1/analysis/chunk_summary.json)" in rendered
