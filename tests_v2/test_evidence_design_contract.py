from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from lunavla.evidence_design import EvidenceDesign


def _design_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "design_id": "language-alpha2",
        "suite": "language",
        "base_config": "configs/v2/transformer_chunk_cpu.yaml",
        "seeds": {
            "train": [11, 22, 33, 44, 55],
            "data": 42,
            "split": 42,
            "evaluation": [1000, 1001, 1002, 1003],
            "bootstrap": 202611,
        },
        "arms": [
            {"id": "control", "role": "control", "mode": "none"},
            {"id": "mask", "role": "intervention", "mode": "mask"},
            {"id": "shuffle", "role": "intervention", "mode": "shuffle"},
            {
                "id": "counterfactual",
                "role": "intervention",
                "mode": "counterfactual",
            },
        ],
        "metrics": [
            {"name": "success_rate", "kind": "binary", "direction": "negative"},
            {
                "name": "final_distance",
                "kind": "continuous",
                "direction": "positive",
            },
        ],
        "budget": {
            "dataset_episodes": 96,
            "batch_size": 32,
            "training_steps": 1000,
            "learning_rate": 0.0003,
            "evaluation_episodes": 4,
            "bootstrap_samples": 10_000,
        },
        "output": {
            "run_root": "outputs/evidence/language-alpha2",
            "snapshot_root": "results/v2/language-alpha2",
        },
    }


def test_evidence_design_round_trip_freezes_every_controlled_input(tmp_path: Path) -> None:
    payload = _design_payload()
    design = EvidenceDesign.from_mapping(payload)

    assert design.seeds.train == (11, 22, 33, 44, 55)
    assert design.seeds.evaluation == (1000, 1001, 1002, 1003)
    assert tuple(arm.mode for arm in design.arms) == (
        "none",
        "mask",
        "shuffle",
        "counterfactual",
    )
    assert design.budget.bootstrap_samples == 10_000
    assert design.output.run_root == "outputs/evidence/language-alpha2"
    assert design.to_dict() == payload
    assert len(design.sha256()) == 64

    path = tmp_path / "design.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    loaded = EvidenceDesign.load(path)
    assert loaded == design
    assert loaded.sha256() == design.sha256()


@pytest.mark.parametrize(
    ("mutate", "error_type", "message"),
    [
        (lambda value: value.update({"typo": 1}), ValueError, "unknown field"),
        (
            lambda value: value["seeds"].update({"typo": 1}),
            ValueError,
            "unknown field.*seeds",
        ),
        (
            lambda value: value["budget"].pop("training_steps"),
            ValueError,
            "missing field.*budget",
        ),
        (
            lambda value: value.update({"schema_version": True}),
            TypeError,
            "schema_version must be an integer",
        ),
        (
            lambda value: value["seeds"].update({"train": [11, 11]}),
            ValueError,
            "train values must be unique",
        ),
        (
            lambda value: value["seeds"].update({"evaluation": [1000, 1000]}),
            ValueError,
            "evaluation values must be unique",
        ),
        (
            lambda value: value["budget"].update({"evaluation_episodes": 3}),
            ValueError,
            "must equal the number of evaluation seeds",
        ),
        (
            lambda value: value["output"].update({"run_root": "../private"}),
            ValueError,
            "output.run_root",
        ),
        (
            lambda value: value["arms"][1].update({"mode": "occlusion"}),
            ValueError,
            "not valid for the language suite",
        ),
        (
            lambda value: value["arms"][1].update({"role": "control"}),
            ValueError,
            "requires role='intervention'",
        ),
    ],
)
def test_evidence_design_rejects_ambiguous_or_uncontrolled_values(
    mutate: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    payload = copy.deepcopy(_design_payload())
    mutate(payload)
    with pytest.raises(error_type, match=message):
        EvidenceDesign.from_mapping(payload)


def test_visual_design_supports_predeclared_state_only_baseline() -> None:
    payload = _design_payload()
    payload["design_id"] = "visual-beta1"
    payload["suite"] = "visual"
    payload["base_config"] = "configs/v2/transformer_visual_cpu.yaml"
    payload["arms"] = [
        {"id": "control", "role": "control", "mode": "none"},
        {"id": "occlusion", "role": "intervention", "mode": "occlusion"},
        {"id": "shuffle", "role": "intervention", "mode": "shuffle"},
        {"id": "state_only", "role": "baseline", "mode": "state_only"},
    ]
    design = EvidenceDesign.from_mapping(payload)
    assert design.suite == "visual"
    assert design.arms[-1].role == "baseline"


def test_evidence_design_reports_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\narms: [unterminated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML"):
        EvidenceDesign.load(path)
