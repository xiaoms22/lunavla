from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from lunavla.artifact_contracts import (
    NUMPY_CHECKPOINT_ROOT_FIELDS,
    RUN_MANIFEST_SCHEMA3_FIELDS,
    TRANSFORMER_SCHEMA3_FIELDS,
    artifact_contract_descriptor,
)
from lunavla.manifest import RunManifest
from model.minivla_policy import NumpyLinearChunkPolicy


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "docs" / "v2" / "artifact_contracts.json"


def test_checked_in_artifact_descriptor_matches_runtime_contract() -> None:
    assert json.loads(GOLDEN.read_text(encoding="utf-8")) == artifact_contract_descriptor()
    assert {item.name for item in fields(RunManifest)} == RUN_MANIFEST_SCHEMA3_FIELDS


def test_numpy_writer_matches_golden_root_fields(tmp_path: Path) -> None:
    checkpoint = NumpyLinearChunkPolicy(2).save_pretrained(tmp_path)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert set(payload) == NUMPY_CHECKPOINT_ROOT_FIELDS


@pytest.mark.torch
def test_transformer_writer_matches_golden_root_fields(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    transformer_policy = pytest.importorskip("lunavla.transformer_policy")
    config = transformer_policy.TransformerPolicyConfig(
        state_dim=2,
        action_dim=2,
        chunk_size=1,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        latent_dim=2,
    )
    policy = transformer_policy.TransformerChunkCVAEPolicy(config)
    payload = pytest.importorskip("torch").load(
        policy.save_checkpoint(tmp_path / "checkpoint.pt"),
        map_location="cpu",
        weights_only=True,
    )
    assert set(payload) == TRANSFORMER_SCHEMA3_FIELDS
