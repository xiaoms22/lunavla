from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lunavla.v3 import (
    PolicyProfileDesignV1,
    PolicyProfileManifestV1,
    run_profile,
    verify_profile,
)
from lunavla.v3.cli import main


def _design(tmp_path: Path) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "config": "configs/v3/fake_pusht_alpha.yaml",
                "warmup": 5,
                "measurements": 20,
                "batch_size": 2,
                "device": "cpu",
                "threads": 1,
                "python_version": "3.12",
                "platforms": ["darwin", "linux"],
                "output_dir": "profile-output",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_profile_contracts_are_strict_and_reject_accelerators(tmp_path: Path) -> None:
    value = yaml.safe_load(_design(tmp_path).read_text(encoding="utf-8"))
    design = PolicyProfileDesignV1.from_mapping(value)
    assert PolicyProfileDesignV1.from_mapping(design.to_dict()).sha256() == design.sha256()
    for field, invalid in (("schema_version", True), ("warmup", 4), ("device", "cuda")):
        payload = design.to_dict()
        payload[field] = invalid
        with pytest.raises((TypeError, ValueError)):
            PolicyProfileDesignV1.from_mapping(payload)
    payload = design.to_dict()
    payload["unknown"] = 1
    with pytest.raises(ValueError, match="exact fields"):
        PolicyProfileDesignV1.from_mapping(payload)


def test_profile_run_verify_and_tamper_detection(tmp_path: Path) -> None:
    output = run_profile(_design(tmp_path))
    manifest = verify_profile(output)
    assert manifest.policy_id == "numpy_linear_chunk"
    assert manifest.comparative_claim_allowed is False
    assert PolicyProfileManifestV1.from_mapping(manifest.to_dict()) == manifest
    assert main(["profile-verify", str(output)]) == 0
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(metrics["training_latency_ms"]) == 20
    assert len(metrics["inference_latency_ms"]) == 20
    metrics["training_latency_ms"][0] += 1
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics_sha256"):
        verify_profile(output)


def test_profile_output_is_atomic_and_requires_overwrite(tmp_path: Path) -> None:
    design = _design(tmp_path)
    output = run_profile(design)
    with pytest.raises(FileExistsError, match="overwrite"):
        run_profile(design)
    replaced = run_profile(design, overwrite=True)
    assert replaced == output
    assert verify_profile(output).policy_id == "numpy_linear_chunk"
