from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lunavla.v3 import (
    ExperimentConfig,
    IntegrationRuntime,
    LeRobotDatasetSourceV3,
    SourceFileRecordV1,
    SourceInventoryV1,
    preflight_source,
    run_integration,
    verify_integration,
)
from lunavla.v3 import integration_workflow as workflow


def _pusht_episodes() -> tuple[Any, ...]:
    config = ExperimentConfig.load("configs/v3/beta2_pusht_integration.yaml")
    rows = [
        {
            "episode_index": 0,
            "frame_index": step,
            "observation.image": np.full((96, 96, 3), step % 256, dtype=np.uint8),
            "observation.state": np.asarray([step, step + 1], dtype=np.float32),
            "action": np.asarray([0.1, -0.1], dtype=np.float32),
        }
        for step in range(161)
    ]
    return LeRobotDatasetSourceV3(
        config.external_dataset_spec,  # type: ignore[arg-type]
        config.feature_schema,
        dataset_factory=lambda **_: rows,
    ).load()


def _inventory(config: ExperimentConfig) -> SourceInventoryV1:
    spec = config.external_dataset_spec
    assert spec is not None
    files = tuple(
        SourceFileRecordV1(path, 1, digest) for path, digest in spec.file_hashes.items()
    )
    return SourceInventoryV1(spec.repo_id, spec.revision, len(files), spec.max_download_bytes, files)


def test_source_preflight_validates_pinned_hub_identity_hashes_and_limit(tmp_path: Path) -> None:
    config = ExperimentConfig.load("configs/v3/beta2_pusht_integration.yaml")
    spec = config.external_dataset_spec
    assert spec is not None
    siblings = [
        SimpleNamespace(
            rfilename=path,
            size=index + 1,
            lfs=SimpleNamespace(size=index + 1, sha256=digest),
        )
        for index, (path, digest) in enumerate(spec.file_hashes.items())
    ]
    api = SimpleNamespace(
        dataset_info=lambda *_args, **_kwargs: SimpleNamespace(
            sha=spec.revision, siblings=siblings
        )
    )
    inventory = preflight_source(config, api=api, metadata_cache=tmp_path)
    assert inventory.repo_id == "lerobot/pusht"
    assert inventory.planned_download_bytes == 6
    siblings[0].lfs.sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 drift"):
        preflight_source(config, api=api, metadata_cache=tmp_path)


def test_integration_run_is_atomic_fail_closed_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = Path("configs/v3/beta2_pusht_integration.yaml")
    config = ExperimentConfig.load(config_path)
    runtime_environment = tmp_path / "fixture-runtime-environment.json"
    runtime_environment.write_text('{"fixture":true}\n', encoding="utf-8")
    lock = tmp_path / "integration.lock"
    lock.write_text("fixture==1 --hash=sha256:" + "1" * 64 + "\n", encoding="utf-8")
    monkeypatch.setattr(workflow, "_LOCK", lock)
    monkeypatch.setattr(workflow, "_git_state", lambda: ("a" * 40, False))

    def runtime(_config: ExperimentConfig, _cache: Path) -> IntegrationRuntime:
        return IntegrationRuntime(
            _inventory(config),
            _pusht_episodes(),
            {"environments": [{"steps": 3}], "closed": True},
            (
                {"policy_id": "act_v3", "loss": 1.0, "gradient_norm": 1.0, "finite": True},
                {"policy_id": "diffusion_v3", "loss": 1.0, "gradient_norm": 1.0, "finite": True},
            ),
            runtime_environment,
            "fixture",
        )

    output = tmp_path / "integration-output"
    result = run_integration(
        config_path,
        cache_dir=tmp_path / "cache",
        output_root=output,
        runtime_factory=runtime,
    )
    manifest = verify_integration(result)
    assert manifest.claim_allowed is False
    assert manifest.benchmark_claim is False
    assert len(manifest.policy_smokes) == 2
    with pytest.raises(FileExistsError, match="overwrite"):
        run_integration(
            config_path,
            cache_dir=tmp_path / "cache",
            output_root=output,
            runtime_factory=runtime,
        )
    metrics = output / "metrics.json"
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    payload["policy_smokes"][0]["loss"] = 0.0
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics_sha256"):
        verify_integration(output)


def test_integration_failure_preserves_no_partial_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "integration.lock"
    lock.write_text("locked\n", encoding="utf-8")
    monkeypatch.setattr(workflow, "_LOCK", lock)
    output = tmp_path / "failed"

    def fail(_config: ExperimentConfig, _cache: Path) -> IntegrationRuntime:
        raise RuntimeError("decode failed")

    with pytest.raises(RuntimeError, match="decode failed"):
        run_integration(
            "configs/v3/beta2_pusht_integration.yaml",
            cache_dir=tmp_path / "cache",
            output_root=output,
            runtime_factory=fail,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.staging-*"))


def test_real_runtime_requires_explicit_hosted_cpu_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LUNAVLA_HOSTED_CPU", raising=False)
    config = ExperimentConfig.load("configs/v3/beta2_pusht_integration.yaml")
    with pytest.raises(RuntimeError, match="LUNAVLA_HOSTED_CPU"):
        workflow._default_runtime(config, tmp_path)


def test_multi_camera_act_payload_consumes_every_declared_camera() -> None:
    config = ExperimentConfig.load("configs/v3/beta2_libero_integration.yaml")
    derived = workflow._policy_payload(config, "act_v3")
    assert derived.policy["parameters"]["camera_features"] == (
        "camera.agentview",
        "camera.wrist",
    )
    assert derived.prompt["camera_order"] == ("camera.agentview", "camera.wrist")
