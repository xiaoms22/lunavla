from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lunavla.v3 import (
    QWEN3_VL_REPO_ID,
    QWEN3_VL_REVISION,
    SMOLVLM2_REPO_ID,
    SMOLVLM2_REVISION,
    DeterministicFixtureExtractor,
    VLMBackendSpecV1,
    build_frozen_feature_cache,
    make_v31_task_dataset,
    preflight_local_model,
    run_qwen_observational_smoke,
    verify_frozen_feature_cache,
)


H0 = "0" * 64
H1 = "1" * 64
H2 = "2" * 64


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _backend(
    backend_id: str = "smolvlm2_500m",
    *,
    file_hash: str = H2,
    total_bytes: int = 5,
) -> VLMBackendSpecV1:
    qwen = backend_id == "qwen3_vl_2b"
    return VLMBackendSpecV1(
        backend_id=backend_id,
        repo_id=QWEN3_VL_REPO_ID if qwen else SMOLVLM2_REPO_ID,
        revision=QWEN3_VL_REVISION if qwen else SMOLVLM2_REVISION,
        spdx_license="Apache-2.0",
        license_scope="model_weights",
        license_evidence_sha256=H0,
        processor_class="AutoProcessor",
        processor_config_sha256=H1,
        model_config_sha256=H2,
        hidden_layer=-1,
        pooling="attention_mask_mean",
        image_token_layout="processor_native",
        camera_order=("camera.primary",),
        model_dtype="float32",
        device="cpu",
        offload_plan="cpu_sharded" if qwen else "none",
        deterministic=True,
        evidence_role="observational" if qwen else "claim_bearing",
        weight_files={"model.safetensors": file_hash},
        total_weight_bytes=total_bytes,
    )


def _dataset():
    return make_v31_task_dataset(data_seed=42, train_per_task=1, held_out_per_cell=2)


def test_preflight_is_local_exact_and_pinned(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    payload = b"model"
    (model / "model.safetensors").write_bytes(payload)
    spec = _backend(file_hash=_sha(payload), total_bytes=len(payload))
    first = preflight_local_model(spec, model)
    second = preflight_local_model(spec, model)
    assert first == second
    assert first.network_accessed is False
    assert first.observed_files == {"model.safetensors": _sha(payload)}
    (model / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory mismatch"):
        preflight_local_model(spec, model)
    (model / "unexpected.json").unlink()
    (model / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        preflight_local_model(spec, model)
    with pytest.raises(ValueError, match="absolute"):
        preflight_local_model(spec, Path("relative"))


def test_feature_cache_is_atomic_deterministic_and_tamper_evident(tmp_path: Path) -> None:
    output = tmp_path / "cache"
    kwargs = {
        "processor_sha256": H0,
        "device_environment_sha256": H1,
    }
    build_frozen_feature_cache(
        _dataset(), _backend(), DeterministicFixtureExtractor(16), output, **kwargs
    )
    first = verify_frozen_feature_cache(output)
    with pytest.raises(FileExistsError):
        build_frozen_feature_cache(
            _dataset(), _backend(), DeterministicFixtureExtractor(16), output, **kwargs
        )
    second_output = tmp_path / "cache-repeat"
    build_frozen_feature_cache(
        _dataset(), _backend(), DeterministicFixtureExtractor(16), second_output, **kwargs
    )
    assert first.sha256() == verify_frozen_feature_cache(second_output).sha256()
    feature = output / "features" / "00000000.npy"
    feature.write_bytes(feature.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="content hash"):
        verify_frozen_feature_cache(output)


def test_failed_overwrite_preserves_previous_generation(tmp_path: Path) -> None:
    output = tmp_path / "cache"
    spec = _backend()
    kwargs = {"processor_sha256": H0, "device_environment_sha256": H1}
    build_frozen_feature_cache(_dataset(), spec, DeterministicFixtureExtractor(), output, **kwargs)
    original = (output / "cache-index.json").read_bytes()

    class FailingExtractor:
        output_dim = 16

        def extract(self, image: np.ndarray, instruction: str) -> np.ndarray:
            raise RuntimeError("injected extraction failure")

    with pytest.raises(RuntimeError, match="injected"):
        build_frozen_feature_cache(
            _dataset(), spec, FailingExtractor(), output, overwrite=True, **kwargs
        )
    assert (output / "cache-index.json").read_bytes() == original
    verify_frozen_feature_cache(output)


def test_qwen_smoke_is_exactly_twelve_rows_and_claim_closed() -> None:
    result = run_qwen_observational_smoke(
        _dataset(), _backend("qwen3_vl_2b"), DeterministicFixtureExtractor()
    )
    assert result.rows == 12
    assert len(set(result.feature_hashes)) == 12
    assert result.observational is True
    assert result.claim_allowed is False
    payload = result.to_dict()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="integer 1"):
        type(result)(**payload)
    with pytest.raises(ValueError, match="observational backend"):
        run_qwen_observational_smoke(_dataset(), _backend(), DeterministicFixtureExtractor())


def test_cache_rejects_manifest_and_inventory_tampering(tmp_path: Path) -> None:
    output = tmp_path / "cache"
    build_frozen_feature_cache(
        _dataset(),
        _backend(),
        DeterministicFixtureExtractor(),
        output,
        processor_sha256=H0,
        device_environment_sha256=H1,
    )
    manifest = output / "manifests" / "00000000.json"
    value = json.loads(manifest.read_text())
    value["sample_id"] = "forged"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest hash"):
        verify_frozen_feature_cache(output)
    manifest.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        verify_frozen_feature_cache(output)
