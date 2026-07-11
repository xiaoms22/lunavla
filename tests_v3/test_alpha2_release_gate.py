from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lunavla.v3 import (
    ALPHA2_PACKAGE_VERSION,
    ALPHA2_TAG,
    Alpha2ReleaseCandidateV1,
    GpuValidationManifestV1,
    LicenseReviewV1,
)
from lunavla.v3.artifacts import ArtifactHashRecordV1, sha256_file
from lunavla.v3.release_contracts import SMOLVLA_WEIGHT_SHA256
from scripts.run_v3_alpha2_release import validate_license


SHA = "1" * 64
GIT_SHA = "2" * 40


def _license_review(evidence_sha256: str = SHA) -> LicenseReviewV1:
    return LicenseReviewV1(
        repo_id="lerobot/smolvla_base",
        revision="d06fce6e38c25c04ac5a6319eefb9fae0e257cb2",
        spdx_license="Apache-2.0",
        evidence_url=(
            "https://huggingface.co/lerobot/smolvla_base/"
            "blob/d06fce6e38c25c04ac5a6319eefb9fae0e257cb2/LICENSE"
        ),
        evidence_sha256=evidence_sha256,
        scope="model_weights",
        reviewer="xiaoms22",
        reviewed_at="2026-07-12",
    )


def _gpu_manifest() -> GpuValidationManifestV1:
    return GpuValidationManifestV1(
        git_sha=GIT_SHA,
        package_version=ALPHA2_PACKAGE_VERSION,
        license_review_sha256=SHA,
        model_source_sha256=SHA,
        dependency_lock_sha256=SHA,
        dispatcher_sha256=SHA,
        runner_labels=("self-hosted", "linux", "x64", "gpu", "lunavla-v3"),
        runner_os="Linux",
        runner_arch="X64",
        gpu_count=1,
        gpu_name="NVIDIA fixture",
        gpu_uuid_sha256=SHA,
        driver_version="fixture",
        cuda_runtime="12.8",
        torch_version="2.11.0+cu128",
        torchvision_version="0.26.0+cu128",
        downloaded_files=(
            ArtifactHashRecordV1("model.safetensors", SMOLVLA_WEIGHT_SHA256),
        ),
        model_bytes=907_000_000,
        train_seed=11,
        loss_before=1.0,
        loss_after=0.5,
        gradient_norm=1.5,
        checkpoint_sha256=SHA,
        restored_action_sha256=SHA,
        resume_rtol=1e-5,
        resume_atol=1e-6,
        optimizer_step_verified=True,
        resume_verified=True,
        inference_verified=True,
    )


def _candidate_assets() -> tuple[ArtifactHashRecordV1, ...]:
    return tuple(
        ArtifactHashRecordV1(path, SHA)
        for path in (
            "dist/lunavla-3.0.0a2-py3-none-any.whl",
            "dist/lunavla-3.0.0a2.tar.gz",
            "environment-requirements.txt",
            "gpu-validation-manifest.json",
            "gpu-attestation-bundle.jsonl",
            "lunavla-v3-alpha2-evidence.tar.gz",
            "sbom.json",
        )
    )


def test_license_review_round_trip_and_official_weight_scope() -> None:
    review = _license_review()
    assert LicenseReviewV1.from_mapping(review.to_dict()) == review
    assert len(review.sha256()) == 64

    payload = review.to_dict()
    payload["evidence_url"] = "https://github.com/huggingface/lerobot/blob/main/LICENSE"
    with pytest.raises(ValueError, match="Hugging Face"):
        LicenseReviewV1.from_mapping(payload)
    payload = review.to_dict()
    payload["scope"] = "source_code"
    with pytest.raises(ValueError, match="model_weights"):
        LicenseReviewV1.from_mapping(payload)


def test_release_contracts_reject_boolean_versions_unknown_fields_and_nonfinite() -> None:
    payload = _license_review().to_dict()
    payload["schema_version"] = True
    with pytest.raises(TypeError, match="integer"):
        LicenseReviewV1.from_mapping(payload)
    payload = _license_review().to_dict()
    payload["unknown"] = "value"
    with pytest.raises(ValueError, match="unknown field"):
        LicenseReviewV1.from_mapping(payload)

    gpu = _gpu_manifest().to_dict()
    gpu["loss_after"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        GpuValidationManifestV1.from_mapping(gpu)
    gpu = _gpu_manifest().to_dict()
    gpu["gpu_count"] = 2
    with pytest.raises(ValueError, match="exactly one GPU"):
        GpuValidationManifestV1.from_mapping(gpu)


def test_gpu_manifest_and_release_candidate_round_trip_fail_closed() -> None:
    gpu = _gpu_manifest()
    assert GpuValidationManifestV1.from_mapping(gpu.to_dict()) == gpu
    assert gpu.claim_allowed is False

    candidate = Alpha2ReleaseCandidateV1(
        expected_tag=ALPHA2_TAG,
        git_sha=GIT_SHA,
        package_version=ALPHA2_PACKAGE_VERSION,
        gpu_manifest_sha256=SHA,
        gpu_attestation_sha256=SHA,
        required_checks_sha256=SHA,
        dispatcher_sha256=SHA,
        assets=_candidate_assets(),
    )
    assert Alpha2ReleaseCandidateV1.from_mapping(candidate.to_dict()) == candidate
    assert candidate.claim_allowed is False
    assert candidate.pypi_published is False

    payload = candidate.to_dict()
    payload["expected_tag"] = "v3.0.0-alpha.1"
    with pytest.raises(ValueError, match="expected_tag"):
        Alpha2ReleaseCandidateV1.from_mapping(payload)


def test_current_license_and_pretrained_gate_fails_before_weight_access(tmp_path: Path) -> None:
    evidence = tmp_path / "LICENSE"
    evidence.write_bytes(b"fixture license text\n")
    review = _license_review(hashlib.sha256(evidence.read_bytes()).hexdigest())
    review_path = review.save(tmp_path / "license-review.json")

    with pytest.raises(RuntimeError, match="remains unverified"):
        validate_license(
            review_path,
            expected_sha256=sha256_file(review_path),
            evidence_path=evidence,
            enable_pretrained_gate=True,
        )

    payload = json.loads(review_path.read_text(encoding="utf-8"))
    payload["evidence_sha256"] = "0" * 64
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence bytes"):
        validate_license(
            review_path,
            expected_sha256=sha256_file(review_path),
            evidence_path=evidence,
            enable_pretrained_gate=True,
        )
