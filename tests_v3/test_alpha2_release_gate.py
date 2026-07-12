from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lunavla.v3 import (
    ALPHA3_PACKAGE_VERSION,
    ALPHA3_TAG,
    Alpha3ReleaseCandidateV1,
    SMOLVLA_VALIDATION_PACKAGE_VERSION,
    SMOLVLA_VALIDATION_TAG,
    SmolVLAValidationCandidateV1,
    GpuValidationManifestV1,
    LicenseReviewV1,
    RunnerQualificationManifestV1,
    WeightLicenseStatusV1,
)
from lunavla.v3.artifacts import ArtifactHashRecordV1, sha256_file
from lunavla.v3.release_contracts import SMOLVLA_WEIGHT_SHA256
from scripts.run_v3_alpha2_release import validate_license


SHA = "1" * 64
GIT_SHA = "2" * 40


def _license_status() -> WeightLicenseStatusV1:
    return WeightLicenseStatusV1(
        repo_id="lerobot/smolvla_base",
        revision="d06fce6e38c25c04ac5a6319eefb9fae0e257cb2",
        weight_sha256="7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb",
        license_status="unverified",
        spdx_license="NOASSERTION",
        pretrained_enabled=False,
        conformance_only=True,
        owner_authorization_scope="local_evaluation_only",
        owner_authorization_is_license_evidence=False,
        release_eligible=False,
        source_snapshots=tuple(
            ArtifactHashRecordV1(path, SHA)
            for path in (
                "docs/v3/release/smolvla-file-inventory-observation.json",
                "docs/v3/release/smolvla-model-card-observation.json",
                "docs/v3/release/smolvla-repository-metadata-observation.json",
            )
        ),
        reviewer="xiaoms22",
        checked_at="2026-07-12",
    )


def _runner_manifest(role: str = "authoritative") -> RunnerQualificationManifestV1:
    return RunnerQualificationManifestV1(
        role=role,
        git_sha=GIT_SHA,
        dependency_lock_sha256=SHA,
        container_image_sha256=SHA,
        runner_name_sha256=SHA,
        runner_labels=("self-hosted", "linux", "x64", "gpu", "lunavla-v3"),
        runner_os="Linux",
        runner_os_version="Ubuntu 22.04.5 LTS",
        runner_arch="X64",
        python_version="3.12.10",
        cpu_count=16,
        memory_bytes=64 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        gpu_count=1,
        cuda_visible_device_count=1,
        gpu_name="NVIDIA A100-SXM4-80GB",
        gpu_uuid_sha256=SHA,
        driver_version="570.124.06",
        cuda_runtime="12.8",
        torch_version="2.11.0+cu128",
        torchvision_version="0.26.0+cu128",
        network_hosts=(
            "api.github.com",
            "download.pytorch.org",
            "github.com",
            "huggingface.co",
            "pypi.org",
        ),
        workspace_clean=True,
        container_isolated=True,
        private_mounts_detected=False,
        ephemeral_declared=True,
        weight_accessed=False,
        release_eligible=False,
        claim_allowed=False,
        checked_at="2026-07-12",
    )


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
        package_version=SMOLVLA_VALIDATION_PACKAGE_VERSION,
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
            "dist/lunavla-3.1.0a1-py3-none-any.whl",
            "dist/lunavla-3.1.0a1.tar.gz",
            "environment-requirements.txt",
            "gpu-validation-manifest.json",
            "gpu-attestation-bundle.jsonl",
            "lunavla-v3-alpha2-evidence.tar.gz",
            "sbom.json",
        )
    )


def _code_candidate_assets() -> tuple[ArtifactHashRecordV1, ...]:
    return tuple(
        ArtifactHashRecordV1(path, SHA)
        for path in (
            "dist/lunavla-3.0.0a3-py3-none-any.whl",
            "dist/lunavla-3.0.0a3.tar.gz",
            "environment-requirements.txt",
            "test-manifest.json",
            "smolvla-conformance-status.json",
            "sbom.json",
            "provenance.jsonl",
            "lunavla-v3-alpha3-code-evidence.tar.gz",
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

    candidate = SmolVLAValidationCandidateV1(
        expected_tag=SMOLVLA_VALIDATION_TAG,
        git_sha=GIT_SHA,
        package_version=SMOLVLA_VALIDATION_PACKAGE_VERSION,
        gpu_manifest_sha256=SHA,
        gpu_attestation_sha256=SHA,
        required_checks_sha256=SHA,
        dispatcher_sha256=SHA,
        assets=_candidate_assets(),
    )
    assert SmolVLAValidationCandidateV1.from_mapping(candidate.to_dict()) == candidate
    assert candidate.claim_allowed is False
    assert candidate.pypi_published is False

    payload = candidate.to_dict()
    payload["expected_tag"] = "v3.0.0-alpha.1"
    with pytest.raises(ValueError, match="expected_tag"):
        SmolVLAValidationCandidateV1.from_mapping(payload)


def test_code_only_alpha3_candidate_requires_conformance_and_no_claims() -> None:
    candidate = Alpha3ReleaseCandidateV1(
        expected_tag=ALPHA3_TAG,
        git_sha=GIT_SHA,
        package_version=ALPHA3_PACKAGE_VERSION,
        public_api_sha256=SHA,
        core_lock_sha256=SHA,
        diffusion_lock_sha256=SHA,
        smolvla_lock_sha256=SHA,
        weight_license_status_sha256=SHA,
        required_checks_sha256=SHA,
        dispatcher_sha256=SHA,
        assets=_code_candidate_assets(),
    )
    assert Alpha3ReleaseCandidateV1.from_mapping(candidate.to_dict()) == candidate
    assert candidate.pretrained_enabled is False
    assert candidate.conformance_only is True
    payload = candidate.to_dict()
    payload["pretrained_enabled"] = True
    with pytest.raises(ValueError, match="pretrained disabled"):
        Alpha3ReleaseCandidateV1.from_mapping(payload)
    payload = candidate.to_dict()
    payload["claim_allowed"] = True
    with pytest.raises(ValueError, match="scientific claims"):
        Alpha3ReleaseCandidateV1.from_mapping(payload)


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


def test_unverified_weight_status_round_trip_is_fail_closed() -> None:
    status = _license_status()
    assert WeightLicenseStatusV1.from_mapping(status.to_dict()) == status
    assert status.spdx_license == "NOASSERTION"
    assert status.pretrained_enabled is False
    assert status.release_eligible is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("license_status", "verified"),
        ("spdx_license", "Apache-2.0"),
        ("pretrained_enabled", True),
        ("conformance_only", False),
        ("owner_authorization_is_license_evidence", True),
        ("release_eligible", True),
    ],
)
def test_unverified_weight_status_cannot_open_gate(field: str, value: object) -> None:
    payload = _license_status().to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        WeightLicenseStatusV1.from_mapping(payload)


def test_runner_qualification_round_trip_for_both_roles() -> None:
    for role in ("authoritative", "secondary"):
        manifest = _runner_manifest(role)
        assert RunnerQualificationManifestV1.from_mapping(manifest.to_dict()) == manifest
        assert manifest.release_eligible is False
        assert manifest.claim_allowed is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_count", 8),
        ("cuda_visible_device_count", 2),
        ("gpu_name", "NVIDIA H100 80GB HBM3"),
        ("driver_version", "565.57.01"),
        ("memory_bytes", 8 * 1024**3),
        ("disk_free_bytes", 20 * 1024**3),
        ("container_isolated", False),
        ("private_mounts_detected", True),
        ("ephemeral_declared", False),
        ("weight_accessed", True),
        ("release_eligible", True),
    ],
)
def test_runner_qualification_rejects_unsafe_or_ineligible_state(
    field: str, value: object
) -> None:
    payload = _runner_manifest().to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        RunnerQualificationManifestV1.from_mapping(payload)
