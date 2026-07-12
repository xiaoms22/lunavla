from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from lunavla.v3.artifacts import sha256_file
from lunavla.v3 import (
    DiagnosticDesignV1,
    DiagnosticTraceRowV1,
    DonorBankV1,
    DonorRecordV1,
    EmbodimentSpec,
    EpisodeRecordV3,
    EvidenceManifestV2,
    ExternalDatasetSpecV1,
    ExperimentConfig,
    FailureRecordV1,
    FeatureNormalizationV1,
    FeatureSchema,
    FeatureSpec,
    ModelSourceContractV1,
    Alpha2ReleaseCandidateV1,
    GpuValidationManifestV1,
    IntegrationManifestV1,
    LicenseReviewV1,
    RunnerQualificationManifestV1,
    WeightLicenseStatusV1,
    NormalizationStatsV1,
    ObservationV3,
    PolicyBatchV3,
    PolicySampleV3,
    PolicySpecV3,
    PromptParityManifestV1,
    PromptParityRecordV1,
    PromptSpecV1,
    StateRouteSpecV1,
    SimulationTaskSpecV1,
    InterventionSpecV1,
    TrainStepResultV3,
    TransitionV3,
    StableEvidenceDesignV1,
    StableEvidenceSummaryV1,
    StableReleaseCandidateV1,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/v3/public_api_contract.json"
LOCK_ALIAS = ROOT / "requirements-v3-core-cpu.lock"
DIFFUSION_LOCK = ROOT / "requirements-v3-diffusion-cpu.lock"
SMOLVLA_LOCK = ROOT / "requirements-v3-smolvla-cpu.lock"
SMOLVLA_GPU_LOCK = ROOT / "requirements-v3-smolvla-gpu-cu128.lock"
RELEASE_LOCK = ROOT / "requirements-v3-release-cpu.lock"
RELEASE_DISPATCHER = ROOT / ".github/workflows/v3-alpha2-release-dispatch.yml"
BETA2_LOCK = ROOT / "requirements-v3-beta2-integration-cu128.lock"
BETA2_DISPATCHER = ROOT / ".github/workflows/v3-beta2-integration-dispatch.yml"
LICENSE_STATUS = ROOT / "docs/v3/release/smolvla-license-status.json"
PUBLIC_TYPES = {
    "FeatureSpec": FeatureSpec,
    "FeatureSchema": FeatureSchema,
    "EmbodimentSpec": EmbodimentSpec,
    "ObservationV3": ObservationV3,
    "TransitionV3": TransitionV3,
    "EpisodeRecordV3": EpisodeRecordV3,
    "ExperimentConfig": ExperimentConfig,
    "ModelSourceContractV1": ModelSourceContractV1,
    "LicenseReviewV1": LicenseReviewV1,
    "WeightLicenseStatusV1": WeightLicenseStatusV1,
    "RunnerQualificationManifestV1": RunnerQualificationManifestV1,
    "GpuValidationManifestV1": GpuValidationManifestV1,
    "Alpha2ReleaseCandidateV1": Alpha2ReleaseCandidateV1,
    "PolicySpecV3": PolicySpecV3,
    "PolicySampleV3": PolicySampleV3,
    "PolicyBatchV3": PolicyBatchV3,
    "TrainStepResultV3": TrainStepResultV3,
    "FeatureNormalizationV1": FeatureNormalizationV1,
    "NormalizationStatsV1": NormalizationStatsV1,
    "PromptSpecV1": PromptSpecV1,
    "StateRouteSpecV1": StateRouteSpecV1,
    "InterventionSpecV1": InterventionSpecV1,
    "DiagnosticDesignV1": DiagnosticDesignV1,
    "DonorRecordV1": DonorRecordV1,
    "DonorBankV1": DonorBankV1,
    "PromptParityRecordV1": PromptParityRecordV1,
    "PromptParityManifestV1": PromptParityManifestV1,
    "DiagnosticTraceRowV1": DiagnosticTraceRowV1,
    "FailureRecordV1": FailureRecordV1,
    "EvidenceManifestV2": EvidenceManifestV2,
    "ExternalDatasetSpecV1": ExternalDatasetSpecV1,
    "SimulationTaskSpecV1": SimulationTaskSpecV1,
    "IntegrationManifestV1": IntegrationManifestV1,
    "StableEvidenceDesignV1": StableEvidenceDesignV1,
    "StableEvidenceSummaryV1": StableEvidenceSummaryV1,
    "StableReleaseCandidateV1": StableReleaseCandidateV1,
}


def descriptor() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "release_stage": "v3.0.0-rc-preparation-stacked-draft",
        "contracts": {
            name: {"signature": str(inspect.signature(value))}
            for name, value in PUBLIC_TYPES.items()
        },
    }


def main() -> int:
    expected = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if descriptor() != expected:
        raise SystemExit("v3 public API descriptor drifted")
    for path in sorted((ROOT / "configs/v3").glob("*.yaml")):
        if path.name.startswith("stable_") and path.name.endswith("_design.yaml"):
            design = StableEvidenceDesignV1.load(path)
            design.validate_stable_matrix()
            continue
        if path.name.endswith("_design.yaml"):
            DiagnosticDesignV1.from_mapping(
                __import__("yaml").safe_load(path.read_text(encoding="utf-8"))
            )
            continue
        ExperimentConfig.load(path)
    status = WeightLicenseStatusV1.from_mapping(
        json.loads(LICENSE_STATUS.read_text(encoding="utf-8"))
    )
    for snapshot in status.source_snapshots:
        path = ROOT / snapshot.path
        if not path.is_file():
            raise SystemExit(f"SmolVLA license source snapshot is missing: {snapshot.path}")
        if sha256_file(path) != snapshot.sha256:
            raise SystemExit(f"SmolVLA license source snapshot hash drifted: {snapshot.path}")
    if LOCK_ALIAS.read_text(encoding="utf-8").splitlines()[-1] != "-r requirements-v2-core-cpu.lock":
        raise SystemExit("v3 CPU lock alias drifted")
    diffusion_lock = DIFFUSION_LOCK.read_text(encoding="utf-8").lower()
    required = {
        "accelerate==1.14.0",
        "diffusers==0.35.2",
        "lerobot==0.6.0",
        "numpy==2.2.6",
        "torch==2.11.0+cpu",
        "torchvision==0.26.0+cpu",
        "transformers==5.5.4",
        "sha256:b38a564fbc441d98380576863bf68635dde5fc2c42ddc2a39d0486640dc9e9a8",
    }
    missing = sorted(item for item in required if item not in diffusion_lock)
    if missing:
        raise SystemExit(f"v3 Diffusion CPU lock is stale; missing {missing}")
    forbidden = ("nvidia-", "nvidia_", "nccl==", "triton==")
    if any(item in diffusion_lock for item in forbidden):
        raise SystemExit("v3 Diffusion CPU lock contains an accelerator-only package")
    smolvla_lock = SMOLVLA_LOCK.read_text(encoding="utf-8").lower()
    smolvla_required = {
        "accelerate==1.14.0",
        "lerobot==0.6.0",
        "numpy==2.2.6",
        "torch==2.11.0+cpu",
        "torchvision==0.26.0+cpu",
        "transformers==5.5.4",
        "sha256:b38a564fbc441d98380576863bf68635dde5fc2c42ddc2a39d0486640dc9e9a8",
    }
    missing = sorted(item for item in smolvla_required if item not in smolvla_lock)
    if missing:
        raise SystemExit(f"v3 SmolVLA CPU lock is stale; missing {missing}")
    if any(item in smolvla_lock for item in forbidden):
        raise SystemExit("v3 SmolVLA CPU lock contains an accelerator-only package")
    gpu_lock = SMOLVLA_GPU_LOCK.read_text(encoding="utf-8").lower()
    gpu_required = {
        "accelerate==1.14.0",
        "lerobot==0.6.0",
        "torch==2.11.0+cu128",
        "torchvision==0.26.0+cu128",
        "transformers==5.5.4",
        "nvidia-cuda-runtime-cu12==12.8.90",
        "triton==3.6.0",
    }
    missing = sorted(item for item in gpu_required if item not in gpu_lock)
    if missing:
        raise SystemExit(f"v3 SmolVLA GPU lock is stale; missing {missing}")
    release_lock = RELEASE_LOCK.read_text(encoding="utf-8").lower()
    release_required = {
        "build==1.5.1",
        "cyclonedx-bom==7",
        "torch==2.11.0+cpu",
        "torchvision==0.26.0+cpu",
        "twine==6.2.0",
    }
    missing = sorted(item for item in release_required if item not in release_lock)
    if missing:
        raise SystemExit(f"v3 release CPU lock is stale; missing {missing}")
    if any(item in release_lock for item in forbidden):
        raise SystemExit("v3 release CPU lock contains an accelerator-only package")
    if "workflow_dispatch:" not in RELEASE_DISPATCHER.read_text(encoding="utf-8"):
        raise SystemExit("v3 Alpha 2 release dispatcher is missing its manual entrypoint")
    beta2_lock = BETA2_LOCK.read_text(encoding="utf-8").lower()
    beta2_required = {
        "accelerate==1.14.0",
        "av==15.1.0",
        "diffusers==0.35.2",
        "gym-pusht==0.1.6",
        "hf-libero==0.1.4",
        "lerobot==0.6.0",
        "numpy==2.2.6",
        "torch==2.11.0+cu128",
        "torchvision==0.26.0+cu128",
        "transformers==5.5.4",
    }
    missing = sorted(item for item in beta2_required if item not in beta2_lock)
    if missing:
        raise SystemExit(f"v3 Beta 2 integration lock is stale; missing {missing}")
    if "workflow_dispatch:" not in BETA2_DISPATCHER.read_text(encoding="utf-8"):
        raise SystemExit("v3 Beta 2 integration dispatcher is missing its manual entrypoint")
    print("v3 alpha contracts, configs, and CPU lock are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
