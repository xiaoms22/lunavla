from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from lunavla.v3 import (
    EmbodimentSpec,
    EpisodeRecordV3,
    ExperimentConfig,
    FeatureNormalizationV1,
    FeatureSchema,
    FeatureSpec,
    ModelSourceContractV1,
    NormalizationStatsV1,
    ObservationV3,
    PolicyBatchV3,
    PolicySampleV3,
    PolicySpecV3,
    TrainStepResultV3,
    TransitionV3,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/v3/public_api_contract.json"
LOCK_ALIAS = ROOT / "requirements-v3-core-cpu.lock"
DIFFUSION_LOCK = ROOT / "requirements-v3-diffusion-cpu.lock"
PUBLIC_TYPES = {
    "FeatureSpec": FeatureSpec,
    "FeatureSchema": FeatureSchema,
    "EmbodimentSpec": EmbodimentSpec,
    "ObservationV3": ObservationV3,
    "TransitionV3": TransitionV3,
    "EpisodeRecordV3": EpisodeRecordV3,
    "ExperimentConfig": ExperimentConfig,
    "ModelSourceContractV1": ModelSourceContractV1,
    "PolicySpecV3": PolicySpecV3,
    "PolicySampleV3": PolicySampleV3,
    "PolicyBatchV3": PolicyBatchV3,
    "TrainStepResultV3": TrainStepResultV3,
    "FeatureNormalizationV1": FeatureNormalizationV1,
    "NormalizationStatsV1": NormalizationStatsV1,
}


def descriptor() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "release_stage": "v3.0.0-alpha.2-diffusion",
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
        ExperimentConfig.load(path)
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
    print("v3 alpha contracts, configs, and CPU lock are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
