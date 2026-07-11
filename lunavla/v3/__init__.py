"""LunaVLA v3 alpha contracts.

The v3 namespace is deliberately separate from the frozen top-level v2 API.
"""

from .config import CONFIG_CONTRACT_REVISION, CONFIG_SCHEMA_VERSION, ExperimentConfig
from .diagnostics import (
    DiagnosticDesignV1,
    DonorBankV1,
    DonorRecordV1,
    FailureRecordV1,
    InterventionSpecV1,
    PromptSpecV1,
    StateRouteSpecV1,
)
from .diagnostic_engine import DiagnosticRouterV1, RoutedObservationV1, typed_episode_key
from .diagnostic_workflow import (
    EvidenceManifestV2,
    run_diagnostic,
    verify_diagnostic_output,
    write_diagnostic_report,
)
from .contracts import (
    DatasetSourceV3,
    EmbodimentSpec,
    EpisodeRecordV3,
    FeatureSchema,
    FeatureSpec,
    ObservationV3,
    TaskEnvV3,
    TransitionV3,
)
from .migration import migrate_v2_mapping, observation_from_v2, observation_to_v2
from .normalization import (
    FeatureNormalizationV1,
    NormalizationStatsV1,
    fit_normalization_stats,
)
from .policy import (
    ModelSourceContractV1,
    PolicyBatchV3,
    PolicySampleV3,
    PolicySpecV3,
    TrainStepResultV3,
    VLAPolicyV3,
)
from .registry import PolicyRegistryV3
from .artifacts import (
    ArtifactHashRecordV1,
    CheckpointEnvelopeV4,
    CheckpointEnvelopeV4R2,
    RunManifestV4,
    RunManifestV4R2,
    RunManifestV4R3,
    checkpoint_envelope_from_mapping,
    run_manifest_from_mapping,
    verify_checkpoint_directory,
    verify_run_directory,
)
from .data import (
    DataAuditManifest,
    DatasetBundle,
    EpisodeHashRecord,
    InMemoryDatasetSourceV3,
    audit_episodes,
    split_episode_ids,
)
from .engine import EngineV3, run_alpha

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "CONFIG_CONTRACT_REVISION",
    "ArtifactHashRecordV1",
    "CheckpointEnvelopeV4",
    "CheckpointEnvelopeV4R2",
    "DataAuditManifest",
    "DatasetBundle",
    "DatasetSourceV3",
    "DiagnosticDesignV1",
    "DiagnosticRouterV1",
    "DonorBankV1",
    "DonorRecordV1",
    "EmbodimentSpec",
    "EvidenceManifestV2",
    "EngineV3",
    "EpisodeRecordV3",
    "EpisodeHashRecord",
    "ExperimentConfig",
    "FeatureSchema",
    "FailureRecordV1",
    "FeatureSpec",
    "FeatureNormalizationV1",
    "InMemoryDatasetSourceV3",
    "InterventionSpecV1",
    "ObservationV3",
    "ModelSourceContractV1",
    "NormalizationStatsV1",
    "PolicyBatchV3",
    "PolicyRegistryV3",
    "PolicySampleV3",
    "PolicySpecV3",
    "PromptSpecV1",
    "RunManifestV4",
    "RunManifestV4R2",
    "RunManifestV4R3",
    "RoutedObservationV1",
    "StateRouteSpecV1",
    "TaskEnvV3",
    "TransitionV3",
    "TrainStepResultV3",
    "VLAPolicyV3",
    "audit_episodes",
    "checkpoint_envelope_from_mapping",
    "fit_normalization_stats",
    "migrate_v2_mapping",
    "observation_from_v2",
    "observation_to_v2",
    "run_alpha",
    "run_diagnostic",
    "run_manifest_from_mapping",
    "split_episode_ids",
    "verify_run_directory",
    "typed_episode_key",
    "verify_checkpoint_directory",
    "verify_diagnostic_output",
    "write_diagnostic_report",
]
