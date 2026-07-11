"""LunaVLA v3 alpha contracts.

The v3 namespace is deliberately separate from the frozen top-level v2 API.
"""

from .config import CONFIG_SCHEMA_VERSION, ExperimentConfig
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
from .artifacts import CheckpointEnvelopeV4, RunManifestV4, verify_run_directory
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
    "CheckpointEnvelopeV4",
    "DataAuditManifest",
    "DatasetBundle",
    "DatasetSourceV3",
    "EmbodimentSpec",
    "EngineV3",
    "EpisodeRecordV3",
    "EpisodeHashRecord",
    "ExperimentConfig",
    "FeatureSchema",
    "FeatureSpec",
    "InMemoryDatasetSourceV3",
    "ObservationV3",
    "RunManifestV4",
    "TaskEnvV3",
    "TransitionV3",
    "audit_episodes",
    "migrate_v2_mapping",
    "observation_from_v2",
    "observation_to_v2",
    "run_alpha",
    "split_episode_ids",
    "verify_run_directory",
]
