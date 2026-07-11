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

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DatasetSourceV3",
    "EmbodimentSpec",
    "EpisodeRecordV3",
    "ExperimentConfig",
    "FeatureSchema",
    "FeatureSpec",
    "ObservationV3",
    "TaskEnvV3",
    "TransitionV3",
    "migrate_v2_mapping",
    "observation_from_v2",
    "observation_to_v2",
]
