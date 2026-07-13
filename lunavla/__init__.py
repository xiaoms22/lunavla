"""Frozen public contracts for LunaVLA v2.0."""

from dataset.vla_dataset import VLARecord
from model.policy_base import ActionChunk

from .ablation import PairedAblationResult, evaluate_action_error_pairs
from .config import ExperimentConfig
from .contracts import DatasetSource, Observation, PolicyBatch, TaskEnv, Transition, VLAPolicy
from .engine import (
    Engine,
    EngineConfig,
    EvaluationResult,
    ObservationIntervention,
    TrainingResult,
)
from .evidence import (
    EvidenceManifest,
    HierarchicalPairedInterval,
    PairedInterval,
    hierarchical_paired_bootstrap_interval,
    paired_bootstrap_interval,
    wilson_interval,
)
from .evidence_design import EvidenceDesign
from .manifest import RunManifest
from .registry import PolicyRegistry, default_policy_registry
from .temporal import TemporalEnsembler


__version__ = "3.0.0"

__all__ = [
    "ActionChunk",
    "DatasetSource",
    "Engine",
    "EngineConfig",
    "EvidenceDesign",
    "EvidenceManifest",
    "EvaluationResult",
    "ExperimentConfig",
    "HierarchicalPairedInterval",
    "Observation",
    "ObservationIntervention",
    "PairedAblationResult",
    "PairedInterval",
    "PolicyBatch",
    "PolicyRegistry",
    "RunManifest",
    "TaskEnv",
    "TemporalEnsembler",
    "TrainingResult",
    "Transition",
    "VLAPolicy",
    "VLARecord",
    "default_policy_registry",
    "evaluate_action_error_pairs",
    "hierarchical_paired_bootstrap_interval",
    "paired_bootstrap_interval",
    "wilson_interval",
]
