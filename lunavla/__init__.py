"""Public contracts for the experimental LunaVLA v2 integration branch."""

from dataset.vla_dataset import VLARecord
from model.policy_base import ActionChunk

from .ablation import PairedAblationResult, evaluate_action_error_pairs
from .config import ExperimentConfig
from .contracts import DatasetSource, Observation, PolicyBatch, TaskEnv, Transition, VLAPolicy
from .engine import Engine, EngineConfig, EvaluationResult, TrainingResult
from .evidence import PairedInterval, paired_bootstrap_interval, wilson_interval
from .manifest import RunManifest
from .registry import PolicyRegistry, default_policy_registry
from .temporal import TemporalEnsembler


__version__ = "2.0.0a0"

__all__ = [
    "ActionChunk",
    "DatasetSource",
    "Engine",
    "EngineConfig",
    "EvaluationResult",
    "ExperimentConfig",
    "Observation",
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
    "paired_bootstrap_interval",
    "wilson_interval",
]
