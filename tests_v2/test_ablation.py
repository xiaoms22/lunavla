from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from lunavla.ablation import evaluate_action_error_pairs
from lunavla.contracts import Observation, PolicyBatch
from lunavla.language_tasks import build_language_examples, make_instruction_ablation_pairs
from model.policy_base import ActionChunk


class _MemorizedInstructionPolicy:
    policy_id = "memorized_instruction"
    device = "cpu"
    action_dim = 2
    chunk_size = 1

    def __init__(self, examples: tuple[Any, ...]) -> None:
        self.actions = {
            str(example.observation.instruction): np.asarray(example.action, dtype=np.float32)
            for example in examples
        }

    def train_batch(self, batch: PolicyBatch, *, learning_rate: float) -> float:
        return 0.0

    def predict_chunk(self, observation: Observation) -> ActionChunk:
        action = self.actions.get(
            str(observation.instruction),
            np.zeros(self.action_dim, dtype=np.float32),
        )
        return ActionChunk(action[None, :], np.asarray([True]))

    def save_checkpoint(
        self,
        path: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return path


def test_paired_language_mask_produces_a_claim_gated_interval() -> None:
    examples = build_language_examples("heldout")
    pairs = make_instruction_ablation_pairs(examples, "mask", seed=4)
    result = evaluate_action_error_pairs(
        _MemorizedInstructionPolicy(examples),
        pairs,
        bootstrap_samples=2_000,
        seed=8,
    )
    assert result.ablation_mode == "mask"
    assert result.interval.paired_n == len(pairs)
    assert result.interval.lower > 0
    assert result.directional_interval_excludes_zero
    assert result.to_dict()["directional_interval_excludes_zero"] is True
    assert result.to_dict()["claim_allowed"] is False


def test_ablation_evaluator_rejects_duplicate_pair_ids() -> None:
    examples = build_language_examples("heldout")
    pair = make_instruction_ablation_pairs(examples, "mask", seed=4)[0]
    with pytest.raises(ValueError, match="pair_id"):
        evaluate_action_error_pairs(
            _MemorizedInstructionPolicy(examples),
            [pair, pair],
            bootstrap_samples=10,
        )
