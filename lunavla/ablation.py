"""Paired modality-ablation evaluation through the public policy contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt

from lunavla.contracts import Observation, VLAPolicy
from lunavla.evidence import PairedInterval, paired_bootstrap_interval


class AblationExample(Protocol):
    observation: Observation
    action: npt.NDArray[np.float32]


class AblationPair(Protocol):
    pair_id: str
    mode: str
    control: AblationExample
    ablated: AblationExample


@dataclass(frozen=True)
class PairedAblationResult:
    ablation_mode: str
    pair_ids: tuple[str, ...]
    control_errors: tuple[float, ...]
    ablated_errors: tuple[float, ...]
    interval: PairedInterval

    @property
    def directional_interval_excludes_zero(self) -> bool:
        """Whether this single contrast has a positive ablated-minus-control interval."""

        return self.interval.supports("positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "ablation_mode": self.ablation_mode,
            "pair_ids": list(self.pair_ids),
            "control_errors": list(self.control_errors),
            "ablated_errors": list(self.ablated_errors),
            "interval": asdict(self.interval),
            "directional_interval_excludes_zero": self.directional_interval_excludes_zero,
            "claim_allowed": False,
            "claim_gate": "requires a predeclared multi-seed paired analysis",
        }


def _first_action(policy: VLAPolicy, observation: Observation) -> npt.NDArray[np.float32]:
    chunk = policy.predict_chunk(observation)
    valid = np.flatnonzero(chunk.valid_mask)
    if valid.size == 0:
        raise ValueError("policy returned an ActionChunk without a valid action")
    return np.asarray(chunk.values[int(valid[0])], dtype=np.float32)


def evaluate_action_error_pairs(
    policy: VLAPolicy,
    pairs: Sequence[AblationPair],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 202_601,
) -> PairedAblationResult:
    """Measure ablated-minus-control first-action MSE on pre-paired examples."""

    source = tuple(pairs)
    if not source:
        raise ValueError("at least one ablation pair is required")
    pair_ids = tuple(str(pair.pair_id) for pair in source)
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("ablation pair_id values must be unique")
    modes = {str(pair.mode) for pair in source}
    if len(modes) != 1:
        raise ValueError("all pairs in one contrast must use the same ablation mode")

    control_errors: list[float] = []
    ablated_errors: list[float] = []
    for pair in source:
        control_target = np.asarray(pair.control.action, dtype=np.float32)
        ablated_target = np.asarray(pair.ablated.action, dtype=np.float32)
        if control_target.shape != (policy.action_dim,):
            raise ValueError(
                f"pair {pair.pair_id} target must have shape {(policy.action_dim,)}"
            )
        if not np.array_equal(control_target, ablated_target):
            raise ValueError(f"pair {pair.pair_id} changes the action target")
        if not np.array_equal(
            pair.control.observation.state,
            pair.ablated.observation.state,
        ):
            raise ValueError(f"pair {pair.pair_id} changes the state")
        control_prediction = _first_action(policy, pair.control.observation)
        ablated_prediction = _first_action(policy, pair.ablated.observation)
        control_errors.append(float(np.mean((control_prediction - control_target) ** 2)))
        ablated_errors.append(float(np.mean((ablated_prediction - control_target) ** 2)))

    interval = paired_bootstrap_interval(
        control_errors,
        ablated_errors,
        metric="first_action_mse",
        samples=bootstrap_samples,
        seed=seed,
    )
    return PairedAblationResult(
        ablation_mode=modes.pop(),
        pair_ids=pair_ids,
        control_errors=tuple(control_errors),
        ablated_errors=tuple(ablated_errors),
        interval=interval,
    )
