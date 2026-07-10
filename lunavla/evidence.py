"""Small deterministic statistics used to gate v2 modality claims."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


Direction = Literal["positive", "negative"]


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binomial proportion."""

    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must satisfy 0 <= successes <= trials")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be positive and finite")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True)
class PairedInterval:
    """A treatment-minus-control paired bootstrap result."""

    metric: str
    paired_n: int
    mean_difference: float
    lower: float
    upper: float
    bootstrap_samples: int
    seed: int

    def supports(self, direction: Direction) -> bool:
        """Whether the interval excludes zero in the declared direction."""

        if direction == "positive":
            return self.lower > 0
        if direction == "negative":
            return self.upper < 0
        raise ValueError(f"unsupported direction: {direction!r}")

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0 or self.upper < 0


def paired_bootstrap_interval(
    control: Sequence[float],
    treatment: Sequence[float],
    *,
    metric: str,
    samples: int = 10_000,
    seed: int = 202_601,
) -> PairedInterval:
    """Bootstrap paired treatment-minus-control differences.

    Inputs must already be aligned by a stable pair identifier. This function
    deliberately does not guess how to join unrelated rows.
    """

    control_array = np.asarray(control, dtype=np.float64)
    treatment_array = np.asarray(treatment, dtype=np.float64)
    if control_array.ndim != 1 or treatment_array.ndim != 1:
        raise ValueError("control and treatment must be rank-1")
    if control_array.shape != treatment_array.shape or control_array.size == 0:
        raise ValueError("control and treatment must be non-empty and equally sized")
    if not np.all(np.isfinite(control_array)) or not np.all(np.isfinite(treatment_array)):
        raise ValueError("control and treatment must contain only finite values")
    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not str(metric).strip():
        raise ValueError("metric cannot be empty")

    differences = treatment_array - control_array
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, differences.size, size=(samples, differences.size))
    means = np.mean(differences[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return PairedInterval(
        metric=str(metric),
        paired_n=int(differences.size),
        mean_difference=float(np.mean(differences)),
        lower=float(lower),
        upper=float(upper),
        bootstrap_samples=samples,
        seed=int(seed),
    )
