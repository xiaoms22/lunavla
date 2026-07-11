from __future__ import annotations

import pytest

from lunavla.evidence import paired_bootstrap_interval, wilson_interval


def test_wilson_interval_is_finite_and_contains_observed_rate() -> None:
    lower, upper = wilson_interval(8, 10)
    assert 0 < lower < 0.8 < upper < 1
    assert wilson_interval(0, 10)[0] == 0


def test_paired_bootstrap_is_deterministic_and_direction_gated() -> None:
    control = [0.1, 0.2, 0.3, 0.4, 0.5]
    ablated = [0.3, 0.4, 0.5, 0.6, 0.7]
    first = paired_bootstrap_interval(
        control, ablated, metric="action_error", samples=2_000, seed=7
    )
    second = paired_bootstrap_interval(
        control, ablated, metric="action_error", samples=2_000, seed=7
    )
    assert first == second
    assert first.supports("positive")
    assert first.excludes_zero
    assert not first.supports("negative")


@pytest.mark.parametrize(
    ("control", "treatment", "message"),
    [
        ([], [], "non-empty"),
        ([1.0], [1.0, 2.0], "equally sized"),
        ([1.0], [float("nan")], "finite"),
    ],
)
def test_paired_bootstrap_rejects_invalid_pairs(
    control: list[float], treatment: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap_interval(control, treatment, metric="error", samples=10)
