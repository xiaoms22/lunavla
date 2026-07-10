from __future__ import annotations

import numpy as np
import pytest

from model.losses import masked_mse, masked_mse_gradient


def test_masked_mse_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(17)
    predictions = rng.normal(size=(2, 3, 2)).astype(np.float32)
    targets = rng.normal(size=(2, 3, 2)).astype(np.float32)
    valid_mask = np.array([[True, True, False], [True, False, False]])
    analytic = masked_mse_gradient(predictions, targets, valid_mask)
    numeric = np.zeros_like(analytic)
    epsilon = 1e-3

    for index in np.ndindex(predictions.shape):
        plus = predictions.copy()
        minus = predictions.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric[index] = (
            masked_mse(plus, targets, valid_mask)
            - masked_mse(minus, targets, valid_mask)
        ) / (2.0 * epsilon)

    relative_error = np.linalg.norm(analytic - numeric) / max(
        np.linalg.norm(analytic) + np.linalg.norm(numeric), 1e-12
    )
    assert relative_error < 1e-3


def test_padding_values_do_not_change_masked_loss_or_gradient() -> None:
    predictions = np.array([[[1.0, 2.0], [4.0, 8.0]]], dtype=np.float32)
    targets_a = np.array([[[0.0, 0.0], [100.0, 100.0]]], dtype=np.float32)
    targets_b = np.array([[[0.0, 0.0], [-100.0, -100.0]]], dtype=np.float32)
    valid_mask = np.array([[True, False]])

    assert masked_mse(predictions, targets_a, valid_mask) == pytest.approx(2.5)
    assert masked_mse(predictions, targets_a, valid_mask) == masked_mse(
        predictions, targets_b, valid_mask
    )
    np.testing.assert_array_equal(
        masked_mse_gradient(predictions, targets_a, valid_mask),
        masked_mse_gradient(predictions, targets_b, valid_mask),
    )


@pytest.mark.parametrize(
    ("predictions", "targets", "mask", "error_type", "message"),
    [
        (np.zeros((1, 2)), np.zeros((1, 3)), None, ValueError, "identical shapes"),
        (np.zeros((2,)), np.zeros((2,)), None, ValueError, "must be"),
        (
            np.zeros((1, 2, 2)),
            np.zeros((1, 2, 2)),
            np.ones((1, 3)),
            ValueError,
            "valid_mask",
        ),
        (
            np.zeros((1, 2, 2)),
            np.zeros((1, 2, 2)),
            np.zeros((1, 2)),
            ValueError,
            "at least one",
        ),
        (
            np.array([[np.nan, 0.0]]),
            np.zeros((1, 2)),
            None,
            ValueError,
            "NaN",
        ),
        (
            np.array([["bad", "input"]]),
            np.array([["bad", "target"]]),
            None,
            TypeError,
            "numeric",
        ),
    ],
)
def test_masked_mse_rejects_invalid_inputs(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray | None,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        masked_mse(predictions, targets, mask)


def test_two_dimensional_loss_accepts_sample_and_element_masks() -> None:
    predictions = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32)
    targets = np.zeros_like(predictions)
    assert masked_mse(predictions, targets, np.array([True, False])) == pytest.approx(2.5)
    assert masked_mse(
        predictions,
        targets,
        np.array([[True, False], [False, False]]),
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.array([0.5]), "boolean, 0, or 1"),
        (np.array([np.nan]), "NaN"),
        (np.array(["yes"]), "boolean or numeric"),
        (np.ones((1, 1, 1)), "valid_mask"),
    ],
)
def test_two_dimensional_loss_rejects_invalid_masks(mask: np.ndarray, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        masked_mse(np.zeros((1, 2)), np.zeros((1, 2)), mask)
