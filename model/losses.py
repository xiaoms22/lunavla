from __future__ import annotations

import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.generic]
Float32Array = npt.NDArray[np.float32]


def _validated_loss_arrays(
    predictions: Array,
    targets: Array,
    valid_mask: Array | None,
) -> tuple[Float32Array, Float32Array, Float32Array]:
    """Validate and normalize arrays used by the NumPy policy losses."""

    predictions_array = np.asarray(predictions)
    targets_array = np.asarray(targets)
    if predictions_array.dtype.kind not in "fiu" or targets_array.dtype.kind not in "fiu":
        raise TypeError("predictions and targets must have numeric dtypes")
    if predictions_array.shape != targets_array.shape:
        raise ValueError(
            "predictions and targets must have identical shapes; "
            f"got {predictions_array.shape} and {targets_array.shape}"
        )
    if predictions_array.ndim not in (2, 3):
        raise ValueError(
            "predictions and targets must be [batch, features] or "
            f"[batch, chunk, action]; got {predictions_array.shape}"
        )

    predictions_float = predictions_array.astype(np.float32, copy=False)
    targets_float = targets_array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(predictions_float)):
        raise ValueError("predictions contain NaN or infinite values")
    if not np.all(np.isfinite(targets_float)):
        raise ValueError("targets contain NaN or infinite values")

    if valid_mask is None:
        element_mask = np.ones(predictions_float.shape, dtype=np.float32)
    else:
        mask_array = np.asarray(valid_mask)
        if mask_array.dtype.kind not in "bfiu":
            raise TypeError("valid_mask must have a boolean or numeric dtype")
        if predictions_float.ndim == 3:
            expected_shape = predictions_float.shape[:2]
            if mask_array.shape != expected_shape:
                raise ValueError(
                    f"valid_mask must have shape {expected_shape}; got {mask_array.shape}"
                )
            element_mask = np.broadcast_to(
                mask_array.astype(np.float32, copy=False)[..., None],
                predictions_float.shape,
            )
        else:
            if mask_array.shape == predictions_float.shape:
                element_mask = mask_array.astype(np.float32, copy=False)
            elif mask_array.shape == predictions_float.shape[:1]:
                element_mask = np.broadcast_to(
                    mask_array.astype(np.float32, copy=False)[:, None],
                    predictions_float.shape,
                )
            else:
                raise ValueError(
                    "valid_mask must match [batch, features] or [batch] for a 2-D loss; "
                    f"got {mask_array.shape}"
                )
        if not np.all(np.isfinite(element_mask)):
            raise ValueError("valid_mask contains NaN or infinite values")
        if not np.all((element_mask == 0.0) | (element_mask == 1.0)):
            raise ValueError("valid_mask values must be boolean, 0, or 1")

    if float(np.sum(element_mask)) <= 0.0:
        raise ValueError("valid_mask must select at least one target element")
    return predictions_float, targets_float, element_mask


def masked_mse(
    predictions: Array,
    targets: Array,
    valid_mask: Array | None = None,
) -> float:
    """Mean squared error over valid action elements only."""

    predictions_array, targets_array, element_mask = _validated_loss_arrays(
        predictions, targets, valid_mask
    )
    squared_error = (predictions_array - targets_array) ** 2
    return float(np.sum(squared_error * element_mask) / np.sum(element_mask))


def masked_mse_gradient(
    predictions: Array,
    targets: Array,
    valid_mask: Array | None = None,
) -> Float32Array:
    """Exact gradient of :func:`masked_mse` with respect to predictions."""

    predictions_array, targets_array, element_mask = _validated_loss_arrays(
        predictions, targets, valid_mask
    )
    normalizer = float(np.sum(element_mask))
    return (2.0 * (predictions_array - targets_array) * element_mask / normalizer).astype(
        np.float32
    )
