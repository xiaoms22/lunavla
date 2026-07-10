from __future__ import annotations

import warnings
from typing import Any

from .minivla_policy import NumpyLinearChunkPolicy


class ACTPolicyWrapper(NumpyLinearChunkPolicy):
    """Deprecated v1.0 alias; this policy is not an ACT Transformer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "ACTPolicyWrapper/'act' is deprecated; use NumpyLinearChunkPolicy/"
            "'numpy_linear_chunk'. The implementation is a linear NumPy policy, not ACT.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
