from __future__ import annotations

from .minivla_policy import MiniVLAPolicy


class ACTPolicyWrapper(MiniVLAPolicy):
    """Small ACT-style policy wrapper with action chunk prediction."""

    policy_name = "act"
