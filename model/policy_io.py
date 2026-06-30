from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .minivla_policy import MiniVLAPolicy
from .policy_base import MiniVLAPolicyBase
from .policy_bc import BCMLPPolicy


def load_policy(path: str | Path) -> tuple[MiniVLAPolicyBase, dict[str, Any]]:
    checkpoint_path = Path(path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    policy_name = str(payload.get("policy_name", "tiny_linear"))
    if policy_name == BCMLPPolicy.policy_name:
        return BCMLPPolicy.load(checkpoint_path)
    return MiniVLAPolicy.load(checkpoint_path)
