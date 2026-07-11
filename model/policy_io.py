from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from .checkpoint_contract import load_json_object
from .minivla_policy import NumpyLinearChunkPolicy
from .policy_base import MiniVLAPolicyBase
from .policy_bc import NumpyBCMLPPolicy


POLICY_ALIASES = {
    "act": NumpyLinearChunkPolicy.policy_name,
    "tiny_linear": NumpyLinearChunkPolicy.policy_name,
    "linear_smoke": NumpyLinearChunkPolicy.policy_name,
    "bc": NumpyBCMLPPolicy.policy_name,
    "bc_mlp": NumpyBCMLPPolicy.policy_name,
}


def canonical_policy_type(policy_type: str, *, warn: bool = True) -> str:
    value = str(policy_type).strip().lower()
    if value in POLICY_ALIASES:
        canonical = POLICY_ALIASES[value]
        if warn:
            warnings.warn(
                f"policy alias {value!r} is deprecated; use {canonical!r}",
                DeprecationWarning,
                stacklevel=2,
            )
        return canonical
    return value


def _resolve_checkpoint(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        canonical = checkpoint_path / "checkpoint.json"
        legacy = checkpoint_path / "checkpoint.pt"
        if canonical.exists():
            return canonical
        if legacy.exists():
            return legacy
        raise FileNotFoundError(f"no checkpoint.json or legacy checkpoint.pt in {checkpoint_path}")
    if not checkpoint_path.exists() and checkpoint_path.name == "checkpoint.pt":
        canonical = checkpoint_path.with_name("checkpoint.json")
        if canonical.exists():
            warnings.warn(
                f"{checkpoint_path.name} was migrated to checkpoint.json",
                DeprecationWarning,
                stacklevel=2,
            )
            return canonical
    return checkpoint_path


def _checkpoint_policy_type(payload: dict[str, Any]) -> str:
    if "schema_version" in payload:
        policy_payload = payload.get("policy")
        if not isinstance(policy_payload, dict) or "type" not in policy_payload:
            raise ValueError("checkpoint is missing policy.type")
        return canonical_policy_type(str(policy_payload["type"]), warn=False)
    if "policy_name" not in payload:
        raise ValueError("legacy checkpoint is missing policy_name")
    return canonical_policy_type(str(payload["policy_name"]), warn=True)


def load_policy(path: str | Path) -> tuple[MiniVLAPolicyBase, dict[str, Any]]:
    checkpoint_path = _resolve_checkpoint(path)
    payload = load_json_object(checkpoint_path)
    policy_type = _checkpoint_policy_type(payload)
    if policy_type == NumpyLinearChunkPolicy.policy_name:
        return NumpyLinearChunkPolicy.load(checkpoint_path)
    if policy_type == NumpyBCMLPPolicy.policy_name:
        return NumpyBCMLPPolicy.load(checkpoint_path)
    raise ValueError(f"unsupported policy type in checkpoint: {policy_type!r}")
