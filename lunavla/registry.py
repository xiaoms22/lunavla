from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping

from .contracts import VLAPolicy


PolicyFactory = Callable[[Mapping[str, Any]], VLAPolicy]
PolicyLoader = Callable[[Path, Mapping[str, Any]], VLAPolicy]

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")


def _policy_name(raw: str) -> str:
    value = str(raw).strip().lower()
    if not _NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "policy names must start with a letter and contain only lowercase letters, "
            "digits, '.', '_', or '-'"
        )
    return value


@dataclass(frozen=True)
class _PolicyRegistration:
    factory: PolicyFactory
    loader: PolicyLoader | None


class PolicyRegistry:
    """Explicit policy construction and checkpoint dispatch without eager heavy imports."""

    def __init__(self) -> None:
        self._entries: dict[str, _PolicyRegistration] = {}
        self._aliases: dict[str, str] = {}
        self._lock = RLock()

    def register(
        self,
        name: str,
        factory: PolicyFactory,
        *,
        loader: PolicyLoader | None = None,
        aliases: tuple[str, ...] = (),
        replace: bool = False,
    ) -> None:
        canonical = _policy_name(name)
        if not callable(factory):
            raise TypeError("factory must be callable")
        if loader is not None and not callable(loader):
            raise TypeError("loader must be callable")
        normalized_aliases = tuple(_policy_name(alias) for alias in aliases)
        if canonical in normalized_aliases:
            raise ValueError("a canonical policy name cannot also be its alias")
        if len(set(normalized_aliases)) != len(normalized_aliases):
            raise ValueError("aliases must be unique")

        with self._lock:
            occupied = set(self._entries) | set(self._aliases)
            if not replace and canonical in occupied:
                raise KeyError(f"policy name is already registered: {canonical}")
            conflicts = sorted(alias for alias in normalized_aliases if alias in occupied)
            if conflicts and not replace:
                raise KeyError("policy alias is already registered: " + ", ".join(conflicts))

            if replace:
                removed_targets = {
                    target
                    for target in (canonical, *normalized_aliases)
                    if target in self._entries
                }
                self._entries.pop(canonical, None)
                self._aliases.pop(canonical, None)
                for alias, target in list(self._aliases.items()):
                    if (
                        target in removed_targets
                        or target == canonical
                        or alias in normalized_aliases
                    ):
                        del self._aliases[alias]
                for alias in normalized_aliases:
                    self._entries.pop(alias, None)

            self._entries[canonical] = _PolicyRegistration(factory=factory, loader=loader)
            self._aliases.update({alias: canonical for alias in normalized_aliases})

    def resolve(self, name: str) -> str:
        normalized = _policy_name(name)
        with self._lock:
            canonical = self._aliases.get(normalized, normalized)
            if canonical not in self._entries:
                available = ", ".join(self.available()) or "none"
                raise KeyError(f"unknown policy {name!r}; available policies: {available}")
            return canonical

    def available(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._entries))

    def create(self, name: str, config: Mapping[str, Any] | None = None) -> VLAPolicy:
        canonical = self.resolve(name)
        with self._lock:
            registration = self._entries[canonical]
        policy = registration.factory(dict(config or {}))
        self._validate_policy(policy, canonical)
        return policy

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        policy_id: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> VLAPolicy:
        checkpoint = self._resolve_checkpoint(path)
        inferred = policy_id or self._checkpoint_policy_id(checkpoint)
        canonical = self.resolve(inferred)
        with self._lock:
            registration = self._entries[canonical]
        if registration.loader is None:
            raise ValueError(f"policy {canonical!r} does not provide a checkpoint loader")
        policy = registration.loader(checkpoint, dict(config or {}))
        self._validate_policy(policy, canonical)
        return policy

    @staticmethod
    def _resolve_checkpoint(path: str | Path) -> Path:
        checkpoint = Path(path)
        if checkpoint.is_dir():
            candidates = (
                checkpoint / "checkpoint.json",
                checkpoint / "checkpoint.pt",
            )
            checkpoint = next((item for item in candidates if item.is_file()), candidates[0])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        return checkpoint

    @staticmethod
    def _checkpoint_policy_id(path: Path) -> str:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError(
                "binary checkpoints require an explicit policy_id for registry dispatch"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"checkpoint is not valid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("checkpoint root must be a JSON object")

        direct = payload.get("policy_id")
        if direct:
            return str(direct)
        policy = payload.get("policy")
        if isinstance(policy, dict) and policy.get("type"):
            return str(policy["type"])
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            adapter = metadata.get("v2_adapter")
            if isinstance(adapter, dict) and adapter.get("policy_id"):
                return str(adapter["policy_id"])
        legacy = payload.get("policy_name")
        if legacy:
            return str(legacy)
        raise ValueError("checkpoint does not declare a policy id")

    @staticmethod
    def _validate_policy(policy: object, requested: str) -> None:
        if not isinstance(policy, VLAPolicy):
            raise TypeError(f"factory for {requested!r} did not return a VLAPolicy")
        if _policy_name(policy.policy_id) != requested:
            raise ValueError(
                f"factory for {requested!r} returned policy_id={policy.policy_id!r}"
            )


_DEFAULT_REGISTRY: PolicyRegistry | None = None
_DEFAULT_LOCK = RLock()


def default_policy_registry() -> PolicyRegistry:
    """Return the light default registry; optional Torch policies register explicitly."""

    global _DEFAULT_REGISTRY
    with _DEFAULT_LOCK:
        if _DEFAULT_REGISTRY is None:
            registry = PolicyRegistry()
            from .numpy_policy import register_numpy_policies

            register_numpy_policies(registry)
            _DEFAULT_REGISTRY = registry
        return _DEFAULT_REGISTRY


def register_transformer_policy(registry: PolicyRegistry) -> None:
    """Opt in to the Torch policy without importing Torch on NumPy-only paths."""

    from .transformer_policy import register_transformer_policy as register_optional

    register_optional(registry)
