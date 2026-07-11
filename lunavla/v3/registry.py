from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Callable, Mapping

from .config import ExperimentConfig
from .normalization import NormalizationStatsV1
from .policy import PolicySpecV3, VLAPolicyV3


PolicyFactoryV3 = Callable[
    [ExperimentConfig, PolicySpecV3, NormalizationStatsV1], VLAPolicyV3
]
PolicyRestorerV3 = Callable[
    [Path, ExperimentConfig, PolicySpecV3, NormalizationStatsV1], VLAPolicyV3
]
_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")


def _name(value: str) -> str:
    normalized = str(value).strip().lower()
    if not _NAME.fullmatch(normalized):
        raise ValueError("policy ids must be lowercase contract identifiers")
    return normalized


@dataclass(frozen=True)
class _Registration:
    factory: PolicyFactoryV3
    restorer: PolicyRestorerV3


class PolicyRegistryV3:
    """Strict v3 policy dispatch without eager optional dependency imports."""

    def __init__(self) -> None:
        self._entries: dict[str, _Registration] = {}
        self._lock = RLock()

    def register(
        self,
        policy_id: str,
        factory: PolicyFactoryV3,
        restorer: PolicyRestorerV3,
        *,
        replace: bool = False,
    ) -> None:
        canonical = _name(policy_id)
        if not callable(factory) or not callable(restorer):
            raise TypeError("policy factory and restorer must be callable")
        with self._lock:
            if canonical in self._entries and not replace:
                raise KeyError(f"policy is already registered: {canonical}")
            self._entries[canonical] = _Registration(factory, restorer)

    def available(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._entries))

    def _registration(self, policy_id: str) -> _Registration:
        canonical = _name(policy_id)
        with self._lock:
            if canonical not in self._entries:
                available = ", ".join(self.available()) or "none"
                raise KeyError(
                    f"unknown v3 policy {canonical!r}; available policies: {available}"
                )
            return self._entries[canonical]

    @staticmethod
    def _validate(policy: object, spec: PolicySpecV3) -> VLAPolicyV3:
        if not isinstance(policy, VLAPolicyV3):
            raise TypeError(f"factory for {spec.policy_id!r} did not return VLAPolicyV3")
        if policy.spec != spec:
            raise ValueError("policy factory returned a mismatched PolicySpecV3")
        return policy

    def create(
        self,
        config: ExperimentConfig,
        spec: PolicySpecV3,
        normalization: NormalizationStatsV1,
    ) -> VLAPolicyV3:
        registration = self._registration(spec.policy_id)
        return self._validate(registration.factory(config, spec, normalization), spec)

    def restore(
        self,
        checkpoint: str | Path,
        config: ExperimentConfig,
        spec: PolicySpecV3,
        normalization: NormalizationStatsV1,
    ) -> VLAPolicyV3:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        registration = self._registration(spec.policy_id)
        return self._validate(
            registration.restorer(path, config, spec, normalization), spec
        )

    def describe(self) -> Mapping[str, str]:
        return {name: "registered" for name in self.available()}
