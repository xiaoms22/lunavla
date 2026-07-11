from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


_ROUTES = {"none", "expert_only", "prompt_only", "dual"}
_INTERVENTION_KINDS = {"prompt", "state", "image", "data", "execution"}
_PHASES = {"train", "rollout", "train_and_rollout"}
_PROMPT_OPERATORS = {
    "control",
    "mask",
    "shuffle",
    "counterfactual",
    "layout_drift",
}
_SUPPORTED_OPERATORS = {
    "prompt": _PROMPT_OPERATORS,
    "state": _ROUTES,
    "image": {"control", "shuffle"},
    "data": {"control"},
    "execution": {"receding_horizon"},
}
_FAILURE_LAYERS = {"language", "vision", "state", "action", "execution"}
_FAILURE_PROVENANCE = {"automatic", "human"}


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    if unknown:
        raise ValueError(f"unknown field(s) in {name}: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing field(s) in {name}: {', '.join(missing)}")
    return copy.deepcopy(dict(value))


def _json_value(value: object, name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return copy.deepcopy(value)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return copy.deepcopy(value)


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _relative_path(value: object, name: str) -> str:
    path = Path(_string(value, name))
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"{name} must be a contained relative path")
    return path.as_posix()


def _render_prompt(
    *,
    instruction: str,
    public_slots: Mapping[str, Any],
    state_values: Mapping[str, Sequence[float]],
    camera_order: Sequence[str],
    assistant_target: str,
    layout_variant: str,
) -> str:
    state: dict[str, list[float]] = {}
    for name, raw_values in state_values.items():
        values = [float(item) for item in raw_values]
        if not all(math.isfinite(item) for item in values):
            raise ValueError(f"prompt state {name} contains NaN or infinite values")
        state[name] = values
    fields: list[tuple[str, Any]] = [
        ("instruction", instruction),
        ("public_slots", {key: _thaw_json(public_slots[key]) for key in sorted(public_slots)}),
        ("state", state),
        ("cameras", list(camera_order)),
        ("assistant_target", assistant_target),
    ]
    if layout_variant == "assistant_before_cameras_v1":
        fields[-2], fields[-1] = fields[-1], fields[-2]
    elif layout_variant != "canonical":
        raise ValueError(f"unsupported prompt layout_variant {layout_variant!r}")
    return json.dumps(
        dict(fields),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


@dataclass(frozen=True)
class PromptSpecV1:
    raw_instruction: str
    public_slots: Mapping[str, Any]
    state_values: Mapping[str, tuple[float, ...]]
    renderer_id: str
    renderer_version: int
    camera_order: tuple[str, ...]
    assistant_target: str
    layout_variant: str = "canonical"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromptSpecV1 schema_version must be integer 1")
        instruction = _string(self.raw_instruction, "raw_instruction")
        renderer_id = _string(self.renderer_id, "renderer_id")
        renderer_version = _integer(
            self.renderer_version, "renderer_version", positive=True
        )
        if renderer_id != "lunavla.canonical_json" or renderer_version != 1:
            raise ValueError("unsupported prompt renderer identity")
        slots = _json_value(dict(self.public_slots), "public_slots")
        if not isinstance(slots, dict):
            raise TypeError("public_slots must be a mapping")
        state: dict[str, tuple[float, ...]] = {}
        for name, raw in self.state_values.items():
            feature = _string(name, "state feature")
            if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Sequence):
                raise TypeError(f"state_values.{feature} must be a sequence")
            values = tuple(float(item) for item in raw)
            if not values or not all(math.isfinite(item) for item in values):
                raise ValueError(f"state_values.{feature} must be non-empty and finite")
            state[feature] = values
        cameras = tuple(_string(item, "camera_order item") for item in self.camera_order)
        if len(cameras) != len(set(cameras)):
            raise ValueError("camera_order cannot contain duplicates")
        target = _string(self.assistant_target, "assistant_target")
        _render_prompt(
            instruction=instruction,
            public_slots=slots,
            state_values=state,
            camera_order=cameras,
            assistant_target=target,
            layout_variant=self.layout_variant,
        )
        object.__setattr__(self, "raw_instruction", instruction)
        object.__setattr__(self, "public_slots", _freeze_json(slots))
        object.__setattr__(self, "state_values", MappingProxyType(state))
        object.__setattr__(self, "renderer_id", renderer_id)
        object.__setattr__(self, "renderer_version", renderer_version)
        object.__setattr__(self, "camera_order", cameras)
        object.__setattr__(self, "assistant_target", target)

    @property
    def rendered_text(self) -> str:
        return _render_prompt(
            instruction=self.raw_instruction,
            public_slots=self.public_slots,
            state_values=self.state_values,
            camera_order=self.camera_order,
            assistant_target=self.assistant_target,
            layout_variant=self.layout_variant,
        )

    @property
    def rendered_sha256(self) -> str:
        return hashlib.sha256(self.rendered_text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "raw_instruction": self.raw_instruction,
            "public_slots": _thaw_json(self.public_slots),
            "state_values": {key: list(value) for key, value in self.state_values.items()},
            "renderer_id": self.renderer_id,
            "renderer_version": self.renderer_version,
            "camera_order": list(self.camera_order),
            "assistant_target": self.assistant_target,
            "layout_variant": self.layout_variant,
            "rendered_text": self.rendered_text,
            "rendered_sha256": self.rendered_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PromptSpecV1":
        payload = _exact(
            value,
            {
                "schema_version",
                "raw_instruction",
                "public_slots",
                "state_values",
                "renderer_id",
                "renderer_version",
                "camera_order",
                "assistant_target",
                "layout_variant",
                "rendered_text",
                "rendered_sha256",
            },
            "PromptSpecV1",
        )
        rendered_text = payload.pop("rendered_text")
        rendered_sha256 = payload.pop("rendered_sha256")
        result = cls(**payload)
        if rendered_text != result.rendered_text or rendered_sha256 != result.rendered_sha256:
            raise ValueError("PromptSpecV1 rendered content or hash was tampered")
        return result

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class StateRouteSpecV1:
    mode: str
    state_features: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("StateRouteSpecV1 schema_version must be integer 1")
        if self.mode not in _ROUTES:
            raise ValueError(f"unsupported state route {self.mode!r}")
        features = tuple(_string(item, "state feature") for item in self.state_features)
        if not features or len(features) != len(set(features)):
            raise ValueError("state_features must be non-empty and unique")
        object.__setattr__(self, "state_features", features)

    @property
    def expert_enabled(self) -> bool:
        return self.mode in {"expert_only", "dual"}

    @property
    def prompt_enabled(self) -> bool:
        return self.mode in {"prompt_only", "dual"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "state_features": list(self.state_features),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StateRouteSpecV1":
        return cls(**_exact(value, {"schema_version", "mode", "state_features"}, "StateRouteSpecV1"))

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class InterventionSpecV1:
    arm_id: str
    kind: str
    operator: str
    phase: str
    parameters: Mapping[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("InterventionSpecV1 schema_version must be integer 1")
        arm = _string(self.arm_id, "arm_id")
        if self.kind not in _INTERVENTION_KINDS:
            raise ValueError(f"unsupported intervention kind {self.kind!r}")
        operator = _string(self.operator, "operator")
        if operator not in _SUPPORTED_OPERATORS[self.kind]:
            raise ValueError(
                f"unsupported {self.kind} intervention operator {operator!r}"
            )
        if self.phase not in _PHASES:
            raise ValueError(f"unsupported intervention phase {self.phase!r}")
        parameters = _json_value(dict(self.parameters), "intervention parameters")
        object.__setattr__(self, "arm_id", arm)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "parameters", _freeze_json(parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arm_id": self.arm_id,
            "kind": self.kind,
            "operator": self.operator,
            "phase": self.phase,
            "parameters": _thaw_json(self.parameters),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InterventionSpecV1":
        return cls(**_exact(value, {"schema_version", "arm_id", "kind", "operator", "phase", "parameters"}, "InterventionSpecV1"))

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class FailureRecordV1:
    layer: str
    label: str
    rule_id: str
    provenance: str
    primary_cause: bool | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("FailureRecordV1 schema_version must be integer 1")
        if self.layer not in _FAILURE_LAYERS:
            raise ValueError(f"unsupported failure layer {self.layer!r}")
        label = _string(self.label, "failure label")
        rule_id = _string(self.rule_id, "failure rule_id")
        if self.provenance not in _FAILURE_PROVENANCE:
            raise ValueError(f"unsupported failure provenance {self.provenance!r}")
        if self.primary_cause is not None and not isinstance(self.primary_cause, bool):
            raise TypeError("primary_cause must be boolean or null")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "rule_id", rule_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layer": self.layer,
            "label": self.label,
            "rule_id": self.rule_id,
            "provenance": self.provenance,
            "primary_cause": self.primary_cause,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FailureRecordV1":
        return cls(
            **_exact(
                value,
                {
                    "schema_version", "layer", "label", "rule_id", "provenance",
                    "primary_cause",
                },
                "FailureRecordV1",
            )
        )


@dataclass(frozen=True)
class DonorRecordV1:
    recipient_id: str
    donor_id: str
    split: str
    step_index: int | None
    recipient_content_sha256: str
    donor_content_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("DonorRecordV1 schema_version must be integer 1")
        recipient = _string(self.recipient_id, "recipient_id")
        donor = _string(self.donor_id, "donor_id")
        if recipient == donor:
            raise ValueError("donor cannot reference the recipient episode")
        split = _string(self.split, "split")
        if split not in {"train", "validation", "test", "evaluation"}:
            raise ValueError("unsupported donor split")
        step = self.step_index
        if step is not None:
            step = _integer(step, "step_index")
        for name in ("recipient_content_sha256", "donor_content_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(item not in "0123456789abcdef" for item in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.recipient_content_sha256 == self.donor_content_sha256:
            raise ValueError("donor content must differ from recipient content")
        object.__setattr__(self, "recipient_id", recipient)
        object.__setattr__(self, "donor_id", donor)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "step_index", step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recipient_id": self.recipient_id,
            "donor_id": self.donor_id,
            "split": self.split,
            "step_index": self.step_index,
            "recipient_content_sha256": self.recipient_content_sha256,
            "donor_content_sha256": self.donor_content_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DonorRecordV1":
        return cls(
            **_exact(
                value,
                {
                    "schema_version", "recipient_id", "donor_id", "split", "step_index",
                    "recipient_content_sha256", "donor_content_sha256",
                },
                "DonorRecordV1",
            )
        )


@dataclass(frozen=True)
class DonorBankV1:
    modality: str
    split: str
    donor_seed: int
    records: tuple[DonorRecordV1, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("DonorBankV1 schema_version must be integer 1")
        if self.modality not in {"instruction", "image"}:
            raise ValueError("donor modality must be instruction or image")
        split = _string(self.split, "donor bank split")
        records = tuple(self.records)
        if not records or any(not isinstance(item, DonorRecordV1) for item in records):
            raise ValueError("donor bank requires DonorRecordV1 records")
        if any(item.split != split for item in records):
            raise ValueError("donor records cannot cross splits")
        keys = [(item.recipient_id, item.step_index) for item in records]
        if len(keys) != len(set(keys)):
            raise ValueError("donor bank contains duplicate recipient/step records")
        if self.modality == "instruction" and any(item.step_index is not None for item in records):
            raise ValueError("instruction donor records cannot declare step_index")
        if self.modality == "image" and any(item.step_index is None for item in records):
            raise ValueError("image donor records require step_index")
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "donor_seed", _integer(self.donor_seed, "donor_seed"))
        object.__setattr__(self, "records", records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "modality": self.modality,
            "split": self.split,
            "donor_seed": self.donor_seed,
            "records": [item.to_dict() for item in self.records],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DonorBankV1":
        payload = _exact(
            value,
            {"schema_version", "modality", "split", "donor_seed", "records"},
            "DonorBankV1",
        )
        payload["records"] = tuple(DonorRecordV1.from_mapping(item) for item in payload["records"])
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class DiagnosticDesignV1:
    design_id: str
    base_config: str
    output_dir: str
    train_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    routes: tuple[StateRouteSpecV1, ...]
    interventions: tuple[InterventionSpecV1, ...]
    donor_seed: int
    analysis_seed: int
    bootstrap_samples: int
    counterfactual_transform_id: str | None
    reduced_design: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("DiagnosticDesignV1 schema_version must be integer 1")
        design_id = _string(self.design_id, "design_id")
        train_seeds = tuple(_integer(item, "train seed") for item in self.train_seeds)
        evaluation_seeds = tuple(
            _integer(item, "evaluation seed") for item in self.evaluation_seeds
        )
        if not train_seeds or len(train_seeds) != len(set(train_seeds)):
            raise ValueError("train_seeds must be non-empty and unique")
        if not evaluation_seeds or len(evaluation_seeds) != len(set(evaluation_seeds)):
            raise ValueError("evaluation_seeds must be non-empty and unique")
        routes = tuple(self.routes)
        arms = tuple(self.interventions)
        if not routes or any(not isinstance(item, StateRouteSpecV1) for item in routes):
            raise ValueError("routes must contain StateRouteSpecV1 records")
        if len({item.mode for item in routes}) != len(routes):
            raise ValueError("diagnostic routes cannot contain duplicates")
        if not arms or any(not isinstance(item, InterventionSpecV1) for item in arms):
            raise ValueError("interventions must contain InterventionSpecV1 records")
        if any(item.phase != "rollout" for item in arms):
            raise ValueError("Beta 1 interventions must use rollout phase")
        kinds = {item.kind for item in arms}
        if len(kinds) != 1 or not kinds <= {"prompt", "image"}:
            raise ValueError("a diagnostic design must contain one prompt or image suite")
        if len({item.arm_id for item in arms}) != len(arms):
            raise ValueError("intervention arm IDs cannot contain duplicates")
        if "control" not in {item.operator for item in arms}:
            raise ValueError("interventions must include control")
        counterfactual = self.counterfactual_transform_id
        if any(item.operator == "counterfactual" for item in arms):
            counterfactual = _string(counterfactual, "counterfactual_transform_id")
        elif counterfactual is not None:
            raise ValueError("counterfactual_transform_id requires a counterfactual arm")
        if not isinstance(self.reduced_design, bool):
            raise TypeError("reduced_design must be boolean")
        object.__setattr__(self, "design_id", design_id)
        object.__setattr__(self, "base_config", _relative_path(self.base_config, "base_config"))
        object.__setattr__(self, "output_dir", _relative_path(self.output_dir, "output_dir"))
        object.__setattr__(self, "train_seeds", train_seeds)
        object.__setattr__(self, "evaluation_seeds", evaluation_seeds)
        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "interventions", arms)
        object.__setattr__(self, "donor_seed", _integer(self.donor_seed, "donor_seed"))
        object.__setattr__(self, "analysis_seed", _integer(self.analysis_seed, "analysis_seed"))
        object.__setattr__(self, "bootstrap_samples", _integer(self.bootstrap_samples, "bootstrap_samples", positive=True))
        object.__setattr__(self, "counterfactual_transform_id", counterfactual)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "design_id": self.design_id,
            "base_config": self.base_config,
            "output_dir": self.output_dir,
            "train_seeds": list(self.train_seeds),
            "evaluation_seeds": list(self.evaluation_seeds),
            "routes": [item.to_dict() for item in self.routes],
            "interventions": [item.to_dict() for item in self.interventions],
            "donor_seed": self.donor_seed,
            "analysis_seed": self.analysis_seed,
            "bootstrap_samples": self.bootstrap_samples,
            "counterfactual_transform_id": self.counterfactual_transform_id,
            "reduced_design": self.reduced_design,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DiagnosticDesignV1":
        payload = _exact(
            value,
            {
                "schema_version", "design_id", "base_config", "output_dir",
                "train_seeds", "evaluation_seeds", "routes", "interventions",
                "donor_seed", "analysis_seed", "bootstrap_samples",
                "counterfactual_transform_id", "reduced_design",
            },
            "DiagnosticDesignV1",
        )
        payload["routes"] = tuple(StateRouteSpecV1.from_mapping(item) for item in payload["routes"])
        payload["interventions"] = tuple(InterventionSpecV1.from_mapping(item) for item in payload["interventions"])
        return cls(**payload)

    def sha256(self) -> str:
        return _stable_hash(self.to_dict())
