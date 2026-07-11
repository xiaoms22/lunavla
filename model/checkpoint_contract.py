"""Strict validation helpers for versioned and legacy NumPy checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from lunavla.artifact_contracts import (
    NUMPY_CHECKPOINT_FORMAT,
    NUMPY_CHECKPOINT_ROOT_FIELDS,
    NUMPY_CHECKPOINT_SCHEMA_VERSION,
    NUMPY_PARAMETER_FIELDS,
    NUMPY_POLICY_FIELDS,
)


def require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, name: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    unknown = sorted(set(value) - expected)
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(unknown)}")
    missing = sorted(expected - set(value))
    if missing:
        raise ValueError(f"missing {name} field(s): {', '.join(missing)}")


def safe_json_value(value: Any, *, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite values")
        return value
    if isinstance(value, (list, tuple)):
        return [
            safe_json_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} keys must be strings")
            result[key] = safe_json_value(item, name=f"{name}.{key}")
        return result
    raise TypeError(f"{name} contains unsupported value type {type(value).__name__}")


def safe_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    result = safe_json_value(value, name="metadata")
    assert isinstance(result, dict)
    return result


def require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_numeric_tree(value: Any, *, name: str) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_numeric_tree(item, name=f"{name}[{index}]")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must contain only numeric JSON values")
    if not math.isfinite(value):
        raise ValueError(f"{name} must contain only finite values")


def load_json_object(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"checkpoint is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError("checkpoint root must be an object")
    return payload


def validate_versioned_checkpoint(
    payload: Mapping[str, Any], *, policy_name: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require_exact_fields(payload, NUMPY_CHECKPOINT_ROOT_FIELDS, name="checkpoint root")
    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != NUMPY_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported checkpoint schema_version: {schema_version!r}")
    if payload["format"] != NUMPY_CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format: {payload['format']!r}")
    policy = payload["policy"]
    if not isinstance(policy, Mapping):
        raise TypeError("checkpoint policy must be a mapping")
    require_exact_fields(policy, NUMPY_POLICY_FIELDS[policy_name], name="checkpoint policy")
    if policy["type"] != policy_name:
        raise ValueError(f"checkpoint contains policy type {policy['type']!r}")
    for dimension in ("input_dim", "action_dim", "chunk_size"):
        require_positive_int(policy[dimension], name=f"policy.{dimension}")
    if policy_name == "numpy_bc_mlp":
        require_positive_int(policy["hidden_dim"], name="policy.hidden_dim")
    parameters = policy["parameters"]
    if not isinstance(parameters, Mapping):
        raise TypeError("checkpoint policy.parameters must be a mapping")
    require_exact_fields(
        parameters,
        NUMPY_PARAMETER_FIELDS[policy_name],
        name="checkpoint policy.parameters",
    )
    normalized_parameters = dict(parameters)
    for name, value in normalized_parameters.items():
        if not isinstance(value, list):
            raise TypeError(f"policy.parameters.{name} must be a JSON array")
        _validate_numeric_tree(value, name=f"policy.parameters.{name}")
    return dict(policy), normalized_parameters, safe_metadata(payload["metadata"])


def validate_legacy_checkpoint(
    payload: Mapping[str, Any], *, policy_name: str, accepted_names: frozenset[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    dimension_fields = {"policy_name", "input_dim", "action_dim", "chunk_size"}
    if policy_name == "numpy_linear_chunk":
        required = dimension_fields | {"weights", "bias"}
    else:
        required = dimension_fields | {"hidden_dim", "weights"}
    allowed = frozenset(required | {"metadata"})
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError("unknown legacy checkpoint field(s): " + ", ".join(unknown))
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("missing legacy checkpoint field(s): " + ", ".join(missing))
    raw_policy_name = payload["policy_name"]
    if not isinstance(raw_policy_name, str) or raw_policy_name not in accepted_names:
        raise ValueError(f"legacy checkpoint contains policy_name {raw_policy_name!r}")
    for dimension in ("input_dim", "action_dim", "chunk_size"):
        require_positive_int(payload[dimension], name=f"legacy.{dimension}")
    if policy_name == "numpy_bc_mlp":
        require_positive_int(payload["hidden_dim"], name="legacy.hidden_dim")
        weights = payload["weights"]
        if not isinstance(weights, Mapping):
            raise TypeError("legacy weights must be a mapping")
        require_exact_fields(
            weights,
            NUMPY_PARAMETER_FIELDS[policy_name],
            name="legacy weights",
        )
        parameter_values = dict(weights)
    else:
        parameter_values = {"weights": payload["weights"], "bias": payload["bias"]}
    for name, value in parameter_values.items():
        if not isinstance(value, list):
            raise TypeError(f"legacy {name} must be a JSON array")
        _validate_numeric_tree(value, name=f"legacy.{name}")
    metadata = safe_metadata(payload.get("metadata", {}))
    return parameter_values, metadata
