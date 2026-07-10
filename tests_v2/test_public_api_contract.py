from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import lunavla
from lunavla.contracts import (
    DatasetSource,
    Observation,
    PolicyBatch,
    TaskEnv,
    Transition,
    VLAPolicy,
)


CONTRACT_PATH = Path(__file__).parents[1] / "docs/v2/public_api_contract.json"
PUBLIC_CONTRACTS = {
    "Observation": Observation,
    "Transition": Transition,
    "PolicyBatch": PolicyBatch,
    "VLAPolicy": VLAPolicy,
    "TaskEnv": TaskEnv,
    "DatasetSource": DatasetSource,
}


def _member_descriptor(contract: type[object]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, value in contract.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, property):
            assert value.fget is not None
            result[name] = {
                "kind": "property",
                "signature": str(inspect.signature(value.fget)),
            }
        elif inspect.isfunction(value):
            result[name] = {
                "kind": "method",
                "signature": str(inspect.signature(value)),
            }
    return result


def test_public_api_matches_machine_readable_rc_candidate_descriptor() -> None:
    descriptor = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert descriptor["schema_version"] == 1
    assert descriptor["release_stage"] == "v2.0.0-rc-candidate"
    assert set(descriptor["contracts"]) == set(PUBLIC_CONTRACTS)

    for name, contract in PUBLIC_CONTRACTS.items():
        expected = descriptor["contracts"][name]
        assert getattr(lunavla, name) is contract
        assert str(inspect.signature(contract)) == expected["signature"]
        if expected["kind"] == "dataclass":
            assert is_dataclass(contract)
            assert [item.name for item in fields(contract)] == expected["fields"]
        assert _member_descriptor(contract) == expected.get("members", {})


def test_public_array_values_are_owned_read_only_and_value_comparable() -> None:
    state = np.asarray([1.0, 2.0], dtype=np.float64)
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    observation = Observation(state, instruction="move", image=image)
    equivalent = Observation(state.copy(), instruction="move", image=image.copy())

    state[:] = -1.0
    image[:] = 0
    np.testing.assert_array_equal(observation.state, [1.0, 2.0])
    np.testing.assert_array_equal(
        observation.image,
        np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
    )
    assert observation == equivalent
    assert observation != Observation(
        np.asarray([1.0, 3.0]), instruction="move", image=equivalent.image
    )
    with pytest.raises(ValueError, match="read-only"):
        observation.state[0] = 4.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        observation.state.setflags(write=True)
    assert observation.image is not None
    with pytest.raises(ValueError, match="read-only"):
        observation.image[0, 0, 0] = 4

    targets = np.asarray([[[0.1, 0.2]]], dtype=np.float32)
    valid_mask = np.asarray([[True]], dtype=bool)
    batch = PolicyBatch((observation,), targets, valid_mask)
    batch_equivalent = PolicyBatch((equivalent,), targets.copy(), valid_mask.copy())
    targets[:] = 99.0
    valid_mask[:] = False
    assert batch == batch_equivalent
    with pytest.raises(ValueError, match="read-only"):
        batch.targets[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        batch.targets.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        batch.valid_mask[0, 0] = False


def test_transition_owns_action_and_recursively_freezes_info() -> None:
    observation = Observation(np.asarray([0.0, 1.0], dtype=np.float32))
    action = np.asarray([0.25, -0.25], dtype=np.float32)
    nested_values: list[int] = [1, 2]
    info: dict[str, Any] = {"nested": {"values": nested_values}}
    transition = Transition(
        observation,
        action,
        -0.5,
        observation,
        False,
        info,
    )
    equivalent = Transition(
        Observation(np.asarray([0.0, 1.0])),
        np.asarray([0.25, -0.25]),
        -0.5,
        Observation(np.asarray([0.0, 1.0])),
        False,
        {"nested": {"values": [1, 2]}},
    )

    action[:] = 9.0
    nested_values.append(3)
    info["new"] = True
    assert transition == equivalent
    assert transition.info == {"nested": {"values": (1, 2)}}
    with pytest.raises(ValueError, match="read-only"):
        transition.action[0] = 0.0
    with pytest.raises(TypeError):
        transition.info["new"] = False  # type: ignore[index]
    nested = transition.info["nested"]
    assert isinstance(nested, Mapping)
    with pytest.raises(TypeError):
        nested["values"] = ()  # type: ignore[index]
