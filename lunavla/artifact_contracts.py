"""Dependency-light golden descriptors for LunaVLA artifact schemas."""

from __future__ import annotations

from typing import Any, Final


RUN_MANIFEST_SCHEMA_VERSION: Final = 3
RUN_MANIFEST_READ_ONLY_SCHEMAS: Final = frozenset({2})
RUN_MANIFEST_SCHEMA3_FIELDS: Final = frozenset(
    {
        "schema_version",
        "run_id",
        "git_sha",
        "git_dirty",
        "source_diff_sha256",
        "config",
        "config_sha256",
        "data_sha256",
        "dataset_split",
        "data_seeds",
        "train_seeds",
        "eval_seeds",
        "python_version",
        "dependencies",
        "policy_id",
        "task_id",
        "checkpoint_sha256",
        "artifact_sha256",
        "command",
        "interventions",
        "pair_ids",
        "paired_intervals",
        "metrics",
        "design_id",
        "design_sha256",
        "condition",
        "eval_fixture",
        "eval_fixture_sha256",
        "paired_data",
        "paired_data_sha256",
        "arms",
        "pairs",
        "runtime_determinism",
    }
)
RUN_MANIFEST_SCHEMA2_FIELDS: Final = RUN_MANIFEST_SCHEMA3_FIELDS - {
    "design_id",
    "design_sha256",
    "condition",
    "eval_fixture",
    "eval_fixture_sha256",
    "paired_data",
    "paired_data_sha256",
    "arms",
    "pairs",
    "runtime_determinism",
}

TRANSFORMER_CHECKPOINT_SCHEMA_VERSION: Final = 3
TRANSFORMER_CHECKPOINT_READ_ONLY_SCHEMAS: Final = frozenset({2})
TRANSFORMER_CHECKPOINT_FORMAT: Final = "lunavla.transformer_chunk_cvae"
TRANSFORMER_IMAGE_SPATIAL_ENCODING: Final = "coordconv_xy_v1"
TRANSFORMER_SCHEMA3_FIELDS: Final = frozenset(
    {
        "schema_version",
        "format",
        "image_spatial_encoding",
        "policy_id",
        "config",
        "state_dict",
        "optimizer_state_dict",
        "latent_rng_state",
        "train_step",
        "metadata",
    }
)
TRANSFORMER_SCHEMA2_FIELDS: Final = TRANSFORMER_SCHEMA3_FIELDS - {
    "image_spatial_encoding"
}

NUMPY_CHECKPOINT_SCHEMA_VERSION: Final = 1
NUMPY_CHECKPOINT_FORMAT: Final = "lunavla.numpy_policy"
NUMPY_CHECKPOINT_ROOT_FIELDS: Final = frozenset(
    {"schema_version", "format", "policy", "metadata"}
)
NUMPY_POLICY_FIELDS: Final = {
    "numpy_linear_chunk": frozenset(
        {"type", "input_dim", "action_dim", "chunk_size", "parameters"}
    ),
    "numpy_bc_mlp": frozenset(
        {
            "type",
            "input_dim",
            "action_dim",
            "chunk_size",
            "hidden_dim",
            "parameters",
        }
    ),
}
NUMPY_PARAMETER_FIELDS: Final = {
    "numpy_linear_chunk": frozenset({"weights", "bias"}),
    "numpy_bc_mlp": frozenset({"w1", "b1", "w2", "b2"}),
}


def artifact_contract_descriptor() -> dict[str, Any]:
    """Return the canonical JSON-ready descriptor checked into ``docs/v2``."""

    return {
        "descriptor_version": 1,
        "numpy_checkpoint": {
            "current_schema": NUMPY_CHECKPOINT_SCHEMA_VERSION,
            "format": NUMPY_CHECKPOINT_FORMAT,
            "legacy_unversioned": "read-only",
            "root_fields": sorted(NUMPY_CHECKPOINT_ROOT_FIELDS),
            "policy_fields": {
                name: sorted(value) for name, value in sorted(NUMPY_POLICY_FIELDS.items())
            },
            "parameter_fields": {
                name: sorted(value)
                for name, value in sorted(NUMPY_PARAMETER_FIELDS.items())
            },
        },
        "run_manifest": {
            "current_schema": RUN_MANIFEST_SCHEMA_VERSION,
            "read_only_schemas": sorted(RUN_MANIFEST_READ_ONLY_SCHEMAS),
            "schema2_root_fields": sorted(RUN_MANIFEST_SCHEMA2_FIELDS),
            "schema3_root_fields": sorted(RUN_MANIFEST_SCHEMA3_FIELDS),
        },
        "transformer_checkpoint": {
            "current_schema": TRANSFORMER_CHECKPOINT_SCHEMA_VERSION,
            "format": TRANSFORMER_CHECKPOINT_FORMAT,
            "image_spatial_encoding": TRANSFORMER_IMAGE_SPATIAL_ENCODING,
            "read_only_schemas": sorted(TRANSFORMER_CHECKPOINT_READ_ONLY_SCHEMAS),
            "schema2_root_fields": sorted(TRANSFORMER_SCHEMA2_FIELDS),
            "schema2_visual": "rejected",
            "schema3_root_fields": sorted(TRANSFORMER_SCHEMA3_FIELDS),
        },
    }
