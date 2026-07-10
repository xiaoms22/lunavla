# v2 release-candidate architecture

LunaVLA v2 uses one modality-preserving boundary for the NumPy baseline and optional PyTorch policies:

```text
DatasetSource ──> Transition / PolicyBatch ──> UnifiedEngine ──> VLAPolicy
                                              │                 │
TaskEnv <──── Observation / ActionChunk <─────┘                 │
                                                               v
                                                versioned checkpoint + run artifacts
```

`Observation` always carries `state` and may carry raw instruction text and an image. Adapters must reject unsupported modalities explicitly; a state-only policy cannot silently discard pixels. `ActionChunk` always contains `[chunk_size, action_dim]` values and a Boolean validity mask.

## Value ownership and equality

The public `Observation`, `Transition`, and `PolicyBatch` boundaries own their array values. Constructors copy arrays, normalize their documented dtype, and expose the copies as read-only. Callers can therefore reuse or mutate their input buffers without changing an already-created contract value. Policies and adapters that need writable storage must make their own explicit copy.

`Transition.info` is recursively detached from caller-owned containers: mappings become read-only mappings, sequences become tuples, and NumPy scalar/array metadata becomes immutable Python values. Contract equality compares normalized array shape, dtype, and contents. It never relies on the ambiguous truth value of an array.

## Environment lifecycle

`TaskEnv` consists of `reset`, `step`, and `close`. The engine owns an environment passed to `Engine.evaluate`: it calls `close()` exactly once after the complete evaluation, including when reset, intervention, prediction, or stepping fails. A caller must not reuse that environment after evaluation. Synthetic, language, and rendered environments implement a no-op close; external adapters release their wrapped environment resources. The experiment runner uses an exactly-once guard around its own `finally` cleanup, so both direct engine use and runner orchestration are safe without closing the underlying environment twice.

The policy registry performs explicit construction and checkpoint dispatch. Unknown policy identifiers fail instead of falling back. Optional PyTorch and LeRobot modules are imported lazily so the dependency-light NumPy path remains usable.

The engine owns the shared train/evaluation loop, execution mode, batching, seeds, and artifact boundary. Policies own optimization and checkpoint serialization. Environments own transition dynamics. Dataset sources own ingestion. This division lets NumPy and PyTorch policies exercise the same contracts without pretending their internal algorithms are identical.

## Versioned configuration contracts

`ExperimentConfig` schema 2 rejects unknown fields at every section and inside
`task.parameters`, `dataset.parameters`, and `evaluation.parameters`. Parameter
allowlists are task- and dataset-specific: options that would otherwise be ignored
cannot cross into training. Types and ranges are normalized during parsing, before an
output directory or model is created.

A resolved config is deeply immutable. Its mappings are read-only and its sequences
are tuples, including nested parameter mappings and seed/image-shape sequences.
`to_dict()` returns a detached tree of ordinary dictionaries and lists for YAML/JSON,
and config hashing is defined over that representation. The v1.1 migration accepts
only an integer schema version; booleans, floats, and strings are not aliases for a
version number.

The machine-readable compatibility surface for both `ExperimentConfig` and
`EvidenceDesign` is tracked in
[`contracts/config-design-schema.json`](contracts/config-design-schema.json).
Tests compare that descriptor with the parser registries, so changing a public field,
version, required parameter, or enum requires an explicit descriptor update.

## Stability

These interfaces are frozen for `v2.0.0-rc.1`. The machine-readable [`public_api_contract.json`](public_api_contract.json) records the public fields and `inspect.signature` values for `ActionChunk`, `Observation`, `VLAPolicy`, `TaskEnv`, `DatasetSource`, `PolicyBatch`, and `Transition`; CI rejects accidental drift. The complete schema and compatibility matrix is in [`contract_freeze.md`](contract_freeze.md). v2.0 stable must preserve this boundary unless an explicit versioned migration is added.
