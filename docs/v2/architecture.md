# v2 experimental architecture

LunaVLA v2 uses one modality-preserving boundary for the NumPy baseline and optional PyTorch policies:

```text
DatasetSource ──> Transition / PolicyBatch ──> UnifiedEngine ──> VLAPolicy
                                              │                 │
TaskEnv <──── Observation / ActionChunk <─────┘                 │
                                                               v
                                                versioned checkpoint + run artifacts
```

`Observation` always carries `state` and may carry raw instruction text and an image. Adapters must reject unsupported modalities explicitly; a state-only policy cannot silently discard pixels. `ActionChunk` always contains `[chunk_size, action_dim]` values and a Boolean validity mask.

The policy registry performs explicit construction and checkpoint dispatch. Unknown policy identifiers fail instead of falling back. Optional PyTorch and LeRobot modules are imported lazily so the dependency-light NumPy path remains usable.

The engine owns the shared train/evaluation loop, execution mode, batching, seeds, and artifact boundary. Policies own optimization and checkpoint serialization. Environments own transition dynamics. Dataset sources own ingestion. This division lets NumPy and PyTorch policies exercise the same contracts without pretending their internal algorithms are identical.

## Stability

This is an alpha contract. v2.0 stable will freeze API signatures, schema version, checkpoint format, and migration behavior only after the language, visual, and adapter gates have controlled evidence and compatibility tests.
