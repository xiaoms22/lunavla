# v3 Alpha compatibility

- Top-level v2 contracts and serialized inputs retain their frozen meaning.
- v3 code lives under `lunavla.v3`; importing v3 does not replace `lunavla.ExperimentConfig` or
  other top-level v2 names.
- v2 configs migrate through `migrate_v2_mapping()` or `lunavla-v3 migrate-config`.
- Migration maps a flat state to `state.proprioception` and one image to `camera.primary`.
- Unknown physical units, frames, and rates become explicit `unspecified_v2` compatibility values;
  real public task adapters may not use those placeholders.
- v2 NumPy checkpoints remain read-only compatible. v3 uses a schema-4 envelope around the strict
  existing checkpoint rather than fabricating tensor conversions.
- v1.x quickstart does not gain LeRobot, LIBERO, PyTorch, or GPU dependencies from Alpha 1.
