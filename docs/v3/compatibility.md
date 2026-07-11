# v3 Alpha compatibility

- Top-level v2 contracts and serialized inputs retain their frozen meaning.
- v3 code lives under `lunavla.v3`; importing v3 does not replace `lunavla.ExperimentConfig` or
  other top-level v2 names.
- v2 configs migrate through `migrate_v2_mapping()` or `lunavla-v3 migrate-config`.
- Migration maps a flat state to `state.proprioception` and one image to `camera.primary`.
- Unknown physical units, frames, and rates become explicit `unspecified_v2` compatibility values;
  real public task adapters may not use those placeholders.
- v2 NumPy checkpoints remain read-only compatible. No v2 tensor checkpoint is relabeled as a v3
  ACT, Diffusion, or SmolVLA checkpoint.
- schema-4 revision-1 run artifacts remain verifiable but are not rewritten. New runs use a
  revision-2 `checkpoint/` directory with hashes for policy, processor, normalization, dependency,
  and training-state files.
- Alpha 1 configs load with explicit Alpha 2 defaults for optimizer, scheduler, precision,
  gradient clipping, and disabled resume.
- `act_v3` is a new policy id. v2 Transformer checkpoints stay read-only and are never relabeled as
  `act_v3`; users must train a native v3 checkpoint.
- the optional `v3-act` profile contains Torch dependencies and does not enter the v1.x or NumPy
  quickstart.
- v1.x quickstart does not gain LeRobot, LIBERO, PyTorch, or GPU dependencies from Alpha 2
  contracts.
