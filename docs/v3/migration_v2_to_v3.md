# v2 to v3 migration guide

The top-level v1/v2 API remains available. v3 lives under `lunavla.v3` and uses schema-3 configs.

Run `lunavla-v3 migrate-config SOURCE --out TARGET`, then validate with
`lunavla-v3 validate-config TARGET`. Migration preserves v2 policy, task, training seed,
evaluation and output semantics. A single image becomes `camera.primary`; flat state becomes
`state.proprioception`.

Values that cannot be inferred become `unspecified_v2`. This marker is acceptable only for the
compatibility path and is rejected by real public task adapters. Users must supply units, frames,
rates, embodiment mappings and camera order before moving a migrated config to PushT or LIBERO.

v2 NumPy checkpoints remain read-only. Tensor checkpoints are never relabelled as `act_v3`,
`diffusion_v3` or `lerobot_smolvla`; retraining is required when architectures are incompatible.
Run manifests and checkpoint envelopes retain their original revision and are verified without
rewriting historical bytes.

