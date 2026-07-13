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
- Alpha 1 configs load with explicit Alpha 2 contract defaults for optimizer, scheduler, precision,
  gradient clipping, and disabled resume.
- ExperimentConfig schema 3 revision 1 retains its previous resolved serialization and hash.
  Revision 2 adds strict root-level `prompt` and `routing` sections; diagnostic execution requires
  revision 2, an instruction-consuming policy, and receding-horizon evaluation.
- `act_v3` is a new policy id. v2 Transformer checkpoints stay read-only and are never relabeled as
  `act_v3`; users must train a native v3 checkpoint.
- the optional `v3-act` profile contains Torch dependencies and does not enter the v1.x or NumPy
  quickstart.
- `diffusion_v3` consumes image and state but not instruction; the config must explicitly declare
  `unused_modalities: [instruction]`. Its optional profile pins LeRobot 0.6.0, Diffusers 0.35.2,
  Transformers 5.5.4, and Accelerate 1.14.0 without entering the v1.x/NumPy quickstart.
- Diffusion checkpoints contain upstream `save_pretrained()` model and processor artifacts plus
  strict optimizer and RNG state. Scheduler, inference-step, normalization, processor, or lock
  drift fails restore or run verification.
- `lerobot_smolvla` is an adapter-contract policy id, not a verified pretrained policy. The fixed
  upstream revision and model hash are recorded, while the weight license remains `unverified` and
  `pretrained_enabled=false`. Default registry creation, optimizer steps, and restore fail closed;
  CI uses only dependency-injected implementations of the documented public methods.
- Alpha 3 uses package `3.0.0a3` and a hosted CPU code-only release. The SmolVLA GPU workflow moves
  to v3.1 and continues to fail closed. Historical v2 manifests retain their original package
  identity; v2 release tooling rejects a v3 package rather than relabeling old assets.
- diagnostic runs use schema-4 revision-3 manifests and EvidenceManifest v2. Revision-1 and
  revision-2 runs remain verifiable. Reduced or incomplete studies always fail closed, and the
  Beta 1 framework does not relabel old runs as controlled evidence.
- the revision-3/V2 diagnostic formats are still Draft contracts. They now bind canonical
  train/eval/deploy preprocessing parity, normalized cell contracts, typed instruction/image
  donors, gate reasons, and optional synthetic-thumbnail hashes without changing revision-1/2
  artifact readers.
- v1.x quickstart does not gain LeRobot, LIBERO, PyTorch, or GPU dependencies from Alpha 3 or
  Beta 1 diagnostic contracts.
- RC preparation adds machine-readable stable evidence and release-candidate contracts only. It
  does not change v1/v2 serialization or promote any Draft v3 contract to a frozen stable API.
- CPU profile and teaching-evidence contracts do not add SmolVLA weights, CUDA, real datasets or
  LIBERO runtime dependencies to v1.x or the NumPy quickstart.
