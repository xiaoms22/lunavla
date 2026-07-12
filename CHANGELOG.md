# Changelog

All notable user-visible changes are recorded here. This project follows semantic versioning after v1.1.

## v3.0.0-beta.1 — diagnostic integration candidate

### Added

- Added canonical prompt rendering, explicit state routes, typed interventions, same-split donor banks, and train/eval/deploy preprocessing parity checks.
- Added schema-4 revision-3 diagnostic manifests, EvidenceManifest v2, per-step traces, independent matrix recomputation, and static reports.
- Added the reduced 40-pair NumPy route/prompt fixture and independent four-pair ACT image-shuffle fixture with deterministic synthetic thumbnails.

### Boundaries

- Every Beta 1 result remains `claim_allowed=false`; reduced fixtures establish framework integrity only.
- No instruction-following, state-route, visual-contribution, task-performance, or robot-deployment claim is made.
- SmolVLA remains conformance-only with unverified weight licensing and pretrained access disabled.

## v3.0.0-alpha.3 — code-only release candidate

### Added

- Added a hosted CPU release path for ACT, Diffusion, and the SmolVLA public-API conformance adapter.
- Pinned the PEP 517 build backend and disabled build isolation so wheel/sdist bytes are reproducible from the recorded release lock.
- Added signed-tag verification, SBOM, provenance, evidence archive, exact checksums, and wheel-install smoke tests.

### Boundaries

- SmolVLA remains `NOASSERTION/unverified`, pretrained loading remains disabled, and no model weight or checkpoint is included.
- The release makes no policy-performance, modality, task, or robot-deployment claim and is not published to PyPI.

## v3.0.0-alpha.2 — signed candidate not released

### Added

- Added versioned model-weight license review, single-GPU validation, and release-candidate contracts.
- Added hash-locked Linux CUDA 12.8 and release CPU profiles plus guarded dispatcher infrastructure.
- Added a dormant official SmolVLA loader, optimizer, checkpoint and restore path that is reachable only after a separate reviewed gate-opening change.

### Boundaries

- The model-weight license remains unverified, the pretrained gate remains disabled, and no qualifying self-hosted runner is registered.
- The signed tag was preserved but no GitHub release was created: reproducibility review found that build isolation resolved an unrecorded setuptools version inconsistent with the lock and SBOM.
- No weight download, PyPI publication, or performance claim was created; the corrected non-destructive successor is Alpha 3.

## v2.0.0 — 2026-07-11

### Added

- Added a fail-closed stable release profile that runs the complete language and visual matrices again on the protected `main` merge SHA, for exactly 15 training runs and 960 arm-episodes.
- Added an isolated generated-results boundary so post-merge evidence cannot overwrite the registered Alpha/Beta snapshots under `results/v2/`.
- Added a same-workflow real LeRobot integration gate whose strict manifest, SHA-256, GitHub provenance bundle, verified signer workflow, source ref, and source digest are bound into the stable candidate.
- Added a combined evidence archive, release-wide checksum verification, stable SBOM/distribution bindings, and the exact `2.0.0` package-to-`v2.0.0` tag contract.

### Boundaries

- Stable evidence remains CPU Linux only, does not upload to PyPI, and does not require or claim GPU support.
- Language and visual claims are copied from the new post-merge `EvidenceManifest` results and remain fail-closed; statistical failure to establish a modality contribution is reported rather than hidden.
- The machine-readable API/schema descriptors still identify `v2.0.0-rc.1` as the point at which the stable public boundary was frozen.

## v2.0.0-rc.1 — 2026-07-11

### Added

- Added experimental `Observation`, `Transition`, `VLAPolicy`, `TaskEnv`, and `DatasetSource` contracts plus a shared policy registry and train/evaluation engine.
- Added a v1 NumPy adapter and an optional PyTorch action-query chunk Transformer with CVAE/KL, padding masks, checkpoint round trips, and temporal ensembling.
- Added instruction-dependent task fixtures with held-out paraphrases and mask/shuffle/counterfactual pair construction.
- Added rendered direct-reach and waypoint-reach fixtures, state-only controls, image occlusion/shuffle pairing, and a lazy LeRobot adapter.
- Added a strict schema-v2 configuration contract, v1.1-to-v2 migration command, versioned dependency lock, CPU gates, and an isolated manual GPU workflow.
- Added hash-locked Linux CPU profiles, a guarded release-evidence entry point, and a real Transformer-through-Engine integration test.
- Added five-seed controlled language and visual studies, hierarchical paired bootstrap reporting, immutable publication registries, and read-only verified review snapshots.
- Added the pinned official LeRobot PushT episode adapter, full 161-frame decode validation, one bounded optimizer step, and a headless Gym PushT smoke.
- Added frozen runtime, configuration, manifest, and checkpoint descriptors plus an RC release profile.

### Hardened

- Invalid task, modality-ablation, state-only, rendered-image, and Transformer head configurations now fail before training starts.
- CPU CI rejects CUDA-only packages and verifies exact NumPy 2.2.6, PyTorch 2.11 CPU, and torchvision 0.26 dependencies.
- Release candidates must come from a clean checkout at an explicitly supplied immutable Git SHA.
- Public arrays, resolved configs, and manifests now have explicit deep ownership/immutability semantics; environments close exactly once.
- Unknown nested parameters, malformed digests, unsafe metadata, non-finite values, boolean schema versions, and incompatible legacy checkpoints now fail closed.

### Boundaries

- The completed language and visual studies did not open their predeclared contribution gates; instruction-following and visual-control contribution remain not established.
- The public runtime API, config schema 2, EvidenceDesign schema 1, RunManifest schema 3, Transformer checkpoint schema 3, and NumPy checkpoint schema 1 are frozen for the RC.
- LeRobot integration establishes adapter connectivity only; it is not a PushT performance, real-robot, production, or CUDA-support claim.

## v1.1.0 — 2026-07-10

### Changed

- Reframed v1.x as a state-based NumPy imitation-learning teaching core, not a visual VLA, ACT Transformer, real PushT benchmark, or robot-deployment project.
- Renamed the implementations to `numpy_linear_chunk` and `numpy_bc_mlp`, and the task to `pusht_style_point_reach`.
- Added explicit action-chunk masks and receding-horizon/open-loop execution modes.
- Made task dynamics, dimensions, seeds, and execution behavior configuration-driven.
- Added episode-disjoint dataset splits, stricter JSONL validation, versioned checkpoints, and run manifests.
- Replaced static result claims with manifest-rendered controlled evidence.

### Deprecated

- Policy name `act` and task name `pusht_mock`; both remain aliases for v1.1 only.
- `predict_action()`; use `predict_chunk()` and select the desired execution behavior.
- Reading legacy JSON checkpoints with a `.pt` suffix; v1.1 writes `checkpoint.json`.

### Corrected

- Historical v1.0 success and BC-versus-chunk numbers were not controlled evidence of a chunking effect. They have been moved to the historical archive and are not used as v1.1 claims.
- Removed PyTorch/CUDA prerequisites and unsupported Diffusion Policy evidence from the current project story.

## v1.0.0 — historical snapshot

The initial teaching release and its completion reports are preserved in `docs/archive/v1.0/`. Their terminology and result interpretation predate the v1.1 corrections.
