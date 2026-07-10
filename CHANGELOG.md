# Changelog

All notable user-visible changes are recorded here. This project follows semantic versioning after v1.1.

## Unreleased — v2.0.0-alpha.1

### Added

- Added experimental `Observation`, `Transition`, `VLAPolicy`, `TaskEnv`, and `DatasetSource` contracts plus a shared policy registry and train/evaluation engine.
- Added a v1 NumPy adapter and an optional PyTorch action-query chunk Transformer with CVAE/KL, padding masks, checkpoint round trips, and temporal ensembling.
- Added instruction-dependent task fixtures with held-out paraphrases and mask/shuffle/counterfactual pair construction.
- Added rendered direct-reach and waypoint-reach fixtures, state-only controls, image occlusion/shuffle pairing, and a lazy LeRobot adapter.
- Added a strict schema-v2 configuration contract, v1.1-to-v2 migration command, versioned dependency lock, CPU gates, and an isolated manual GPU workflow.
- Added hash-locked Linux CPU profiles, a guarded release-evidence entry point, and a real Transformer-through-Engine integration test.

### Hardened

- Invalid task, modality-ablation, state-only, rendered-image, and Transformer head configurations now fail before training starts.
- CPU CI rejects CUDA-only packages and verifies exact NumPy 2.2.6, PyTorch 2.11 CPU, and torchvision 0.26 dependencies.
- Release candidates must come from a clean checkout at an explicitly supplied immutable Git SHA.

### Boundaries

- v2 modality adapters are experimental inputs. They are not evidence that language or vision contributes until controlled paired confidence intervals exclude zero.
- Public v2 APIs and checkpoint schemas are not frozen until the stable release gate.
- Alpha evidence establishes implementation integrity only; it is not a language, vision, LeRobot performance, or CUDA support claim.

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
