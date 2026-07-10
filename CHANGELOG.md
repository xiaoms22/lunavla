# Changelog

All notable user-visible changes are recorded here. This project follows semantic versioning after v1.1.

## Unreleased — v1.1.0

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
