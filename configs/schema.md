# Experiment Config Schema v1

Runnable configs are parsed by `trainer.config.ExperimentConfig`. Unknown fields fail instead
of being ignored.

- `schema_version`: must be `1`.
- `project_name` and `framework`: stable run identity; framework is `lunavla`.
- `task`: `pusht_style_point_reach`.
- `policy`: `type`, observation/instruction/action dimensions, `chunk_size`, and optional
  `hidden_dim`. Supported types are `numpy_linear_chunk` and `numpy_bc_mlp`.
- `dataset`: source, generation or JSONL path, data seed, split seed, episode-level split
  fractions, and instruction/task parameters.
- `training`: CPU device, batch size, optimization steps, learning rate, seed, and log interval.
- `eval`: episodes, seed(s), horizon, goal/start range, action clip, success distance, and
  `open_loop_chunk` or `receding_horizon` execution mode.
- `artifacts`: output directory, optional report path, and `checkpoint.json`.

The deprecated `model.*`, `policy.name`, `act`, `bc_mlp`, `pusht_mock`, and JSON
`checkpoint.pt` inputs are migrated with a `DeprecationWarning` for one v1.x compatibility
window. NumPy policies reject CUDA configuration explicitly.

Datasets are split by complete episode IDs into train, validation, and test sets. JSONL records
use `VLARecord`, including `next_observation` and `terminated`, and are checked for consistent
dimensions, finite values, unique `(episode_id, timestep)` pairs, and contiguous timesteps.
