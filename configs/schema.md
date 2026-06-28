# Config Schema

All runnable configs keep the same top-level shape:

- `model`: MiniMind-style model contract, including model name, policy type, observation dimension, instruction dimension, action dimension, and action chunk size.
- `project_name`: stable experiment name.
- `framework`: framework label for the lightweight scaffold.
- `policy`: policy name and action chunk size.
- `task`: task id, currently `pusht`.
- `dataset`: data source, size, seed, and optional language instruction.
- `training`: device hint, batch size, steps, learning rate, seed, logging interval.
- `eval`: evaluation episodes, rollout horizon, and success distance.
- `artifacts`: output directory, checkpoint name, and report path.

The runnable path uses `dataset.source: mock_pusht` so the repository can be verified without a simulator install. Use `dataset.source: jsonl` for small custom demonstrations that follow the same record schema.
