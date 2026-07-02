# Config Schema

All runnable configs keep the same top-level shape:

- `model`: LunaVLA model contract, including model name, policy type, observation dimension, instruction dimension, action dimension, and action chunk size.
- `project_name`: stable experiment name.
- `framework`: framework label for the lightweight scaffold.
- `policy`: policy name and action chunk size.
- `task`: task id, currently `pusht`.
- `dataset`: data source, size, seed, and optional language instruction.
- `training`: device hint, batch size, steps, learning rate, seed, logging interval.
- `eval`: evaluation episodes, rollout horizon, and success distance.
- `artifacts`: output directory, checkpoint name, and report path.

The runnable path uses `dataset.source: mock_pusht` so the repository can be verified without a simulator install. Use `dataset.source: jsonl` for small custom demonstrations that follow the same record schema. The public JSONL smoke configs are `configs/act_pusht_jsonl_smoke.yaml` and `configs/act_pusht_jsonl_noisy_smoke.yaml`.

The BC tuning smoke configs are `configs/bc_pusht_cpu_smoke.yaml` and `configs/bc_pusht_hidden64_smoke.yaml`. They keep the dataset, training steps, and evaluation settings fixed while changing `policy.hidden_dim`.

Optional generator fields for mock/JSONL smoke data include `goal`, `start_low`, `start_high`, `action_gain`, `action_clip`, `action_noise_std`, and `success_distance`.

Run `python scripts/validate_configs.py` after editing configs. It checks required sections, supported dataset sources, ACT chunk-size consistency, numeric training/eval fields, and artifact paths before a training run starts.
