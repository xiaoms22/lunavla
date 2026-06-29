# Data Pipeline

LunaVLA keeps the data path deliberately small:

```text
PushT-style generator -> VLA records -> training batch -> checkpoint -> rollout eval -> report/demo
```

## Record Shape

Each record contains:

- `observation`: current state features.
- `action`: expert action for imitation learning.
- `episode_id`: demonstration id.
- `timestep`: step inside an episode.
- `success`: whether the state is close enough to the goal.
- `task_id`: structured task name, such as `pusht_mock`.
- `subtask_id`: coarse task stage, such as `approach_block`, `align_push`, `push_to_goal`, or `settle`.
- `phase`: display-friendly alias for the current task stage.
- `language_instruction`: optional task text carried with the sample.
- `metadata`: task-specific details such as distance to goal.

This Task Layer is intentionally small. It makes the IL/VA core inspectable before LunaVLA adds optional LLM planning or world-model diagnostics.

## Supported Sources

- `mock_pusht`: generated demonstrations used by the smoke test and baseline.
- `jsonl`: a local file with one VLA record per line.

The generator is intentionally simple so learners can inspect every step and modify the data distribution without setting up external robotics infrastructure.

Inspect one generated sample:

```bash
python scripts/inspect_dataset.py
```

This writes `outputs/dataset_inspection.md` and prints the raw VLA record, model input vector, and flattened action chunk target.

## Why Rollout Matters

Training loss only checks whether actions match demonstrations one step at a time. Rollout evaluation feeds predicted actions back into the environment state, so it exposes behavior drift, overshooting, and unstable action chunks.
