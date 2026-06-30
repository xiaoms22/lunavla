# Data Card

## Dataset

LunaVLA uses a PushT-style demonstration generator for a tiny observation-to-action task. It can also export the same record schema to a small local JSONL file for the optional stronger-data smoke path.

## Schema

Each record contains:

- `observation`
- `action`
- `episode_id`
- `timestep`
- `success`
- `language_instruction`
- `metadata`

## Purpose

The data exists to make the full VLA-style learning loop runnable without simulator setup. It is designed for teaching, smoke tests, ablations, and project reports.

## Generation

The generator samples initial positions, defines a fixed goal, and creates expert actions that move toward that goal. See `dataset/pusht_dataset.py`.

## Local JSONL Smoke

Run `python scripts/run_jsonl_data_smoke.py` to export `data/examples/pusht_jsonl_smoke.jsonl`, reload it with `dataset.source: jsonl`, and generate a separate train/eval/report folder under `outputs/act_pusht_jsonl_smoke/`.

## Action Statistics

Run `python scripts/generate_action_statistics.py` to compute demonstration action mean, standard deviation, min/max, percentiles, and clipping fraction. The stats are saved with each run and used for reporting action scale and normalization boundaries.

## Limitations

- It is not real robot data.
- It does not contain images.
- The JSONL sample is still generated teaching data; it is not a LeRobot or real-world robot dataset.
- It does not evaluate broad manipulation generalization.

## Recommended Claim

> Used PushT-style demonstrations to validate a VLA training and rollout-evaluation workflow.
