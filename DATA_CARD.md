# Data Card

## Dataset

LunaVLA uses a PushT-style demonstration generator for a tiny observation-to-action task.

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

## Limitations

- It is not real robot data.
- It does not contain images.
- It does not evaluate broad manipulation generalization.

## Recommended Claim

> Used PushT-style demonstrations to validate a VLA training and rollout-evaluation workflow.
