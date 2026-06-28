# Evaluation

VLA projects cannot be evaluated by training loss alone. A policy can fit demonstration actions and still behave poorly when its own predictions are rolled out over time.

## Metrics

- `success_rate`: fraction of episodes that reach the goal threshold.
- `mean_final_distance`: average final distance to the goal.
- `rollout_length`: number of steps before success or timeout.
- `action_smoothness`: average action delta across a rollout.
- `failure_cases`: labeled failed episodes.

## Current Implementation

`eval_vla.py` evaluates saved checkpoints on a PushT-style rollout. It writes:

- `eval_summary.json`
- optional `rollouts/episode_*.json`
- `failure_cases.jsonl`

`scripts/compare_runs.py` compares baseline and ablation run directories.

## Why Failure Taxonomy Matters

Failure labels help turn a weak result into a useful learning artifact. A failed rollout can reveal wrong direction, boundary clipping, poor coverage, oscillation, or action smoothing issues.

See `docs/failure_taxonomy.md`.
